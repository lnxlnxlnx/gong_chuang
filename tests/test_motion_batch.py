"""跨模块"各自发出、一起等"：机械臂多轴 + 夹爪 + 转盘，一个 join 点。

这是 ``robot.motion`` 的验收用例：三个真模块（真的 RobotArm / GripperServo /
TurntableServo，只有串口和时钟是假的）用 ``wait=False`` 各自发命令，再由一个
``MotionBatch.wait()`` 统一收尾。全程离线。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from modules.arm.config import ArmConfig, AxisConfig
from modules.arm.drive.kinematics import CylindricalPose
from modules.arm.runtime import RobotArm
from modules.motion import MotionBatch, MotionController
from modules.motor.drive.emm42 import Direction
from modules.motor.drive.motion import DEFAULT_INTER_FRAME_SECONDS, estimate_move_seconds
from modules.servo.config import GripperConfig, TurntableConfig
from modules.servo.drive.bus_servo import BusServoConfig
from modules.servo.drive.turntable import TurntableServo
from modules.servo.runtime import GripperServo
from robot import Robot

POSITION_COMMAND = 0xFD
FRAME_TAIL = 0x6B
BUFFERED_FOR_TRIGGER = 1
PULSES_PER_REVOLUTION = 3200
SAFE_TEST_RPM = 10
SAFE_TEST_ACCELERATION = 5
SYNC_TRIGGER_FRAME = bytes((0x00, 0xFF, 0x66, FRAME_TAIL))

ARM_CONFIG = ArmConfig(
    pulses_per_revolution=PULSES_PER_REVOLUTION,
    rpm=SAFE_TEST_RPM,
    acceleration=SAFE_TEST_ACCELERATION,
    default_pulses=800,
    max_pulses=9000,
    axes={
        "base": AxisConfig(motor_id=5, reduction_ratio=6),
        # 两个直线轴带上毫米标定，位姿才算得出来（缺一个就整体变未知）。
        "lift": AxisConfig(motor_id=6, mm_per_motor_rev=61.0),
        "extension": AxisConfig(motor_id=7, mm_per_motor_rev=61.0),
    },
)
GRIPPER_CONFIG = GripperConfig(
    servo_id=1, open_position=200, closed_position=0, duration_ms=1000
)
TURNTABLE_CONFIG = TurntableConfig(
    servo_id=2,
    duration_ms=1500,
    stations={"home": 0, "station_2": 500, "station_3": 1000},
)
SERVO_BUS = BusServoConfig(port="FAKE_SERVO", baudrate=115200, timeout_seconds=0.1)


class FakeClock:
    """假单调时钟：睡多久就往前走多久，所以票据的等待会**到点**而不是空转。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def total(self) -> float:
        return sum(self.slept)


class FakeEmm42Bus:
    """只写不读的假步进总线：机械臂三轴是 ``expect_ack=False``，不需要回包。"""

    def __init__(self) -> None:
        self.writes: list = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        return b""

    def close(self) -> None:
        self.is_open = False


def buffered_frame(*, motor_id: int, direction: Direction, pulses: int) -> bytes:
    return (
        bytes((motor_id, POSITION_COMMAND, int(direction)))
        + SAFE_TEST_RPM.to_bytes(2, "big")
        + bytes((SAFE_TEST_ACCELERATION,))
        + pulses.to_bytes(4, "big")
        + bytes((0, BUFFERED_FOR_TRIGGER, FRAME_TAIL))
    )


class CrossModuleMotionTests(unittest.TestCase):
    """三个模块同时发命令，一个 ``batch.wait()`` 收尾。"""

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.bus = FakeEmm42Bus()
        self.arm = RobotArm._from_settings(
            config=ARM_CONFIG,
            port="FAKE_STEPPER",
            bus=self.bus,
            sleep=self.clock.sleep,
            clock=self.clock,
        )
        self.bus.writes.clear()

        self.gripper_servo = Mock()
        self.gripper_servo.read_position.return_value = 40  # 闭到 40：夹住物料了
        self.gripper = GripperServo(
            config=GRIPPER_CONFIG,
            port=SERVO_BUS.port,
            servo=self.gripper_servo,
            sleep=self.clock.sleep,
            clock=self.clock,
        )

        self.turntable_servo = Mock()
        self.turntable_servo.read_position.return_value = 500  # 正好停在 station_2
        self.turntable = TurntableServo(
            "",
            config=TURNTABLE_CONFIG,
            bus_config=SERVO_BUS,
            servo=self.turntable_servo,
            sleep=self.clock.sleep,
            clock=self.clock,
        )

        self.robot = Robot(
            arm=self.arm,
            gripper=self.gripper,
            turntable=self.turntable,
            motion=MotionController(clock=self.clock, sleep=self.clock.sleep),
        )

    def test_the_readme_style_batch_starts_three_modules_and_joins_once(self) -> None:
        robot = self.robot
        batch = robot.motion.begin()
        batch.add(robot.arm.move_axes({"lift": 4000, "extension": -8000}, wait=False))
        batch.add(robot.gripper.set_position(40, wait=False))
        batch.add(robot.turntable.go_to_station("station_2", wait=False))

        # 发命令阶段：三轴缓存 + 一条广播触发；两个舵机各收到一条定点命令。
        self.assertEqual(
            self.bus.writes,
            [
                buffered_frame(motor_id=6, direction=Direction.CW, pulses=4000),
                buffered_frame(motor_id=7, direction=Direction.CCW, pulses=8000),
                SYNC_TRIGGER_FRAME,
            ],
        )
        self.gripper_servo.set_position.assert_called_once_with(40, GRIPPER_CONFIG.duration_ms)
        self.turntable_servo.set_position.assert_called_once_with(500, TURNTABLE_CONFIG.duration_ms)
        self.assertEqual(len(batch), 3)
        self.assertTrue(all(ticket.pending for ticket in batch))

        report = batch.wait()

        self.assertTrue(report.ok)
        self.assertEqual(
            [item.subject for item in report.items],
            ["arm:lift+extension", "gripper:set", "turntable:station_2"],
        )
        # 整组只等到**最慢**的那一项（这里是最长的伸缩轴），不是三段等待之和。
        slowest = max(
            estimate_move_seconds(
                8000, SAFE_TEST_RPM, PULSES_PER_REVOLUTION, SAFE_TEST_ACCELERATION
            ),
            GRIPPER_CONFIG.duration_ms / 1000.0 + 0.2,
            TURNTABLE_CONFIG.duration_ms / 1000.0 + 0.2,
        )
        self.assertAlmostEqual(
            self.clock.total(), DEFAULT_INTER_FRAME_SECONDS + slowest, places=6
        )
        self.assertLess(self.clock.total(), GRIPPER_CONFIG.duration_ms / 1000.0 + slowest + 1.0)

    def test_the_join_happens_on_the_caller_thread_so_no_thread_is_started(self) -> None:
        """动作在硬件上并行，join 在调用线程上：这是本项目"不开线程"的约定。"""
        import threading

        before = {thread.name for thread in threading.enumerate()}
        batch = self.robot.motion.begin()
        batch.add(self.robot.arm.move_axes({"lift": 800}, wait=False))
        batch.add(self.robot.gripper.set_position(40, wait=False))
        batch.wait()

        self.assertEqual({thread.name for thread in threading.enumerate()}, before)

    def test_a_failing_item_does_not_stop_the_others(self) -> None:
        """一项收尾失败不阻断其它项——和 ``close`` 是同一条规矩。"""
        self.turntable_servo.read_position.side_effect = RuntimeError("舵机不回包")
        batch = self.robot.motion.begin()
        batch.add(self.robot.arm.move_axes({"lift": 800}, wait=False))
        batch.add(self.robot.turntable.go_to_station("station_2", wait=False))

        with self.assertRaises(Exception) as caught:
            batch.wait()

        report = caught.exception.report
        self.assertEqual([item.ok for item in report.items], [True, False])
        self.assertIn("舵机不回包", str(caught.exception))

    def test_each_ticket_can_also_be_waited_on_its_own(self) -> None:
        arm_ticket = self.robot.arm.move_axes({"lift": 800}, wait=False)
        gripper_ticket = self.robot.gripper.set_position(40, wait=False)

        gripper_report = gripper_ticket.wait()
        arm_reports = arm_ticket.wait()

        self.assertEqual(gripper_report.target, 40)
        self.assertEqual([item.motor_id for item in arm_reports], [6])

    def test_the_arm_pose_survives_a_waited_async_group(self) -> None:
        """异步不等于"位姿永远未知"：等完之后位姿按实际下发的脉冲对上。"""
        self.arm.pose_state.record_measurement(CylindricalPose(0.0, 250.0, 100.0))

        ticket = self.robot.arm.move_axes({"base": 3200, "lift": 100}, wait=False)
        self.assertFalse(self.arm.pose_state.is_known)

        batch = MotionBatch(clock=self.clock, sleep=self.clock.sleep)
        batch.add(ticket)
        batch.wait()

        self.assertTrue(self.arm.pose_state.is_known)
        self.assertAlmostEqual(self.arm.pose_state.pose.base_degrees, 60.0, places=9)

    def test_closing_the_session_stops_the_axes_but_leaves_a_borrowed_bus_open(self) -> None:
        batch = self.robot.motion.begin()
        batch.add(self.robot.arm.move_axes({"lift": 800}, wait=False))
        batch.wait()
        self.bus.writes.clear()

        self.robot.close()

        self.assertTrue(self.robot.closed)
        # 三轴各一条停止 + 三轴各一条失能。
        self.assertEqual(len(self.bus.writes), 6)
        # 这条总线不是本会话开的（是测试借给它的），所以不该被物理关闭。
        self.assertTrue(self.bus.is_open)


if __name__ == "__main__":
    unittest.main()

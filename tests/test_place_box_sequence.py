"""``app.place_box_to_repo`` 的帧序列：三轴各发**绝对角度**帧、一起等、到位才松爪。

全部离线：串口、时钟与两个舵机都是内存假件。这里守的是"以前那种 ``wait=False``
发完就跑"不会再回来：

- 每条机械臂命令都是**普通位置帧**（多机同步标志 = 00），收到帧就自己走；
- 松爪必须发生在三轴都收尾**之后**；
- 回初始姿态同样是三轴一起发、一起等。
"""

from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import Mock

import app.gripper as app_gripper
from modules.arm.drive.kinematics import motor_pulses_for_degrees
from modules.arm.runtime import RobotArm
from modules.motion import MotionController
from modules.motor.drive.motion import estimate_move_seconds
from modules.servo.drive.turntable import TurntableServo
from modules.servo.runtime import GripperServo
from robot import Robot
from tests.test_motion_batch import (
    ARM_CONFIG,
    GRIPPER_CONFIG,
    PULSES_PER_REVOLUTION,
    SAFE_TEST_ACCELERATION,
    SAFE_TEST_RPM,
    SERVO_BUS,
    TURNTABLE_CONFIG,
    FakeClock,
    FakeEmm42Bus,
)

POSITION_COMMAND = 0xFD
FRAME_TAIL = 0x6B
PLAIN_POSITION = 0  # 同步标志 = 00：不缓存，收到帧立刻开始走

BASE, LIFT, EXTENSION = 5, 6, 7

#: 多机同步的广播触发帧；本文件一路都不该出现它（那是 move_axes 的路子）。
SYNC_TRIGGER_FRAME = bytes((0x00, 0xFF, 0x66, FRAME_TAIL))


def pulses(degrees: float) -> int:
    """绝对电机角度 → 脉冲，和 ``RobotArm.move_axis_to_motor_degrees`` 同一把尺。"""
    return motor_pulses_for_degrees(
        degrees, pulses_per_revolution=PULSES_PER_REVOLUTION
    )


def position_frames(writes) -> list:
    """从总线写入里挑出位置帧（13 字节、功能码 0xFD）。"""
    return [item for item in writes if len(item) == 13 and item[1] == POSITION_COMMAND]


def frame_pulses(frame: bytes) -> int:
    return int.from_bytes(frame[6:10], "big")


class PlaceBoxSequenceTests(unittest.TestCase):
    """一条 ``place_box_to_repo`` 到底发了哪些帧、按什么顺序。"""

    def setUp(self) -> None:
        app_gripper.current_position_index = 0
        self.clock = FakeClock()
        self.bus = FakeEmm42Bus()
        self.arm = RobotArm._from_settings(
            config=dataclasses.replace(ARM_CONFIG, max_pulses=11000),
            port="FAKE_STEPPER",
            bus=self.bus,
            sleep=self.clock.sleep,
            clock=self.clock,
        )
        self.bus.writes.clear()

        self.gripper_servo = Mock()
        self.gripper_servo.read_position.return_value = GRIPPER_CONFIG.open_position
        self.gripper = GripperServo(
            config=GRIPPER_CONFIG,
            port=SERVO_BUS.port,
            servo=self.gripper_servo,
            sleep=self.clock.sleep,
            clock=self.clock,
        )

        home = TURNTABLE_CONFIG.stations["home"]
        self.turntable_servo = Mock()
        self.turntable_servo.read_position.return_value = home
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

    def test_three_axes_are_sent_as_plain_absolute_frames_in_one_group(self) -> None:
        app_gripper.place_box_to_repo(self.robot)

        frames = position_frames(self.bus.writes)
        # 放料 3 帧 + 回位 3 帧 + 回位后 move_forward 1 帧
        self.assertEqual(len(frames), 7)

        # 同步标志必须是 00：这些帧不缓存、不靠广播触发，收到就开始走。
        flags = [frame[-2] for frame in frames]
        self.assertEqual(flags, [PLAIN_POSITION] * len(frames))
        self.assertNotIn(SYNC_TRIGGER_FRAME, self.bus.writes)

        # 放料前那一组：抬升 → 伸缩 → 底座，目标与绝对角度换算一致。
        self.assertEqual([frame[0] for frame in frames[:3]], [LIFT, EXTENSION, BASE])
        self.assertEqual(
            [frame_pulses(frame) for frame in frames[:3]],
            [pulses(900), pulses(10), pulses(1048)],
        )

        # 回位那一组：675 / 270 / 0，最后才是相对前进 3200 脉冲。
        self.assertEqual(
            [frame[0] for frame in frames[3:]], [LIFT, EXTENSION, BASE, EXTENSION]
        )
        self.assertEqual(
            [frame_pulses(frame) for frame in frames[3:]],
            [pulses(225), pulses(260), pulses(1048), 3200],
        )

    def test_jaws_open_only_after_all_three_axes_have_finished(self) -> None:
        opened_at = []
        self.gripper_servo.set_position.side_effect = (
            lambda position, duration: opened_at.append(self.clock.now)
        )

        app_gripper.place_box_to_repo(self.robot)

        slowest = max(
            estimate_move_seconds(
                pulses(degrees),
                SAFE_TEST_RPM,
                PULSES_PER_REVOLUTION,
                SAFE_TEST_ACCELERATION,
            )
            for degrees in (900, 10, 1048)
        )
        self.assertEqual(len(opened_at), 1, "一条 place 只松一次爪")
        self.assertGreaterEqual(
            opened_at[0], slowest, "松爪时最慢的底座 1048° 必须已经到点"
        )

    def test_the_axis_odometer_keeps_absolute_semantics(self) -> None:
        app_gripper.place_box_to_repo(self.robot)

        self.assertEqual(self.arm.turns_state.of("base").pulses, 0)  # 1048 → 0
        self.assertEqual(self.arm.turns_state.of("lift").pulses, 6000)  # 675°
        self.assertEqual(self.arm.turns_state.of("extension").pulses, 5600)


if __name__ == "__main__":
    unittest.main()

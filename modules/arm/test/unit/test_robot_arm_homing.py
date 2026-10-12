"""回零：帧内容、轴顺序、等待时长，以及"必须先使能"这条前提。"""

from __future__ import annotations

import unittest

from modules.arm.config import ArmConfig, AxisConfig, HomingConfig
from modules.arm.drive.robot_arm import RobotArm
from modules.motor.drive.emm42 import HomeMode

PULSES_PER_REVOLUTION = 3200
FRAME_TAIL = 0x6B

#: 目标轴地址，独立于控制器实现写出来：字段写错时测试必须失败。
MOTOR_IDS = {"extension": 7, "lift": 6, "base": 5}
TORQUE_COMMAND = 0xF3
HOME_COMMAND = 0x9A


def home_frame(*, motor_id: int, mode: HomeMode) -> bytes:
    """手册 6.3.2：地址 + 0x9A + 回零模式 + 多机同步标志 + 校验。"""
    return bytes((motor_id, HOME_COMMAND, int(mode), 0x00, FRAME_TAIL))


def torque_frame(*, motor_id: int, enabled: bool) -> bytes:
    return bytes((motor_id, TORQUE_COMMAND, 0xAB, int(enabled), 0, FRAME_TAIL))


class FakeSerial:
    """在内存里应答每一个命令帧的 Emm42 总线。"""

    def __init__(self, *, answer: bool = False) -> None:
        self._answer = answer
        self._receive_buffer = bytearray()
        self.writes: list = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        if self._answer:
            self._receive_buffer.extend(bytes((frame[0], frame[1], 0x02, 0x6B)))
        return len(frame)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return data

    def close(self) -> None:
        self.is_open = False


def fixture_config(*, homing: HomingConfig = None, **overrides) -> ArmConfig:
    settings = dict(
        pulses_per_revolution=PULSES_PER_REVOLUTION,
        rpm=10,
        acceleration=5,
        default_pulses=800,
        max_pulses=5000,
        axes={
            "base": AxisConfig(motor_id=5, reduction_ratio=5),
            "lift": AxisConfig(motor_id=6, mm_per_motor_rev=40.0),
            "extension": AxisConfig(motor_id=7, mm_per_motor_rev=40.0),
        },
        homing=homing,
    )
    settings.update(overrides)
    return ArmConfig(**settings)


class HomingTestCase(unittest.TestCase):
    def make_arm(self, config=None):
        self.sleep_calls: list = []
        self.serial = FakeSerial()
        self.arm = RobotArm(
            "/dev/ttyUSB0",
            config=config if config is not None else fixture_config(
                homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST)
            ),
            serial_port=self.serial,
            sleep=self.sleep_calls.append,
        )
        return self.arm


class HomingFrameTests(HomingTestCase):
    def test_one_axis_sends_the_documented_home_frame(self) -> None:
        arm = self.make_arm()

        arm.home_axis("lift")

        self.assertEqual(
            self.serial.writes,
            [home_frame(motor_id=MOTOR_IDS["lift"], mode=HomeMode.SINGLE_TURN_NEAREST)],
        )

    def test_the_configured_mode_is_forwarded_verbatim(self) -> None:
        for mode in (
            HomeMode.SINGLE_TURN_DIRECTION,
            HomeMode.MULTI_TURN_COLLISION,
            HomeMode.MULTI_TURN_LIMIT_SWITCH,
        ):
            with self.subTest(mode=mode):
                arm = self.make_arm(
                    config=fixture_config(homing=HomingConfig(mode=mode))
                )
                arm.home_axis("base")
                self.assertEqual(
                    self.serial.writes,
                    [home_frame(motor_id=MOTOR_IDS["base"], mode=mode)],
                )

    def test_the_arm_drivers_do_not_wait_for_an_ack(self) -> None:
        """5/6/7 是 Response=None：回零帧发出去就没有任何回包，不能去读 ACK。"""
        arm = self.make_arm()

        arm.home_all()

        self.assertEqual(len(self.serial.writes), 3)


class HomingOrderTests(HomingTestCase):
    def test_home_all_retracts_and_lifts_before_turning_the_base(self) -> None:
        """伸出状态下转底座会把整条臂横扫过工作区，所以顺序必须是 伸缩 → 升降 → 底座。"""
        arm = self.make_arm()

        arm.home_all()

        self.assertEqual(
            [frame[0] for frame in self.serial.writes],
            [MOTOR_IDS["extension"], MOTOR_IDS["lift"], MOTOR_IDS["base"]],
        )

    def test_the_order_is_exposed_so_callers_can_reason_about_it(self) -> None:
        self.assertEqual(RobotArm.HOME_ORDER, ("extension", "lift", "base"))


class HomingWaitTests(HomingTestCase):
    def test_each_axis_waits_the_configured_duration(self) -> None:
        arm = self.make_arm(
            config=fixture_config(
                homing=HomingConfig(
                    mode=HomeMode.SINGLE_TURN_NEAREST, expected_seconds=3.0, settle_seconds=0.5
                )
            )
        )

        waited = arm.home_all()

        self.assertEqual(self.sleep_calls, [3.5, 3.5, 3.5])
        self.assertEqual(tuple(waited), (3.5, 3.5, 3.5))

    def test_wait_false_sends_every_frame_without_blocking(self) -> None:
        arm = self.make_arm()

        arm.home_all(wait=False)

        self.assertEqual(len(self.serial.writes), 3)
        self.assertEqual(self.sleep_calls, [])

    def test_the_wait_is_a_conservative_estimate_not_a_measurement(self) -> None:
        """回零结束没有回包，所以等的是一个够长的保守时长——这一点必须写在配置里。"""
        homing = HomingConfig(
            mode=HomeMode.SINGLE_TURN_NEAREST, expected_seconds=12.0, settle_seconds=0.5
        )
        self.assertAlmostEqual(homing.wait_seconds, 12.5, places=9)
        self.assertTrue(homing.is_single_turn)

    def test_multi_turn_modes_are_not_single_turn(self) -> None:
        for mode in (HomeMode.MULTI_TURN_COLLISION, HomeMode.MULTI_TURN_LIMIT_SWITCH):
            with self.subTest(mode=mode):
                self.assertFalse(HomingConfig(mode=mode).is_single_turn)


class HomingPreconditionTests(HomingTestCase):
    def test_an_unconfigured_arm_refuses_to_home(self) -> None:
        arm = self.make_arm(config=fixture_config(homing=None))

        with self.assertRaisesRegex(ValueError, "homing"):
            arm.home_all()

        self.assertEqual(self.serial.writes, [])

    def test_homing_is_a_separate_step_from_enabling_torque(self) -> None:
        """手册写明触发回零的前提是"已使能扭矩"，所以控制器不替调用方使能。"""
        arm = self.make_arm()

        arm.enable_all()
        arm.home_all()

        self.assertEqual(
            self.serial.writes[:3],
            [torque_frame(motor_id=motor_id, enabled=True) for motor_id in (5, 6, 7)],
        )
        self.assertEqual([frame[1] for frame in self.serial.writes[3:]], [HOME_COMMAND] * 3)


class HomingConfigWarningTests(unittest.TestCase):
    def test_single_turn_homing_beyond_one_motor_revolution_is_flagged(self) -> None:
        from modules.arm.drive.kinematics import TravelLimit

        config = fixture_config(
            homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST),
            base_limit=TravelLimit(-110.0, 110.0),
        )

        warnings = config.homing_warnings()

        self.assertTrue(any("超过一个电机圈" in item for item in warnings), warnings)
        self.assertTrue(any("72.0" in item for item in warnings), warnings)

    def test_travel_within_one_motor_revolution_is_accepted(self) -> None:
        from modules.arm.drive.kinematics import CylindricalPose, TravelLimit

        config = fixture_config(
            homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST),
            base_limit=TravelLimit(-36.0, 36.0),
            safe_pose=CylindricalPose(0.0, 40.0, 60.0),
        )

        self.assertEqual(config.homing_warnings(), ())

    def test_multi_turn_homing_has_no_travel_restriction(self) -> None:
        from modules.arm.drive.kinematics import CylindricalPose, TravelLimit

        config = fixture_config(
            homing=HomingConfig(mode=HomeMode.MULTI_TURN_LIMIT_SWITCH),
            base_limit=TravelLimit(-170.0, 170.0),
            safe_pose=CylindricalPose(0.0, 40.0, 60.0),
        )

        self.assertEqual(config.homing_warnings(), ())

    def test_missing_safe_pose_is_flagged_because_homing_lands_on_the_zeros(self) -> None:
        config = fixture_config(homing=HomingConfig(mode=HomeMode.MULTI_TURN_LIMIT_SWITCH))

        warnings = config.homing_warnings()

        self.assertTrue(any("safe_pose" in item for item in warnings), warnings)


if __name__ == "__main__":
    unittest.main()

"""回零：帧内容、轴顺序、等待时长，以及"必须先使能"这条前提。

臂通过 ``RobotArm._from_settings`` 装配（真实驱动 + 内存假串口）：装配时会按统一生命
周期使能三个轴，所以夹具把那三条使能帧清掉，只留本文件要看的回零 / 归位帧。
"""

from __future__ import annotations

import unittest

from modules.arm.config import ArmConfig, AxisConfig, HomingConfig
from modules.arm.drive.kinematics import CylindricalPose, TravelLimit
from modules.arm.runtime import RobotArm
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
        # 建立绝对参考的硬前提（计划 §5.2）：safe_pose 与三轴 limits 都要有，而且每个轴的
        # 两端距参考位都**严格小于半个电机圈**（底座 360/5/2 = 36°、直线轴 40/2 = 20 mm）。
        # 帧 / 顺序 / 等待时长的用例都要走完 ``home_all(wait=True)`` 的"记参考"那一步，
        # 所以夹具给一套合格的值；不合格的边界另有专门用例。
        base_limit=TravelLimit(-30.0, 30.0),
        radial_limit=TravelLimit(30.0, 50.0),
        height_limit=TravelLimit(50.0, 70.0),
        safe_pose=CylindricalPose(0.0, 40.0, 60.0),
        homing=homing,
    )
    settings.update(overrides)
    return ArmConfig(**settings)


class HomingTestCase(unittest.TestCase):
    def make_arm(self, config=None):
        self.sleep_calls: list = []
        self.serial = FakeSerial()
        # 装配接缝：真实驱动 + 注入的假串口，不打开任何真实设备。
        self.arm = RobotArm._from_settings(
            config=config if config is not None else fixture_config(
                homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST)
            ),
            port="/dev/ttyUSB0",
            bus=self.serial,
            sleep=self.sleep_calls.append,
        )
        # 装配时已经使能三轴（3 条 0xF3 帧）；本文件只看之后的回零 / 归位帧。
        self.serial.writes.clear()
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

    def test_an_ambiguous_single_turn_reference_is_refused_before_the_first_frame(self) -> None:
        """40 mm/圈、参考 0、行程 [0, 30]：不足一圈，但从 30 就近回零会走到 40。

        所以建立绝对参考必须在**第一条回零帧之前**失败，不能回完零拿一个不知道是哪一圈的
        位姿继续定位（计划 §5.2）。
        """
        arm = self.make_arm(
            config=fixture_config(
                homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST),
                safe_pose=CylindricalPose(0.0, 40.0, 0.0),
                base_limit=TravelLimit(-30.0, 30.0),
                radial_limit=TravelLimit(30.0, 50.0),
                # 升降 40 mm/圈 -> 半圈 20 mm，而行程上端 30 mm 距参考 0 有 30 mm。
                height_limit=TravelLimit(0.0, 30.0),
            )
        )

        with self.assertRaisesRegex(ValueError, "半个电机圈"):
            arm.home_all()

        self.assertEqual(self.serial.writes, [], "预检失败必须在第一条回零帧之前")

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
        config = fixture_config(
            homing=HomingConfig(mode=HomeMode.MULTI_TURN_LIMIT_SWITCH), safe_pose=None
        )

        warnings = config.homing_warnings()

        self.assertTrue(any("safe_pose" in item for item in warnings), warnings)


def absolute_frame(*, motor_id: int, pulses: int, rpm: int = 10, acceleration: int = 5) -> bytes:
    """手册 6.3.1：地址 + 0xFD + 方向 + 速度(2B) + 加速度 + 脉冲数(4B) + 绝对标志 + 同步标志 + 校验。"""
    direction = 0x00 if pulses >= 0 else 0x01
    return (
        bytes((motor_id, 0xFD, direction))
        + rpm.to_bytes(2, "big")
        + bytes((acceleration,))
        + abs(pulses).to_bytes(4, "big")
        + bytes((0x01, 0x00, FRAME_TAIL))
    )


class ZeroingFrameTests(HomingTestCase):
    def make_zeroable_arm(self):
        from modules.arm.drive.kinematics import TravelLimit

        return self.make_arm(
            config=fixture_config(
                homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST),
                base_limit=TravelLimit(-110.0, 110.0),
                radial_limit=TravelLimit(40.0, 160.0),
                height_limit=TravelLimit(-5.0, 120.0),
            )
        )

    def test_one_axis_goes_to_absolute_pulse_zero(self) -> None:
        arm = self.make_zeroable_arm()

        arm.move_axis_to_zero("base")

        self.assertEqual(
            self.serial.writes,
            [absolute_frame(motor_id=MOTOR_IDS["base"], pulses=0)],
        )

    def test_all_axes_retract_and_lift_before_turning_the_base(self) -> None:
        """顺序和回零一致：伸出状态下转底座会把整条臂横扫过工作区。"""
        arm = self.make_zeroable_arm()

        arm.move_all_to_zero()

        self.assertEqual(
            [frame[0] for frame in self.serial.writes],
            [MOTOR_IDS["extension"], MOTOR_IDS["lift"], MOTOR_IDS["base"]],
        )

    def test_every_frame_uses_absolute_mode(self) -> None:
        arm = self.make_zeroable_arm()

        arm.move_all_to_zero()

        # 帧布局：id FD dir rpm(2B) acc 脉冲数(4B) 绝对标志 同步标志 6B —— 绝对标志在第 10 字节。
        for frame in self.serial.writes:
            self.assertEqual(frame[10], 0x01, frame.hex(" "))


class ZeroingWaitTests(HomingTestCase):
    def make_zeroable_arm(self):
        from modules.arm.drive.kinematics import TravelLimit

        return self.make_arm(
            config=fixture_config(
                homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST),
                base_limit=TravelLimit(-110.0, 110.0),
                radial_limit=TravelLimit(40.0, 160.0),
                height_limit=TravelLimit(-5.0, 120.0),
            )
        )

    def test_the_wait_is_the_worst_case_travel_of_that_axis(self) -> None:
        """绝对模式不知道当前位置，所以等的是按行程跨度算出的最坏情况。"""
        from modules.motor.drive.motion import estimate_move_seconds

        arm = self.make_zeroable_arm()

        waited = arm.move_axis_to_zero("base")

        # 底座行程 220°、g=5 -> 220 * 16000 / 360 = 9778 脉冲。
        expected = estimate_move_seconds(9778, arm.config.rpm, PULSES_PER_REVOLUTION, arm.config.acceleration)
        self.assertAlmostEqual(waited, expected, places=9)
        self.assertEqual(self.sleep_calls, [expected])

    def test_linear_axes_use_their_own_travel_span(self) -> None:
        arm = self.make_zeroable_arm()

        arm.move_axis_to_zero("extension")

        # 伸缩行程 120 mm、40 mm/电机圈 -> 120 * 3200 / 40 = 9600 脉冲。
        self.assertEqual(len(self.sleep_calls), 1)
        self.assertGreater(self.sleep_calls[0], 0.0)
        self.assertEqual(
            self.serial.writes,
            [absolute_frame(motor_id=MOTOR_IDS["extension"], pulses=0)],
        )

    def test_wait_false_sends_the_frame_without_blocking(self) -> None:
        arm = self.make_zeroable_arm()

        waited = arm.move_all_to_zero(wait=False)

        self.assertEqual(self.sleep_calls, [])
        self.assertEqual(tuple(waited), (0.0, 0.0, 0.0))
        self.assertEqual(len(self.serial.writes), 3)

    def test_uncalibrated_travel_refuses_to_guess_the_wait(self) -> None:
        """行程没标定就估不出归位耗时——宁可让调用方自己等，也不假装知道。"""
        arm = self.make_arm(
            config=fixture_config(
                homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST),
                base_limit=None,
            )
        )

        with self.assertRaisesRegex(ValueError, "wait=False"):
            arm.move_axis_to_zero("base")

        self.assertEqual(self.serial.writes, [])

    def test_uncalibrated_linear_constant_is_reported_by_name(self) -> None:
        from modules.arm.drive.kinematics import TravelLimit

        arm = self.make_arm(
            config=fixture_config(
                homing=HomingConfig(mode=HomeMode.SINGLE_TURN_NEAREST),
                radial_limit=TravelLimit(40.0, 160.0),
                axes={
                    "base": AxisConfig(motor_id=5, reduction_ratio=5),
                    "lift": AxisConfig(motor_id=6, mm_per_motor_rev=40.0),
                    "extension": AxisConfig(motor_id=7),
                },
            )
        )

        with self.assertRaisesRegex(ValueError, "mm_per_motor_rev"):
            arm.move_axis_to_zero("extension")

        self.assertEqual(self.serial.writes, [])


if __name__ == "__main__":
    unittest.main()

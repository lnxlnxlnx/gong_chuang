"""带符号的单位制动作：正负方向、零脉冲短路、未标定轴拒绝毫米。

夹具刻意独立于 ``test_robot_arm.py``：命令帧的字面量在这里重新拼一遍，
这样即使控制器写错了字段，本文件仍然会失败。

臂通过 ``RobotArm._from_settings`` 装配（真实驱动 + 内存假串口），构造器本身不做 I/O；
装配时会按统一生命周期使能三个轴，所以夹具把那三条使能帧清掉，只留本文件要看的动作帧。
"""

from __future__ import annotations

import unittest

from modules.arm.config import ArmConfig, AxisConfig
from modules.arm.drive.kinematics import CylindricalPose
from modules.arm.runtime import RobotArm
from modules.motor.drive.emm42 import Direction

POSITION_COMMAND = 0xFD
PULSES_PER_REVOLUTION = 3200
BASE_REDUCTION = 5
MM_PER_MOTOR_REV = 40.0
RPM = 10
ACCELERATION = 5
FRAME_TAIL = 0x6B


def fixture_config(**overrides) -> ArmConfig:
    settings = dict(
        pulses_per_revolution=PULSES_PER_REVOLUTION,
        rpm=RPM,
        acceleration=ACCELERATION,
        default_pulses=800,
        max_pulses=5000,
        axes={
            "base": AxisConfig(motor_id=5, reduction_ratio=BASE_REDUCTION),
            "lift": AxisConfig(motor_id=6, mm_per_motor_rev=MM_PER_MOTOR_REV),
            "extension": AxisConfig(motor_id=7, mm_per_motor_rev=MM_PER_MOTOR_REV),
        },
    )
    settings.update(overrides)
    return ArmConfig(**settings)


def relative_position_frame(
    *, motor_id: int, direction: Direction, pulses: int
) -> bytes:
    return (
        bytes((motor_id, POSITION_COMMAND, int(direction)))
        + RPM.to_bytes(2, "big")
        + bytes((ACCELERATION,))
        + pulses.to_bytes(4, "big")
        + bytes((0, 0, FRAME_TAIL))
    )


class FakeSerial:
    """在内存里应答每一个命令帧的 Emm42 总线。"""

    def __init__(self) -> None:
        self._receive_buffer = bytearray()
        self.writes: list = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
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


class UnitMoveTestCase(unittest.TestCase):
    def make_arm(self, config=None):
        self.sleep_calls: list = []
        self.serial = FakeSerial()
        # 装配接缝：真实驱动 + 注入的假串口，不打开任何真实设备。
        self.arm = RobotArm._from_settings(
            config=config if config is not None else fixture_config(),
            port="/dev/ttyUSB0",
            bus=self.serial,
            sleep=self.sleep_calls.append,
        )
        # 装配时已经使能三轴（3 条 0xF3 帧）；本文件只看之后的动作帧。
        self.serial.writes.clear()
        return self.arm


class BaseUnitMoveTests(UnitMoveTestCase):
    def test_positive_degrees_turn_cw_and_negative_degrees_turn_ccw(self) -> None:
        arm = self.make_arm()

        arm.move_base_degrees(22.5)
        arm.move_base_degrees(-22.5)

        self.assertEqual(
            self.serial.writes,
            [
                relative_position_frame(motor_id=5, direction=Direction.CW, pulses=1000),
                relative_position_frame(motor_id=5, direction=Direction.CCW, pulses=1000),
            ],
        )

    def test_the_report_carries_the_output_angle(self) -> None:
        arm = self.make_arm()

        report = arm.move_base_degrees(22.5)

        self.assertEqual(report.pulses, 1000)
        self.assertAlmostEqual(report.motor_degrees, 112.5, places=9)
        self.assertAlmostEqual(report.output_degrees, 22.5, places=9)

    def test_a_sub_pulse_angle_sends_nothing_and_does_not_wait(self) -> None:
        arm = self.make_arm()

        report = arm.move_base_degrees(0.001)

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep_calls, [])
        self.assertEqual(report.pulses, 0)
        self.assertEqual(report.motor_id, 5)
        self.assertEqual(report.estimated_seconds, 0.0)

    def test_exactly_zero_degrees_is_a_no_op_rather_than_a_zero_pulse_frame(self) -> None:
        arm = self.make_arm()

        arm.move_base_degrees(0.0)

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep_calls, [])


class LinearUnitMoveTests(UnitMoveTestCase):
    def test_a_full_motor_revolution_is_mm_per_motor_rev(self) -> None:
        arm = self.make_arm()

        arm.move_lift_mm(MM_PER_MOTOR_REV)

        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=6, direction=Direction.CW, pulses=3200)],
        )

    def test_negative_millimetres_reverse_the_documented_direction(self) -> None:
        arm = self.make_arm()

        arm.move_lift_mm(-20.0)
        arm.move_radial_mm(10.0)
        arm.move_radial_mm(-10.0)

        self.assertEqual(
            self.serial.writes,
            [
                relative_position_frame(motor_id=6, direction=Direction.CCW, pulses=1600),
                relative_position_frame(motor_id=7, direction=Direction.CW, pulses=800),
                relative_position_frame(motor_id=7, direction=Direction.CCW, pulses=800),
            ],
        )

    def test_an_uncalibrated_linear_axis_refuses_millimetres_before_writing(self) -> None:
        arm = self.make_arm(
            config=fixture_config(
                axes={
                    "base": AxisConfig(motor_id=5, reduction_ratio=BASE_REDUCTION),
                    "lift": AxisConfig(motor_id=6),
                    "extension": AxisConfig(motor_id=7),
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "标定"):
            arm.move_lift_mm(10.0)
        with self.assertRaisesRegex(ValueError, "标定"):
            arm.move_radial_mm(10.0)

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep_calls, [])

    def test_a_single_move_beyond_the_pulse_cap_is_still_rejected(self) -> None:
        """保守上限照旧生效：需要长距离时由调用方分几次走或提高上限。"""
        arm = self.make_arm()

        with self.assertRaisesRegex(ValueError, "pulses"):
            arm.move_radial_mm(100.0)  # 8000 脉冲 > max_pulses 5000

        self.assertEqual(self.serial.writes, [])

    def test_millimetres_round_to_pulses_using_the_configured_constant(self) -> None:
        arm = self.make_arm(config=fixture_config(pulses_per_revolution=4000))

        arm.move_radial_mm(10.0)

        # 10 mm / 40 mm-per-rev × 4000 脉冲每圈 = 1000
        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=7, direction=Direction.CW, pulses=1000)],
        )


class UnitMoveWaitTests(UnitMoveTestCase):
    def test_unit_moves_wait_for_the_estimated_time_by_default(self) -> None:
        arm = self.make_arm()

        report = arm.move_radial_mm(20.0)

        self.assertEqual(len(self.sleep_calls), 1)
        self.assertAlmostEqual(self.sleep_calls[0], report.estimated_seconds, places=9)
        self.assertGreater(report.estimated_seconds, 0.0)

    def test_wait_false_still_sends_the_frame_without_blocking(self) -> None:
        arm = self.make_arm()

        arm.move_radial_mm(20.0, wait=False)

        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=7, direction=Direction.CW, pulses=1600)],
        )
        self.assertEqual(self.sleep_calls, [])


class RelativePoseMoveTests(UnitMoveTestCase):
    """一次多轴的相对入口：``move_cylindrical_by`` / ``move_by``。

    这一组钉住两件事，两件都不是实现细节：

    - **执行顺序**（抬升在前、下降在最后、平面先伸缩再底座）——那是碰撞保护；
    - **X/Y 分解依赖当前位姿**——那是极坐标的几何事实，不是保守。
    """

    def record(self, arm, *, base_degrees: float, radial_mm: float, height_mm: float) -> None:
        arm.record_measurement(
            CylindricalPose(
                base_degrees=base_degrees, radial_mm=radial_mm, height_mm=height_mm
            )
        )

    def test_the_cylindrical_entry_raises_then_extends_then_turns(self) -> None:
        arm = self.make_arm()

        reports = arm.move_cylindrical_by(
            base_degrees=22.5, radial_mm=10.0, height_mm=5.0
        )

        self.assertEqual(
            self.serial.writes,
            [
                relative_position_frame(motor_id=6, direction=Direction.CW, pulses=400),
                relative_position_frame(motor_id=7, direction=Direction.CW, pulses=800),
                relative_position_frame(motor_id=5, direction=Direction.CW, pulses=1000),
            ],
        )
        self.assertEqual([report.motor_id for report in reports], [6, 7, 5])

    def test_a_descent_is_left_for_last(self) -> None:
        """下降排在转底座之后：低位转底盘才是撞到台面物块的那一步。"""
        arm = self.make_arm()

        arm.move_cylindrical_by(base_degrees=22.5, height_mm=-20.0)

        self.assertEqual(
            self.serial.writes,
            [
                relative_position_frame(motor_id=5, direction=Direction.CW, pulses=1000),
                relative_position_frame(motor_id=6, direction=Direction.CCW, pulses=1600),
            ],
        )

    def test_zero_increments_send_nothing(self) -> None:
        arm = self.make_arm()

        reports = arm.move_cylindrical_by()

        self.assertEqual(reports, ())
        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep_calls, [])

    def test_move_by_splits_a_tangential_step_with_the_current_pose(self) -> None:
        """半径 100 mm、底座角 0° 时 +Y 是纯切向：只转底座，不伸缩。"""
        arm = self.make_arm()
        self.record(arm, base_degrees=0.0, radial_mm=100.0, height_mm=50.0)

        arm.move_by(y_mm=10.0)

        # 10 mm / 100 mm = 0.1 rad = 5.7296°；按 16000 脉冲/圈 = 254.6 -> 255
        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=5, direction=Direction.CW, pulses=255)],
        )

    def test_move_by_along_the_current_radius_is_pure_extension(self) -> None:
        arm = self.make_arm()
        self.record(arm, base_degrees=0.0, radial_mm=100.0, height_mm=50.0)

        arm.move_by(x_mm=10.0)

        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=7, direction=Direction.CW, pulses=800)],
        )

    def test_the_height_component_of_move_by_needs_no_pose(self) -> None:
        """位姿未知时 z 仍然能相对走：高度与方位无关。"""
        arm = self.make_arm()

        reports = arm.move_by(z_mm=-2.0)

        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=6, direction=Direction.CCW, pulses=160)],
        )
        self.assertEqual([report.motor_id for report in reports], [6])

    def test_x_and_y_without_a_pose_are_refused_before_writing(self) -> None:
        arm = self.make_arm()

        with self.assertRaisesRegex(RuntimeError, "位姿"):
            arm.move_by(x_mm=1.0)

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep_calls, [])

    def test_a_radius_too_small_for_xy_is_refused(self) -> None:
        """基座轴线附近 1 mm 的切向增量就是几十度：宁可拒绝，不要分解出这种动作。"""
        arm = self.make_arm()
        self.record(arm, base_degrees=0.0, radial_mm=0.5, height_mm=50.0)

        with self.assertRaisesRegex(ValueError, "半径"):
            arm.move_by(y_mm=1.0)

        self.assertEqual(self.serial.writes, [])

    def test_an_uncalibrated_linear_axis_refuses_millimetres(self) -> None:
        arm = self.make_arm(
            config=fixture_config(
                axes={
                    "base": AxisConfig(motor_id=5, reduction_ratio=BASE_REDUCTION),
                    "lift": AxisConfig(motor_id=6),
                    "extension": AxisConfig(motor_id=7),
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "标定"):
            arm.move_cylindrical_by(radial_mm=10.0)
        with self.assertRaisesRegex(ValueError, "标定"):
            arm.move_by(z_mm=10.0)

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep_calls, [])

    def test_a_boolean_or_non_finite_increment_is_rejected(self) -> None:
        arm = self.make_arm()

        for bad in (True, float("nan"), float("inf"), "10"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    arm.move_cylindrical_by(base_degrees=bad)

        self.assertEqual(self.serial.writes, [])


if __name__ == "__main__":
    unittest.main()

"""每轴累计转角与脉冲：状态类的行为，以及它在 RobotArm 里只有一个执行点。

`:class:`modules.arm.arm_state.MotorTurnsState`` 回答"电机转了多少"，和回答"爪口在哪"的
``ArmPose`` 是兄弟。这里同时钉住两件事：

1. 状态类本身：带符号累加、**不归一**、原点由调用方决定；
2. ``RobotArm`` 只在一处推进它（``_send``），所以方向原语、单位制方法和绝对规划帧
   走的是同一条账。
"""

from __future__ import annotations

import unittest

from modules.arm.arm_state import MotorTurnsState
from modules.arm.config import ArmConfig, AxisConfig
from modules.arm.drive.kinematics import CylindricalPose, TravelLimit
from modules.arm.runtime import RobotArm

PULSES_PER_REVOLUTION = 3200


def base_pulses_for(degrees: float, *, reduction: int = 5) -> int:
    """Independent fixture: the documented pulse count for a base rotation."""
    return round(degrees * PULSES_PER_REVOLUTION * reduction / 360.0)


class FakeSerial:
    """Records frames; the arm's 5/6/7 drivers never answer control frames."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
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


class RecordingSleep:
    """Records requested waits instead of sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def turns_config() -> ArmConfig:
    """A fully calibrated arm so the absolute-planning path is usable too."""
    return ArmConfig(
        pulses_per_revolution=PULSES_PER_REVOLUTION,
        rpm=10,
        acceleration=5,
        default_pulses=800,
        max_pulses=5000,
        axes={
            "base": AxisConfig(motor_id=5, reduction_ratio=5),
            "lift": AxisConfig(motor_id=6, mm_per_motor_rev=61.0),
            "extension": AxisConfig(motor_id=7, mm_per_motor_rev=61.0),
        },
        base_limit=TravelLimit(-180.0, 180.0),
        radial_limit=TravelLimit(180.0, 359.1),
        height_limit=TravelLimit(10.0, 183.5),
        # 绝对规划要先有一个安全位（转角度前回到它），所以夹具也得给。
        safe_pose=CylindricalPose(0.0, 200.0, 100.0),
    )


class MotorTurnsStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.turns = MotorTurnsState(
            PULSES_PER_REVOLUTION, ("base", "lift", "extension")
        )

    def test_every_axis_starts_at_zero(self) -> None:
        for axis in ("base", "lift", "extension"):
            with self.subTest(axis=axis):
                self.assertEqual(self.turns.of(axis).pulses, 0)

    def test_a_full_motor_revolution_is_360_degrees(self) -> None:
        reading = self.turns.record("lift", PULSES_PER_REVOLUTION)

        self.assertEqual(reading.motor_degrees, 360.0)
        self.assertEqual(reading.motor_revolutions, 1.0)

    def test_the_counter_grows_and_shrinks_by_signed_pulses(self) -> None:
        self.turns.record("base", 1000)
        self.turns.record("base", 500)
        self.assertEqual(self.turns.of("base").pulses, 1500)

        self.turns.record("base", -2000)

        self.assertEqual(self.turns.of("base").pulses, -500)

    def test_the_count_is_never_normalised(self) -> None:
        """底座能转整圈：绕两圈必须看得见，归一化之后就和没转一样了。

        这正是 ``ArmPose.base_degrees`` 覆盖不了的那件事（它归一到 ``(-180, 180]``），
        也是单独一个里程表存在的理由。
        """
        self.turns.record("base", 16000)
        self.turns.record("base", 16000)

        reading = self.turns.of("base")

        self.assertEqual(reading.pulses, 32000)
        self.assertEqual(reading.motor_degrees, 3600.0)
        self.assertEqual(reading.motor_revolutions, 10.0)

    def test_each_axis_keeps_its_own_count(self) -> None:
        self.turns.record("base", 16000)
        self.turns.record("lift", 3200)
        self.turns.record("extension", -1600)

        self.assertEqual(self.turns.of("base").pulses, 16000)
        self.assertEqual(self.turns.of("lift").pulses, 3200)
        self.assertEqual(self.turns.of("extension").pulses, -1600)

    def test_marking_the_origin_clears_every_axis(self) -> None:
        self.turns.record("base", 16000)
        self.turns.record("lift", 3200)

        self.turns.mark_origin()

        self.assertEqual(
            [self.turns.of(axis).pulses for axis in ("base", "lift", "extension")],
            [0, 0, 0],
        )

    def test_marking_one_axis_leaves_the_others_alone(self) -> None:
        """单轴回零只让那一根轴停在零点上，别的轴的原点不该被顺手抹掉。"""
        self.turns.record("base", 16000)
        self.turns.record("lift", 3200)

        self.turns.mark_origin("lift")

        self.assertEqual(self.turns.of("lift").pulses, 0)
        self.assertEqual(self.turns.of("base").pulses, 16000)

    def test_an_unknown_axis_is_rejected_rather_than_silently_zero(self) -> None:
        with self.assertRaises(KeyError):
            self.turns.of("nope")
        with self.assertRaises(KeyError):
            self.turns.record("nope", 100)

    def test_non_integer_pulses_are_rejected(self) -> None:
        for bad in (1.5, True, "100"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.turns.record("base", bad)  # type: ignore[arg-type]

    def test_a_non_positive_pulses_per_revolution_is_rejected(self) -> None:
        for bad in (0, -3200):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MotorTurnsState(bad, ("base",))

    def test_summary_names_every_axis(self) -> None:
        text = self.turns.summary()
        for axis in ("base", "lift", "extension"):
            self.assertIn(axis, text)


class RobotArmTurnsTests(unittest.TestCase):
    """``RobotArm`` 在 ``_send`` 一处推进里程，所以每条路都记账、且只记一次。"""

    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.sleep = RecordingSleep()
        self.arm = RobotArm._from_settings(
            config=turns_config(),
            port="/dev/ttyUSB0",
            bus=self.serial,
            sleep=self.sleep,
        )
        self.serial.writes.clear()

    def test_a_relative_move_advances_the_count_and_reports_it(self) -> None:
        report = self.arm.move_up(pulses=800)

        self.assertIsNotNone(report.turns)
        assert report.turns is not None
        self.assertEqual(report.turns.axis_name, "lift")
        self.assertEqual(report.turns.pulses, 800)
        self.assertEqual(self.arm.turns_state.of("lift").pulses, 800)

    def test_opposite_directions_cancel_out(self) -> None:
        self.arm.move_up(pulses=800)
        self.arm.move_down(pulses=300)

        self.assertEqual(self.arm.turns_state.of("lift").pulses, 500)

    def test_a_unit_converted_move_advances_by_the_actual_pulses(self) -> None:
        """``move_lift_mm`` 传的是毫米，记的必须是**换算后的整数脉冲**。"""
        report = self.arm.move_lift_mm(61.0)

        self.assertEqual(report.pulses, PULSES_PER_REVOLUTION)
        self.assertEqual(self.arm.turns_state.of("lift").pulses, PULSES_PER_REVOLUTION)

    def test_equally_named_methods_do_not_double_count(self) -> None:
        """毫米版内部调的是 ``_signed_move``；上一层的名字不该再记一遍。"""
        report = self.arm.move_radial_mm(61.0)

        self.assertEqual(self.arm.turns_state.of("extension").pulses, report.pulses)

    def test_the_planned_absolute_path_uses_the_same_counter(self) -> None:
        """绝对规划走 ``_execute_frame``，但它同样经过 ``_send`` —— 账只有一本。

        这条如果红了，说明有人把累计挪到了上一层，绝对定位那条路就会漏记。
        """
        self.arm.record_measurement(CylindricalPose(0.0, 200.0, 100.0))

        self.arm.move_cylindrical(base_degrees=30.0, radial_mm=200.0, height_mm=100.0)

        self.assertEqual(
            self.arm.turns_state.of("base").pulses, base_pulses_for(30.0)
        )

    def test_a_rejected_move_is_not_counted(self) -> None:
        """越界的脉冲在发帧之前就被拒了，账上不该多出来。"""
        with self.assertRaises(ValueError):
            self.arm.rotate_cw(pulses=self.arm.config.max_pulses + 1)

        self.assertEqual(self.arm.turns_state.of("base").pulses, 0)
        self.assertEqual(self.serial.writes, [])

    def test_zero_pulses_reports_the_current_reading_without_moving(self) -> None:
        self.arm.move_up(pulses=800)

        report = self.arm.move_down(pulses=0)

        self.assertEqual(report.pulses, 0)
        assert report.turns is not None
        self.assertEqual(report.turns.pulses, 800)
        self.assertEqual(self.arm.turns_state.of("lift").pulses, 800)

    def test_absolute_zeroing_puts_the_axis_back_on_the_origin(self) -> None:
        """绝对归位和回零停在同一个物理点（驱动器计数器 0），所以原点同此。"""
        self.arm.move_up(pulses=800)

        self.arm.move_axis_to_zero("lift")

        self.assertEqual(self.arm.turns_state.of("lift").pulses, 0)

    def test_an_unwaited_move_still_counts_because_the_frame_was_sent(self) -> None:
        """``wait=False`` 不推进位姿（不知道走没走到），但帧确实发出去了。"""
        self.arm.move_up(pulses=800, wait=False)

        self.assertFalse(self.arm.pose_state.is_known)
        self.assertEqual(self.arm.turns_state.of("lift").pulses, 800)

    def test_direction_uses_the_same_axis_as_the_physical_primitive(self) -> None:
        self.arm.rotate_cw(pulses=4000)
        self.arm.rotate_ccw(pulses=1000)

        reading = self.arm.turns_state.of("base")
        self.assertEqual(reading.pulses, 3000)
        self.assertEqual(reading.motor_degrees, 337.5)


if __name__ == "__main__":
    unittest.main()

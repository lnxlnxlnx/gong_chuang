"""环心 → 底座绝对角度：纯数学求解器 + 会话入口的三条拒绝。

这里测两件事，分开测：

1. **数值**：``plan_ring_base`` 的步长、绝对脉冲、角度换算、残差与参数校验；
2. **会话**：``Grasp.plan_ring`` 怎么定位、怎么核对比例尺、什么时候拒绝。

全部不需要机械臂与相机：假臂只提供里程表与减速比，假观测源只回放一个 ``RingTarget``。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from modules.grasp.drive.ring_plan import RingBasePlan, plan_ring_base
from modules.grasp.drive.servo_model import (
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoTarget,
)
from modules.grasp.runtime import Grasp
from modules.vision.drive.numbered_ring import RingTarget

#: 实测雅可比（modules/grasp/eye_in_hand.json）。
JACOBIAN = ServoJacobian(
    base_dx=-0.08174204217107724,
    base_dy=-0.007583697824866524,
    radial_dx=-0.0002735146342596506,
    radial_dy=0.01812172800295296,
)
PULSES_PER_REV = 3200
REDUCTION = 5


def plan(reading=(0.0, 0.0), target=(0.0, 0.0), *, base_pulses=0, radial_pulses=0,
         jacobian=JACOBIAN, tolerance_mm=4.0) -> RingBasePlan:
    return plan_ring_base(
        reading=reading,
        target=target,
        jacobian=jacobian,
        base_pulses=base_pulses,
        radial_pulses=radial_pulses,
        pulses_per_revolution=PULSES_PER_REV,
        reduction_ratio=REDUCTION,
        tolerance_mm=tolerance_mm,
    )


class SolverTests(unittest.TestCase):
    def test_already_on_target_sends_nothing(self) -> None:
        result = plan(reading=(3.0, -2.0), target=(3.0, -2.0))
        self.assertEqual(result.base_pulses, 0)
        self.assertEqual(result.radial_pulses, 0)
        self.assertTrue(result.aligned)

    def test_absolute_angle_is_current_plus_step(self) -> None:
        """绝对值 = 里程表读数 + 增量；角度是电机轴口径（不取模）。"""
        result = plan(reading=(10.0, 0.0), base_pulses=2000)
        expected_pulses = 2000 + result.base_pulses
        self.assertEqual(result.base_motor_pulses, expected_pulses)
        self.assertAlmostEqual(
            result.base_motor_degrees, expected_pulses * 360.0 / PULSES_PER_REV
        )
        self.assertAlmostEqual(
            result.base_output_degrees, result.base_motor_degrees / REDUCTION
        )

    def test_the_base_step_moves_along_the_measured_column(self) -> None:
        """读数偏 +10 mm 时，按实测那一列算出来的步长应当约 +121 脉冲。"""
        result = plan(reading=(10.0, 0.0))
        self.assertGreater(result.base_pulses, 0)
        self.assertLessEqual(abs(result.base_pulses - 121), 3, result.base_pulses)

    def test_both_axes_together_land_inside_the_tolerance(self) -> None:
        result = plan(reading=(3.0, -4.0))
        self.assertLess(result.residual_mm, 1.0)
        self.assertTrue(result.aligned)

    def test_negative_reading_turns_the_other_way(self) -> None:
        positive = plan(reading=(10.0, 0.0))
        negative = plan(reading=(-10.0, 0.0))
        self.assertEqual(positive.base_pulses, -negative.base_pulses)

    def test_singular_jacobian_is_refused(self) -> None:
        singular = ServoJacobian(base_dx=0.01, base_dy=0.02, radial_dx=0.02, radial_dy=0.04)
        self.assertTrue(singular.is_singular)
        with self.assertRaisesRegex(ServoError, "奇异"):
            plan(reading=(1.0, 1.0), jacobian=singular)

    def test_bad_arguments_are_rejected_before_any_math(self) -> None:
        for bad in ((1.0,), ("a", "b"), (1.0, float("nan")), (1.0, 2.0, 3.0)):
            with self.subTest(reading=bad):
                with self.assertRaises(ValueError):
                    plan(reading=bad)
        with self.assertRaises(ValueError):
            plan_ring_base(
                reading=(0.0, 0.0), target=(0.0, 0.0), jacobian=JACOBIAN,
                base_pulses=0, radial_pulses=0,
                pulses_per_revolution=0, reduction_ratio=REDUCTION,
            )
        with self.assertRaises(ValueError):
            plan_ring_base(
                reading=(0.0, 0.0), target=(0.0, 0.0), jacobian=JACOBIAN,
                base_pulses=0, radial_pulses=0,
                pulses_per_revolution=PULSES_PER_REV, reduction_ratio=0,
            )

    def test_summary_mentions_the_absolute_angle(self) -> None:
        text = plan(reading=(10.0, 0.0)).summary()
        self.assertIn("底座绝对", text)
        self.assertIn("脉冲", text)


# ---------------------------------------------------------------------- 会话
def ring_target(*, digit=2, score=0.93, clipped=False, reading=(10.0, 0.0),
                scale_mm_per_px=0.111, scale_source="depth/focal") -> RingTarget:
    return RingTarget(
        center_px=(700.0, 400.0),
        radius_px=250.0,
        digit=digit,
        score=score,
        clipped=clipped,
        x_mm=None if reading is None else reading[0],
        y_mm=None if reading is None else reading[1],
        scale_mm_per_px=scale_mm_per_px,
        scale_source=scale_source,
        detail="测试",
    )


class FakeArm:
    """只提供 ``plan_ring`` 用到的那几项：里程表、脉冲数、减速比。"""

    def __init__(self, *, base_pulses=2000, radial_pulses=-300) -> None:
        self.turns_state = Mock()
        self.turns_state.of.side_effect = lambda name: Mock(
            pulses=base_pulses if name == "base" else radial_pulses
        )
        self.pulses_per_revolution = PULSES_PER_REV
        self.config = Mock()
        self.config.base_reduction = REDUCTION


class FakeSource:
    def __init__(self, target) -> None:
        self._target = target
        self.calls = 0

    def locate_ring(self, *, timeout=None):
        self.calls += 1
        return self._target


def make_grasp(target) -> tuple[Grasp, FakeArm, FakeSource]:
    grasp = Grasp(
        config=ServoConfig(),
        selector=ObjectSelector(),
        target=ServoTarget(x_mm=0.0, y_mm=0.0),
        jacobian=JACOBIAN,
        lift_scale=None,
    )
    arm = FakeArm()
    source = FakeSource(target)
    grasp._arm = arm
    grasp._source = source
    return grasp, arm, source


class SessionPlanTests(unittest.TestCase):
    def test_one_frame_and_one_plan(self) -> None:
        grasp, arm, source = make_grasp(ring_target())
        result = grasp.plan_ring()
        self.assertEqual(source.calls, 1, "只该取一帧")
        self.assertEqual(result.digit, 2)
        self.assertAlmostEqual(result.score, 0.93)
        self.assertEqual(result.reading, (10.0, 0.0))
        self.assertEqual(result.base_motor_pulses, 2000 + result.base_pulses)

    def test_the_target_reading_comes_from_the_session_config(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target(reading=(5.0, 0.0)))
        grasp._target = ServoTarget(x_mm=5.0, y_mm=0.0)
        result = grasp.plan_ring()
        self.assertEqual(result.base_pulses, 0, "读数等于目标，不该动")

    def test_no_ring_in_view_is_an_error(self) -> None:
        grasp, _arm, _source = make_grasp(None)
        with self.assertRaisesRegex(ServoError, "没有环"):
            grasp.plan_ring()

    def test_clipped_ring_refuses_to_give_an_angle(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target(clipped=True))
        with self.assertRaisesRegex(ServoError, "贴到画面边缘"):
            grasp.plan_ring()

    def test_missing_scale_refuses_to_give_an_angle(self) -> None:
        grasp, _arm, _source = make_grasp(
            ring_target(reading=None, scale_mm_per_px=None, scale_source="none")
        )
        with self.assertRaisesRegex(ServoError, "比例尺"):
            grasp.plan_ring()

    def test_a_different_ring_number_is_refused(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target(digit=3))
        with self.assertRaisesRegex(ServoError, "3 号"):
            grasp.plan_ring(require_digit=2)

    def test_the_required_ring_number_passes(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target(digit=3))
        self.assertEqual(grasp.plan_ring(require_digit=3).digit, 3)

    def test_a_scale_that_does_not_match_the_jacobian_is_refused(self) -> None:
        """环心用 station.mm_per_px、雅可比按针孔标定：角度会整体差一个倍数。"""
        grasp, _arm, _source = make_grasp(
            ring_target(scale_mm_per_px=0.495, scale_source="station.mm_per_px")
        )
        with self.assertRaisesRegex(ServoError, "station.mm_per_px"):
            grasp.plan_ring()

    def test_an_explicit_scale_rescales_the_reading(self) -> None:
        """给了与雅可比一致的 mm/px，就按它把读数换过去（0.495 -> 0.099 是 1/5）。"""
        grasp, _arm, _source = make_grasp(
            ring_target(reading=(10.0, 0.0), scale_mm_per_px=0.495,
                        scale_source="station.mm_per_px")
        )
        result = grasp.plan_ring(scale_mm_per_px=0.099)
        self.assertAlmostEqual(result.reading[0], 2.0, places=6)
        self.assertLess(abs(result.base_pulses), 30)

    def test_a_bad_explicit_scale_is_rejected(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target())
        with self.assertRaises(ServoError):
            grasp.plan_ring(scale_mm_per_px=0.0)

    def test_a_source_without_ring_vision_says_so(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target())
        grasp._source = Mock(spec=["frame"])  # TCP 观测源：没有环心定位
        with self.assertRaisesRegex(ServoError, "没有环心定位"):
            grasp.plan_ring()

    def test_a_closed_session_refuses(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target())
        grasp._closed = True
        with self.assertRaises(RuntimeError):
            grasp.plan_ring()


if __name__ == "__main__":
    unittest.main()

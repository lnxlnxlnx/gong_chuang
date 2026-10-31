"""环心 → 底座绝对角度：纯数学求解器 + 会话入口的三条拒绝。

这里测两件事，分开测：

1. **数值**：``plan_ring_base`` 的步长、绝对脉冲、角度换算、残差与参数校验；
2. **会话**：``Grasp.plan_ring`` 怎么定位、怎么核对比例尺、什么时候拒绝。

全部不需要机械臂与相机：假臂只提供里程表与减速比，假观测源只回放一个 ``RingTarget``。
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import Mock, patch

from modules.grasp.drive.ring_plan import RingBasePlan, plan_ring_base
from modules.grasp.drive.visual_servo_model import (
    MaterialSelector,
    VisualServoConfig,
    VisualServoError,
    VisualServoJacobian,
    VisualServoTarget,
)
from modules.grasp.runtime import Grasp
from modules.vision.drive.numbered_ring import RingTarget

#: 实测雅可比（modules/grasp/eye_in_hand.json）。
JACOBIAN = VisualServoJacobian(
    base_dx=-0.08174204217107724,
    base_dy=-0.007583697824866524,
    radial_dx=-0.0002735146342596506,
    radial_dy=0.01812172800295296,
)
PULSES_PER_REV = 3200
REDUCTION = 5


def plan(reading=(0.0, 0.0), target=(0.0, 0.0), *, base_pulses=0, extension_pulses=0,
         jacobian=JACOBIAN, tolerance_mm=4.0) -> RingBasePlan:
    return plan_ring_base(
        reading=reading,
        target=target,
        jacobian=jacobian,
        base_pulses=base_pulses,
        extension_pulses=extension_pulses,
        pulses_per_revolution=PULSES_PER_REV,
        reduction_ratio=REDUCTION,
        tolerance_mm=tolerance_mm,
    )


class SolverTests(unittest.TestCase):
    def test_already_on_target_sends_nothing(self) -> None:
        result = plan(reading=(3.0, -2.0), target=(3.0, -2.0))
        self.assertEqual(result.base_pulses, 0)
        self.assertEqual(result.extension_pulses, 0)
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
        singular = VisualServoJacobian(base_dx=0.01, base_dy=0.02, radial_dx=0.02, radial_dy=0.04)
        self.assertTrue(singular.is_singular)
        with self.assertRaisesRegex(VisualServoError, "奇异"):
            plan(reading=(1.0, 1.0), jacobian=singular)

    def test_bad_arguments_are_rejected_before_any_math(self) -> None:
        for bad in ((1.0,), ("a", "b"), (1.0, float("nan")), (1.0, 2.0, 3.0)):
            with self.subTest(reading=bad):
                with self.assertRaises(ValueError):
                    plan(reading=bad)
        with self.assertRaises(ValueError):
            plan_ring_base(
                reading=(0.0, 0.0), target=(0.0, 0.0), jacobian=JACOBIAN,
                base_pulses=0, extension_pulses=0,
                pulses_per_revolution=0, reduction_ratio=REDUCTION,
            )
        with self.assertRaises(ValueError):
            plan_ring_base(
                reading=(0.0, 0.0), target=(0.0, 0.0), jacobian=JACOBIAN,
                base_pulses=0, extension_pulses=0,
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
    """只提供 ``plan_ring`` 用到的那几项：里程表、脉冲数、减速比、绝对定位。"""

    def __init__(self, *, base_pulses=2000, extension_pulses=-300) -> None:
        self.turns_state = Mock()
        self.turns_state.of.side_effect = lambda name: Mock(
            pulses=base_pulses if name == "base" else extension_pulses
        )
        self.pulses_per_revolution = PULSES_PER_REV
        self.config = Mock()
        self.config.base_reduction = REDUCTION
        self.moves: list[tuple[str, int]] = []

    def move_axis_to_motor_pulses(self, axis_name, pulses, *, rpm=None, acceleration=None):
        self.moves.append((axis_name, int(pulses)))
        return f"report({axis_name})"


class FakeSource:
    def __init__(self, target) -> None:
        self._target = target
        self.calls = 0
        #: 每次定位收到的 ``digit`` —— 检查"按号筛框"有没有真的下传。
        self.digits: list = []

    def locate_ring(self, *, timeout=None, digit=None):
        self.calls += 1
        self.digits.append(digit)
        return self._target


def make_grasp(
    target, config: VisualServoConfig | None = None
) -> tuple[Grasp, FakeArm, FakeSource]:
    grasp = Grasp(
        config=config if config is not None else VisualServoConfig(),
        selector=MaterialSelector(),
        target=VisualServoTarget(x_mm=0.0, y_mm=0.0),
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
        grasp._target = VisualServoTarget(x_mm=5.0, y_mm=0.0)
        result = grasp.plan_ring()
        self.assertEqual(result.base_pulses, 0, "读数等于目标，不该动")

    def test_the_ring_offset_shifts_the_ring_target_only(self) -> None:
        """落料补偿：读数正好等于「target + 补偿」时不该再动（料会落在环心）。"""
        grasp, _arm, _source = make_grasp(
            ring_target(reading=(7.5, 0.0)),
            config=VisualServoConfig(ring_offset_x_mm=7.5),
        )
        # 补偿只加在环心那一次上：物料用的 target 一个数都没变。
        self.assertEqual((grasp._target.x_mm, grasp._target.y_mm), (0.0, 0.0))
        self.assertEqual(grasp._ring_target().x_mm, 7.5)
        self.assertEqual(
            grasp.plan_ring().base_pulses, 0, "读数等于补偿后的目标，不该动"
        )

    def test_without_an_offset_the_same_reading_still_moves(self) -> None:
        """不配补偿（默认 0）时行为与加这个字段之前完全一致。"""
        grasp, _arm, _source = make_grasp(ring_target(reading=(7.5, 0.0)))
        self.assertEqual(grasp._ring_target().x_mm, 0.0)
        self.assertNotEqual(grasp.plan_ring().base_pulses, 0)

    def test_the_offset_follows_a_retaught_target(self) -> None:
        """补偿是相对 ``target`` 的偏移：重标 target 之后环心目标跟着走。"""
        grasp, _arm, _source = make_grasp(
            ring_target(reading=(5.0, 0.0)),
            config=VisualServoConfig(ring_offset_x_mm=7.5),
        )
        grasp._target = VisualServoTarget(x_mm=-2.5, y_mm=0.0)
        self.assertEqual(grasp._ring_target().x_mm, 5.0)
        self.assertEqual(
            grasp.plan_ring().base_pulses, 0, "读数等于 target + 补偿，不该动"
        )

    def test_no_ring_in_view_is_an_error(self) -> None:
        grasp, _arm, _source = make_grasp(None)
        with self.assertRaisesRegex(VisualServoError, "没有环"):
            grasp.plan_ring()

    def test_clipped_ring_refuses_to_give_an_angle(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target(clipped=True))
        with self.assertRaisesRegex(VisualServoError, "贴到画面边缘"):
            grasp.plan_ring()

    def test_missing_scale_refuses_to_give_an_angle(self) -> None:
        grasp, _arm, _source = make_grasp(
            ring_target(reading=None, scale_mm_per_px=None, scale_source="none")
        )
        with self.assertRaisesRegex(VisualServoError, "比例尺"):
            grasp.plan_ring()

    def test_the_required_digit_filters_the_frame(self) -> None:
        """一帧里可能同时有好几个环：按号筛框，而不是"取分最高的那个再看是不是它"。"""
        grasp, _arm, source = make_grasp(ring_target(digit=2))
        grasp.plan_ring(require_digit=2)
        grasp.plan_ring()

        self.assertEqual(source.digits, [2, None])

    def test_a_different_ring_number_is_refused(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target(digit=3))
        with self.assertRaisesRegex(VisualServoError, "3 号"):
            grasp.plan_ring(require_digit=2)

    def test_the_required_ring_number_passes(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target(digit=3))
        self.assertEqual(grasp.plan_ring(require_digit=3).digit, 3)

    def test_a_reading_in_physical_millimetres_is_accepted(self) -> None:
        """``ring box``（已知外径 / 模型框）也是物理毫米，与 jacobian 同一把尺。"""
        grasp, _arm, _source = make_grasp(
            ring_target(scale_mm_per_px=95.0 / 520.0, scale_source="ring box")
        )
        self.assertIsNotNone(grasp.plan_ring())

    def test_a_scale_that_does_not_match_the_jacobian_is_refused(self) -> None:
        """环心用 station.mm_per_px、雅可比按针孔标定：角度会整体差一个倍数。"""
        grasp, _arm, _source = make_grasp(
            ring_target(scale_mm_per_px=0.495, scale_source="station.mm_per_px")
        )
        with self.assertRaisesRegex(VisualServoError, "station.mm_per_px"):
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
        with self.assertRaises(VisualServoError):
            grasp.plan_ring(scale_mm_per_px=0.0)

    def test_a_source_without_ring_vision_says_so(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target())
        grasp._source = Mock(spec=["frame"])  # TCP 观测源：没有环心定位
        with self.assertRaisesRegex(VisualServoError, "没有环心定位"):
            grasp.plan_ring()

    def test_a_closed_session_refuses(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target())
        grasp._closed = True
        with self.assertRaises(RuntimeError):
            grasp.plan_ring()


class RingFrameRetryTests(unittest.TestCase):
    """模型偶尔把相邻两个环读成同一号、整帧编号作废：坏帧要重读，别整段失败。"""

    class _Flaky:
        def __init__(self, blanks: int) -> None:
            self.blanks = blanks
            self.calls = 0

        def locate_ring(self, *, timeout=None, digit=None):
            self.calls += 1
            return None if self.calls <= self.blanks else ring_target()

    def _grasp_with(self, source):
        grasp, _arm, _source = make_grasp(ring_target())
        grasp._source = source
        return grasp

    def test_a_bad_frame_is_retried(self) -> None:
        source = self._Flaky(blanks=1)
        with patch("time.sleep"):
            plan = self._grasp_with(source).plan_ring(require_digit=2)

        self.assertEqual(source.calls, 2)
        self.assertEqual(plan.digit, 2)

    def test_it_gives_up_after_the_attempts(self) -> None:
        source = self._Flaky(blanks=99)
        with patch("time.sleep"), self.assertRaisesRegex(VisualServoError, "没有环"):
            self._grasp_with(source).plan_ring()

        self.assertEqual(source.calls, Grasp.RING_FRAME_ATTEMPTS)


class AbsolutePositionTests(unittest.TestCase):
    """"先算、后走"：把各电机的绝对位置存下来，中间干完别的再执行。"""

    def test_absolute_positions_are_per_axis(self) -> None:
        grasp, _arm, _source = make_grasp(ring_target(reading=(10.0, 0.0)))
        plan = grasp.plan_ring()
        self.assertEqual(plan.pulses["base"], 2000 + plan.base_pulses)
        self.assertEqual(plan.pulses["extension"], -300 + plan.extension_pulses)
        self.assertAlmostEqual(plan.degrees["base"], plan.base_motor_degrees)
        self.assertAlmostEqual(plan.degrees["extension"], plan.extension_motor_degrees)

    def test_apply_moves_every_axis_to_its_absolute_position(self) -> None:
        grasp, arm, _source = make_grasp(ring_target(reading=(10.0, 0.0)))
        plan = grasp.plan_ring()
        plan.apply(arm)
        self.assertEqual(
            arm.moves,
            [
                ("base", plan.pulses["base"]),
                ("extension", plan.pulses["extension"]),
            ],
        )

    def test_apply_can_rotate_the_base_only(self) -> None:
        grasp, arm, _source = make_grasp(ring_target(reading=(10.0, 0.0)))
        plan = grasp.plan_ring()
        plan.apply(arm, axes=("base",))
        self.assertEqual(arm.moves, [("base", plan.pulses["base"])])

    def test_apply_rejects_an_axis_that_is_not_planned(self) -> None:
        grasp, arm, _source = make_grasp(ring_target())
        with self.assertRaisesRegex(ValueError, "lift"):
            grasp.plan_ring().apply(arm, axes=("lift",))


class SimulatedPlantArm:
    """假臂 + 假环境：增量脉冲按**真实的**雅可比作用到"环心读数"上。

    用它就能离线验证 :meth:`Grasp.align_ring` 的闭环：把 ``plant`` 设成配置雅可比的
    一半，等价于"配置过期/尺不对"，闭环应当在线学出响应比并两三步入位。
    """

    def __init__(self, plant: VisualServoJacobian, *, reading=(30.0, 20.0),
                 base_pulses=0, extension_pulses=0) -> None:
        self.plant = plant
        self.reading = [float(reading[0]), float(reading[1])]
        self.pulses = {"base": base_pulses, "extension": extension_pulses}
        self.moves: list[tuple[str, int]] = []
        self.turns_state = Mock()
        self.turns_state.of.side_effect = lambda name: Mock(pulses=self.pulses[name])
        self.pulses_per_revolution = PULSES_PER_REV
        self.config = Mock(base_reduction=REDUCTION)

    def _apply(self, axis: str, signed: int) -> None:
        self.moves.append((axis, signed))
        self.pulses[axis] += signed
        moved = self.plant.displacement_for(
            signed if axis == "base" else 0,
            signed if axis == "extension" else 0,
        )
        self.reading[0] += moved[0]
        self.reading[1] += moved[1]

    # ``_move_axis`` 只会调这四个方向原语，且 ``wait=False``（不等，返回的耗时是 0）。
    def rotate_cw(self, pulses, wait=False, **_kwargs):
        self._apply("base", int(pulses))
        return Mock(estimated_seconds=0.0)

    def rotate_ccw(self, pulses, wait=False, **_kwargs):
        self._apply("base", -int(pulses))
        return Mock(estimated_seconds=0.0)

    def move_forward(self, pulses, wait=False, **_kwargs):
        self._apply("extension", int(pulses))
        return Mock(estimated_seconds=0.0)

    def move_backward(self, pulses, wait=False, **_kwargs):
        self._apply("extension", -int(pulses))
        return Mock(estimated_seconds=0.0)


class LiveRingSource:
    """回放"当前读数"的观测源：每次 ``locate_ring`` 都按假臂的最新状态造一条。"""

    def __init__(self, arm: SimulatedPlantArm) -> None:
        self.arm = arm
        self.calls = 0

    def locate_ring(self, *, timeout=None, digit=None):
        self.calls += 1
        return RingTarget(
            center_px=(640.0, 360.0),
            radius_px=250.0,
            digit=2,
            score=0.95,
            clipped=False,
            # 比例尺 1.0、来源 depth/focal：读数原样进入解算，测试只看闭环本身
            x_mm=self.arm.reading[0],
            y_mm=self.arm.reading[1],
            scale_mm_per_px=1.0,
            scale_source="depth/focal",
        )


def make_closed_loop(plant_scale: float = 1.0):
    """配置雅可比 = JACOBIAN；真实灵敏度 = ``plant_scale`` × JACOBIAN。"""
    plant = VisualServoJacobian(
        base_dx=JACOBIAN.base_dx * plant_scale,
        base_dy=JACOBIAN.base_dy * plant_scale,
        radial_dx=JACOBIAN.radial_dx * plant_scale,
        radial_dy=JACOBIAN.radial_dy * plant_scale,
    )
    arm = SimulatedPlantArm(plant)
    grasp = Grasp(
        config=VisualServoConfig(),
        selector=MaterialSelector(),
        target=VisualServoTarget(x_mm=0.0, y_mm=0.0),
        jacobian=JACOBIAN,
        lift_scale=None,
    )
    grasp._arm = arm
    grasp._source = LiveRingSource(arm)
    return grasp, arm


class ClosedLoopTests(unittest.TestCase):
    """``align_ring``：一步不够时就靠"实测 / 预测"在线折算，两三步入位。"""

    def test_a_matching_machine_lands_in_one_step(self) -> None:
        grasp, arm = make_closed_loop(1.0)
        result = grasp.align_ring(gain=1.0)   # 一步到位：gain=1 意味着全量修正
        self.assertTrue(result.aligned, result.summary())
        self.assertEqual(len(arm.moves), 2, arm.moves)   # 底座 + 伸缩，各一次
        self.assertAlmostEqual(result.response or 0.0, 1.0, places=6)
        self.assertLess(result.final_error_mm, 1.0)

    def test_the_closed_loop_lands_on_the_compensated_target(self) -> None:
        """落料补偿改的是闭环的**终点**：读数停在「target + 补偿」，不是停在 0。"""
        grasp, arm = make_closed_loop(1.0)
        grasp._config = VisualServoConfig(ring_offset_x_mm=7.5)
        result = grasp.align_ring(gain=1.0)
        self.assertTrue(result.aligned, result.summary())
        # 脉冲要取整，落点差着半个脉冲的量级（0.036 mm），所以按 delta 比。
        self.assertAlmostEqual(arm.reading[0], 7.5, delta=0.1)
        self.assertAlmostEqual(arm.reading[1], 0.0, delta=0.1)
        self.assertLess(result.final_error_mm, 0.1, "残差要跟补偿后的目标同一口径")

    def test_a_half_sensitive_machine_converges_by_learning_the_response(self) -> None:
        """真机只有配置的一半灵敏：闭环按实测算出响应比 0.5，第二步就补上了。"""
        grasp, arm = make_closed_loop(0.5)
        result = grasp.align_ring(max_iterations=4, gain=1.0)
        self.assertTrue(result.aligned, result.summary())
        # 响应比是**机器本身**的性质（真实灵敏度 / 配置灵敏度），不随折算变化
        self.assertAlmostEqual(result.response or 0.0, 0.5, places=2)
        # 每一步都朝同一个方向走（不是来回摆）
        bases = [pulses for axis, pulses in arm.moves if axis == "base"]
        self.assertGreaterEqual(len(bases), 1)
        self.assertTrue(all(pulse > 0 for pulse in bases), bases)

    def test_a_double_sensitive_machine_also_converges(self) -> None:
        grasp, _arm = make_closed_loop(2.0)
        result = grasp.align_ring(max_iterations=4, gain=0.6)
        self.assertTrue(result.aligned, result.summary())
        self.assertGreater(result.response or 0.0, 0.5)

    def test_a_pair_of_stale_frames_does_not_report_a_stalled_axis(self) -> None:
        """一步之后连续读到旧位置时，先复测，不能把旧帧当成电机没有响应。"""
        grasp, arm = make_closed_loop(1.0)

        class DelayedSource(LiveRingSource):
            stale_reads = 0
            previous_target = None

            def locate_ring(self, **kwargs):
                if len(self.arm.moves) == 4 and self.stale_reads < 2:
                    self.stale_reads += 1
                    return self.previous_target
                target = super().locate_ring(**kwargs)
                self.previous_target = target
                return target

        grasp._source = DelayedSource(arm)
        result = grasp.align_ring(max_iterations=6, gain=0.6, tolerance_mm=2.0)

        self.assertTrue(result.aligned, result.summary())
        self.assertLessEqual(math.hypot(*arm.reading), 2.0)

    def test_one_weak_base_move_does_not_amplify_the_next_correction(self) -> None:
        """复现现场小步几乎没响应：不能把一次低响应倒数当作下一步的大增益。"""
        grasp, arm = make_closed_loop(1.0)
        arm.reading = [-24.33, -2.88]
        grasp._config = VisualServoConfig(ring_offset_x_mm=6.5, ring_offset_y_mm=-7.0)
        original_apply = arm._apply

        def apply_with_one_weak_base(axis, signed):
            if axis == "base" and len(arm.moves) == 2:
                arm.moves.append((axis, signed))
                arm.pulses[axis] += signed
                moved = arm.plant.displacement_for(signed * 0.1, 0)
                arm.reading[0] += moved[0]
                arm.reading[1] += moved[1]
                return
            original_apply(axis, signed)

        arm._apply = apply_with_one_weak_base
        result = grasp.align_ring(max_iterations=6, gain=0.6, tolerance_mm=2.0)

        for step in result.steps:
            predicted = JACOBIAN.displacement_for(step.base_pulses, step.extension_pulses)
            self.assertLessEqual(math.hypot(*predicted), step.error_mm + 0.1)
        self.assertTrue(result.aligned, result.summary())

    def test_a_stationary_machine_still_aborts_after_remeasurement(self) -> None:
        """复测只排除旧帧；设备持续不动时必须停止，不能靠追加运动掩盖故障。"""
        grasp, arm = make_closed_loop(0.0)
        result = grasp.align_ring(max_iterations=6, gain=0.6, tolerance_mm=2.0)

        self.assertFalse(result.aligned)
        self.assertIn("几乎没有", result.note)
        self.assertAlmostEqual(result.final_error_mm, math.hypot(30.0, 20.0))
        self.assertLessEqual(len(arm.moves), 4, "一轮修正后只允许退回，不能继续盲走")

    def test_a_reversed_axis_stops_instead_of_walking_away(self) -> None:
        """方向反了（接线/标定问题）必须停下并说清楚，而不是继续发帧。

        停完之后还要退回误差最小那一次（这里是起点）：第一步已经把它推远了，
        留在那儿等于挑最差的结果。
        """
        grasp, arm = make_closed_loop(-1.0)
        result = grasp.align_ring(max_iterations=4)
        self.assertFalse(result.aligned)
        self.assertIn("方向相反", result.note)
        self.assertIn("误差最小", result.note)
        # 第一步（两根轴）+ 退回第一步（两根轴），到此为止，不再试探第二步
        self.assertLessEqual(len(arm.moves), 4, "只该走一步、再退回")
        self.assertAlmostEqual(arm.reading[0], 30.0, delta=1.0)

    def test_base_only_leaves_the_extension_alone(self) -> None:
        grasp, arm = make_closed_loop(1.0)
        result = grasp.align_ring(axes=("base",), max_iterations=4)
        self.assertEqual({axis for axis, _ in arm.moves}, {"base"})
        self.assertIsNotNone(result.final_error_mm)

    def test_it_goes_back_to_the_smallest_error_when_the_steps_run_out(self) -> None:
        """走满步数不报错：退回**误差最小**那一次的姿态，而不是停在过冲的最后一步。

        ``gain=4`` 故意过冲：误差 36 mm → 108 mm。两步入数走完，臂必须回到量到 36 mm
        时那个位置（读数 (30, 20)），报的残差也是它。
        """
        grasp, arm = make_closed_loop(1.0)

        result = grasp.align_ring(max_iterations=2, gain=4.0, tolerance_mm=0.5)

        self.assertFalse(result.aligned)
        self.assertIn("误差最小", result.note)
        self.assertAlmostEqual(result.final_error_mm, math.hypot(30.0, 20.0), delta=1.0)
        self.assertAlmostEqual(arm.reading[0], 30.0, delta=1.0)
        self.assertAlmostEqual(arm.reading[1], 20.0, delta=1.0)

    def test_a_small_step_does_not_update_the_response_scale(self) -> None:
        """1 mm 的步子被背隙/噪声主导：只走，不拿它去学响应比（否则 scale 被带飞）。"""
        grasp, arm = make_closed_loop(1.0)
        arm.reading = [1.0, 0.0]

        result = grasp.align_ring(tolerance_mm=0.2, max_iterations=2, gain=1.0)

        self.assertTrue(result.aligned, result.summary())
        self.assertIsNone(result.response, "小于约 2 mm 预测位移的步子不参与学习")

    def test_an_unknown_axis_is_rejected_before_any_motion(self) -> None:
        grasp, arm = make_closed_loop(1.0)
        with self.assertRaisesRegex(ValueError, "lift"):
            grasp.align_ring(axes=("lift",))
        self.assertEqual(arm.moves, [])

    def test_the_result_carries_the_final_plan_like_plan_ring(self) -> None:
        """闭环结果里的 ``plan`` 与 ``plan_ring()`` 是同一套字段，可直接存下来。"""
        grasp, arm = make_closed_loop(0.5)
        result = grasp.align_ring(max_iterations=4, gain=1.0)
        self.assertIsNotNone(result.plan, "闭环必须把最后那一份计划交出来")
        plan = result.plan
        # 收敛之后"该走到的绝对位置"就是当前里程表（剩下的步长已经是 0）
        self.assertEqual(plan.base_motor_pulses, arm.pulses["base"])
        self.assertEqual(plan.extension_motor_pulses, arm.pulses["extension"])
        # 转发属性与 plan 一致，用起来和 plan_ring() 一样
        self.assertAlmostEqual(result.base, plan.base)
        self.assertAlmostEqual(result.extension, plan.extension)
        self.assertEqual(result.pulses, plan.pulses)
        self.assertEqual(result.degrees, plan.degrees)
        self.assertIn("最终绝对位置", result.summary())

    def test_a_closed_loop_without_ring_vision_says_so(self) -> None:
        grasp, _arm = make_closed_loop(1.0)
        grasp._source = Mock(spec=["frame"])
        with self.assertRaisesRegex(VisualServoError, "没有环心定位"):
            grasp.align_ring()


class AssumedDigitTests(unittest.TestCase):
    """按示教姿态逐环放的流程：这一格是几号环**已经知道**，模型读错号不该让整段失败。

    2026-10-29 现场就是这种取景：一张转过角度的 1 号靶标，模型 5 帧全读成 2/3 号，环心
    却一直是准的（框内椭圆精修散布 0.7 px）。按号筛框（``require_digit``）只会得到
    "画面里没有环"。
    """

    class Misread:
        """模型永远报 2 号，环心跟着假臂走。"""

        def __init__(self, arm) -> None:
            self.arm = arm
            self.digits: list = []

        def locate_ring(self, *, timeout=None, digit=None):
            self.digits.append(digit)
            return RingTarget(
                center_px=(640.0, 360.0),
                radius_px=250.0,
                digit=2,
                score=0.8,
                clipped=False,
                x_mm=self.arm.reading[0],
                y_mm=self.arm.reading[1],
                scale_mm_per_px=1.0,
                scale_source="depth/focal",
            )

    def build(self):
        """1 号环的落料补偿是 7.5 mm、2 号是 0：从落点就能看出用的是哪一份。"""
        arm = SimulatedPlantArm(JACOBIAN)
        grasp = Grasp(
            config=VisualServoConfig(
                ring_offsets_mm=((7.5, 0.0), (0.0, 0.0), (0.0, 0.0)),
            ),
            selector=MaterialSelector(),
            target=VisualServoTarget(x_mm=0.0, y_mm=0.0),
            jacobian=JACOBIAN,
            lift_scale=None,
        )
        source = self.Misread(arm)
        grasp._arm = arm
        grasp._source = source
        return grasp, arm, source

    def test_a_misread_digit_still_aligns_with_the_taught_ring_offset(self) -> None:
        grasp, arm, source = self.build()

        result = grasp.align_ring(assume_digit=1, gain=1.0)

        self.assertTrue(result.aligned, result.summary())
        self.assertEqual(result.digit, 1)
        # 补偿取 1 号那份（7.5 mm），不是模型说的 2 号（0 mm）
        self.assertAlmostEqual(arm.reading[0], 7.5, delta=0.1)
        self.assertIn("模型读成 2 号", result.note)
        self.assertIn(1, source.digits, "先按姿态给的号找")
        self.assertIn(None, source.digits, "读不出这个号才退回画面里那个环")

    def test_a_matching_digit_is_not_reported_as_a_misread(self) -> None:
        grasp, _arm = make_closed_loop(1.0)   # 这个源报的就是 2 号

        result = grasp.align_ring(assume_digit=2, gain=1.0)

        self.assertTrue(result.aligned, result.summary())
        self.assertNotIn("模型读成", result.note)

    def test_a_strict_ring_number_still_refuses_a_misread(self) -> None:
        grasp, arm, _source = self.build()

        with self.assertRaisesRegex(VisualServoError, "不是要求的 1 号"):
            grasp.align_ring(require_digit=1, gain=1.0)
        self.assertEqual(arm.moves, [], "号不对时一步都不该走")

    def test_the_two_notions_of_ring_number_cannot_be_mixed(self) -> None:
        grasp, _arm, _source = self.build()

        with self.assertRaisesRegex(ValueError, "只能给一个"):
            grasp.align_ring(require_digit=1, assume_digit=1)


class RingSettleTests(unittest.TestCase):
    """底座摆完还在抖：读数没稳就拿来算步长，抖动会被放大成来回摆。"""

    class Jittery:
        """前几帧读数在抖，之后稳住；每帧都记一次调用。"""

        def __init__(self, readings) -> None:
            self.readings = list(readings)
            self.calls = 0

        def locate_ring(self, *, timeout=None, digit=None):
            x_mm, y_mm = self.readings[min(self.calls, len(self.readings) - 1)]
            self.calls += 1
            return ring_target(reading=(x_mm, y_mm))

    def locate(self, source):
        grasp, _arm, _source = make_grasp(ring_target())
        return grasp._locate_ring_for_plan(
            source, timeout=None, require_digit=None, settle=True
        )

    def test_two_matching_frames_end_the_wait(self) -> None:
        source = self.Jittery([(3.0, 0.0)])
        target = self.locate(source)

        self.assertEqual(source.calls, 2)   # 两帧一致就收工，不读第三次
        self.assertAlmostEqual(target.reading()[0], 3.0, places=6)

    def test_it_reads_through_a_jittering_frame(self) -> None:
        """三帧读数取中位数，不是用最后一帧——最后一帧只是"够近"，不代表最可信。"""
        source = self.Jittery([(3.0, 0.0), (5.0, 0.0), (5.05, 0.02)])
        target = self.locate(source)

        self.assertEqual(source.calls, 3)
        self.assertAlmostEqual(target.reading()[0], 5.0, places=6)
        self.assertAlmostEqual(target.reading()[1], 0.0, places=6)

    def test_never_settling_still_stops_at_the_read_limit(self) -> None:
        source = self.Jittery([(1.0, 0.0), (50.0, 0.0), (100.0, 0.0)])
        target = self.locate(source)

        self.assertEqual(source.calls, 3)   # 有上限：等不到稳定就读满三帧
        self.assertAlmostEqual(target.reading()[0], 50.0, places=6)


if __name__ == "__main__":
    unittest.main()

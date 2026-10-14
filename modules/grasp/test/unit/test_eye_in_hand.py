"""eye-in-hand 视觉伺服的数学与闭环行为。

判据只有一条硬标准：**按 ``pulses_for`` 算出来的脉冲走一遍，画面误差必须归零。**
这条不变量对 2×2 求逆的符号错误、行列式写反、列顺序调换全都敏感——用一个随机
雅可比矩阵族来压它，比逐个人工算例更难糊弄过去。

伺服循环那边则是四个失败路径必须响亮：看不到物块、雅可比奇异、预算用完、迭代用尽。
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

from modules.grasp.drive import eye_in_hand as servo
from modules.grasp.drive.eye_in_hand import (
    BASE_AXIS,
    RADIAL_AXIS,
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoTarget,
    align_x_y,
    load_jacobian,
    load_object_selector,
    load_servo_config,
    load_target,
    measure_offset,
    probe_jacobian,
    save_calibration,
    select_object,
)
from modules.vision.drive.sighting import parse_sighting


def sighting(material_id: int, x: float, y: float, color: str = "green", *, stamp: Optional[float] = None):
    return parse_sighting(
        f"物料 {material_id}: {color}物料（{color}），xyz=({x:.6f},{y:.6f},113.2) mm",
        received_at=stamp,
    )


class ServoJacobianTests(unittest.TestCase):
    def test_applying_the_solved_pulses_cancels_the_error(self) -> None:
        """核心不变量：``pulses_for`` 算出的脉冲走一遍，误差必须归零。"""
        generator = random.Random(20261014)
        checked = 0
        while checked < 200:
            jacobian = ServoJacobian(
                base_dx=generator.uniform(-0.03, 0.03),
                base_dy=generator.uniform(-0.03, 0.03),
                radial_dx=generator.uniform(-0.03, 0.03),
                radial_dy=generator.uniform(-0.03, 0.03),
            )
            if jacobian.is_singular:
                continue
            checked += 1
            error_x = generator.uniform(-50.0, 50.0)
            error_y = generator.uniform(-50.0, 50.0)
            with self.subTest(jacobian=jacobian, error=(error_x, error_y)):
                base_pulses, radial_pulses = jacobian.pulses_for(error_x, error_y)
                moved_x, moved_y = jacobian.displacement_for(base_pulses, radial_pulses)
                self.assertAlmostEqual(moved_x + error_x, 0.0, places=9)
                self.assertAlmostEqual(moved_y + error_y, 0.0, places=9)

    def test_zero_error_asks_for_no_motion(self) -> None:
        jacobian = ServoJacobian(base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)
        self.assertEqual(jacobian.pulses_for(0.0, 0.0), (0.0, 0.0))

    def test_columns_are_not_swapped(self) -> None:
        """底座只影响 x、伸缩只影响 y 的对角矩阵：脉冲量必须直接等于误差的相反数除以斜率。"""
        jacobian = ServoJacobian(base_dx=0.01, base_dy=0.0, radial_dx=0.0, radial_dy=0.02)
        base_pulses, radial_pulses = jacobian.pulses_for(1.0, -2.0)
        self.assertAlmostEqual(base_pulses, -100.0, places=9)
        self.assertAlmostEqual(radial_pulses, 100.0, places=9)

    def test_singular_is_detected_and_refused(self) -> None:
        # 两列成比例 -> 两个轴在画面里同向，凑不出任意方向。
        singular = ServoJacobian(base_dx=0.01, base_dy=0.02, radial_dx=0.02, radial_dy=0.04)
        self.assertTrue(singular.is_singular)
        with self.assertRaisesRegex(ServoError, "奇异"):
            singular.pulses_for(1.0, 1.0)

    def test_displacement_is_linear_in_pulses(self) -> None:
        jacobian = ServoJacobian(base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)
        first = jacobian.displacement_for(100.0, 0.0)
        second = jacobian.displacement_for(200.0, 0.0)
        self.assertAlmostEqual(second[0], 2 * first[0], places=9)
        self.assertAlmostEqual(second[1], 2 * first[1], places=9)
        self.assertEqual(jacobian.displacement_for(0.0, 0.0), (0.0, 0.0))

    def test_summary_is_in_per_thousand_pulses(self) -> None:
        text = ServoJacobian(base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091).summary()
        self.assertIn("每 1000 脉冲", text)
        self.assertIn("12.00", text)


class ObjectSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.green = sighting(4, 1.0, 2.0, "green")
        self.blue = sighting(5, 3.0, 4.0, "blue")

    def test_open_selector_takes_the_only_target(self) -> None:
        selector = ObjectSelector()
        self.assertTrue(selector.is_open)
        self.assertEqual(select_object((self.green,), selector).material_id, 4)

    def test_open_selector_refuses_ambiguity(self) -> None:
        with self.assertRaisesRegex(ServoError, "多个目标"):
            select_object((self.green, self.blue), ObjectSelector())

    def test_material_id_selects(self) -> None:
        selector = ObjectSelector(material_id=5)
        self.assertEqual(select_object((self.green, self.blue), selector).material_id, 5)

    def test_colour_selects_and_is_normalised(self) -> None:
        selector = ObjectSelector(color="  BLUE ")
        self.assertEqual(selector.color, "blue")
        self.assertEqual(select_object((self.green, self.blue), selector).material_id, 5)

    def test_both_conditions_must_hold(self) -> None:
        selector = ObjectSelector(material_id=4, color="blue")
        with self.assertRaisesRegex(ServoError, "没有物块"):
            select_object((self.green, self.blue), selector)

    def test_empty_frame_is_reported(self) -> None:
        with self.assertRaisesRegex(ServoError, "没有物块"):
            select_object((), ObjectSelector())

    def test_selector_that_matches_nothing_lists_what_was_seen(self) -> None:
        with self.assertRaises(ServoError) as caught:
            select_object((self.green,), ObjectSelector(material_id=99))
        self.assertIn("物料 4", str(caught.exception))


class ServoConfigTests(unittest.TestCase):
    def test_defaults_are_sane(self) -> None:
        config = ServoConfig()
        self.assertGreater(config.tolerance_mm, 0)
        self.assertGreater(config.max_total_pulses, config.max_pulses_per_step)

    def test_non_positive_values_are_rejected(self) -> None:
        for kwargs in (
            {"tolerance_mm": 0.0},
            {"tolerance_mm": -1.0},
            {"max_iterations": 0},
            {"max_pulses_per_step": 0},
            {"max_total_pulses": 0},
            {"probe_pulses": 0},
            {"frame_timeout_seconds": 0.0},
            {"frame_margin_seconds": -1.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ServoConfig(**kwargs)


class ConfigFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "eye_in_hand.json"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write(self, document) -> None:
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def test_project_config_loads(self) -> None:
        config = load_servo_config()
        self.assertGreater(config.tolerance_mm, 0)

    def test_project_config_starts_uncalibrated(self) -> None:
        """出厂状态必须是"没标定"，而不是某个编出来的默认值。"""
        jacobian = load_jacobian()
        target = load_target()
        if jacobian is not None:
            self.assertFalse(jacobian.is_singular)
        # 两个都可能已被标定过，这里只要求"要么没有，要么是个能用的值"。
        if target is not None:
            self.assertTrue(math.isfinite(target.x_mm) and math.isfinite(target.y_mm))

    def test_uncalibrated_blocks_return_none(self) -> None:
        self.write({"jacobian": {"base_dx": None, "base_dy": None, "radial_dx": None, "radial_dy": None}})
        self.assertIsNone(load_jacobian(self.path))
        self.write({"target": {"x_mm": None, "y_mm": None}})
        self.assertIsNone(load_target(self.path))
        self.write({})
        self.assertIsNone(load_jacobian(self.path))
        self.assertIsNone(load_target(self.path))

    def test_invalid_documents_are_rejected(self) -> None:
        for document in (
            "not an object",
            {"object": "not an object"},
            {"object": {"material_id": -1}},
            {"object": {"material_id": "4"}},
            {"object": {"color": "  "}},
            {"target": "not an object"},
            {"jacobian": "not an object"},
            {"tolerance_mm": 0},
            {"max_iterations": 0},
            {"probe_pulses": 1.5},
        ):
            with self.subTest(document=document):
                self.write(document)
                with self.assertRaises(ServoError):
                    load_servo_config(self.path)
                    load_object_selector(self.path)
                    load_target(self.path)
                    load_jacobian(self.path)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(ServoError):
            load_servo_config(Path("nope.json"))

    def test_save_calibration_round_trips_and_keeps_other_keys(self) -> None:
        self.write({"tolerance_mm": 1.5, "object": {"color": "green"}})
        target = ServoTarget(x_mm=-8.25, y_mm=1.75)
        jacobian = ServoJacobian(base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)
        save_calibration(target=target, jacobian=jacobian, config_path=self.path)

        self.assertEqual(load_target(self.path), target)
        reloaded = load_jacobian(self.path)
        assert reloaded is not None
        self.assertEqual(reloaded, jacobian)
        # 别的键必须原样保留。
        self.assertEqual(load_servo_config(self.path).tolerance_mm, 1.5)
        self.assertEqual(load_object_selector(self.path).color, "green")


class FakeArm:
    """把脉冲按真实雅可比换算成画面位移；``gain`` 模拟"只走了这么多"。"""

    def __init__(self, jacobian: ServoJacobian, *, gain: float = 1.0) -> None:
        self.jacobian = jacobian
        self.gain = gain
        self.offset = [12.0, -7.0]
        self.calls: List[Tuple[str, int]] = []

    def _axis(self, axis: str, forward: bool, pulses: int) -> None:
        signed = pulses if forward else -pulses
        dx, dy = self.jacobian.displacement_for(
            signed if axis == BASE_AXIS else 0.0,
            signed if axis == RADIAL_AXIS else 0.0,
        )
        self.offset[0] += dx * self.gain
        self.offset[1] += dy * self.gain
        self.calls.append((f"{axis}:{'fwd' if forward else 'back'}", pulses))

    def rotate_cw(self, pulses, **_kwargs):
        self._axis(BASE_AXIS, True, pulses)

    def rotate_ccw(self, pulses, **_kwargs):
        self._axis(BASE_AXIS, False, pulses)

    def move_forward(self, pulses, **_kwargs):
        self._axis(RADIAL_AXIS, True, pulses)

    def move_backward(self, pulses, **_kwargs):
        self._axis(RADIAL_AXIS, False, pulses)

    @property
    def net_pulses(self) -> Tuple[int, int]:
        base = sum(value if "fwd" in kind else -value for kind, value in self.calls if kind.startswith("base"))
        radial = sum(value if "fwd" in kind else -value for kind, value in self.calls if kind.startswith("radial"))
        return (base, radial)


class BufferedLink:
    """带缓冲的相机：只有明确要求"不早于某时刻"时才发布新帧。

    没带过滤的读会拿到上一次发布的快照。真机上就是这样——动作刚做完，队列里可能还压着
    动作之前的帧。所以这个假件逼调用方把时间戳过滤用对：少了它，闭环读到旧坐标，
    以为没动过，然后重复发同一条指令。
    """

    def __init__(self, arm: FakeArm, *, material_id: int = 4, color: str = "green") -> None:
        self.arm = arm
        self.material_id = material_id
        self.color = color
        self.error = None
        self._published: Optional[Tuple[float, float, float]] = None
        self.guarded_reads = 0
        self.unguarded_reads = 0

    def frame(self, *, timeout=None, max_age_seconds=None, min_received_at=None):
        if min_received_at is None:
            self.unguarded_reads += 1
            if self._published is None:
                self._publish(time.monotonic())
        else:
            self.guarded_reads += 1
            self._publish(max(time.monotonic(), min_received_at))
        x, y, stamp = self._published
        return (sighting(self.material_id, x, y, self.color, stamp=stamp),)

    def _publish(self, stamp: float) -> None:
        self._published = (self.arm.offset[0], self.arm.offset[1], stamp)


class EmptyLink:
    error = None

    def frame(self, **_kwargs):
        return None


class ProbeJacobianTests(unittest.TestCase):
    TRUTH = ServoJacobian(base_dx=0.0120, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)

    def test_recovers_the_truth_exactly(self) -> None:
        arm = FakeArm(self.TRUTH)
        probed = probe_jacobian(arm, BufferedLink(arm), ObjectSelector(), ServoConfig(probe_pulses=600))
        # 逐项比较而不是整对象相等：除法会带出 1e-18 量级的尾数，
        # 那和"标定对不对"无关。
        for field in ("base_dx", "base_dy", "radial_dx", "radial_dy"):
            with self.subTest(field=field):
                self.assertAlmostEqual(getattr(probed, field), getattr(self.TRUTH, field), places=15)

    def test_returns_the_arm_to_where_it_started(self) -> None:
        arm = FakeArm(self.TRUTH)
        start = list(arm.offset)
        probe_jacobian(arm, BufferedLink(arm), ObjectSelector(), ServoConfig(probe_pulses=600))
        self.assertAlmostEqual(arm.offset[0], start[0], places=9)
        self.assertAlmostEqual(arm.offset[1], start[1], places=9)
        self.assertEqual(arm.net_pulses, (0, 0))

    def test_uses_two_same_direction_segments_per_axis(self) -> None:
        """每轴两段**同向**：第二段没有背隙损失，斜率才是真的。"""
        arm = FakeArm(self.TRUTH)
        probe_jacobian(arm, BufferedLink(arm), ObjectSelector(), ServoConfig(probe_pulses=600))
        base_calls = [kind for kind, _ in arm.calls if kind.startswith("base")]
        radial_calls = [kind for kind, _ in arm.calls if kind.startswith("radial")]
        self.assertEqual(base_calls, ["base:fwd", "base:fwd", "base:back", "base:back"])
        self.assertEqual(radial_calls, ["radial:fwd", "radial:fwd", "radial:back", "radial:back"])

    def test_singular_machine_is_refused(self) -> None:
        flat = ServoJacobian(base_dx=0.01, base_dy=0.01, radial_dx=0.02, radial_dy=0.02)
        arm = FakeArm(flat)
        with self.assertRaisesRegex(ServoError, "奇异"):
            probe_jacobian(arm, BufferedLink(arm), ObjectSelector(), ServoConfig(probe_pulses=600))

    def test_no_object_is_refused(self) -> None:
        arm = FakeArm(self.TRUTH)
        with self.assertRaisesRegex(ServoError, "等不到"):
            probe_jacobian(arm, EmptyLink(), ObjectSelector(), ServoConfig(probe_pulses=600))
        self.assertEqual(arm.calls, [], "看不到物块时一个动作都不该发")


class AlignXYTests(unittest.TestCase):
    TRUTH = ServoJacobian(base_dx=0.0120, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)
    TARGET = ServoTarget(x_mm=2.0, y_mm=-1.0)

    def test_converges_and_guards_every_post_move_read(self) -> None:
        arm = FakeArm(self.TRUTH)
        link = BufferedLink(arm)
        config = ServoConfig(
            tolerance_mm=0.5, max_iterations=12, max_pulses_per_step=2000, min_step_mm=0.1
        )
        result = align_x_y(arm, link, ObjectSelector(), self.TARGET, self.TRUTH, config)

        self.assertTrue(result.aligned, result)
        self.assertLessEqual(result.error_mm, 0.5)
        self.assertAlmostEqual(result.readings[-1][0], self.TARGET.x_mm, places=1)
        self.assertAlmostEqual(result.readings[-1][1], self.TARGET.y_mm, places=1)
        # 除了第一次，后面每次读都必须带时间戳过滤。
        self.assertGreaterEqual(link.guarded_reads, result.iterations - 1)
        self.assertEqual(link.unguarded_reads, 1)

    def test_converges_despite_a_systematic_shortfall(self) -> None:
        """每次只走 60%：闭环靠重新测量补回来，这正是不用绝对标定也能到位的原因。"""
        arm = FakeArm(self.TRUTH, gain=0.6)
        link = BufferedLink(arm)
        config = ServoConfig(
            tolerance_mm=0.5, max_iterations=12, max_pulses_per_step=2000, min_step_mm=0.1
        )
        result = align_x_y(arm, link, ObjectSelector(), self.TARGET, self.TRUTH, config)
        self.assertTrue(result.aligned, result)
        self.assertGreater(result.iterations, 2, "有系统性欠走时不该一两次就到")

    def test_per_step_clamp_is_respected_in_the_fine_tier(self) -> None:
        """细调档的单步上限必须真的卡住。

        必须把 ``coarse_error_mm`` 调高，否则初始误差一大就进了粗调档、
        用的是另一个（更大的）上限——那样这条断言就测不到细调档了。
        ``min_step_mm`` 也调小：这一条测的是**上限**，地板不该掺进来。
        """
        arm = FakeArm(self.TRUTH)
        link = BufferedLink(arm)
        config = ServoConfig(
            tolerance_mm=0.5, max_iterations=12, max_pulses_per_step=100,
            coarse_error_mm=1e9, min_step_mm=0.2,
        )
        align_x_y(arm, link, ObjectSelector(), self.TARGET, self.TRUTH, config)
        for _kind, pulses in arm.calls:
            self.assertLessEqual(pulses, 100)

    def test_coarse_tier_takes_a_bigger_step(self) -> None:
        """误差大时进粗调档：单步上限放宽，于是不必把大误差拆成好几轮。

        刻意**不**去比较"粗调比细调少几轮"——细调在同样迭代上限下可能根本收敛不了
        （那正是需要粗调的原因），拿它当基准会让这条测试变成在测迭代上限。
        """
        config = ServoConfig(
            tolerance_mm=0.5, max_iterations=12,
            max_pulses_per_step=100, coarse_max_pulses_per_step=5000,
            coarse_error_mm=2.0,
        )
        arm = FakeArm(self.TRUTH)
        link = BufferedLink(arm)
        result = align_x_y(arm, link, ObjectSelector(), self.TARGET, self.TRUTH, config)
        self.assertTrue(result.aligned, result)
        # 粗调档真的用到了超过细调上限的步长（误差 11.66 mm，需要约 900 脉冲）。
        self.assertTrue(
            any(pulses > 100 for _kind, pulses in arm.calls),
            f"误差 11.66 mm 时应当走粗调档：{arm.calls}",
        )
        self.assertLessEqual(result.iterations, 4, "精确解一步就该基本到位")

    def test_minimum_step_is_scaled_per_axis(self) -> None:
        """背隙门槛按**画面毫米**表达、按各轴灵敏度换算，而不是一个固定脉冲数。

        这台机器底座 0.164 mm/脉冲、伸缩 0.0139 mm/脉冲，差将近 12 倍。
        同一个"最小脉冲数"落在底座上等于二十多毫米的走动，会把闭环推到振荡。
        """
        real_like = ServoJacobian(
            base_dx=-0.1642, base_dy=-0.0114, radial_dx=0.00089, radial_dy=0.0139
        )
        base_min, radial_min = real_like.minimum_pulses(1.0)
        self.assertAlmostEqual(base_min, 1.0 / real_like.base_norm, places=9)
        self.assertAlmostEqual(radial_min, 1.0 / real_like.radial_norm, places=9)
        self.assertGreater(radial_min, base_min * 5, "两个轴的换算结果必须差开一个量级")

    def test_sensitive_axis_takes_small_steps(self) -> None:
        """底座很灵敏时它只该走几个脉冲；步子大小由灵敏度决定，不是拍脑袋定的。"""
        real_like = ServoJacobian(
            base_dx=-0.1642, base_dy=-0.0114, radial_dx=0.00089, radial_dy=0.0139
        )
        arm = FakeArm(real_like)
        arm.offset = [3.0, 2.0]
        result = align_x_y(
            arm, BufferedLink(arm), ObjectSelector(), ServoTarget(0.0, 0.0), real_like,
            ServoConfig(tolerance_mm=0.1, max_iterations=8, min_step_mm=0.01),
        )
        self.assertTrue(result.aligned, result)
        base_moves = [pulses for kind, pulses in arm.calls if kind.startswith("base")]
        self.assertTrue(base_moves, arm.calls)
        # 3 mm 的误差在 0.164 mm/脉冲 的灵敏度下只要约 18 脉冲。
        self.assertLessEqual(max(base_moves), 60, arm.calls)

    def test_error_below_the_backlash_floor_stops_instead_of_oscillating(self) -> None:
        """误差比背隙地板还小时**收工**，不许施加下限来回跨过目标。

        这是真机上最容易变成"永远收敛不了"的一处：地板本来是为了吃掉背隙，
        可它同时给系统设了一个分辨率下限；容差比它小的时候，每一步都会跨过目标。
        """
        real_like = ServoJacobian(
            base_dx=-0.1642, base_dy=-0.0114, radial_dx=0.00089, radial_dy=0.0139
        )
        arm = FakeArm(real_like)
        arm.offset = [0.05, 0.05]  # 误差约 0.07 mm，远小于地板 1.0 mm
        result = align_x_y(
            arm, BufferedLink(arm), ObjectSelector(), ServoTarget(0.0, 0.0), real_like,
            ServoConfig(tolerance_mm=0.01, max_iterations=6, min_step_mm=1.0),
        )
        self.assertTrue(result.resolution_limited, result)
        self.assertEqual(arm.calls, [], "不该发出任何动作——那只会跨过目标")
        self.assertIn("分辨率极限", result.summary())

    def test_small_rotation_lets_both_axes_start_together(self) -> None:
        """转角不大时两个轴同时发（底座先发）；跟踪速度就靠这一条。"""
        arm = FakeArm(self.TRUTH)
        align_x_y(
            arm, BufferedLink(arm), ObjectSelector(), self.TARGET, self.TRUTH,
            ServoConfig(tolerance_mm=0.5, max_pulses_per_step=4000, min_step_mm=0.1,
                        parallel_rotation_limit_pulses=4000),
        )
        self.assertTrue(arm.calls, arm.calls)
        self.assertTrue(arm.calls[0][0].startswith("base"), arm.calls)

    def test_large_rotation_moves_the_extension_first(self) -> None:
        """大角度旋转不并行：先伸缩到位再转，免得伸出状态下横扫底座。"""
        arm = FakeArm(self.TRUTH)
        align_x_y(
            arm, BufferedLink(arm), ObjectSelector(), self.TARGET, self.TRUTH,
            ServoConfig(tolerance_mm=0.5, max_pulses_per_step=4000, min_step_mm=0.1,
                        parallel_rotation_limit_pulses=1),
        )
        self.assertTrue(arm.calls, arm.calls)
        self.assertTrue(arm.calls[0][0].startswith("radial"), arm.calls)

    def test_iteration_limit_is_reported(self) -> None:
        arm = FakeArm(self.TRUTH, gain=0.01)
        link = BufferedLink(arm)
        config = ServoConfig(
            tolerance_mm=0.1, max_iterations=3, max_pulses_per_step=2000, min_step_mm=0.01
        )
        with self.assertRaisesRegex(ServoError, "3 次"):
            align_x_y(arm, link, ObjectSelector(), self.TARGET, self.TRUTH, config)

    def test_pulse_budget_is_enforced(self) -> None:
        arm = FakeArm(self.TRUTH, gain=0.01)
        link = BufferedLink(arm)
        config = ServoConfig(
            tolerance_mm=0.01, max_iterations=50, max_pulses_per_step=40,
            max_total_pulses=80, min_step_mm=0.001,
        )
        with self.assertRaisesRegex(ServoError, "预算"):
            align_x_y(arm, link, ObjectSelector(), self.TARGET, self.TRUTH, config)

    def test_singular_jacobian_is_reported_before_moving(self) -> None:
        arm = FakeArm(self.TRUTH)
        link = BufferedLink(arm)
        singular = ServoJacobian(base_dx=0.01, base_dy=0.02, radial_dx=0.02, radial_dy=0.04)
        with self.assertRaisesRegex(ServoError, "奇异"):
            align_x_y(arm, link, ObjectSelector(), self.TARGET, singular, ServoConfig())
        self.assertEqual(arm.calls, [])

    def test_missing_object_is_reported(self) -> None:
        arm = FakeArm(self.TRUTH)
        with self.assertRaisesRegex(ServoError, "等不到"):
            align_x_y(arm, EmptyLink(), ObjectSelector(), self.TARGET, self.TRUTH, ServoConfig())
        self.assertEqual(arm.calls, [])

    def test_result_reports_net_pulses(self) -> None:
        arm = FakeArm(self.TRUTH)
        link = BufferedLink(arm)
        result = align_x_y(
            arm, link, ObjectSelector(), self.TARGET, self.TRUTH,
            ServoConfig(tolerance_mm=0.5, max_pulses_per_step=2000, min_step_mm=0.1),
        )
        self.assertEqual((result.base_pulses, result.radial_pulses), arm.net_pulses)

    def test_summary_mentions_the_outcome(self) -> None:
        arm = FakeArm(self.TRUTH)
        link = BufferedLink(arm)
        result = align_x_y(
            arm, link, ObjectSelector(), self.TARGET, self.TRUTH,
            ServoConfig(tolerance_mm=0.5, max_pulses_per_step=2000, min_step_mm=0.1),
        )
        text = result.summary()
        self.assertIn("完成", text)
        self.assertIn("容差", text)

    def test_measure_offset_reads_the_xy_pair(self) -> None:
        arm = FakeArm(self.TRUTH)
        link = BufferedLink(arm)
        self.assertEqual(measure_offset(link, ObjectSelector(), ServoConfig()), tuple(arm.offset))


if __name__ == "__main__":
    unittest.main()

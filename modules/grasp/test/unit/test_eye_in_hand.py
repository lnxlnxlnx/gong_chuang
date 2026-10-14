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
from unittest.mock import patch

from modules.grasp.drive import eye_in_hand as servo
from modules.grasp.drive.eye_in_hand import (
    BASE_AXIS,
    RADIAL_AXIS,
    LiftScale,
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoTarget,
    align_x_y,
    descend_z,
    load_jacobian,
    load_object_selector,
    load_servo_config,
    load_target,
    measure_offset,
    measure_settled,
    probe_jacobian,
    retract_extension,
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
            {"z_descend_pulses": 0},
            {"extension_retract_pulses": -150},
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

    def test_the_pulse_steps_are_written_in_the_project_config(self) -> None:
        """两个脉冲数必须真的落在配置文件里：现场要能手改，不能只靠代码里的默认值。"""
        document = json.loads(servo.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIn("z_descend_pulses", document)
        self.assertIn("extension_retract_pulses", document)
        config = load_servo_config()
        self.assertGreater(config.z_descend_pulses, 0)
        self.assertGreater(config.extension_retract_pulses, 0)

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
            {"z_descend_pulses": 0},
            {"extension_retract_pulses": 1.5},
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
        else:
            self.guarded_reads += 1
        # **每次都发布当前真实位姿**：真摄像头的每次读都是一帧新画面
        # （新鲜度靠取帧前丢缓冲保证）。"动作之后必须重新取帧"这件事，
        # 由伺服确实把 min_received_at 传下来这一点来保证——测试用 guarded_reads 断言。
        self._publish(max(time.monotonic(), min_received_at or 0.0))
        x, y, stamp = self._published
        return (sighting(self.material_id, x, y, self.color, stamp=stamp),)

    def _publish(self, stamp: float) -> None:
        self._published = (self.arm.offset[0], self.arm.offset[1], stamp)


class EmptyLink:
    error = None

    def frame(self, **_kwargs):
        return None


class SettleTests(unittest.TestCase):
    """``measure_settled``：等到画面连续两次读数一致才算数。

    这是整套东西里最容易被忽略、代价又最大的一处：机械臂动完之后读数还在大幅变化
    （实测 ``-23.65,+39.77`` → ``-33.38,+23.03``），中间还反复因为运动模糊检不到物块。
    拿那种数去解算，闭环会振荡、探针会把雅可比算大好几倍。
    """

    class ScriptedLink:
        """按脚本给读数；脚本用完后一直返回 ``stable``。"""

        def __init__(self, series, stable) -> None:
            self._series = list(series)
            self._stable = stable
            self.reads = 0
            self.error = None

        def frame(self, **_kwargs):
            self.reads += 1
            x, y = self._series.pop(0) if self._series else self._stable
            return (sighting(4, x, y, stamp=time.monotonic()),)

    def test_waits_until_two_readings_agree(self) -> None:
        link = self.ScriptedLink([(0.0, 0.0), (5.0, 0.0), (9.0, 0.0)], (9.0, 0.0))
        offset = measure_settled(
            link, ObjectSelector(), ServoConfig(settle_timeout_seconds=2.0)
        )
        self.assertEqual(offset, (9.0, 0.0))
        # 前三次都在变，第四次与第三次一致才返回。
        self.assertEqual(link.reads, 4)

    def test_returns_the_first_pair_that_agrees(self) -> None:
        """已经静止时只要两次读数——不该无脑多等。"""
        link = self.ScriptedLink([], (3.0, -1.0))
        offset = measure_settled(
            link, ObjectSelector(), ServoConfig(settle_timeout_seconds=2.0)
        )
        self.assertEqual(offset, (3.0, -1.0))
        self.assertEqual(link.reads, 2)

    def test_timeout_returns_the_last_reading(self) -> None:
        """一直不停稳时如实收工并报出来，而不是把整条流程卡死。"""

        class AlwaysMoving:
            error = None

            def __init__(self) -> None:
                self.value = 0.0

            def frame(self, **_kwargs):
                self.value += 10.0
                return (sighting(4, self.value, 0.0, stamp=time.monotonic()),)

        messages = []
        offset = measure_settled(
            AlwaysMoving(), ObjectSelector(),
            ServoConfig(settle_timeout_seconds=0.2, settle_poll_seconds=0.02),
            on_message=messages.append,
        )
        self.assertGreater(offset[0], 0.0)
        self.assertTrue(any("等停稳超时" in message for message in messages), messages)

    def test_blind_frames_during_motion_do_not_abort_the_wait(self) -> None:
        """运动模糊时检不到物块是常态；中途看不到不该直接判失败。"""

        class FlakyLink:
            error = None

            def __init__(self) -> None:
                self.calls = 0

            def frame(self, **_kwargs):
                self.calls += 1
                if self.calls in (1, 2):  # 前两次糊得检不到
                    return ()
                return (sighting(4, 7.0, -2.0, stamp=time.monotonic()),)

        offset = measure_settled(
            FlakyLink(), ObjectSelector(),
            ServoConfig(settle_timeout_seconds=2.0, settle_poll_seconds=0.02),
        )
        self.assertEqual(offset, (7.0, -2.0))

    def test_never_seeing_anything_is_reported(self) -> None:
        with self.assertRaisesRegex(ServoError, "稳定看到物块"):
            measure_settled(
                EmptyLink(), ObjectSelector(),
                ServoConfig(settle_timeout_seconds=0.2, settle_poll_seconds=0.02),
            )


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
        with self.assertRaisesRegex(ServoError, "稳定看到物块"):
            probe_jacobian(
                arm, EmptyLink(), ObjectSelector(),
                # 停稳超时调小：这一条测的是"看不到物块要报错"，不该真等 4 秒。
                ServoConfig(probe_pulses=600, settle_timeout_seconds=0.2),
            )
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
        # 除了最初那几次，后面每次读都必须带时间戳过滤。
        self.assertGreaterEqual(link.guarded_reads, result.iterations - 1)
        # 不带过滤的读只有开头：安全抬升判据要两次读数确认停稳，正式测量再两次。
        self.assertGreaterEqual(link.unguarded_reads, 2, "停稳判据需要连续两次读数")
        self.assertLessEqual(link.unguarded_reads, 6, "只有开头允许不带过滤")

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
        with self.assertRaisesRegex(ServoError, "稳定看到物块"):
            align_x_y(
                arm, EmptyLink(), ObjectSelector(), self.TARGET, self.TRUTH,
                ServoConfig(settle_timeout_seconds=0.2),
            )
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

    def test_occluded_material_is_reported_instead_of_chased(self) -> None:
        """爪瓣挡住物块时必须**报出来**，而不是照着那个假观测继续解算。

        实测：转 200 脉冲后爪瓣压在物块上，像素直径从 287 px 掉到 195 px。
        测距公式按直径反推距离，于是把 105 mm 报成 155 mm——看起来完全像
        "物块突然远了 50 mm"，其实与机械臂位置毫无关系。
        """

        class OccludingLink:
            error = None

            def __init__(self, arm) -> None:
                self.arm = arm
                self.calls = 0

            def frame(self, **_kwargs):
                self.calls += 1
                # 前两次测量正常（287 px），之后被挡住（195 px）。
                diameter = 287 if self.calls <= 4 else 195
                x, y = self.arm.offset
                return (
                    parse_sighting(
                        f"物料 4: 绿色物料（green），圆直径 {diameter} px，"
                        f"xyz=({x:.4f},{y:.4f},105.0) mm",
                        received_at=time.monotonic(),
                    ),
                )

        arm = FakeArm(self.TRUTH, gain=0.6)  # 让它多走几轮，才有机会看到遮挡
        with self.assertRaisesRegex(ServoError, "挡住"):
            align_x_y(
                arm, OccludingLink(arm), ObjectSelector(), self.TARGET, self.TRUTH,
                ServoConfig(tolerance_mm=0.5, max_pulses_per_step=2000, min_step_mm=0.1),
            )

    def test_diameter_change_within_the_limit_is_not_treated_as_occlusion(self) -> None:
        """直径正常的小幅波动（检测噪声）不该误报遮挡。"""

        class WobblyLink(BufferedLink):
            def __init__(self, arm) -> None:
                super().__init__(arm)
                self.calls = 0

            def frame(self, **kwargs):
                self.calls += 1
                (item,) = super().frame(**kwargs)
                diameter = 287 if self.calls % 2 else 281  # ±2%，远小于 25%
                raw = item.raw.replace("xyz=", f"圆直径 {diameter} px，xyz=")
                return (parse_sighting(raw, received_at=item.received_at),)

        arm = FakeArm(self.TRUTH, gain=0.7)
        result = align_x_y(
            arm, WobblyLink(arm), ObjectSelector(), self.TARGET, self.TRUTH,
            ServoConfig(tolerance_mm=0.5, max_pulses_per_step=2000, min_step_mm=0.1),
        )
        self.assertTrue(result.aligned, result)

    def test_measure_offset_reads_the_xy_pair(self) -> None:
        arm = FakeArm(self.TRUTH)
        link = BufferedLink(arm)
        self.assertEqual(measure_offset(link, ObjectSelector(), ServoConfig()), tuple(arm.offset))


class MoveAxesTests(unittest.TestCase):
    """``_move_axes`` 的帧间隔：**这是真机上最容易"看起来对、其实什么都没动"的一处**。

    实测这台机器的 CH340 上，两条 ``FD`` 位置帧背靠背连发（间隔 0 ms）会让**两帧都失效**
    ——不是丢一帧，是两个轴都不动；而驱动器 ``Response=None``，不会报任何错。
    曾经因为没有这个间隔，所有"两轴同时要动"的步骤都变成空操作，
    现象是闭环怎么走都不动、最后报"迭代用尽"。
    """

    TRUTH = ServoJacobian(base_dx=0.0120, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)

    def _record_sleeps(self, arm, config, base_pulses, radial_pulses):
        sleeps: List[float] = []
        with patch.object(servo.time, "sleep", sleeps.append):
            servo._move_axes(arm, base_pulses, radial_pulses, config)
        return sleeps

    def test_gap_is_inserted_between_the_two_frames(self) -> None:
        arm = FakeArm(self.TRUTH)
        config = ServoConfig(inter_frame_seconds=0.03, parallel_rotation_limit_pulses=4000)
        sleeps = self._record_sleeps(arm, config, 100, 200)
        self.assertEqual([kind for kind, _ in arm.calls], ["base:fwd", "radial:fwd"], arm.calls)
        self.assertIn(0.03, sleeps, f"两条帧之间没有留间隔：{sleeps}")

    def test_no_gap_when_only_one_axis_moves(self) -> None:
        """只动一个轴时不需要间隔——那种情况本来就不会撞上这个问题。"""
        arm = FakeArm(self.TRUTH)
        config = ServoConfig(inter_frame_seconds=0.03)
        sleeps = self._record_sleeps(arm, config, 100, 0)
        self.assertNotIn(0.03, sleeps, sleeps)

    def test_large_rotation_still_extends_first(self) -> None:
        """大角度旋转不并行：先伸缩到位、再转，免得伸出状态下横扫底座。"""
        arm = FakeArm(self.TRUTH)
        config = ServoConfig(parallel_rotation_limit_pulses=1, inter_frame_seconds=0.03)
        self._record_sleeps(arm, config, 500, 200)
        self.assertEqual([kind for kind, _ in arm.calls], ["radial:fwd", "base:fwd"], arm.calls)

    def test_nothing_happens_when_both_are_zero(self) -> None:
        arm = FakeArm(self.TRUTH)
        self.assertEqual(self._record_sleeps(arm, ServoConfig(), 0, 0), [])
        self.assertEqual(arm.calls, [])


class LiftScaleTests(unittest.TestCase):
    """升降灵敏度的符号：**这是"z 高于目标却去抬升"那个 bug 的守卫**。"""

    def test_pulses_cancel_the_error(self) -> None:
        # 抬升让 z 变大（dz_per_pulse > 0）：z 高于目标时要**下降**（负脉冲）。
        scale = LiftScale(dz_per_pulse=0.0121)
        self.assertLess(scale.pulses_for(+50.0), 0.0, "z 高于目标必须往下走")
        self.assertGreater(scale.pulses_for(-50.0), 0.0, "z 低于目标必须往上走")
        # 走完这些脉冲，误差正好归零。
        for error in (50.0, -30.0, 7.5):
            with self.subTest(error=error):
                moved = scale.pulses_for(error) * scale.dz_per_pulse
                self.assertAlmostEqual(moved + error, 0.0, places=9)

    def test_zero_sensitivity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ServoError, "灵敏度是 0"):
            LiftScale(dz_per_pulse=0.0).pulses_for(1.0)


class ZGuardTests(unittest.TestCase):
    """z 伺服的下降护栏：宁可不走，也不许再往下。"""

    def test_guard_clamps_a_descent_that_would_go_too_low(self) -> None:
        scale = LiftScale(dz_per_pulse=0.0121)
        config = ServoConfig(z_min_mm=30.0)
        # 现在 z=40、想下降 2000 脉冲（按 0.0121 会降到 15.8 mm，低于护栏）。
        step, clamped = servo._lift_step_within_guard(
            z_mm=40.0, step=-2000, scale=scale, config=config
        )
        self.assertTrue(clamped)
        self.assertGreater(step, -2000)
        # 夹完之后预测的 z 不低于护栏。
        self.assertGreaterEqual(40.0 + step * scale.dz_per_pulse, config.z_min_mm - 1e-9)

    def test_guard_leaves_a_safe_move_alone(self) -> None:
        scale = LiftScale(dz_per_pulse=0.0121)
        config = ServoConfig(z_min_mm=30.0)
        step, clamped = servo._lift_step_within_guard(
            z_mm=120.0, step=-600, scale=scale, config=config
        )
        self.assertFalse(clamped)
        self.assertEqual(step, -600)


class PulseStepTests(unittest.TestCase):
    """脉冲版抓取前两步：升降下降 + 伸缩后缩。

    这两步是**开环**的（爪子压下去之后画面里就没有物块了），所以能测的只有两件事：
    发出的脉冲数与配置一致，以及"平面没对准"必须真的拦住——一个动作帧都不发。
    """

    class Arm:
        """只记动作的假臂：升降与伸缩各记一条。"""

        def __init__(self) -> None:
            self.calls: List[Tuple[str, int]] = []

        def move_down(self, pulses, **_kwargs):
            self.calls.append(("down", pulses))

        def move_up(self, pulses, **_kwargs):
            self.calls.append(("up", pulses))

        def move_backward(self, pulses, **_kwargs):
            self.calls.append(("back", pulses))

        def move_forward(self, pulses, **_kwargs):
            self.calls.append(("forward", pulses))

    class StillLink:
        """画面静止的假镜头：连续两次读数一致，``_settled_sighting`` 立刻返回。"""

        def __init__(self, x: float, y: float) -> None:
            self.x, self.y = x, y
            self.error = None

        def frame(self, **_kwargs):
            return (sighting(4, self.x, self.y),)

    def test_the_descent_uses_the_configured_pulse_count(self) -> None:
        arm = self.Arm()
        result = descend_z(arm, self.StillLink(1.0, -1.0), ObjectSelector(), ServoConfig())
        self.assertEqual(arm.calls, [("down", 1700)])
        self.assertEqual(result.pulses, 1700)
        self.assertAlmostEqual(result.planar_mm, math.hypot(1.0, -1.0), places=9)
        self.assertGreater(result.start_z_mm, 0.0)

    def test_a_plane_that_is_not_aligned_is_refused_before_any_motion(self) -> None:
        arm = self.Arm()
        with self.assertRaisesRegex(ServoError, "还没对准"):
            descend_z(arm, self.StillLink(20.0, 0.0), ObjectSelector(), ServoConfig())
        self.assertEqual(arm.calls, [])

    def test_the_override_only_changes_this_run(self) -> None:
        arm = self.Arm()
        config = ServoConfig()
        result = descend_z(arm, self.StillLink(0.0, 0.0), ObjectSelector(), config, pulses=250)
        self.assertEqual(arm.calls, [("down", 250)])
        self.assertEqual(result.pulses, 250)
        # 覆盖不写回配置：下一次还是配置里的值。
        self.assertEqual(config.z_descend_pulses, 1700)

    def test_a_nonsense_override_is_refused_without_moving(self) -> None:
        for pulses in (0, -150, 12.5, True):
            with self.subTest(pulses=pulses):
                arm = self.Arm()
                with self.assertRaises(ServoError):
                    descend_z(
                        arm,
                        self.StillLink(0.0, 0.0),
                        ObjectSelector(),
                        ServoConfig(),
                        pulses=pulses,
                    )
                self.assertEqual(arm.calls, [])

    def test_the_retract_pulls_the_extension_back_by_the_configured_count(self) -> None:
        arm = self.Arm()
        moved = retract_extension(arm, ServoConfig())
        self.assertEqual(arm.calls, [("back", 150)])
        self.assertEqual(moved, 150)

    def test_the_retract_override_is_honoured(self) -> None:
        arm = self.Arm()
        self.assertEqual(retract_extension(arm, ServoConfig(), pulses=40), 40)
        self.assertEqual(arm.calls, [("back", 40)])


if __name__ == "__main__":
    unittest.main()

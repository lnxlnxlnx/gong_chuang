"""eye-in-hand 视觉伺服：四个归属地的数学、闭环行为与就位主链。

判据只有一条硬标准：**按 ``pulses_for`` 算出来的脉冲走一遍，画面误差必须归零。**
这条不变量对 2×2 求逆的符号错误、行列式写反、列顺序调换全都敏感——用一个随机
雅可比矩阵族来压它，比逐个人工算例更难糊弄过去。

伺服循环那边则是四个失败路径必须响亮：看不到物块、雅可比奇异、预算用完、迭代用尽。

## 名字现在分四个家

原先一个 ``eye_in_hand.py`` 住着四件事，现在按"谁改谁"分开：

- 值对象 / 配置读写 / 结果类型 → :mod:`modules.grasp.drive.servo_model`；
- 观测判据（选取、停稳、可见性探测） → :mod:`modules.grasp.drive.servo_observation`；
- 脉冲原语（真的发帧并等） → :mod:`modules.grasp.drive.servo_motion`；
- 实际控制算法（探针、安全抬升、悬停、x/y 闭环、下降、后缩） →
  :mod:`modules.grasp.drive.eye_in_hand`。

**主编排不再是自由函数**：``align_and_descend`` 只剩
:meth:`modules.grasp.runtime.Grasp.align_and_descend` 一个定义（实例方法），
所以它在这里的测试也变成实例测试。

## patch 要打在"被执行的那个名字"上

``align_x_y`` 等函数在**自己的模块**里查找 ``_move_axes`` / ``measure_offset``，
所以脉冲与观测断言指着 ``servo_motion`` / ``servo_observation``；而
``Grasp`` 的方法是在**方法体内**从 ``modules.grasp.drive.eye_in_hand``
导入算法的，所以算法断言指着那个真实定义处。
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

from modules.grasp.drive import eye_in_hand as algorithms
from modules.grasp.drive import servo_model as model
from modules.grasp.drive import servo_motion as motion
from modules.grasp.drive import servo_observation as observation
from modules.grasp.drive.eye_in_hand import (
    align_x_y,
    descend_z,
    find_object,
    hover_above_object,
    probe_jacobian,
    retract_extension,
)
from modules.grasp.drive.servo_model import (
    BASE_AXIS,
    RADIAL_AXIS,
    GraspReadyResult,
    HoverResult,
    LiftScale,
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoResult,
    ServoTarget,
    ZDescentResult,
    load_jacobian,
    load_object_selector,
    load_servo_config,
    load_target,
    save_calibration,
)
from modules.grasp.drive.servo_observation import (
    VISIBILITY_PROBE_FRAMES,
    measure_offset,
    measure_settled,
    select_object,
)
from modules.grasp import runtime as grasp_runtime
from modules.grasp.runtime import Grasp
from modules.vision.drive.sighting import parse_sighting


def sighting(material_id: int, x: float, y: float, color: str = "green", *, stamp: Optional[float] = None):
    return parse_sighting(
        f"物料 {material_id}: {color}物料（{color}），xyz=({x:.6f},{y:.6f},113.2) mm",
        received_at=stamp,
    )


def sighting_at(x: float, y: float, z: float):
    """和 :func:`sighting` 一样，但 z 可以指定——升降的测试要 z 真的会变。"""
    return parse_sighting(f"物料 4: green物料（green），xyz=({x:.6f},{y:.6f},{z:.6f}) mm")


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
        document = json.loads(model.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
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
            {"hover_z_mm": 0},
            {"hover_z_mm": "110"},
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


def align_result() -> ServoResult:
    """一份"平面已对准"的假结果（误差落在容差内，``aligned`` 为真）。"""
    return ServoResult(
        target=ServoTarget(x_mm=0.0, y_mm=0.0),
        readings=((0.4, -0.5),),
        error_x=0.4,
        error_y=-0.5,
        base_pulses=10,
        radial_pulses=-4,
        iterations=1,
        tolerance_mm=2.0,
    )


def ready_result() -> GraspReadyResult:
    """一份"四步都跑完了"的假结果，装配假会话时用它。"""
    return GraspReadyResult(
        raised_pulses=250,
        aligned=align_result(),
        descent=ZDescentResult(pulses=2200, planar_mm=0.64, start_z_mm=100.0, seconds=1.2),
        retracted_pulses=150,
        lift_seconds=1.0,
        align_seconds=1.1,
        descend_seconds=1.2,
        retract_seconds=1.3,
        hover_pulses=-120,
    )


class GraspSessionRig:
    """装配一个假会话：假臂 + 假观测源 + 真会话 + 被 patch 的五个算法。

    只替换真实的机械臂构造点 ``modules.grasp.runtime._new_arm``（这里直接塞假臂）与
    观测源；会话自己的校验、动作编排、耗时记录都跑真代码——这一层才是本文件要测的。

    ``with`` 里挂上的是**真实定义处**（``modules.grasp.drive.eye_in_hand``）的五个名字，
    因为会话是在动作方法体内从那里导入算法的：patch 别处拦不住。
    """

    #: 与 :class:`AlignXYTests` 用的是同一个雅可比；刻意再写一遍而不是互相引用，
    #: 免得两个类的加载顺序变成隐含依赖。
    TRUTH = ServoJacobian(base_dx=0.0120, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)
    TARGET = ServoTarget(x_mm=0.0, y_mm=0.0)
    STAGES = ("hover", "raise", "align", "down", "back")

    def __init__(self, *, hover_z_mm: Optional[float] = None, strict_align: bool = True):
        """``strict_align``：主链必须传 ``ensure_safe_height=False``。

        单独对准（``Grasp.align_xy``）走的是**默认** ``True``，所以那一组用
        ``strict_align=False``——那正是"安全高度检查没被顺手关掉"的证据。
        """
        self.strict_align = strict_align
        self.arm = FakeArm(self.TRUTH)
        self.source = BufferedLink(self.arm)
        self.config = ServoConfig(hover_z_mm=hover_z_mm)
        self.grasp = Grasp(
            config=self.config,
            selector=ObjectSelector(),
            target=self.TARGET,
            jacobian=self.TRUTH,
            lift_scale=LiftScale(dz_per_pulse=0.0121),
            on_message=lambda _message: None,
        )
        self.grasp._arm = self.arm
        self.grasp._source = self.source
        self._mocks = []
        self._patches = []

    # ---------------------------------------------------------------- 装/拆
    def __enter__(self):
        self._patches = [
            patch.object(algorithms, name)
            for name in ("hover_above_object", "raise_before_rotating", "align_x_y",
                         "descend_z", "retract_extension")
        ]
        hover, lift, align, down, back = (item.start() for item in self._patches)
        self._mocks = [hover, lift, align, down, back]
        hover.side_effect = lambda *_a, **_k: HoverResult(
            configured=True, pulses=-120, start_z_mm=120.0, end_z_mm=100.0
        )
        lift.side_effect = lambda *_a, **_k: 250
        align.side_effect = self._align
        down.side_effect = lambda *_a, **_k: ZDescentResult(
            pulses=2200, planar_mm=0.64, start_z_mm=100.0, seconds=1.2
        )
        back.side_effect = lambda *_a, **_k: 150
        return self

    def __exit__(self, *_exc):
        for item in reversed(self._patches):
            item.stop()
        self._mocks = []
        return False

    # ---------------------------------------------------------------- 假算法
    def _align(self, *_args, **kwargs):
        if not self.strict_align:
            return align_result()
        if kwargs.get("ensure_safe_height", True) is not False:
            # 主链必须关掉那次重复的安全高度检查；详情见
            # ``test_align_is_asked_to_skip_the_duplicate_safe_height_check``。
            raise AssertionError("主链必须传 ensure_safe_height=False")
        return align_result()

    # ---------------------------------------------------------------- 观测
    def stage_clock(self, *, durations=None) -> List[float]:
        """把会话四段耗时用到的 ``time.monotonic`` 换成一个受控时钟。

        返回四段应有的耗时。会话是在 ``runtime`` 里**按模块名**调
        ``time.monotonic()`` 的，所以这里必须 patch 那个查找处——
        给自己挂一个 ``_clock`` 属性是拦不住的（那只是测试自己的想象）。
        """
        durations = [1.0, 2.5, 1.25, 2.0] if durations is None else durations
        ticks: List[float] = []
        for index, seconds in enumerate(durations):
            ticks.append(100.0 * (index + 1))
            ticks.append(100.0 * (index + 1) + seconds)
        remaining = iter(ticks)

        def clock():
            return next(remaining)

        self._patches.append(patch.object(grasp_runtime.time, "monotonic", clock))
        self._patches[-1].start()
        return list(durations)

    # ---------------------------------------------------------------- 断言助手
    def called_stages(self) -> List[str]:
        return [name for name, item in zip(self.STAGES, self._mocks) if item.called]

    def mock(self, stage: str):
        return self._mocks[self.STAGES.index(stage)]


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
        # ``sleep`` 要在**真正执行它的模块**里换掉：``_move_axes`` 是 servo_motion 的
        # 函数，它查的是自己模块的名字，不是 eye_in_hand 的。
        with patch.object(motion.time, "sleep", sleeps.append):
            motion._move_axes(arm, base_pulses, radial_pulses, config)
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
        step, clamped = algorithms._lift_step_within_guard(
            z_mm=40.0, step=-2000, scale=scale, config=config
        )
        self.assertTrue(clamped)
        self.assertGreater(step, -2000)
        # 夹完之后预测的 z 不低于护栏。
        self.assertGreaterEqual(40.0 + step * scale.dz_per_pulse, config.z_min_mm - 1e-9)

    def test_guard_leaves_a_safe_move_alone(self) -> None:
        scale = LiftScale(dz_per_pulse=0.0121)
        config = ServoConfig(z_min_mm=30.0)
        step, clamped = algorithms._lift_step_within_guard(
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


class LiftRig:
    """升降假臂：z 随脉冲线性变化，动作记在 ``calls`` 里。"""

    def __init__(self, *, z_mm: float = 67.0, slope: float = 0.0121) -> None:
        self.z = z_mm
        self.slope = slope
        self.calls: List[Tuple[str, int]] = []

    def move_up(self, pulses, **_kwargs):
        self.calls.append(("up", pulses))
        self.z += pulses * self.slope

    def move_down(self, pulses, **_kwargs):
        self.calls.append(("down", pulses))
        self.z -= pulses * self.slope

    @property
    def up_pulses(self) -> List[int]:
        return [pulses for kind, pulses in self.calls if kind == "up"]


class CoupledZLink:
    """读数直接取假臂当前的 z——画面静止，所以一次停稳测量就是两次读。"""

    def __init__(self, rig: LiftRig, *, x: float = 0.0, y: float = 0.0) -> None:
        self.rig = rig
        self.x, self.y = x, y
        self.error = None
        self.reads = 0

    def frame(self, **_kwargs):
        self.reads += 1
        return (sighting_at(self.x, self.y, self.rig.z),)


class FirstFrameHigh(CoupledZLink):
    """第一帧读得偏大（模拟运动模糊），之后读真值。"""

    def __init__(self, rig: LiftRig, *, offset_mm: float) -> None:
        super().__init__(rig)
        self.offset_mm = offset_mm

    def frame(self, **_kwargs):
        self.reads += 1
        z = self.rig.z + (self.offset_mm if self.reads == 1 else 0.0)
        return (sighting_at(0.0, 0.0, z),)


class BlindLink:
    """画面里什么都没有的假镜头。"""

    error = None

    def __init__(self) -> None:
        self.reads = 0

    def frame(self, **_kwargs):
        self.reads += 1
        return None


class SafeLiftTests(unittest.TestCase):
    """转底盘前抬到安全高度：标过灵敏度就一步算到位，没标才一段段试。

    为什么值得单独测：每一步后面都跟一次停稳测量（实测约 537 ms），
    旧写法最坏要 20 步，而按灵敏度算通常 2 步就到。
    """

    def make(self, *, z_mm: float = 67.0, slope: float = 0.0121, **overrides):
        rig = LiftRig(z_mm=z_mm, slope=slope)
        link = CoupledZLink(rig)
        return rig, link, ServoConfig(**overrides)

    def test_the_calibrated_scale_reaches_the_safe_height_in_two_segments(self) -> None:
        rig, link, config = self.make()
        raised = algorithms.raise_before_rotating(
            rig, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )

        self.assertGreaterEqual(rig.z, config.safe_lift_z_mm)
        self.assertEqual(raised, sum(rig.up_pulses))
        # 第一段被单段上限夹住（按灵敏度算出来要 2782），第二段是算出来的零头。
        self.assertEqual(rig.up_pulses[0], config.safe_lift_max_pulses_per_step)
        self.assertEqual(len(rig.up_pulses), 2)
        # 三次测量（进段一次、修正一次、确认一次），每次两次定位。
        self.assertLessEqual(link.reads, 8)

    def test_without_the_scale_it_walks_in_fixed_segments(self) -> None:
        rig, link, config = self.make()
        raised = algorithms.raise_before_rotating(rig, link, ObjectSelector(), config)

        self.assertGreaterEqual(rig.z, config.safe_lift_z_mm)
        self.assertEqual(set(rig.up_pulses), {config.safe_lift_step_pulses})
        # 固定步长要明显更多段——这正是"按灵敏度算"省下来的地方。
        self.assertGreater(len(rig.up_pulses), 2)
        self.assertEqual(raised, sum(rig.up_pulses))

    def test_the_scale_path_can_be_switched_off(self) -> None:
        rig, link, config = self.make(safe_lift_use_scale=False)
        algorithms.raise_before_rotating(
            rig, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )
        self.assertEqual(set(rig.up_pulses), {config.safe_lift_step_pulses})

    def test_the_computed_segment_is_capped(self) -> None:
        rig, link, config = self.make(safe_lift_max_pulses_per_step=400)
        algorithms.raise_before_rotating(
            rig, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )
        self.assertEqual(rig.up_pulses[0], 400)

    def test_already_high_enough_costs_one_measurement_and_no_motion(self) -> None:
        rig, link, config = self.make(z_mm=105.0)
        raised = algorithms.raise_before_rotating(
            rig, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )

        self.assertEqual(raised, 0)
        self.assertEqual(rig.calls, [])
        self.assertEqual(link.reads, 2)  # 一次停稳测量 = 两次定位

    def test_the_quick_check_passes_on_a_single_frame_when_clearly_safe(self) -> None:
        """每一轮都要查的调用方（--follow）用单帧快判：够高就只花一帧。"""
        rig, link, config = self.make(z_mm=110.0)
        raised = algorithms.raise_before_rotating(rig, link, ObjectSelector(), config, quick=True)

        self.assertEqual(raised, 0)
        self.assertEqual(rig.calls, [])
        self.assertEqual(link.reads, 1)

    def test_the_quick_check_falls_back_when_the_margin_is_not_met(self) -> None:
        rig, link, config = self.make(z_mm=102.0)  # 过了安全高度，但不够余量
        raised = algorithms.raise_before_rotating(rig, link, ObjectSelector(), config, quick=True)

        self.assertEqual(raised, 0)
        # 一帧快判 + 一次停稳测量（两次定位）
        self.assertGreaterEqual(link.reads, 3)

    def test_a_first_frame_that_reads_high_inside_the_margin_does_not_pass(self) -> None:
        """余量就是为这一条存在的：模糊帧把 z 读大、"以为够高"正是低位转底盘的成因。"""
        rig, _unused, config = self.make(z_mm=99.0)  # 真值低于安全高度
        link = FirstFrameHigh(rig, offset_mm=4.0)  # 第一帧读成 103，仍在余量（5 mm）内
        raised = algorithms.raise_before_rotating(rig, link, ObjectSelector(), config, quick=True)

        self.assertGreater(len(rig.up_pulses), 0)  # 没被那一帧骗过去：该抬还是抬
        self.assertEqual(raised, sum(rig.up_pulses))

    def test_a_lift_that_changes_nothing_is_reported_instead_of_pushed(self) -> None:
        rig, link, config = self.make(slope=0.0)  # 已经到行程端点：走多少都不变
        raised = algorithms.raise_before_rotating(
            rig, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )

        self.assertEqual(len(rig.up_pulses), 1)  # 只试一段就停，不接着往上顶
        self.assertEqual(raised, config.safe_lift_max_pulses_per_step)

    def test_a_lift_that_barely_moves_is_also_treated_as_an_endstop(self) -> None:
        """到端点时物块几乎不动、只剩画面噪声：判据不能只看"是不是恰好 0"。"""
        rig, link, config = self.make(slope=0.00005)
        raised = algorithms.raise_before_rotating(
            rig, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )

        self.assertEqual(len(rig.up_pulses), 1)
        self.assertEqual(raised, config.safe_lift_max_pulses_per_step)

    def test_without_a_scale_a_dead_lift_stops_after_two_segments(self) -> None:
        rig, link, config = self.make(slope=0.0)
        raised = algorithms.raise_before_rotating(rig, link, ObjectSelector(), config)

        # 没有灵敏度可参照，所以第一段没法判；两段都不动就收手（而不是顶满 20 段）。
        self.assertEqual(len(rig.up_pulses), 2)
        self.assertEqual(raised, 2 * config.safe_lift_step_pulses)


class HoverTests(unittest.TestCase):
    """起始悬停高度（``hover_z_mm``）：配了才动；敢降要有标定；降不到安全高度以下。"""

    def rig(self, *, z_mm: float = 120.0, slope: float = 0.0121, **overrides):
        arm = LiftRig(z_mm=z_mm, slope=slope)
        link = CoupledZLink(arm)
        return arm, link, ServoConfig(**overrides)

    def test_without_the_key_nothing_moves(self) -> None:
        arm, link, config = self.rig(z_mm=80.0)
        result = hover_above_object(
            arm, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )

        self.assertFalse(result.configured)
        self.assertEqual(result.pulses, 0)
        self.assertEqual(arm.calls, [])

    def test_it_lowers_to_the_configured_height(self) -> None:
        arm, link, config = self.rig(z_mm=120.0, hover_z_mm=100.0)
        result = hover_above_object(
            arm, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )

        self.assertTrue(result.configured)
        self.assertLess(result.pulses, 0)  # 往下走
        self.assertAlmostEqual(arm.z, 100.0, delta=config.z_tolerance_mm)
        self.assertGreaterEqual(arm.z, config.safe_lift_z_mm)

    def test_it_raises_to_the_configured_height(self) -> None:
        arm, link, config = self.rig(z_mm=80.0, hover_z_mm=110.0)
        result = hover_above_object(
            arm, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )

        self.assertGreater(result.pulses, 0)
        self.assertAlmostEqual(arm.z, 110.0, delta=config.z_tolerance_mm)

    def test_already_at_the_height_costs_no_motion(self) -> None:
        arm, link, config = self.rig(z_mm=110.5, hover_z_mm=110.0)
        result = hover_above_object(
            arm, link, ObjectSelector(), config, scale=LiftScale(dz_per_pulse=0.0121)
        )

        self.assertEqual(result.pulses, 0)
        self.assertEqual(arm.calls, [])

    def test_without_a_scale_it_refuses_to_lower(self) -> None:
        """没标定灵敏度时往下走是在朝物块靠近，宁可不动。"""
        arm, link, config = self.rig(z_mm=130.0, hover_z_mm=110.0)
        result = hover_above_object(arm, link, ObjectSelector(), config)

        self.assertTrue(result.stopped_without_scale)
        self.assertEqual(arm.calls, [])

    def test_without_a_scale_it_still_raises(self) -> None:
        arm, link, config = self.rig(z_mm=80.0, hover_z_mm=110.0)
        result = hover_above_object(arm, link, ObjectSelector(), config)

        self.assertGreater(len(arm.up_pulses), 0)
        self.assertGreaterEqual(arm.z, 110.0 - config.z_tolerance_mm)

    def test_the_config_refuses_a_hover_below_the_safe_height(self) -> None:
        """悬停得比"转底盘前的安全高度"还低 = 低位转底盘，配置层直接拦。"""
        with self.assertRaises(ValueError):
            ServoConfig(hover_z_mm=90.0, safe_lift_z_mm=100.0)
        with self.assertRaises(ValueError):
            ServoConfig(hover_z_mm=0.0)


class FindObjectTests(unittest.TestCase):
    def test_a_missing_object_does_not_pay_the_settle_wait_at_every_step(self) -> None:
        """看不到物块时只快速探几帧；旧写法每步都要等到 settle_timeout 才认输。"""
        rig = LiftRig()
        link = BlindLink()
        config = ServoConfig(
            settle_timeout_seconds=30.0, settle_poll_seconds=0.25, find_max_steps=3
        )

        started = time.monotonic()
        found = find_object(rig, link, ObjectSelector(), config)
        elapsed = time.monotonic() - started

        self.assertFalse(found)
        self.assertLess(elapsed, 1.0)  # 旧写法这里要 3 × 30 s
        self.assertEqual(link.reads, 3 * VISIBILITY_PROBE_FRAMES)
        self.assertEqual(len(rig.up_pulses), 3)  # 每一步仍然照抬


class EnsureSafeHeightTests(unittest.TestCase):
    """``align_x_y`` 的"要不要先查安全高度"开关：省一次测量，但不能省错地方。"""

    def rig(self):
        jacobian = ServoJacobian(
            base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091
        )
        arm = FakeArm(jacobian)
        link = BufferedLink(arm)
        target = ServoTarget(x_mm=arm.offset[0], y_mm=arm.offset[1])
        return arm, link, target, jacobian

    def test_the_check_is_skipped_only_when_the_caller_asks(self) -> None:
        arm, link, target, jacobian = self.rig()
        config = ServoConfig()

        with patch.object(algorithms, "raise_before_rotating") as check:
            result = align_x_y(
                arm, link, ObjectSelector(), target, jacobian, config,
                ensure_safe_height=False,
            )
        check.assert_not_called()
        self.assertTrue(result.aligned)

        with patch.object(algorithms, "raise_before_rotating") as check:
            align_x_y(arm, link, ObjectSelector(), target, jacobian, config)
        check.assert_called_once()

    def test_the_calibrated_scale_is_handed_to_the_check(self) -> None:
        arm, link, target, jacobian = self.rig()
        scale = LiftScale(dz_per_pulse=0.0121)

        with patch.object(algorithms, "raise_before_rotating") as check:
            align_x_y(
                arm, link, ObjectSelector(), target, jacobian, ServoConfig(),
                lift_scale=scale,
            )

        self.assertEqual(check.call_args.kwargs["scale"], scale)


class GraspAlignXYSessionTests(unittest.TestCase):
    """``Grasp.align_xy``：配了 ``hover_z_mm`` 才先悬停；安全高度检查不能被顺手关掉。

    ``ensure_safe_height`` 的默认值是 ``True`` 这件事在这里必须被钉住：主链
    （:meth:`Grasp.align_and_descend`）才关它，单独对准时**一次都不能省**——
    低位转底盘会把台面上的物块撞倒。
    """

    def build(self, *, hover_z_mm: Optional[float]) -> GraspSessionRig:
        # ``strict_align=False``：这一组走的是 ``align_x_y`` 的默认参数，
        # 也就是"底层那次安全高度检查仍然发生"。
        return GraspSessionRig(hover_z_mm=hover_z_mm, strict_align=False)

    def test_a_configured_hover_runs_first_and_then_aligns(self) -> None:
        rig = self.build(hover_z_mm=105.0)
        with rig as session:
            result = session.grasp.align_xy()
            hover = session.mock("hover")
            align = session.mock("align")

            hover.assert_called_once()
            align.assert_called_once()
            self.assertEqual(align.call_args.args[0:3], (session.arm, session.source, ObjectSelector()))
            # 单独对准时**不能**关掉任何安全检查：``ensure_safe_height`` 一个都别出现。
            self.assertNotIn("ensure_safe_height", align.call_args.kwargs)
            session.mock("raise").assert_not_called()
            session.mock("down").assert_not_called()
            session.mock("back").assert_not_called()

        self.assertTrue(result.aligned)

    def test_without_the_key_it_skips_hover_but_still_aligns(self) -> None:
        rig = self.build(hover_z_mm=None)
        with rig as session:
            result = session.grasp.align_xy()
            session.mock("hover").assert_not_called()
            session.mock("align").assert_called_once()
            self.assertNotIn("ensure_safe_height", session.mock("align").call_args.kwargs)

        self.assertTrue(result.aligned)


class GraspAlignAndDescendTests(unittest.TestCase):
    """一键就位现在是 ``Grasp`` 的**实例方法**（原来的自由函数已删除）。

    四步的顺序、``ensure_safe_height=False``、两个脉冲覆盖、以及四段耗时的记录，
    一条条都要还在：第 3、4 步是开环的（爪子压下去之后画面里就没有物块了），
    第 2 步的结果是它们**唯一**的安全依据。
    """

    def test_the_steps_run_in_the_documented_order(self) -> None:
        rig = GraspSessionRig()
        with rig as session:
            durations = session.stage_clock()
            result = session.grasp.align_and_descend()
            called = session.called_stages()

        self.assertEqual(called, ["hover", "raise", "align", "down", "back"])
        # 完整字段：每一段的结果都要带出来，不能被换成"只报成功"。
        self.assertEqual(result.hover_pulses, -120)
        self.assertEqual(result.raised_pulses, 250)
        self.assertEqual(result.retracted_pulses, 150)
        self.assertEqual(result.descent.pulses, 2200)
        self.assertTrue(result.aligned.aligned)
        self.assertEqual(
            (
                result.lift_seconds,
                result.align_seconds,
                result.descend_seconds,
                result.retract_seconds,
            ),
            tuple(durations),
        )
        self.assertAlmostEqual(result.total_seconds, sum(durations))
        for keyword in ("就位完成", "悬停 -120", "抬升 250", "下降 2200", "后缩 150"):
            self.assertIn(keyword, result.summary())

    def test_align_is_asked_to_skip_the_duplicate_safe_height_check(self) -> None:
        """刚刚才查过安全高度（悬停 + 抬升），这几步又不改高度，只跳过那一次重复。"""
        rig = GraspSessionRig()
        with rig as session:
            session.stage_clock()
            session.grasp.align_and_descend()
            self.assertIs(
                session.mock("align").call_args.kwargs["ensure_safe_height"], False
            )

    def test_the_overrides_reach_the_two_pulse_steps(self) -> None:
        rig = GraspSessionRig()
        with rig as session:
            session.stage_clock()
            session.grasp.align_and_descend(z_pulses=800, retract_pulses=40)
            self.assertEqual(session.mock("down").call_args.kwargs["pulses"], 800)
            self.assertEqual(session.mock("back").call_args.kwargs["pulses"], 40)

    def test_without_overrides_the_two_steps_get_none(self) -> None:
        """``None`` 表示"用配置里的值"：会话不能替调用方填数。"""
        rig = GraspSessionRig()
        with rig as session:
            session.stage_clock()
            session.grasp.align_and_descend()
            self.assertIsNone(session.mock("down").call_args.kwargs["pulses"])
            self.assertIsNone(session.mock("back").call_args.kwargs["pulses"])

    def test_every_stage_gets_the_configured_lift_scale(self) -> None:
        """没标定升降灵敏度时"往下走"是在朝物块靠近，所以尺度要一路传下去。"""
        rig = GraspSessionRig()
        scale = rig.grasp._lift_scale
        with rig as session:
            session.stage_clock()
            session.grasp.align_and_descend()
            for stage in ("hover", "raise", "down"):
                with self.subTest(stage=stage):
                    self.assertIs(session.mock(stage).call_args.kwargs["scale"], scale)

    def test_a_failing_hover_stops_before_anything_else(self) -> None:
        self._assert_stage_failure("hover")

    def test_a_failing_lift_stops_before_aligning(self) -> None:
        self._assert_stage_failure("raise")

    def test_a_failing_align_stops_before_descending(self) -> None:
        self._assert_stage_failure("align")

    def test_a_failing_descent_stops_before_retracting(self) -> None:
        self._assert_stage_failure("down")

    def test_a_failing_retract_is_reported(self) -> None:
        self._assert_stage_failure("back")

    def _assert_stage_failure(self, stage: str) -> None:
        """某一阶段失败时：后面的一步都不许发，错误原样传出去。"""
        rig = GraspSessionRig()
        with rig as session:
            session.stage_clock()
            session.mock(stage).side_effect = ServoError(f"{stage} 失败")
            with self.assertRaisesRegex(ServoError, f"{stage} 失败"):
                session.grasp.align_and_descend()
            called = session.called_stages()

        index = rig.STAGES.index(stage)
        self.assertEqual(called, list(rig.STAGES[: index + 1]), called)

    def test_the_algorithms_are_imported_from_the_real_definition_site(self) -> None:
        """会话在**方法体内**从 eye_in_hand 导入算法；patch 定义处才拦得住。

        这条同时守着一件事：一次就位只对准**一次**。``align_and_descend``
        不经过 ``self.align_xy()``——那会重复悬停与安全检查，改变原来只查一次的主链。
        """
        rig = GraspSessionRig()
        with rig as session:
            session.stage_clock()
            session.grasp.align_and_descend()
            self.assertTrue(session.mock("align").called)
            # 多一次就是走了 self.align_xy() 那条重复路。
            self.assertEqual(session.mock("align").call_count, 1)


if __name__ == "__main__":
    unittest.main()

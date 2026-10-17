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

    def test_the_ambiguity_error_names_both_ways_out(self) -> None:
        """报错要说清两条路：按次传（推荐），或者填默认值。"""
        with self.assertRaises(ServoError) as caught:
            select_object((self.green, self.blue), ObjectSelector())
        message = str(caught.exception)
        self.assertIn("want=", message)
        self.assertIn("eye_in_hand.json", message)

    def test_from_want_speaks_the_same_language_as_locate(self) -> None:
        """``locate("red")`` 与 ``align_xy(want="red")`` 必须是同一套说法。"""
        by_colour = ObjectSelector.from_want("  GREEN ")
        self.assertEqual(by_colour.color, "green")
        self.assertFalse(by_colour.is_open)
        self.assertEqual(select_object((self.green, self.blue), by_colour).material_id, 4)

        by_number = ObjectSelector.from_want(2)          # 编号从 1 开始，与「物料 N」一致
        self.assertEqual(by_number.material_id, 2)

        self.assertTrue(ObjectSelector.from_want(None).is_open)

    def test_from_want_rejects_the_same_bad_writes_as_locate(self) -> None:
        for bad in (0, -3, "", "   ", 1.5, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ObjectSelector.from_want(bad)


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


class JumpyDepthLink:
    """x/y 一动不动，**z 每帧乱跳**——真机上 z 就是这样。

    ``z`` 是按像素半径反推的（``z = f·D/2r``，本机约 0.42 mm/px），所以 1 px 的半径
    抖动就顶掉整个细档 0.4 mm。实测真机帧（``output/debug/now.png``、``p1_after.png``，
    加 σ=1–2 灰阶噪声）连续两帧的 Δz 中位数是 1.7–2.7 mm、≤0.4 mm 的只占 13%，
    而 Δxy 中位数是 0.02–0.26 mm、≤0.4 mm 的占 74–100%。
    """

    error = None

    def __init__(self, depths=(100.0, 103.0), *, x: float = 0.4, y: float = -0.5, arm=None) -> None:
        self._depths = tuple(depths)
        self._x = x
        self._y = y
        #: 给了假机械臂就跟着它的真实位移报 x/y（闭环才会真的收敛），只让 z 乱跳。
        self._arm = arm
        self.reads = 0

    def frame(self, **_kwargs):
        self.reads += 1
        depth = self._depths[(self.reads - 1) % len(self._depths)]
        x, y = self._arm.offset if self._arm is not None else (self._x, self._y)
        return (sighting_at(x, y, depth),)


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

    def test_a_jumping_depth_does_not_hold_the_planar_gate(self) -> None:
        """``include_depth=False``：z 乱跳时两帧就该收工。

        这是"对准有时要等好几秒"的正解——x/y 对准一个 z 都不用，没有理由等 z 停下来。
        """
        link = JumpyDepthLink()
        settled = observation._settled_sighting(
            link, ObjectSelector(),
            ServoConfig(settle_tolerance_mm=0.4, settle_timeout_seconds=2.0,
                        settle_poll_seconds=0.01),
            include_depth=False,
        )

        self.assertEqual(link.reads, 2, "只比 x/y 时第二帧就该收工")
        self.assertEqual((settled.x_mm, settled.y_mm), (0.4, -0.5))

    def test_the_same_jumping_depth_still_gates_the_steps_that_measure_z(self) -> None:
        """默认（三维）判据一个字没改：测 z 的那几条路照旧要等 z 停稳。"""
        link = JumpyDepthLink()
        messages = []
        settled = observation._settled_sighting(
            link, ObjectSelector(),
            ServoConfig(settle_tolerance_mm=0.4, settle_timeout_seconds=0.05,
                        settle_poll_seconds=0.005),
            on_message=messages.append,
        )

        self.assertGreater(link.reads, 2, "z 一直跳，三维判据下不该两帧就收工")
        self.assertTrue(any("等停稳超时" in message for message in messages), messages)
        self.assertEqual((settled.x_mm, settled.y_mm), (0.4, -0.5))


class AlignDoesNotWaitForDepthTests(unittest.TestCase):
    """x/y 对准**不该等 z 停下来**——真机上那一次等待就是"好几秒"的来源。

    修好之前：细档判据要求连续两帧在三维上都一致（含 z），而 z 的抖动必然超过 0.4 mm，
    于是每次测量都要等满 ``settle_timeout_seconds``（默认 4 s）才拿最后读数走人。
    """

    JACOBIAN = ServoJacobian(base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)

    @staticmethod
    def config(**overrides) -> ServoConfig:
        values = dict(
            tolerance_mm=4.0,
            settle_tolerance_mm=0.4,
            coarse_settle_tolerance_mm=3.0,
            coarse_error_mm=10.0,
            min_step_mm=1.0,
            max_iterations=12,
            # 特意给得很大：判据要是把 z 算进去，这条用例就会真的等满它。
            settle_timeout_seconds=5.0,
            settle_poll_seconds=0.01,
        )
        values.update(overrides)
        return ServoConfig(**values)

    def test_an_already_aligned_object_returns_after_two_frames(self) -> None:
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(x=0.4, y=-0.5)  # 误差 0.64 mm，已经在容差里
        started = time.monotonic()
        result = align_x_y(
            arm, link, ObjectSelector(), ServoTarget(x_mm=0.0, y_mm=0.0), self.JACOBIAN,
            self.config(), ensure_safe_height=False,
        )
        elapsed = time.monotonic() - started

        self.assertEqual(link.reads, 2, "x/y 已经对准：两帧就该收工")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(arm.calls, [], "已经对准就不该再发动作")
        self.assertLess(elapsed, 1.0, "判据若把 z 算进去，这里会等满 settle_timeout_seconds")

    def test_a_jumping_depth_does_not_multiply_the_measurements(self) -> None:
        """闭环该走几步就走几步；z 每帧乱跳不该让每一步都多等一次超时。

        目标在 (0,0)，假机械臂从 (12, -7) 出发：雅可比是精确的，所以第一步就该到位，
        然后第二步确认并收工。**每次停稳测量两帧**（判据是"连续两帧一致"），
        所以两次迭代 = 四帧；修好之前这里会是"两帧 + 两次等满 5 s"。
        """
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(arm=arm)
        started = time.monotonic()
        result = align_x_y(
            arm, link, ObjectSelector(), ServoTarget(x_mm=0.0, y_mm=0.0), self.JACOBIAN,
            self.config(), ensure_safe_height=False,
        )
        elapsed = time.monotonic() - started

        self.assertEqual(link.reads, 2 * result.iterations, "每次停稳测量两帧，没有额外空转")
        self.assertEqual(result.iterations, 2)
        self.assertLessEqual(result.error_mm, self.config().tolerance_mm)
        self.assertLess(elapsed, 1.0)


class SingleStepTests(unittest.TestCase):
    """**单帧一步**：最多取一帧、走一步、不验证——跟随循环的基础。

    ``locate`` 一次 + ``align_xy(reading=...)`` 的组合就是"YOLO 像素 → 实际 x/y →
    机械臂动一下"，每轮只取一帧；放进循环就是跟踪。
    """

    JACOBIAN = ServoJacobian(base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)

    @staticmethod
    def config(**overrides) -> ServoConfig:
        values = dict(
            tolerance_mm=4.0,
            min_step_mm=1.0,
            coarse_error_mm=10.0,
            max_pulses_per_step=1200,
            coarse_max_pulses_per_step=4000,
            max_total_pulses=10000,
            settle_timeout_seconds=5.0,
        )
        values.update(overrides)
        return ServoConfig(**values)

    def call(self, arm, link, **kwargs):
        return align_x_y(
            arm, link, ObjectSelector(), ServoTarget(x_mm=0.0, y_mm=0.0), self.JACOBIAN,
            kwargs.pop("config", self.config()), ensure_safe_height=False, **kwargs
        )

    def test_a_given_reading_moves_once_without_taking_any_frame(self) -> None:
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(arm=arm)
        result = self.call(arm, link, reading=(12.0, -7.0))

        self.assertEqual(link.reads, 0, "调用方已经量过了：这一步不该再取帧")
        self.assertTrue(result.unverified)
        self.assertEqual(result.iterations, 1)
        # 走的这一步应当正好把误差抵消（线性化下 J·p = −e）；脉冲取整会留下
        # 不到半个脉冲的残差（实测量级 0.002 mm），所以比的是容差而不是"正好 0"。
        moved_x, moved_y = self.JACOBIAN.displacement_for(
            result.base_pulses, result.radial_pulses
        )
        self.assertAlmostEqual(moved_x + 12.0, 0.0, delta=0.05)
        self.assertAlmostEqual(moved_y - 7.0, 0.0, delta=0.05)

    def test_a_given_reading_inside_the_tolerance_moves_nothing(self) -> None:
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(arm=arm)
        result = self.call(arm, link, reading=(1.0, -2.0))

        self.assertEqual(arm.calls, [])
        self.assertEqual((result.base_pulses, result.radial_pulses), (0, 0))
        self.assertEqual(result.error_mm, math.hypot(1.0, -2.0))

    def test_without_a_reading_it_takes_exactly_one_frame(self) -> None:
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(arm=arm)          # x/y 跟着假机械臂走
        result = self.call(arm, link, single_step=True)

        self.assertEqual(link.reads, 1, "单帧一步只取一帧，不等停稳")
        self.assertTrue(result.unverified)
        self.assertTrue(arm.calls, "误差 13.9 mm，应当走了一步")

    def test_a_single_step_is_clamped_by_the_per_step_limit(self) -> None:
        """单步幅度照样被 step_limit_for 夹住：撞上去的风险由此兜住。"""
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(arm=arm)
        config = self.config()
        result = self.call(arm, link, reading=(200.0, 200.0), config=config)

        self.assertLessEqual(abs(result.base_pulses), config.coarse_max_pulses_per_step)
        self.assertLessEqual(abs(result.radial_pulses), config.coarse_max_pulses_per_step)

    def test_bad_readings_are_rejected_before_anything_moves(self) -> None:
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(arm=arm)
        for bad in ("12", b"12", (1.0,), (1.0, 2.0, 3.0), 3.0, (1.0, "x")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.call(arm, link, reading=bad)
        self.assertEqual(arm.calls, [], "参数错了就一个动作帧都不该发")
        self.assertEqual(link.reads, 0)

    def test_the_summary_says_it_was_not_verified(self) -> None:
        arm = FakeArm(self.JACOBIAN)
        result = self.call(arm, JumpyDepthLink(arm=arm), reading=(12.0, -7.0))
        self.assertIn("单帧一步", result.summary())
        self.assertIn("没有再验证", result.summary())
        # 读数说的是动作**之前**那一次——这一点必须写在结果里，别让调用方误读。
        self.assertEqual(result.readings, ((12.0, -7.0),))
        self.assertEqual(result.error_mm, math.hypot(12.0, -7.0))

    def test_locate_then_align_in_a_loop_converges(self) -> None:
        """用户要的那个循环：**量一次 → 走一步**，几轮之内把误差收进容差。

        ``reading`` 就是 ``vision.locate()`` 给出的 ``(report.best.x_mm, y_mm)``；
        这里直接读假机械臂的位置来代表它。给了读数，这一路**一帧都不取**。
        """
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(arm=arm)
        config = self.config()
        target = ServoTarget(x_mm=0.0, y_mm=0.0)

        rounds = 0
        for _ in range(8):
            reading = (arm.offset[0], arm.offset[1])
            if math.hypot(*reading) <= config.tolerance_mm:
                break
            align_x_y(
                arm, link, ObjectSelector(), target, self.JACOBIAN, config,
                ensure_safe_height=False, reading=reading,
            )
            rounds += 1

        self.assertLessEqual(math.hypot(*arm.offset), config.tolerance_mm)
        self.assertLessEqual(rounds, 2, "雅可比精确时两轮之内就该收工")
        self.assertEqual(link.reads, 0, "给了读数就不该再取帧")

    def test_the_loop_follows_a_block_that_gets_pushed(self) -> None:
        """物块被推走 → 循环把它收回画面里的目标读数上（这就是跟踪）。

        "被推走"在相机看来就是读数多了一个偏移（机械臂自己没动）；伺服要做的正是把
        这个偏移收掉——所以收敛之后**读数**应当回到目标，而不是机械臂回到原位。

        推得远时**一轮收不完**，这是设计使然：单步幅度被 ``step_limit_for`` 夹住
        （这里实测 51 mm 的偏移一步只能收掉 46 mm），所以循环还要再转一轮。
        这正是"单帧一步 + 外层循环"这个分工的样子。
        """
        arm = FakeArm(self.JACOBIAN)
        link = JumpyDepthLink(arm=arm)
        config = self.config()
        target = ServoTarget(x_mm=0.0, y_mm=0.0)

        for push in ((15.0, -12.0), (-25.0, 20.0), (3.0, 3.0)):
            with self.subTest(push=push):
                reading = (arm.offset[0] + push[0], arm.offset[1] + push[1])
                for _ in range(4):
                    if math.hypot(*reading) <= config.tolerance_mm:
                        break
                    align_x_y(
                        arm, link, ObjectSelector(), target, self.JACOBIAN, config,
                        ensure_safe_height=False, reading=reading,
                    )
                    reading = (arm.offset[0] + push[0], arm.offset[1] + push[1])

                self.assertLessEqual(
                    math.hypot(*reading), config.tolerance_mm,
                    "循环转几轮之后，画面里的偏移应当被收进容差",
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
    """``Grasp.align_xy``：**默认只动平面**，高度完全由调用方负责。

    这条默认值是刻意选的：物块放得高时悬停会一路往上抬到行程端点（抬升方向没有上限），
    所以"单独对准"不再自己升降。代价是**转底盘前的安全高度检查也默认关掉了**——
    低位转底盘会把台面上的物块撞倒（现场发生过），调用方必须先自己点动到够高，
    或者显式传 ``ensure_safe_height=True`` 把那次检查要回来。
    主链 :meth:`Grasp.align_and_descend` 的默认值**没变**（仍然悬停 + 仍然查安全高度）。
    """

    def build(self, *, hover_z_mm: Optional[float]) -> GraspSessionRig:
        # ``strict_align=False``：这一组走的是 ``align_x_y`` 的默认参数那一支。
        return GraspSessionRig(hover_z_mm=hover_z_mm, strict_align=False)

    def test_the_default_does_not_touch_the_height(self) -> None:
        """默认：既不悬停，也不查安全高度——高度是调用方的事。"""
        rig = self.build(hover_z_mm=105.0)
        with rig as session:
            result = session.grasp.align_xy()
            session.mock("hover").assert_not_called()
            self.assertIs(
                session.mock("align").call_args.kwargs["ensure_safe_height"], False
            )

        self.assertTrue(result.aligned)

    def test_a_requested_hover_runs_first_and_then_aligns(self) -> None:
        rig = self.build(hover_z_mm=105.0)
        with rig as session:
            result = session.grasp.align_xy(hover=True)
            hover = session.mock("hover")
            align = session.mock("align")

            hover.assert_called_once()
            align.assert_called_once()
            self.assertEqual(align.call_args.args[0:3], (session.arm, session.source, ObjectSelector()))
            session.mock("raise").assert_not_called()
            session.mock("down").assert_not_called()
            session.mock("back").assert_not_called()

        self.assertTrue(result.aligned)

    def test_the_safe_height_check_can_be_asked_back(self) -> None:
        """要回那次检查时，底层必须拿到它自己的默认值（关键字一个都不传）。"""
        rig = self.build(hover_z_mm=105.0)
        with rig as session:
            session.grasp.align_xy(ensure_safe_height=True)
            self.assertNotIn(
                "ensure_safe_height", session.mock("align").call_args.kwargs
            )

    def test_without_the_key_it_skips_hover_but_still_aligns(self) -> None:
        rig = self.build(hover_z_mm=None)
        with rig as session:
            result = session.grasp.align_xy()
            session.mock("hover").assert_not_called()
            session.mock("align").assert_called_once()

        self.assertTrue(result.aligned)

    def test_a_given_reading_and_single_step_reach_align_x_y(self) -> None:
        """跟随循环要的两个口子必须**原样**传到 ``align_x_y``，不能被中间层吃掉。

        吃掉的后果很隐蔽：循环照样转，只是每轮多等两帧、而且用的是自己刚测过的旧读数。
        """
        rig = self.build(hover_z_mm=None)
        with rig as session:
            session.grasp.align_xy(reading=(3.0, -4.0))
            self.assertEqual(
                session.mock("align").call_args.kwargs["reading"], (3.0, -4.0)
            )

            session.mock("align").reset_mock()
            session.grasp.align_xy(single_step=True)
            self.assertIs(session.mock("align").call_args.kwargs["single_step"], True)

            # 两个都不给时也不该凭空塞值进去（底层默认必须是它自己的默认）。
            session.mock("align").reset_mock()
            session.grasp.align_xy()
            kwargs = session.mock("align").call_args.kwargs
            self.assertIsNone(kwargs["reading"])
            self.assertIs(kwargs["single_step"], False)

    def test_a_per_call_want_replaces_the_configured_selector(self) -> None:
        """一次程序里抓多种物料：``want`` 按次生效，配置文件一个字都不用改。"""
        rig = self.build(hover_z_mm=None)
        with rig as session:
            session.grasp.align_xy(want="blue")
            selector = session.mock("align").call_args.args[2]
            self.assertEqual(selector.color, "blue")
            self.assertFalse(selector.is_open)

            session.mock("align").reset_mock()
            session.grasp.align_xy(want=2)
            self.assertEqual(session.mock("align").call_args.args[2].material_id, 2)

            # 不传就用配置里那份（这里的会话配置是"画面里唯一的目标"）。
            session.mock("align").reset_mock()
            session.grasp.align_xy()
            self.assertTrue(session.mock("align").call_args.args[2].is_open)

    def test_a_per_call_want_reaches_every_step_of_align_and_descend(self) -> None:
        """整轮就位（悬停 → 安全高度 → x/y → 下降）每一步都得看同一个 want。

        漏掉任何一步，那一步就会去对准**别的**物料——而且是静默的。
        （``retract_extension`` 不收 selector，所以不在这里查。）
        """
        rig = self.build(hover_z_mm=105.0)
        with rig as session:
            session.grasp.align_and_descend(want="red")

            wanted = ObjectSelector(color="red")
            for name in ("hover", "raise", "align", "down"):
                with self.subTest(step=name):
                    # 这四个底层函数的前三个位置参数都是 (arm, source, selector)。
                    self.assertEqual(session.mock(name).call_args.args[2], wanted)


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


class GraspManualHeightTests(unittest.TestCase):
    """``hover`` / ``ensure_safe_height``：高度可以整段交给调用方。

    为什么要有这两个开关（现场实测）：物块放得高时，悬停伺服要把"镜头到物块"的距离压到
    ``hover_z_mm``，于是**一路往上抬**——抬升方向没有上限，只有"走了脉冲 z 却没变"的
    事后判据，所以它会顶在行程端点上，电机堵转。

    两个开关的边界必须清楚：``hover=False`` 只让开"悬停"；``ensure_safe_height=False``
    才是"转底盘前的高度我自己负责"——低位转底盘会撞到台面上的物块。
    """

    def test_skipping_both_leaves_the_height_untouched(self) -> None:
        rig = GraspSessionRig(hover_z_mm=105.0)
        with rig as session:
            session.stage_clock()
            result = session.grasp.align_and_descend(
                hover=False, ensure_safe_height=False
            )
            called = session.called_stages()
            session.mock("hover").assert_not_called()
            session.mock("raise").assert_not_called()

        self.assertEqual(called, ["align", "down", "back"])
        self.assertEqual(result.hover_pulses, 0)
        self.assertEqual(result.raised_pulses, 0)
        # 跳过悬停时摘要不能说"悬停 0 脉冲"，也不能假装抬过。
        self.assertNotIn("悬停", result.summary())
        self.assertIn("抬升 0 脉冲", result.summary())

    def test_the_default_still_hovers_and_raises(self) -> None:
        rig = GraspSessionRig(hover_z_mm=105.0)
        with rig as session:
            session.stage_clock()
            session.grasp.align_and_descend()
            self.assertEqual(
                session.called_stages(),
                ["hover", "raise", "align", "down", "back"],
            )

    def test_skipping_hover_keeps_the_safe_height_check(self) -> None:
        """``hover=False`` 不是"关掉安全检查"：转底盘前照样要够高。"""
        rig = GraspSessionRig(hover_z_mm=105.0)
        with rig as session:
            session.stage_clock()
            result = session.grasp.align_and_descend(hover=False)

            session.mock("hover").assert_not_called()
            session.mock("raise").assert_called_once()

        self.assertEqual(result.raised_pulses, 250)
        self.assertEqual(result.hover_pulses, 0)

    def test_the_align_step_still_skips_its_duplicate_check(self) -> None:
        """主链只查一次安全高度这件事与两个开关无关。"""
        rig = GraspSessionRig(hover_z_mm=105.0)
        with rig as session:
            session.stage_clock()
            session.grasp.align_and_descend(hover=False, ensure_safe_height=False)
            self.assertIs(
                session.mock("align").call_args.kwargs["ensure_safe_height"], False
            )

    def test_align_xy_can_skip_both_height_steps(self) -> None:
        """单独对准也能不动高度：``align_xy(hover=False, ensure_safe_height=False)``。"""
        rig = GraspSessionRig(hover_z_mm=105.0, strict_align=False)
        with rig as session:
            result = session.grasp.align_xy(hover=False, ensure_safe_height=False)

            session.mock("hover").assert_not_called()
            self.assertIs(
                session.mock("align").call_args.kwargs["ensure_safe_height"], False
            )

        self.assertTrue(result.aligned)


class OcclusionTests(unittest.TestCase):
    """物块中途看不见：**只有"一个目标都没有"才可能是爪子挡光**，别的必须报错。

    现场现象：对准到最后，爪子正好悬在物块上方，自己把光挡住，画面里就没有那块料了。
    这时如果照旧报错，流程会死在一个已经对准的位置上；但如果一律"看不见就当对准了"，
    就等于放行一个没量过的落点。所以只开一个窄口子：**已经测到过 + 那一次已经进入
    "可以下降"的平面精度**，才按最后一次读数收工，并且在结果里写明"未再验证"。
    """

    TRUTH = ServoJacobian(base_dx=0.0120, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)
    TARGET = ServoTarget(x_mm=2.0, y_mm=-1.0)

    class OccludedLink:
        """前 ``visible_reads`` 次读得到物块，之后画面里什么都没有（被爪子挡住）。"""

        error = None

        def __init__(self, arm: "FakeArm", *, visible_reads: int) -> None:
            self.arm = arm
            self.remaining = visible_reads

        def frame(self, **_kwargs):
            if self.remaining <= 0:
                return ()
            self.remaining -= 1
            return (
                sighting(
                    4, self.arm.offset[0], self.arm.offset[1], stamp=time.monotonic()
                ),
            )

    def config(self) -> ServoConfig:
        # 超时给得很短：这条路径要等满 settle_timeout 才认输，测试不该真的等 4 s。
        return ServoConfig(
            tolerance_mm=0.5,
            max_iterations=12,
            max_pulses_per_step=2000,
            min_step_mm=0.1,
            z_center_tolerance_mm=6.0,
            settle_timeout_seconds=0.05,
            settle_poll_seconds=0.01,
        )

    def test_a_blind_frame_after_a_close_reading_finishes_with_a_flag(self) -> None:
        """走了两步之后被挡住，而最后一次误差已经在 6 mm 内 → 收工，且写明未再验证。"""
        arm = FakeArm(self.TRUTH, gain=0.6)
        link = self.OccludedLink(arm, visible_reads=4)
        messages: List[str] = []

        result = align_x_y(
            arm, link, ObjectSelector(), self.TARGET, self.TRUTH, self.config(),
            ensure_safe_height=False, on_message=messages.append,
        )

        self.assertTrue(result.occluded)
        # 用的是第 2 次（也就是最后一次有效）的读数，不是凭空造一个。
        self.assertEqual(len(result.readings), 2)
        self.assertAlmostEqual(result.error_x, result.readings[-1][0] - self.TARGET.x_mm, places=9)
        self.assertAlmostEqual(result.error_mm, 4.66, delta=0.3)
        self.assertIn("挡住", result.summary())
        self.assertTrue(any("未再验证" in message for message in messages), messages)

    def test_a_blind_frame_while_still_far_off_still_raises(self) -> None:
        """一次有效读数之后就被挡住、但误差还有十几毫米 → 必须报错，不能猜。"""
        arm = FakeArm(self.TRUTH)
        link = self.OccludedLink(arm, visible_reads=2)

        with self.assertRaisesRegex(ServoError, "没有物块"):
            align_x_y(
                arm, link, ObjectSelector(), self.TARGET, self.TRUTH, self.config(),
                ensure_safe_height=False,
            )

    def test_a_blind_first_frame_raises_without_any_history(self) -> None:
        """一帧都没测到过：没有"最后一次读数"可用，只能报错。"""
        arm = FakeArm(self.TRUTH)
        link = self.OccludedLink(arm, visible_reads=0)

        with self.assertRaisesRegex(ServoError, "没有物块"):
            align_x_y(
                arm, link, ObjectSelector(), self.TARGET, self.TRUTH, self.config(),
                ensure_safe_height=False,
            )

    def test_several_targets_are_never_treated_as_occlusion(self) -> None:
        """画面里有两个目标时**不能**当遮挡收工：那会朝错的目标走，必须问人。"""

        class TwoTargets:
            error = None

            def frame(self, **_kwargs):
                stamp = time.monotonic()
                return (
                    sighting(1, 30.0, 0.0, stamp=stamp),
                    sighting(2, -30.0, 0.0, stamp=stamp),
                )

        arm = FakeArm(self.TRUTH)
        with self.assertRaisesRegex(ServoError, "多个目标"):
            align_x_y(
                arm, TwoTargets(), ObjectSelector(), self.TARGET, self.TRUTH, self.config(),
                ensure_safe_height=False,
            )

    def test_the_timeout_message_carries_the_last_inner_reason(self) -> None:
        """超时那句必须带上最后一次内层原因——否则三种成因长得一模一样。"""
        arm = FakeArm(self.TRUTH)
        link = self.OccludedLink(arm, visible_reads=0)

        with self.assertRaises(ServoError) as caught:
            measure_settled(
                link,
                ObjectSelector(),
                ServoConfig(settle_timeout_seconds=0.05, settle_poll_seconds=0.01),
            )

        message = str(caught.exception)
        self.assertIn("也没能稳定看到物块", message)
        self.assertIn("这一段的最后一次失败是", message)
        self.assertIn("没有物块", message)


class StepDirectionTests(unittest.TestCase):
    """每一步走完，实测位移与雅可比预测**方向相反**时必须立刻停下。

    这是"雅可比过时 / 装反"的唯一签名。没有这条判据时，闭环只会照着雅可比一直走，
    直到脉冲预算（``max_total_pulses``）用完——那时报的是"预算用完"，现场看到的是
    "越走越偏、方向不对"，离真正的原因已经很远。

    判据必须窄：只拦**方向相反**；同向但只走了一部分（背隙、物块被推动、雅可比偏小）
    不许误伤，否则一个能收敛的闭环会被无谓地打断。
    """

    TRUTH = ServoJacobian(base_dx=0.0120, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)
    TARGET = ServoTarget(x_mm=0.0, y_mm=0.0)

    def config(self) -> ServoConfig:
        return ServoConfig(
            tolerance_mm=0.5, max_iterations=12, max_pulses_per_step=2000, min_step_mm=0.1
        )

    def inverted(self) -> ServoJacobian:
        """与机器相反的那份雅可比（相机装反 / 接线改过之后的典型状态）。"""
        return ServoJacobian(
            base_dx=-self.TRUTH.base_dx,
            base_dy=-self.TRUTH.base_dy,
            radial_dx=-self.TRUTH.radial_dx,
            radial_dy=-self.TRUTH.radial_dy,
        )

    def test_an_inverted_jacobian_stops_after_two_contradicting_steps(self) -> None:
        """存下来的雅可比和机器相反（相机装反/接线改过）→ 连着两步相反就报，不再往下走。

        **一步不算**：物块被爪子推动、检测跳到别的圆上都会让单独一步看起来相反，
        所以判据要求连续两步；代价是多走一步，换来的是不误伤能收敛的闭环。
        """
        arm = FakeArm(self.TRUTH)

        with self.assertRaises(ServoError) as caught:
            align_x_y(
                arm, BufferedLink(arm), ObjectSelector(), self.TARGET, self.inverted(),
                self.config(), ensure_safe_height=False,
            )

        message = str(caught.exception)
        # 报错里要同时给出实测与预测两个数，现场才能核对；并给出重探的命令。
        self.assertIn("连着两步", message)
        self.assertIn("实测", message)
        self.assertIn("预测", message)
        self.assertIn("--calibrate", message)
        # 走了两步（每步两个轴各一条帧），没有把预算烧完。
        self.assertEqual(len(arm.calls), 4, arm.calls)

    def test_the_reversed_step_criterion_ignores_small_and_same_direction_steps(self) -> None:
        """判据本身：方向相反才为真；预测很小（噪声量级）或同向都为假。"""
        config = self.config()
        # 预测 10 mm、实测 -5 mm：明显相反。
        self.assertTrue(
            algorithms._step_is_reversed(
                self.TRUTH, step=(1000, 0), measured=(-5.0, 0.0), config=config
            )
        )
        # 同向，只是走少了一半（背隙/被推动）：不算相反。
        self.assertFalse(
            algorithms._step_is_reversed(
                self.TRUTH, step=(1000, 0), measured=(5.0, 0.0), config=config
            )
        )
        # 预测本来就很小（细调阶段 1 mm 量级）：噪声就能翻转符号，不算数。
        self.assertFalse(
            algorithms._step_is_reversed(
                self.TRUTH, step=(10, 0), measured=(-2.0, 0.0), config=config
            )
        )

    def test_a_partial_step_in_the_right_direction_is_not_flagged(self) -> None:
        """背隙/物块被推动让这一步只走了一部分，方向仍然一致 → 照旧收敛。"""
        arm = FakeArm(self.TRUTH, gain=0.3)

        result = align_x_y(
            arm, BufferedLink(arm), ObjectSelector(), self.TARGET, self.TRUTH,
            self.config(), ensure_safe_height=False,
        )

        self.assertTrue(result.aligned, result.summary())

    def test_a_consistent_jacobian_reports_measured_against_predicted(self) -> None:
        """方向一致时也要把"实测 vs 预测"打出来，供现场核对雅可比还准不准。"""
        # gain=0.5：一步走不完，于是第二、第三轮都会经过核对那一行。
        arm = FakeArm(self.TRUTH, gain=0.5)
        messages: List[str] = []

        result = align_x_y(
            arm, BufferedLink(arm), ObjectSelector(), self.TARGET, self.TRUTH,
            self.config(), ensure_safe_height=False, on_message=messages.append,
        )

        self.assertTrue(result.aligned, result.summary())
        self.assertTrue(
            any("实测画面位移" in message and "雅可比预测" in message for message in messages),
            messages,
        )


if __name__ == "__main__":
    unittest.main()

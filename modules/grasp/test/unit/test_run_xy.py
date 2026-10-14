"""x/y 对准的闭环编排：判别、收敛、安全绕行、失败路径。

全部用假臂 + 假观测，不碰串口也不碰 socket：这一层的行为（什么时候重新测、
什么时候先缩回、出错时发没发过动作）正是真机上最难复现、也最该被钉住的部分。
"""

from __future__ import annotations

import math
import time
import unittest
from typing import List, Optional, Tuple

from modules.arm.config import ArmConfig, AxisConfig
from modules.arm.drive.kinematics import TravelLimit
from modules.grasp.config import CorrectionSettings, GraspConfig
from modules.grasp.drive.run_xy import (
    GripperMarker,
    XYAlignConfig,
    XYAligner,
    XYAlignError,
    load_gripper_marker,
    load_xy_align_config,
)
from modules.vision.drive.calibration import PlaneCalibration
from modules.vision.drive.sighting import parse_sighting
from modules.vision.drive.sighting_plane import SightingPlane

#: 夹具用的镜头系标定：基座轴心在 (100, -50)，θ 零点偏 90°。
#: 于是"镜头 +x 方向"对应基座的 θ=+90°，换算关系是
#: ``镜头点 = 轴心 + r·(cos(θ−90°), sin(θ−90°))``。
CENTER_X, CENTER_Y = 100.0, -50.0
ROTATION_DEGREES = 90.0

#: 夹具里的物料（镜头系毫米）。
MATERIAL_XY = (110.0, -30.0)
#: 它对应的基座圆柱坐标：相对轴心 (10, 20) -> θ = atan2(20,10)+90°，r = √500。
MATERIAL_BASE = 153.434948822922
MATERIAL_RADIAL = math.hypot(10.0, 20.0)


def bench_plane() -> SightingPlane:
    return SightingPlane(
        plane=PlaneCalibration(
            center_x_mm=CENTER_X, center_y_mm=CENTER_Y, rotation_degrees=ROTATION_DEGREES
        ),
        table_depth_mm=300.0,
    )


def bench_arm_config() -> ArmConfig:
    return ArmConfig(
        pulses_per_revolution=3200,
        rpm=10,
        acceleration=5,
        default_pulses=800,
        max_pulses=5000,
        axes={
            "base": AxisConfig(motor_id=5, reduction_ratio=5),
            "lift": AxisConfig(motor_id=6, mm_per_motor_rev=40.0),
            "extension": AxisConfig(motor_id=7, mm_per_motor_rev=40.0),
        },
        base_limit=TravelLimit(-180.0, 180.0),
        radial_limit=TravelLimit(15.0, 200.0),
        height_limit=TravelLimit(-5.0, 120.0),
    )


class FakeArm:
    """真值位姿 + 两种误差模型；``move_lift_mm`` 直接失败，用来证明高度没被碰过。

    两种误差的**闭环行为完全不同**，测试必须把两者分开：

    - ``backlash``：**一次性**的（真实背隙就是这样）。第一次少走，之后正常走，
      所以闭环第二次测量就能把它修回来——这是闭环存在的意义。
    - ``radial_bias``：**每次都发生**的系统偏差（传动常数标错、皮带打滑）。
      闭环修不掉它，因为每修一次又欠一次，这正是"不收敛"该有的样子。
    """

    def __init__(
        self,
        *,
        base: float,
        radial: float,
        backlash: float = 0.0,
        radial_bias: float = 1.0,
    ) -> None:
        self.base = base
        self.radial = radial
        self.height = 60.0
        self._backlash = backlash
        self.radial_bias = radial_bias
        self.calls: List[Tuple[str, float]] = []

    def move_base_degrees(self, degrees: float, **_kwargs) -> None:
        self.calls.append(("base", round(degrees, 6)))
        self.base += degrees

    def move_radial_mm(self, millimeters: float, **_kwargs) -> None:
        lost = min(abs(millimeters), self._backlash) if self._backlash > 0 else 0.0
        self._backlash -= lost
        moved = (millimeters - (lost if millimeters > 0 else -lost)) * self.radial_bias
        self.calls.append(("radial", round(millimeters, 6)))
        self.radial += moved

    def move_lift_mm(self, millimeters: float, **_kwargs) -> None:
        raise AssertionError("x/y 对准不许动高度")


class FakeLink:
    """按假臂的**真实**位姿合成观测；点数不够时返回 ``None``（模拟等不到帧）。"""

    def __init__(
        self,
        arm: FakeArm,
        material_xy: Tuple[float, float] = MATERIAL_XY,
        *,
        frames_available: Optional[int] = None,
        material_id: int = 3,
        marker_id: int = 9,
    ) -> None:
        self.arm = arm
        self.material_xy = material_xy
        self.error: Optional[str] = None
        self._remaining = frames_available
        self.material_id = material_id
        self.marker_id = marker_id
        self.frames_served = 0

    def frame(self, **_kwargs):
        if self._remaining is not None:
            if self._remaining <= 0:
                return None
            self._remaining -= 1
        self.frames_served += 1
        radians = math.radians(self.arm.base - ROTATION_DEGREES)
        arm_x = CENTER_X + self.arm.radial * math.cos(radians)
        arm_y = CENTER_Y + self.arm.radial * math.sin(radians)
        now = time.monotonic()
        marker = parse_sighting(
            f"物料 {self.marker_id}: 红色物料（red），xyz=({arm_x:.6f},{arm_y:.6f},295.0) mm",
            received_at=now,
        )
        material = parse_sighting(
            f"物料 {self.material_id}: 绿色物料（green），"
            f"xyz=({self.material_xy[0]},{self.material_xy[1]},285.0) mm",
            received_at=now,
        )
        return (marker, material)


class IdenticalLink(FakeLink):
    """把靶标和物料合成**同一条**观测，用来验证"判别不出就拒绝"。"""

    def frame(self, **_kwargs):
        now = time.monotonic()
        item = parse_sighting(
            f"物料 {self.marker_id}: 红色物料（red），"
            f"xyz=({MATERIAL_XY[0]},{MATERIAL_XY[1]},285.0) mm",
            received_at=now,
        )
        return (item,)


def build_aligner(
    arm: FakeArm,
    *,
    link=None,
    marker: Optional[GripperMarker] = None,
    tolerance_mm: float = 1.0,
    iterations: int = 4,
    retract_radius_mm: Optional[float] = None,
    corridor_degrees: float = 5.0,
) -> XYAligner:
    return XYAligner(
        arm=arm,
        link=link if link is not None else FakeLink(arm),
        plane=bench_plane(),
        marker=marker if marker is not None else GripperMarker(material_id=9),
        config=XYAlignConfig(tolerance_mm=tolerance_mm, iterations=iterations),
        grasp_config=GraspConfig(
            motion_rpm=None,
            retract_radius_mm=retract_radius_mm,
            correction=CorrectionSettings(rotation_corridor_degrees=corridor_degrees),
        ),
        arm_config=bench_arm_config(),
    )


class GripperMarkerTests(unittest.TestCase):
    def test_requires_some_way_to_identify_the_gripper(self) -> None:
        """判别不出靶标时必须拒绝构造：把物料当成爪子会得出"已对准"的静默错误。"""
        with self.assertRaisesRegex(XYAlignError, "gripper_marker"):
            GripperMarker()

    def test_matches_by_material_id(self) -> None:
        marker = GripperMarker(material_id=9)
        self.assertTrue(marker.matches(parse_sighting("物料 9: xyz=(0,0,100) mm")))
        self.assertFalse(marker.matches(parse_sighting("物料 3: xyz=(0,0,100) mm")))

    def test_matches_by_colour_case_insensitively(self) -> None:
        marker = GripperMarker(color="RED")
        self.assertEqual(marker.color, "red")
        self.assertTrue(marker.matches(parse_sighting("物料 1: 红色物料（red），xyz=(0,0,1) mm")))

    def test_both_conditions_must_hold(self) -> None:
        marker = GripperMarker(material_id=9, color="red")
        self.assertTrue(
            marker.matches(parse_sighting("物料 9: 红色物料（red），xyz=(0,0,1) mm"))
        )
        self.assertFalse(
            marker.matches(parse_sighting("物料 9: 绿色物料（green），xyz=(0,0,1) mm"))
        )
        self.assertFalse(
            marker.matches(parse_sighting("物料 8: 红色物料（red），xyz=(0,0,1) mm"))
        )

    def test_describe_mentions_the_configured_conditions(self) -> None:
        text = GripperMarker(material_id=9, color="red").describe()
        self.assertIn("9", text)
        self.assertIn("red", text)


class XYAlignConfigTests(unittest.TestCase):
    def test_project_config_loads(self) -> None:
        config = load_xy_align_config()
        self.assertGreater(config.tolerance_mm, 0)
        self.assertGreaterEqual(config.iterations, 1)
        self.assertGreater(config.max_frame_age_ms, 0)

    def test_project_config_declares_a_gripper_marker(self) -> None:
        """配置里必须能认出靶标，否则真机第一步就跑不起来。"""
        marker = load_gripper_marker()
        self.assertTrue(marker.material_id is not None or marker.color is not None)

    def test_invalid_values_are_rejected(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xy_align.json"
            for document in (
                {"tolerance_mm": 0},
                {"tolerance_mm": -1},
                {"iterations": 0},
                {"iterations": 1.5},
                {"max_frame_age_ms": 0},
                {"gripper_marker": {"material_id": -1}},
                {"gripper_marker": {"material_id": "9"}},
                {"gripper_marker": {"color": "   "}},
                {"gripper_marker": "not an object"},
                "not an object",
            ):
                with self.subTest(document=document):
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(XYAlignError):
                        load_xy_align_config(path)
                        load_gripper_marker(path)


class AlreadyAlignedTests(unittest.TestCase):
    def test_no_motion_when_already_aligned(self) -> None:
        """已经对准时一次测量就要收工，不许"抖"出任何动作。"""
        arm = FakeArm(base=MATERIAL_BASE, radial=MATERIAL_RADIAL, backlash=2.0)
        result = build_aligner(arm).align()
        self.assertTrue(result.aligned, result)
        self.assertEqual(result.measurements, 1)
        self.assertEqual(result.waypoints, ())
        self.assertEqual(arm.calls, [], arm.calls)
        self.assertLess(result.error_mm, 1e-6)


class ConvergenceTests(unittest.TestCase):
    def test_converges_from_a_distant_pose(self) -> None:
        arm = FakeArm(base=0.0, radial=120.0)
        result = build_aligner(arm).align()
        self.assertTrue(result.aligned, result)
        self.assertAlmostEqual(arm.base, MATERIAL_BASE, places=6)
        self.assertAlmostEqual(arm.radial, MATERIAL_RADIAL, places=6)

    def test_backlash_is_corrected_by_re_measuring(self) -> None:
        """背隙让第一次动作少走 4 mm；闭环必须靠第二次测量把它修回来。"""
        arm = FakeArm(base=0.0, radial=120.0, backlash=4.0)
        result = build_aligner(arm).align()
        self.assertTrue(result.aligned, result)
        self.assertGreaterEqual(result.measurements, 2, "有背隙时一次测量不可能够")
        self.assertAlmostEqual(arm.radial, MATERIAL_RADIAL, places=6)

    def test_reports_failure_instead_of_drifting_when_it_cannot_converge(self) -> None:
        """每次都欠 60% 的系统偏差：闭环修不掉，必须报错而不是一直走下去。

        （用一次性背隙测不出这件事：第二次测量就补回来了。）
        """
        arm = FakeArm(base=0.0, radial=200.0, radial_bias=0.4)
        aligner = build_aligner(arm, tolerance_mm=0.5, iterations=3)
        with self.assertRaisesRegex(XYAlignError, "容差"):
            aligner.align()

    def test_height_is_never_touched(self) -> None:
        """高度是下一步的事；这一层碰它会让"先 x/y 再 z"的顺序失去意义。"""
        arm = FakeArm(base=-100.0, radial=150.0, backlash=3.0)
        build_aligner(arm).align()
        self.assertEqual(arm.height, 60.0)
        for kind, _value in arm.calls:
            self.assertIn(kind, ("base", "radial"), arm.calls)


class SafetyCorridorTests(unittest.TestCase):
    def test_large_rotation_retracts_first(self) -> None:
        """伸出状态下转大角度：必须先缩到配置的半径，再转，最后再伸出去。"""
        arm = FakeArm(base=-150.0, radial=120.0)
        build_aligner(arm, retract_radius_mm=50.0).align()

        trace: List[Tuple[float, float]] = []
        radial, base = 120.0, -150.0
        for kind, value in arm.calls:
            if kind == "radial":
                radial += value
            else:
                base += value
            trace.append((round(radial, 6), round(base, 6)))

        turn_index = next(index for index, (_r, b) in enumerate(trace) if abs(b + 150.0) > 1e-6)
        self.assertAlmostEqual(trace[turn_index][0], 50.0, msg=f"旋转时径向不是缩回值：{trace}")

    def test_corridor_rotation_takes_the_short_way(self) -> None:
        """−150° → 153.43° 应当走 −56.57°；直接跳向目标的绝对角会绕 +303°。"""
        arm = FakeArm(base=-150.0, radial=120.0)
        build_aligner(arm, retract_radius_mm=50.0).align()
        applied = sum(value for kind, value in arm.calls if kind == "base")
        self.assertAlmostEqual(applied, -56.565052, places=4)

    def test_small_rotation_does_not_retract(self) -> None:
        """小角度就地转是安全的：爪子本来就悬在物料上方几十毫米。"""
        arm = FakeArm(base=MATERIAL_BASE - 2.0, radial=MATERIAL_RADIAL)
        build_aligner(arm, retract_radius_mm=50.0).align()
        radial_calls = [value for kind, value in arm.calls if kind == "radial"]
        # 目标径向和当前一致，所以除了修正误差以外不该出现"先缩回 50"这种动作。
        for value in radial_calls:
            self.assertLess(abs(value), 20.0, arm.calls)

    def test_no_corridor_configured_means_no_retraction(self) -> None:
        """``retract_radius_mm`` 为 ``None`` 是"明确不缩回"，不是"用一个默认半径"。"""
        arm = FakeArm(base=-150.0, radial=120.0)
        build_aligner(arm, retract_radius_mm=None).align()
        # 动作应当只有"转"和"伸"两次，中间没有缩回。
        self.assertEqual(len(arm.calls), 2, arm.calls)
        self.assertAlmostEqual(arm.calls[0][0], "radial", msg=arm.calls)
        self.assertAlmostEqual(arm.calls[1][0], "base", msg=arm.calls)

    def test_corridor_radius_outside_travel_is_rejected_before_moving(self) -> None:
        arm = FakeArm(base=-150.0, radial=120.0)
        aligner = XYAligner(
            arm=arm,
            link=FakeLink(arm),
            plane=bench_plane(),
            marker=GripperMarker(material_id=9),
            config=XYAlignConfig(tolerance_mm=1.0, iterations=4),
            grasp_config=GraspConfig(
                retract_radius_mm=5.0,  # 径向行程下限是 15
                correction=CorrectionSettings(rotation_corridor_degrees=5.0),
            ),
            arm_config=bench_arm_config(),
        )
        with self.assertRaisesRegex(XYAlignError, "径向行程"):
            aligner.align()
        self.assertEqual(arm.calls, [], arm.calls)


class IdentifyTests(unittest.TestCase):
    def test_missing_marker_raises_and_moves_nothing(self) -> None:
        arm = FakeArm(base=0.0, radial=100.0)

        class NoMarker(FakeLink):
            def frame(self, **_kwargs):
                now = time.monotonic()
                return (
                    parse_sighting("物料 3: 绿色物料（green），xyz=(110,-30,285) mm", received_at=now),
                )

        aligner = build_aligner(arm, link=NoMarker(arm))
        with self.assertRaisesRegex(XYAlignError, "靶标"):
            aligner.align()
        self.assertEqual(arm.calls, [])

    def test_missing_material_raises(self) -> None:
        arm = FakeArm(base=0.0, radial=100.0)

        class NoMaterial(FakeLink):
            def frame(self, **_kwargs):
                now = time.monotonic()
                return (
                    parse_sighting("物料 9: 红色物料（red），xyz=(100,-50,295) mm", received_at=now),
                )

        aligner = build_aligner(arm, link=NoMaterial(arm))
        with self.assertRaisesRegex(XYAlignError, "没有物料"):
            aligner.align()
        self.assertEqual(arm.calls, [])

    def test_multiple_materials_are_refused(self) -> None:
        """一次只对准一个；猜错目标会让机械臂朝错误的位置动，而动作看起来完全正常。"""
        arm = FakeArm(base=0.0, radial=100.0)

        class TwoMaterials(FakeLink):
            def frame(self, **_kwargs):
                now = time.monotonic()
                return (
                    parse_sighting("物料 9: 红色物料（red），xyz=(100,-50,295) mm", received_at=now),
                    parse_sighting("物料 3: 绿色物料（green），xyz=(110,-30,285) mm", received_at=now),
                    parse_sighting("物料 4: 蓝色物料（blue），xyz=(120,-20,285) mm", received_at=now),
                )

        aligner = build_aligner(arm, link=TwoMaterials(arm))
        with self.assertRaisesRegex(XYAlignError, "多个物料"):
            aligner.align()
        self.assertEqual(arm.calls, [])

    def test_no_observations_at_all_is_reported_with_the_link_error(self) -> None:
        arm = FakeArm(base=0.0, radial=100.0)
        link = FakeLink(arm, frames_available=0)
        link.error = "连接被重置"
        aligner = build_aligner(arm, link=link)
        with self.assertRaises(XYAlignError) as caught:
            aligner.align()
        self.assertIn("连接被重置", str(caught.exception))
        self.assertEqual(arm.calls, [])


class ReachabilityTests(unittest.TestCase):
    def test_target_outside_the_radial_travel_is_refused_before_moving(self) -> None:
        arm = FakeArm(base=0.0, radial=100.0)
        aligner = build_aligner(arm, link=FakeLink(arm, material_xy=(400.0, -50.0)))
        with self.assertRaisesRegex(XYAlignError, "行程"):
            aligner.align()
        self.assertEqual(arm.calls, [], "越界时一个动作帧都不该发出去")

    def test_target_inside_the_travel_is_accepted(self) -> None:
        arm = FakeArm(base=0.0, radial=100.0)
        aligner = build_aligner(arm, link=FakeLink(arm, material_xy=(130.0, -50.0)))
        self.assertTrue(aligner.align().aligned)


class StoppingConditionTests(unittest.TestCase):
    def test_iterations_bound_the_number_of_measurements(self) -> None:
        arm = FakeArm(base=0.0, radial=200.0, radial_bias=0.4)
        aligner = build_aligner(arm, tolerance_mm=0.5, iterations=2)
        with self.assertRaises(XYAlignError) as caught:
            aligner.align()
        self.assertIn("2 次", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

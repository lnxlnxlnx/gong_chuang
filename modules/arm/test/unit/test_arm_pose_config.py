"""机械臂配置里新增的标定字段：行程范围、安全位、每轴毫米换算。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.arm.config import load_arm_config
from modules.arm.drive.kinematics import CylindricalPose, TravelLimit


def write_config(path: Path, **overrides) -> None:
    document = {
        "motion": {
            "pulses_per_revolution": 3200,
            "rpm": 10,
            "acceleration": 5,
            "default_pulses": 800,
            "max_pulses": 5000,
        },
        "axes": {
            "base": {"motor_id": 5, "reduction_ratio": 5},
            "lift": {"motor_id": 6},
            "extension": {"motor_id": 7},
        },
    }
    document.update(overrides)
    path.write_text(json.dumps(document), encoding="utf-8")


class ArmPoseConfigTestCase(unittest.TestCase):
    def load(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            write_config(path, **overrides)
            return load_arm_config(path, hardware_path=path)


class ArmLimitConfigTests(ArmPoseConfigTestCase):
    def test_missing_limits_block_means_uncalibrated_not_unlimited(self) -> None:
        config = self.load()
        self.assertIsNone(config.base_limit)
        self.assertIsNone(config.radial_limit)
        self.assertIsNone(config.height_limit)

    def test_limits_are_read_as_inclusive_ranges(self) -> None:
        config = self.load(
            limits={
                "base_degrees": [-110.0, 110.0],
                "radial_mm": [40.0, 160.0],
                "height_mm": [-5.0, 120.0],
            }
        )
        self.assertEqual(config.base_limit, TravelLimit(-110.0, 110.0))
        self.assertEqual(config.radial_limit, TravelLimit(40.0, 160.0))
        self.assertEqual(config.height_limit, TravelLimit(-5.0, 120.0))

    def test_null_limits_stay_uncalibrated(self) -> None:
        config = self.load(limits={"base_degrees": None, "radial_mm": None, "height_mm": None})
        self.assertIsNone(config.base_limit)
        self.assertIsNone(config.radial_limit)
        self.assertIsNone(config.height_limit)

    def test_a_malformed_range_is_rejected(self) -> None:
        for limits in (
            {"base_degrees": [1.0]},
            {"base_degrees": [1.0, 2.0, 3.0]},
            {"base_degrees": 5.0},
            {"base_degrees": [10.0, 5.0]},
            {"base_degrees": ["low", "high"]},
        ):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    self.load(limits=limits)


class ArmSafePoseConfigTests(ArmPoseConfigTestCase):
    def test_missing_safe_pose_block_is_reported_by_the_accessor(self) -> None:
        config = self.load()
        self.assertIsNone(config.safe_pose)
        with self.assertRaisesRegex(ValueError, "safe_pose"):
            config.require_safe_pose()

    def test_a_complete_safe_pose_is_read(self) -> None:
        config = self.load(
            safe_pose={"base_degrees": 0.0, "radial_mm": 40.0, "height_mm": 60.0}
        )
        self.assertEqual(config.safe_pose, CylindricalPose(0.0, 40.0, 60.0))
        self.assertEqual(config.require_safe_pose().radial_mm, 40.0)

    def test_an_all_null_safe_pose_stays_uncalibrated(self) -> None:
        config = self.load(
            safe_pose={"base_degrees": None, "radial_mm": None, "height_mm": None}
        )
        self.assertIsNone(config.safe_pose)

    def test_a_partial_safe_pose_is_rejected_rather_than_defaulted(self) -> None:
        """只填一半等于没标定，不能悄悄用 0 补上另外两个。"""
        for document in (
            {"base_degrees": 0.0, "radial_mm": None, "height_mm": None},
            {"base_degrees": None, "radial_mm": 40.0, "height_mm": None},
            {"base_degrees": 0.0, "radial_mm": 40.0},
        ):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    self.load(safe_pose=document)


class ArmLinearCalibrationConfigTests(ArmPoseConfigTestCase):
    def test_missing_mm_per_motor_rev_is_reported_with_the_experiment_to_run(self) -> None:
        config = self.load()
        self.assertIsNone(config.mm_per_motor_rev("lift"))
        with self.assertRaisesRegex(ValueError, "mm_per_motor_rev"):
            config.require_mm_per_motor_rev("extension")

    def test_linear_axes_accept_a_mm_per_motor_rev(self) -> None:
        config = self.load(
            axes={
                "base": {"motor_id": 5, "reduction_ratio": 5},
                "lift": {"motor_id": 6, "mm_per_motor_rev": 40.0},
                "extension": {"motor_id": 7, "mm_per_motor_rev": 12.5},
            }
        )
        self.assertEqual(config.mm_per_motor_rev("lift"), 40.0)
        self.assertEqual(config.require_mm_per_motor_rev("extension"), 12.5)

    def test_non_positive_calibration_values_are_rejected(self) -> None:
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.load(
                        axes={
                            "base": {"motor_id": 5, "reduction_ratio": 5},
                            "lift": {"motor_id": 6, "mm_per_motor_rev": value},
                            "extension": {"motor_id": 7},
                        }
                    )

    def test_the_base_axis_rejects_a_linear_constant(self) -> None:
        with self.assertRaisesRegex(ValueError, "mm_per_motor_rev"):
            self.load(
                axes={
                    "base": {"motor_id": 5, "reduction_ratio": 5, "mm_per_motor_rev": 40.0},
                    "lift": {"motor_id": 6},
                    "extension": {"motor_id": 7},
                }
            )

    def test_linear_axes_reject_a_reduction_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "reduction_ratio"):
            self.load(
                axes={
                    "base": {"motor_id": 5, "reduction_ratio": 5},
                    "lift": {"motor_id": 6, "reduction_ratio": 3},
                    "extension": {"motor_id": 7},
                }
            )


class ArmInitialStateConfigTests(ArmPoseConfigTestCase):
    """``initial_motor_degrees``：程序启动那一刻每根轴在哪（状态声明，不是传动常数）。"""

    def test_a_missing_block_means_every_axis_starts_on_the_origin(self) -> None:
        config = self.load()
        self.assertEqual(dict(config.initial_motor_degrees), {})
        for axis in ("base", "lift", "extension"):
            with self.subTest(axis=axis):
                self.assertEqual(config.initial_motor_pulses(axis), 0)

    def test_the_declared_angle_is_read_and_converted_to_pulses(self) -> None:
        """换爪子之后 7 号轴的机械初始位在 270°，也就是 2400 脉冲。"""
        config = self.load(initial_motor_degrees={"extension": 270.0})
        self.assertEqual(config.initial_motor_degrees["extension"], 270.0)
        self.assertEqual(config.initial_motor_pulses("extension"), 2400)
        self.assertEqual(config.initial_motor_pulses("base"), 0)

    def test_a_negative_angle_is_kept_signed(self) -> None:
        """同一圈上 -90° 是另一个位置（-800 脉冲），不做归一。"""
        config = self.load(initial_motor_degrees={"extension": -90.0})
        self.assertEqual(config.initial_motor_pulses("extension"), -800)

    def test_an_unknown_axis_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_motor_degrees"):
            self.load(initial_motor_degrees={"nope": 90.0})

    def test_a_non_number_reading_is_rejected(self) -> None:
        for bad in ("270", True, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.load(initial_motor_degrees={"extension": bad})

    def test_a_reading_beyond_the_axis_travel_is_rejected(self) -> None:
        """伸缩行程 179.1 mm / 61 mm 每圈 = 1057°；声明 1200° 对应不到真实位置。"""
        with self.assertRaisesRegex(ValueError, "行程"):
            self.load(
                axes={
                    "base": {"motor_id": 5, "reduction_ratio": 5},
                    "lift": {"motor_id": 6, "mm_per_motor_rev": 61.0},
                    "extension": {"motor_id": 7, "mm_per_motor_rev": 61.0},
                },
                limits={
                    "base_degrees": [-180.0, 180.0],
                    "radial_mm": [180.0, 359.1],
                    "height_mm": [10.0, 183.5],
                },
                initial_motor_degrees={"extension": 1200.0},
            )

    def test_the_travel_endpoint_itself_is_still_allowed(self) -> None:
        """边界：正好等于总行程（底座 360° × g = 5 → 1800°）不算超。"""
        config = self.load(
            limits={"base_degrees": [-180.0, 180.0]},
            initial_motor_degrees={"base": 1800.0},
        )
        self.assertEqual(config.initial_motor_pulses("base"), 16000)

    def test_an_uncalibrated_axis_skips_the_upper_bound_check(self) -> None:
        """缺行程标定就无从判断，跳过——与 ``limits`` 缺失等于没有保护的既有约定一致。"""
        config = self.load(initial_motor_degrees={"extension": 5000.0})
        self.assertGreater(config.initial_motor_pulses("extension"), 0)

    def test_an_unknown_axis_name_is_reported_by_the_accessor(self) -> None:
        config = self.load()
        with self.assertRaises(KeyError):
            config.initial_motor_pulses("nope")


class ProjectArmConfigTests(unittest.TestCase):
    """仓库配置里已经**实测**填上的标定值。

    2026-10-16 实测（操作流程见 ``only_for_test/README.md``）：

    - 底座：16000 脉冲回到原位 -> ``reduction_ratio = 5``（实验 A0）；
    - 升降 / 伸缩：3200 脉冲 = 61 mm -> ``mm_per_motor_rev = 61``（实验 B/C）；
    - 行程：底座 360°、升降 1024° = 173.5 mm（底端离台面 10 mm）、
      伸缩 1057° = 179.1 mm（全缩回时半径 180 mm）（实验 F）。

    这些都是**量出来的**，所以钉住它们：改回占位值（升降曾经写的是 ``10``，差 6.1 倍）
    必须让测试红，而不是像以前那样只断言"应该是 None"。
    """

    def test_the_repository_config_carries_the_measured_transmission_constants(self) -> None:
        config = load_arm_config()
        self.assertEqual(config.base_reduction, 5)
        self.assertEqual(config.mm_per_motor_rev("lift"), 61.0)
        self.assertEqual(config.mm_per_motor_rev("extension"), 61.0)

    def test_the_repository_config_carries_the_measured_travel_limits(self) -> None:
        config = load_arm_config()
        self.assertEqual(config.base_limit, TravelLimit(-180.0, 180.0))
        self.assertEqual(config.height_limit, TravelLimit(10.0, 183.5))
        self.assertEqual(config.radial_limit, TravelLimit(180.0, 359.1))

    def test_the_pulse_ceiling_stays_inside_the_measured_travel(self) -> None:
        """``max_pulses`` 曾经是 10000，比两个直线轴的实测行程都长（9102 / 9396）。

        那意味着一条 ``move_up(10000)`` 会在顶到机械止挡之后再顶 17 mm —— 而当时
        ``limits`` 全是 ``null``，没有任何软件保护。这条守住"脉冲上限不得超过实测行程"。
        """
        config = load_arm_config()
        for axis in ("lift", "extension"):
            limit = config.axis_travel_span(axis)
            assert limit is not None
            span_mm = limit.maximum - limit.minimum
            travel_pulses = round(
                span_mm
                * config.pulses_per_revolution
                / config.require_mm_per_motor_rev(axis)
            )
            with self.subTest(axis=axis):
                self.assertLessEqual(config.max_pulses, travel_pulses)

    def test_the_repository_config_declares_the_extension_starting_angle(self) -> None:
        """2026-10-18 换了爪子：7 号轴的机械初始位在原点外 270° = 2400 脉冲 = 45.75 mm。

        这是**状态声明**（程序启动那一刻轴在哪），不是标定常数：回零 / 绝对归零把原点
        重新定义之后它就失效了。机构改回"启动即在原点"时，把这里改回 0 即可。
        """
        config = load_arm_config()
        self.assertEqual(config.initial_motor_pulses("extension"), 2400)
        self.assertEqual(config.initial_motor_pulses("base"), 0)
        self.assertEqual(config.initial_motor_pulses("lift"), 0)

    def test_safe_pose_is_still_honest_about_being_unmeasured(self) -> None:
        """``safe_pose`` 只能由相机测出零点姿态后填，不许默认一个值。"""
        config = load_arm_config()
        self.assertIsNone(config.safe_pose)


if __name__ == "__main__":
    unittest.main()

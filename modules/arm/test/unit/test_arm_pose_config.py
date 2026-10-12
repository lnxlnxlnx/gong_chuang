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
            return load_arm_config(path)


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


class ProjectArmConfigTests(unittest.TestCase):
    def test_the_repository_config_loads_and_is_honest_about_being_uncalibrated(self) -> None:
        config = load_arm_config()
        self.assertEqual(config.base_reduction, 5)
        self.assertIsNone(config.safe_pose)
        self.assertIsNone(config.base_limit)
        self.assertIsNone(config.mm_per_motor_rev("lift"))
        self.assertIsNone(config.mm_per_motor_rev("extension"))


if __name__ == "__main__":
    unittest.main()

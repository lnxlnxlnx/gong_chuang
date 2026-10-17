"""抓取策略参数：默认值、边界校验，以及"不读机器常数"的分工。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.grasp.config import CorrectionSettings, GraspConfig, load_grasp_config


def write_config(path: Path, document: object) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


class GraspConfigDefaultsTests(unittest.TestCase):
    def test_the_repository_file_loads(self) -> None:
        config = load_grasp_config()
        self.assertGreater(config.clearance_mm, 0.0)
        self.assertGreaterEqual(config.correction.iterations, 1)
        self.assertGreater(config.correction.tolerance_mm, 0.0)
        self.assertTrue(0.0 < config.correction.gain <= 1.0)

    def test_an_empty_document_falls_back_to_the_documented_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grasp.json"
            write_config(path, {})
            config = load_grasp_config(path)

        self.assertEqual(config.clearance_mm, 25.0)
        self.assertEqual(config.grip_ratio, 0.5)
        self.assertEqual(config.marker_to_grasp_mm, 0.0)
        self.assertIsNone(config.motion_rpm)
        self.assertEqual(config.correction, CorrectionSettings())

    def test_the_strategy_file_does_not_carry_machine_constants(self) -> None:
        """传动比、行程、相机标定各有归属，不该混进抓取策略文件。"""
        config = load_grasp_config()
        for forbidden in ("mm_per_motor_rev", "reduction_ratio", "focal_px", "safe_pose"):
            self.assertFalse(hasattr(config, forbidden), forbidden)


class GraspConfigValidationTests(unittest.TestCase):
    def load(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grasp.json"
            write_config(path, document)
            return load_grasp_config(path)

    def test_out_of_range_values_are_rejected(self) -> None:
        for document in (
            {"clearance_mm": -1.0},
            {"grip_ratio": 1.5},
            {"grip_ratio": -0.1},
            {"marker_to_grasp_mm": "up"},
            {"motion_rpm": 0},
            {"motion_rpm": 99999},
            {"correction": {"tolerance_mm": 0}},
            {"correction": {"gain": 0.0}},
            {"correction": {"gain": 1.5}},
            {"correction": {"iterations": 0}},
            {"correction": {"iterations": -1}},
            {"correction": {"iterations": 2.5}},
        ):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    self.load(document)

    def test_a_valid_override_is_read_back(self) -> None:
        config = self.load(
            {
                "clearance_mm": 30.0,
                "grip_ratio": 0.25,
                "marker_to_grasp_mm": -12.5,
                "motion_rpm": 8,
                "correction": {"iterations": 5, "tolerance_mm": 1.5, "gain": 0.5},
            }
        )
        self.assertEqual(config.clearance_mm, 30.0)
        self.assertEqual(config.grip_ratio, 0.25)
        self.assertEqual(config.marker_to_grasp_mm, -12.5)
        self.assertEqual(config.motion_rpm, 8)
        self.assertEqual(
            config.correction, CorrectionSettings(iterations=5, tolerance_mm=1.5, gain=0.5)
        )

    def test_missing_correction_block_keeps_the_defaults(self) -> None:
        config = self.load({"clearance_mm": 10.0})
        self.assertEqual(config.correction, CorrectionSettings())


if __name__ == "__main__":
    unittest.main()

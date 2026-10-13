"""视觉两个配置文件的读取与写入：目标参数（手改）与标定结果（机器写）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.vision.config import (
    DEFAULT_MATERIAL_DIAMETER_MM,
    TargetSettings,
    load_vision_config,
    save_vision_calibration,
)
from modules.vision.drive.calibration import PlaneCalibration


def write_config(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


class VisionTargetSettingsTests(unittest.TestCase):
    def test_missing_targets_block_falls_back_to_the_documented_material_diameter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            write_config(config_path, {"windows": {"camera_id": 1}})

            config = load_vision_config(config_path, Path(directory) / "absent.json")

        self.assertEqual(config.targets.material_diameter_mm, DEFAULT_MATERIAL_DIAMETER_MM)
        self.assertIsNone(config.targets.marker_diameter_mm)
        self.assertFalse(config.targets.can_split_by_color)

    def test_targets_are_read_from_the_hand_edited_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            write_config(
                config_path,
                {
                    "jetson": {"camera_id": 0},
                    "targets": {
                        "material_diameter_mm": 30.0,
                        "marker_diameter_mm": 20.0,
                        "material_bgr": [200, 120, 60],
                        "marker_bgr": [245, 245, 245],
                        "color_tolerance": 45.0,
                    },
                },
            )

            config = load_vision_config(config_path, Path(directory) / "absent.json")

        self.assertEqual(config.camera_id("jetson"), 0)
        self.assertEqual(config.targets.marker_diameter_mm, 20.0)
        self.assertEqual(config.targets.material_bgr, (200.0, 120.0, 60.0))
        self.assertTrue(config.targets.can_split_by_color)
        self.assertEqual(config.targets.color_tolerance, 45.0)

    def test_invalid_target_values_are_rejected(self) -> None:
        for targets in (
            {"material_diameter_mm": 0},
            {"marker_diameter_mm": -5},
            {"material_bgr": [1, 2]},
            {"color_tolerance": 0},
            {"working_distance_mm": [85, 60]},
            {"working_distance_mm": [0, 80]},
            {"working_distance_mm": [70]},
            {"working_distance_mm": "60,80"},
        ):
            with self.subTest(targets=targets):
                with tempfile.TemporaryDirectory() as directory:
                    config_path = Path(directory) / "config.json"
                    write_config(config_path, {"windows": {"camera_id": 1}, "targets": targets})
                    with self.assertRaises(ValueError):
                        load_vision_config(config_path, Path(directory) / "absent.json")

    def test_working_distance_is_read_as_a_near_far_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            write_config(
                config_path,
                {"windows": {"camera_id": 1},
                 "targets": {"working_distance_mm": [60, 85.5]}},
            )

            config = load_vision_config(config_path, Path(directory) / "absent.json")

        self.assertEqual(config.targets.working_distance_mm, (60.0, 85.5))

    def test_missing_working_distance_means_no_prior(self) -> None:
        self.assertIsNone(TargetSettings().working_distance_mm)

    def test_unknown_platform_is_reported_by_the_accessor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            write_config(config_path, {"windows": {"camera_id": 3}})
            config = load_vision_config(config_path, Path(directory) / "absent.json")
        self.assertEqual(config.camera_id("windows"), 3)
        with self.assertRaisesRegex(ValueError, "jetson"):
            config.camera_id("jetson")

    def test_unconfigured_marker_and_table_are_reported_by_the_accessors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            write_config(config_path, {"windows": {"camera_id": 1}})
            config = load_vision_config(config_path, Path(directory) / "absent.json")

        with self.assertRaisesRegex(ValueError, "marker_diameter_mm"):
            config.require_marker_diameter()
        with self.assertRaisesRegex(ValueError, "table_depth_mm"):
            config.require_table_depth()
        with self.assertRaisesRegex(ValueError, "平面标定"):
            config.require_plane()


class VisionCalibrationFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.config_path = self.directory / "config.json"
        self.calibration_path = self.directory / "calibration.json"
        write_config(self.config_path, {"windows": {"camera_id": 1}})

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_project_calibration_file_parses_as_uncalibrated(self) -> None:
        """仓库里那份占位文件必须能被读成"还没标定"，而不是抛异常。"""
        config = load_vision_config()
        self.assertIsNone(config.plane)
        self.assertIsNone(config.focal_px)
        self.assertIsNone(config.table_depth_mm)

    def test_round_trip_preserves_every_written_field(self) -> None:
        plane = PlaneCalibration(
            center_x_mm=101.5,
            center_y_mm=-22.25,
            rotation_degrees=13.75,
            rms_mm=1.5,
            samples=12,
        )
        save_vision_calibration(
            calibration_path=self.calibration_path,
            plane=plane,
            focal_px=1018.0,
            table_depth_mm=305.5,
        )

        config = load_vision_config(self.config_path, self.calibration_path)

        self.assertEqual(config.focal_px, 1018.0)
        self.assertEqual(config.table_depth_mm, 305.5)
        self.assertIsNotNone(config.plane)
        self.assertAlmostEqual(config.plane.center_x_mm, 101.5, places=9)
        self.assertAlmostEqual(config.plane.center_y_mm, -22.25, places=9)
        self.assertAlmostEqual(config.plane.rotation_degrees, 13.75, places=9)
        self.assertAlmostEqual(config.plane.rms_mm, 1.5, places=9)
        self.assertEqual(config.plane.samples, 12)

    def test_writing_one_field_keeps_the_others(self) -> None:
        """标完焦距不能把已有的平面标定抹掉，反之亦然。"""
        save_vision_calibration(
            calibration_path=self.calibration_path,
            focal_px=999.0,
            table_depth_mm=300.0,
        )
        save_vision_calibration(
            calibration_path=self.calibration_path,
            plane=PlaneCalibration(center_x_mm=1.0, center_y_mm=2.0, rotation_degrees=3.0),
        )

        config = load_vision_config(self.config_path, self.calibration_path)

        self.assertEqual(config.focal_px, 999.0)
        self.assertEqual(config.table_depth_mm, 300.0)
        self.assertIsNotNone(config.plane)
        self.assertAlmostEqual(config.plane.rotation_degrees, 3.0, places=9)

    def test_invalid_calibration_values_are_rejected(self) -> None:
        for document in (
            {"focal_px": 0},
            {"table_depth_mm": -1},
            {"plane": {"center_x_mm": 1.0, "center_y_mm": 2.0, "rotation_degrees": "north"}},
        ):
            with self.subTest(document=document):
                write_config(self.calibration_path, document)
                with self.assertRaises(ValueError):
                    load_vision_config(self.config_path, self.calibration_path)

    def test_all_null_plane_block_reads_as_uncalibrated(self) -> None:
        write_config(
            self.calibration_path,
            {
                "focal_px": None,
                "table_depth_mm": None,
                "plane": {
                    "center_x_mm": None,
                    "center_y_mm": None,
                    "rotation_degrees": None,
                    "rms_mm": None,
                    "samples": 0,
                },
            },
        )
        config = load_vision_config(self.config_path, self.calibration_path)
        self.assertIsNone(config.plane)


if __name__ == "__main__":
    unittest.main()

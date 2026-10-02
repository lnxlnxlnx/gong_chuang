"""COIN-D6 configuration loading and validation checks."""

import json
import os
import unittest
from pathlib import Path

from lidar.config import (
    MAX_DISTANCE_MM,
    default_profile_name,
    describe_config,
    load_lidar_config,
)
from tests.scratch import scratch_directory


def document(**overrides) -> dict:
    """A complete, valid configuration document that tests can break."""
    root = {
        "profiles": {"windows": {"port": "COM7"}, "jetson": {"port": "/dev/ttyUSB1"}},
        "device": {
            "baudrate": 230400,
            "ack_timeout_seconds": 1.0,
            "read_timeout_seconds": 0.2,
            "strict_checksum": True,
            "require_ack": True,
        },
        "scan": {
            "revolutions": 2,
            "angle_min_degrees": -90.0,
            "angle_max_degrees": 90.0,
            "min_distance_mm": 50,
            "max_distance_mm": 8000,
            "clockwise": True,
            "angle_offset_degrees": 0.0,
            "max_points": 5000,
            "timeout_seconds": 4.0,
        },
        "visualization": {
            "range_mm": 3000.0,
            "color_by": "intensity",
            "point_size": 5.0,
            "colormap": "viridis",
            "show_window": False,
        },
    }
    root.update(overrides)
    return root


class ShippedConfigTests(unittest.TestCase):
    def test_the_project_config_loads_for_both_platforms(self):
        windows = load_lidar_config("windows")
        jetson = load_lidar_config("jetson")

        self.assertEqual(windows.profile, "windows")
        self.assertTrue(windows.device.port)
        self.assertTrue(jetson.device.port)
        self.assertEqual(windows.device.baudrate, 230400)
        self.assertTrue(windows.device.strict_checksum)
        self.assertGreater(windows.scan.revolutions, 0)
        self.assertLess(windows.scan.angle_min_degrees, windows.scan.angle_max_degrees)
        self.assertLessEqual(windows.scan.max_distance_mm, MAX_DISTANCE_MM)

    def test_the_default_profile_matches_the_operating_system(self):
        self.assertEqual(default_profile_name(), "windows" if os.name == "nt" else "jetson")
        self.assertEqual(load_lidar_config().profile, default_profile_name())

    def test_describe_mentions_the_port_and_window(self):
        text = describe_config(load_lidar_config("windows"))

        self.assertIn(load_lidar_config("windows").device.port, text)
        self.assertIn("角度", text)


class LoaderTests(unittest.TestCase):
    def load(self, payload, profile="windows"):
        with scratch_directory() as directory:
            path = Path(directory) / "coin_d6.json"
            if isinstance(payload, str):
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(json.dumps(payload), encoding="utf-8")
            return load_lidar_config(profile, path)

    def test_reads_every_documented_field(self):
        config = self.load(document())

        self.assertEqual(config.device.port, "COM7")
        self.assertEqual(config.device.read_timeout_seconds, 0.2)
        self.assertTrue(config.device.require_ack)
        self.assertEqual(config.scan.revolutions, 2)
        self.assertEqual(config.scan.angle_min_degrees, -90.0)
        self.assertEqual(config.scan.min_distance_mm, 50)
        self.assertTrue(config.scan.clockwise)
        self.assertEqual(config.view.color_by, "intensity")
        self.assertFalse(config.view.show_window)

    def test_missing_optional_sections_fall_back_to_defaults(self):
        config = self.load({"profiles": {"windows": {"port": "COM9"}}})

        self.assertEqual(config.device.baudrate, 230400)
        self.assertEqual(config.scan.revolutions, 3)
        self.assertEqual(config.scan.angle_min_degrees, -180.0)
        self.assertEqual(config.view.color_by, "distance")
        self.assertTrue(config.view.show_window)

    def test_rejects_a_missing_file(self):
        with self.assertRaisesRegex(ValueError, "cannot read"):
            load_lidar_config("windows", Path("nowhere") / "coin_d6.json")

    def test_rejects_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            self.load("{not json")

    def test_rejects_a_non_object_root(self):
        with self.assertRaisesRegex(ValueError, "root must be an object"):
            self.load([])

    def test_rejects_an_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "unknown profile"):
            self.load(document(), "laboratory")

    def test_rejects_a_missing_or_empty_port(self):
        with self.assertRaisesRegex(ValueError, "port"):
            self.load({"profiles": {"windows": {}}})
        with self.assertRaisesRegex(ValueError, "port"):
            self.load({"profiles": {"windows": {"port": "  "}}})

    def test_rejects_an_impossible_scan_window(self):
        broken = document()
        broken["scan"]["angle_min_degrees"] = 90.0
        broken["scan"]["angle_max_degrees"] = -90.0
        with self.assertRaisesRegex(ValueError, "angle_min_degrees"):
            self.load(broken)

        broken = document()
        broken["scan"]["min_distance_mm"] = 9000
        with self.assertRaisesRegex(ValueError, "min_distance_mm"):
            self.load(broken)

    def test_rejects_out_of_range_numbers(self):
        for section, key, value in (
            ("device", "baudrate", 10),
            ("scan", "revolutions", 0),
            ("scan", "max_distance_mm", MAX_DISTANCE_MM + 1),
            ("visualization", "range_mm", 0),
            ("visualization", "color_by", "rainbow"),
        ):
            broken = document()
            broken[section][key] = value
            with self.assertRaises(ValueError, msg=f"{section}.{key}"):
                self.load(broken)

    def test_rejects_non_boolean_switches(self):
        broken = document()
        broken["device"]["strict_checksum"] = "yes"
        with self.assertRaisesRegex(ValueError, "strict_checksum"):
            self.load(broken)

        broken = document()
        broken["scan"]["clockwise"] = 1
        with self.assertRaisesRegex(ValueError, "clockwise"):
            self.load(broken)


if __name__ == "__main__":
    unittest.main()

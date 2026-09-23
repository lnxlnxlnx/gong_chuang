"""Verify loading of platform-specific mecanum hardware test profiles."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chassis import MecanumChassis
from hardware_tests.mecanum.config import create_chassis, load_profile
from mecanum import WheelPosition


def write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "vehicle": {
                    "motor_ids": {"FL": 1, "FR": 2, "RL": 3, "RR": 4},
                    "wheel_signs": {"FL": 1, "FR": -1, "RL": 1, "RR": 1},
                    "acceleration": 5,
                    "pattern": "X",
                    "rotation_gain": 1.25,
                },
                "profiles": {
                    "jetson": {
                        "port": "/dev/ttyUSB0",
                        "individual_wheels": {"rpm": 20, "duration": 0.75},
                        "chassis_motion": {"rpm": 30, "duration": 1.0, "pause": 0.5},
                    },
                    "windows": {
                        "port": "COM3",
                        "individual_wheels": {"rpm": 15, "duration": 0.5},
                        "chassis_motion": {"rpm": 25, "duration": 0.8, "pause": 0.25},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class MecanumHardwareConfigTests(unittest.TestCase):
    def test_loads_windows_profile_with_shared_vehicle_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)

            profile = load_profile("windows", config_path)

        self.assertEqual(profile.port, "COM3")
        self.assertEqual(profile.motor_ids[WheelPosition.FL], 1)
        self.assertEqual(profile.wheel_signs[WheelPosition.FR], -1)
        self.assertEqual(profile.acceleration, 5)
        self.assertEqual(profile.pattern, "X")
        self.assertEqual(profile.rotation_gain, 1.25)
        self.assertEqual(profile.motion_transform, "standard")
        self.assertEqual(profile.individual_rpm, 15)
        self.assertEqual(profile.chassis_pause, 0.25)

    def test_rejects_an_unknown_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)

            with self.assertRaisesRegex(ValueError, "unknown profile"):
                load_profile("laboratory", config_path)

    def test_rejects_duplicate_motor_addresses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["vehicle"]["motor_ids"]["RR"] = 1
            config_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be unique"):
                load_profile("jetson", config_path)

    def test_create_chassis_wraps_profile_drive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)
            profile = load_profile("jetson", config_path)

        with patch("hardware_tests.mecanum.config.create_drive") as create_drive:
            drive = object()
            create_drive.return_value = drive
            chassis = create_chassis(profile)

        self.assertIsInstance(chassis, MecanumChassis)
        self.assertIs(chassis.drive, drive)
        create_drive.assert_called_once_with(profile, port=None, acceleration=None)

def test_installed_profiles_use_calibrated_motion_transform(self) -> None:
        self.assertEqual(load_profile("windows").motion_transform, "calibrated")


if __name__ == "__main__":
    unittest.main()

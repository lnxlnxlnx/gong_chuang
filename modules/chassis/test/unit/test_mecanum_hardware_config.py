"""Verify loading of platform-specific mecanum hardware test profiles."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.chassis.drive.chassis import MecanumChassis
from modules.chassis.config import create_chassis, create_drive, load_profile
from modules.chassis.drive.mecanum import WheelPosition


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
                        "individual_wheels": {"rpm": 20, "duration": 0.75},
                        "chassis_motion": {"rpm": 30, "duration": 1.0, "pause": 0.5},
                    },
                    "windows": {
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

        with patch("modules.chassis.config.create_drive") as create_drive:
            drive = object()
            create_drive.return_value = drive
            chassis = create_chassis(profile)

        self.assertIsInstance(chassis, MecanumChassis)
        self.assertIs(chassis.drive, drive)
        create_drive.assert_called_once_with(profile, port=None, acceleration=None)

    def test_installed_profiles_use_calibrated_motion_transform(self) -> None:
        """The shipped profile must keep the calibrated transform, not the test default."""
        self.assertEqual(load_profile("windows").motion_transform, "calibrated")

    def test_drive_port_comes_from_the_shared_stepper_bus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)
            profile = load_profile("windows", config_path)

        with patch(
            "modules.chassis.config.load_bus_config",
            return_value=SimpleNamespace(port="COM42"),
        ), patch("modules.chassis.config.MecanumDrive") as drive:
            create_drive(profile)

        self.assertEqual(drive.call_args.args[0], "COM42")

    def test_explicit_port_overrides_the_shared_stepper_bus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)
            profile = load_profile("windows", config_path)

        with patch(
            "modules.chassis.config.load_bus_config",
            return_value=SimpleNamespace(port="COM42"),
        ), patch("modules.chassis.config.MecanumDrive") as drive:
            create_drive(profile, port="COM7")

        self.assertEqual(drive.call_args.args[0], "COM7")


if __name__ == "__main__":
    unittest.main()

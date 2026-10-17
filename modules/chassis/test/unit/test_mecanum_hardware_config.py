"""Verify loading of platform-specific mecanum hardware test profiles."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.chassis.config import load_profile
from modules.chassis.drive.mecanum import WheelPosition
from modules.chassis.runtime import MecanumChassis

#: 装配 arm+chassis 时那条共享步进连接的替身。
BUS = SimpleNamespace(close=lambda: None, is_open=True)


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

    def test_open_assembles_the_profile_drive_and_takes_the_shared_bus_port(self) -> None:
        """装配只有一处：``MecanumChassis.open`` 读 profile、从 motor 配置取端口、建驱动。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)
            expected = load_profile("windows", config_path)

            with patch(
                "modules.chassis.runtime.load_bus_config",
                return_value=SimpleNamespace(port="COM42"),
            ), patch("modules.chassis.runtime.MecanumDrive") as drive:
                chassis = MecanumChassis.open(config_path=config_path)

        self.assertIsInstance(chassis, MecanumChassis)
        self.assertIs(chassis.drive, drive.return_value)
        self.assertEqual(drive.call_args.args[0], "COM42")
        # profile 的校准值真的传到了驱动上。
        self.assertEqual(drive.call_args.kwargs["motion_transform"], expected.motion_transform)
        self.assertEqual(drive.call_args.kwargs["motor_ids"], expected.motor_ids)
        drive.return_value.initialize.assert_called_once_with()
        drive.return_value.enable_all.assert_called_once_with(True)

    def test_the_module_level_motor_config_path_override_is_used(self) -> None:
        """``motor_config_path`` 只对本次会话生效，并真的传给加载器。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)
            motor_path = Path(temporary_directory) / "motor.json"

            with patch(
                "modules.chassis.runtime.load_bus_config",
                return_value=SimpleNamespace(port="COM42"),
            ) as load_bus, patch("modules.chassis.runtime.MecanumDrive") as drive:
                MecanumChassis.open(
                    config_path=config_path, motor_config_path=motor_path
                )

        self.assertEqual(load_bus.call_args.kwargs["config_path"], motor_path)
        self.assertEqual(drive.call_args.args[0], "COM42")

    def test_installed_profiles_use_calibrated_motion_transform(self) -> None:
        """The shipped profile must keep the calibrated transform, not the test default."""
        self.assertEqual(load_profile("windows").motion_transform, "calibrated")

    def test_borrowed_bus_is_handed_to_the_drive_and_not_owned(self) -> None:
        """借来的步进连接只**传进去**：底盘不物理关闭它（arm 还在用同一条）。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "mecanum_hardware.json"
            write_config(config_path)
            profile = load_profile("windows", config_path)

        with patch("modules.chassis.runtime.MecanumDrive") as drive:
            MecanumChassis._from_settings(
                profile=profile, port="COM42", bus=BUS, owns_bus=False
            )

        self.assertIs(drive.call_args.kwargs["serial_port"], BUS)
        # 借来的连接：显式告诉驱动"你不拥有它"。
        self.assertFalse(drive.return_value._owns_serial)


if __name__ == "__main__":
    unittest.main()

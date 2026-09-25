"""Verify editable robot-arm hardware settings."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robot_arm_config import load_arm_config


def write_config(path: Path, *, default_pulses: int = 600, max_pulses: int = 1000) -> None:
    path.write_text(
        json.dumps(
            {
                "motion": {
                    "pulses_per_revolution": 3200,
                    "rpm": 10,
                    "acceleration": 5,
                    "default_pulses": default_pulses,
                    "max_pulses": max_pulses,
                },
                "axes": {
                    "base": {"motor_id": 5, "reduction_ratio": 6},
                    "lift": {"motor_id": 6},
                    "extension": {"motor_id": 7},
                },
            }
        ),
        encoding="utf-8",
    )


class RobotArmConfigTests(unittest.TestCase):
    def test_loads_motion_limits_and_axis_addresses_from_one_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot_arm.json"
            write_config(config_path, default_pulses=1200, max_pulses=5000)

            config = load_arm_config(config_path)

        self.assertEqual(config.default_pulses, 1200)
        self.assertEqual(config.max_pulses, 5000)
        self.assertEqual(config.rpm, 10)
        self.assertEqual(config.acceleration, 5)
        self.assertEqual(config.motor_id("base"), 5)
        self.assertEqual(config.motor_id("lift"), 6)
        self.assertEqual(config.motor_id("extension"), 7)
        self.assertEqual(config.base_reduction, 6)

    def test_rejects_default_pulses_above_the_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot_arm.json"
            write_config(config_path, default_pulses=5001, max_pulses=5000)

            with self.assertRaisesRegex(ValueError, "default_pulses"):
                load_arm_config(config_path)


if __name__ == "__main__":
    unittest.main()

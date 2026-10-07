"""The manual bus-servo test must never move before explicit confirmation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules.servo.drive.bus_servo import load_bus_config
from tests.manual import servo


class BusServoConfigTests(unittest.TestCase):
    @staticmethod
    def _write(directory: str, document: dict) -> Path:
        path = Path(directory) / "servo.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_loads_port_baudrate_and_servo_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                {"windows": {"port": "COM14"}, "bus": {"baudrate": 115200, "servo_id": 2}},
            )
            config = load_bus_config("windows", path)
        self.assertEqual((config.port, config.baudrate, config.servo_id), ("COM14", 115200, 2))

    def test_port_must_be_configured_before_opening_serial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, {"jetson": {"port": ""}, "bus": {}})
            with self.assertRaisesRegex(ValueError, "jetson.port"):
                load_bus_config("jetson", path)

    def test_bus_section_and_servo_id_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, {"windows": {"port": "COM14"}})
            with self.assertRaisesRegex(ValueError, "bus"):
                load_bus_config("windows", path)

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                {"windows": {"port": "COM14"}, "bus": {"baudrate": 115200, "servo_id": 0}},
            )
            with self.assertRaisesRegex(ValueError, "bus.servo_id"):
                load_bus_config("windows", path)

    def test_project_file_configures_the_verified_direct_bus(self) -> None:
        """The project file must supply a usable direct-bus bus config, not just a port."""
        config = load_bus_config("windows")
        self.assertEqual(config.baudrate, 115200)
        self.assertTrue(config.port.strip())
        self.assertTrue(1 <= config.servo_id <= 253)


class ServoManualTests(unittest.TestCase):
    def test_read_only_and_rejected_confirmation_never_move(self) -> None:
        actuator = Mock()
        actuator.read_position.return_value = 500
        servo.run_manual(actuator, 1, prompt=lambda _: "q")
        actuator.set_position.assert_not_called()

        answers = iter(["520", "800", "no"])
        servo.run_manual(actuator, 1, prompt=lambda _: next(answers))
        actuator.set_position.assert_not_called()

    def test_confirmed_target_moves_once_and_checks_position(self) -> None:
        actuator = Mock()
        actuator.read_position.side_effect = [500, 520]
        answers = iter(["520", "800", "YES"])
        with patch.object(servo.time, "sleep"):
            servo.run_manual(actuator, 1, prompt=lambda _: next(answers))
        actuator.set_position.assert_called_once_with(520, 800)
        self.assertEqual(actuator.read_position.call_count, 2)

    def test_main_opens_configured_port_baudrate_and_id(self) -> None:
        config = Mock(port="COM14", baudrate=115200, servo_id=2)
        with patch.object(servo, "load_bus_config", return_value=config), \
                patch.object(servo, "run_manual"), \
                patch.object(servo, "BusServo") as bus_servo, \
                patch("builtins.input", side_effect=["q"]):
            servo.main()
        bus_servo.assert_called_once_with("COM14", 2, baudrate=115200, timeout=0.5)
        bus_servo.return_value.initialize.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

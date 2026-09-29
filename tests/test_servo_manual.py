"""The manual bus-servo test must never move before explicit confirmation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.manual import servo


class ServoManualTests(unittest.TestCase):
    def test_port_must_be_configured_before_opening_serial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servo.json"
            path.write_text(json.dumps({"jetson": {"port": ""}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "jetson.port"):
                servo.load_port("jetson", path)

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
        actuator.set_position.assert_called_once_with(1, position=520, duration_ms=800)
        self.assertEqual(actuator.read_position.call_count, 2)


if __name__ == "__main__":
    unittest.main()

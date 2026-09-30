"""Check gripper configuration and confirmed action dispatch without hardware."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from gripper import Gripper, GripperConfig, load_gripper_config
from tests.manual.gripper import run_manual


class GripperTests(unittest.TestCase):
    def test_project_config_loads_gripper_positions(self) -> None:
        config = load_gripper_config("windows")
        self.assertEqual(
            (config.servo_id, config.open_position, config.closed_position, config.duration_ms),
            (1, 300, 500, 1000),
        )

    def test_controller_port_comes_from_servo_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servo.json"
            path.write_text(json.dumps({"windows": {"port": "COM77"}}), encoding="utf-8")
            self.assertEqual(load_gripper_config("windows", servo_config_path=path).port, "COM77")

    def test_invalid_endpoint_is_rejected_before_serial_opens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gripper.json"
            path.write_text(json.dumps({
                "servo_id": 1,
                "open_position": 300,
                "closed_position": 1001,
                "duration_ms": 1000,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closed_position"):
                load_gripper_config("windows", path)

    def test_actions_use_distinct_configured_positions_and_close_port(self) -> None:
        controller = Mock()
        config = GripperConfig("COM30", 1, 300, 500, 1000)
        gripper = Gripper(config, controller=controller).initialize()
        gripper.open_gripper()
        gripper.close_gripper()
        gripper.close()
        self.assertEqual(controller.set_position.call_args_list[0].args, (1,))
        self.assertEqual(controller.set_position.call_args_list[0].kwargs, {"position": 300, "duration_ms": 1000})
        self.assertEqual(controller.set_position.call_args_list[1].args, (1,))
        self.assertEqual(controller.set_position.call_args_list[1].kwargs, {"position": 500, "duration_ms": 1000})
        controller.close.assert_called_once()

    def test_manual_actions_require_individual_yes(self) -> None:
        gripper = Mock()
        gripper.config = GripperConfig("COM30", 1, 300, 500, 1000)
        gripper.read_position.return_value = 400
        answers = iter(["no", "YES"])
        run_manual(gripper, prompt=lambda _: next(answers))
        gripper.open_gripper.assert_not_called()
        gripper.close_gripper.assert_called_once()
        gripper.read_position.assert_called_once()

    def test_manual_does_not_move_when_servo_does_not_answer(self) -> None:
        gripper = Mock()
        gripper.config = GripperConfig("COM30", 1, 300, 500, 1000)
        gripper.read_position.side_effect = TimeoutError("no response")
        with self.assertRaises(TimeoutError):
            run_manual(gripper, prompt=lambda _: "YES")
        gripper.open_gripper.assert_not_called()
        gripper.close_gripper.assert_not_called()


if __name__ == "__main__":
    unittest.main()

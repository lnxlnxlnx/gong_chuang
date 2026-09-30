"""Check gripper configuration and confirmed action dispatch without hardware."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from gripper import Gripper, GripperConfig, load_gripper_config
from tests.manual import gripper as gripper_manual
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
            path.write_text(json.dumps({
                "windows": {"port": "COM77"},
                "gripper": {
                    "servo_id": 1, "open_position": 300,
                    "closed_position": 500, "duration_ms": 1000,
                },
            }), encoding="utf-8")
            self.assertEqual(load_gripper_config("windows", path).port, "COM77")

    def test_invalid_endpoint_is_rejected_before_serial_opens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servo.json"
            path.write_text(json.dumps({
                "windows": {"port": "COM77"},
                "gripper": {
                    "servo_id": 1,
                    "open_position": 300,
                    "closed_position": 1001,
                    "duration_ms": 1000,
                },
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
        servo = Mock()
        servo.read_position.return_value = 400
        config = GripperConfig("COM14", 1, 300, 500, 1000)
        answers = iter(["no", "YES"])
        with patch.object(gripper_manual.time, "sleep"):
            run_manual(servo, config, prompt=lambda _: next(answers))
        # 只有第二项得到 YES：第一次不会发送任何位置指令
        servo.set_position.assert_called_once_with(500, 1000)
        # 起始位置读一次，张开的 YES 被拒绝所以没有读回，闭合后读回一次
        self.assertEqual(servo.read_position.call_count, 2)

    def test_manual_does_not_move_when_servo_does_not_answer(self) -> None:
        servo = Mock()
        servo.read_position.side_effect = TimeoutError("no response")
        config = GripperConfig("COM14", 1, 300, 500, 1000)
        with self.assertRaises(TimeoutError):
            run_manual(servo, config, prompt=lambda _: "YES")
        servo.set_position.assert_not_called()

    def test_main_opens_configured_gripper_id_on_the_bus_port(self) -> None:
        config = GripperConfig("COM14", 1, 300, 500, 1000)
        bus = Mock(port="COM14", baudrate=115200, servo_id=2)
        with patch.object(gripper_manual, "load_gripper_config", return_value=config), \
                patch.object(gripper_manual, "load_bus_config", return_value=bus), \
                patch.object(gripper_manual, "run_manual"), \
                patch.object(gripper_manual, "BusServo") as bus_servo:
            gripper_manual.main()
        bus_servo.assert_called_once_with("COM14", 1, baudrate=115200, timeout=0.5)
        bus_servo.return_value.initialize.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

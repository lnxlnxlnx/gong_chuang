"""Direct trial sends only confirmed positions and waits for each move."""

import unittest
from unittest.mock import Mock, patch

from modules.servo.drive.gripper import GripperConfig
from tests.manual import gripper_direct
from tests.manual.gripper_direct import run_direct_test


class GripperDirectTests(unittest.TestCase):
    def test_confirmed_moves_send_sequential_positions(self) -> None:
        servo = Mock()
        servo.read_position.side_effect = [506, 300, 499]
        config = GripperConfig("COM30", 1, 300, 500, 1000)
        answers = iter(["YES", "YES"])
        sleep = Mock()
        run_direct_test(servo, config, prompt=lambda _: next(answers), sleep=sleep)
        self.assertEqual([call.args for call in servo.set_position.call_args_list], [(300, 1000), (500, 1000)])
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(servo.read_position.call_count, 3)

    def test_rejected_actions_never_send_a_position(self) -> None:
        servo = Mock()
        config = GripperConfig("COM30", 1, 300, 500, 1000)
        run_direct_test(servo, config, prompt=lambda _: "no", sleep=Mock())
        servo.set_position.assert_not_called()

    def test_main_defaults_to_unified_servo_port(self) -> None:
        config = GripperConfig("COM77", 1, 300, 500, 1000)
        with patch("sys.argv", ["gripper_direct.py"]), \
                patch.object(gripper_direct, "load_gripper_config", return_value=config), \
                patch.object(gripper_direct, "BusServo") as bus_servo, \
                patch.object(gripper_direct, "run_direct_test"):
            gripper_direct.main()
        bus_servo.assert_called_once_with("COM77", 1, timeout=0.5)


if __name__ == "__main__":
    unittest.main()

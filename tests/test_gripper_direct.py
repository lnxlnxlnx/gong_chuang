"""Direct trial sends only confirmed positions and waits for each move."""

import unittest
from unittest.mock import Mock

from gripper import GripperConfig
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


if __name__ == "__main__":
    unittest.main()

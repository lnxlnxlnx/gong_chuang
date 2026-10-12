"""夹爪应用层：开合、等待、读回到位判据，以及"夹住东西"的检测。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from modules.servo.config import GripperConfig
from modules.servo.drive.bus_servo import BusServoConfig
from modules.servo.drive.gripper import (
    CLOSE_COMMAND,
    OPEN_COMMAND,
    DEFAULT_POSITION_TOLERANCE,
    GripperServo,
)

CONFIG = GripperConfig(servo_id=1, open_position=300, closed_position=500, duration_ms=1000)
BUS = BusServoConfig(port="FAKE", baudrate=115200, servo_id=1)


def make_gripper(positions, **overrides) -> tuple[GripperServo, Mock, list]:
    servo = Mock()
    servo.read_position.side_effect = list(positions)
    slept: list = []
    settings = dict(
        config=CONFIG,
        bus_config=BUS,
        servo=servo,
        sleep=slept.append,
        settle_seconds=0.2,
    )
    settings.update(overrides)
    return GripperServo("", **settings), servo, slept


class GripperLifecycleTests(unittest.TestCase):
    def test_initialize_opens_the_bus_and_holds_torque(self) -> None:
        gripper, servo, _ = make_gripper([])

        gripper.initialize()

        servo.initialize.assert_called_once_with()
        servo.set_torque.assert_called_once_with(True)

    def test_close_only_closes_the_bus_and_leaves_the_jaws_alone(self) -> None:
        gripper, servo, _ = make_gripper([])

        gripper.close()

        servo.close.assert_called_once_with()
        servo.set_position.assert_not_called()

    def test_the_context_manager_initializes_and_closes(self) -> None:
        gripper, servo, _ = make_gripper([])

        with gripper:
            pass

        servo.initialize.assert_called_once_with()
        servo.close.assert_called_once_with()

    def test_injected_servo_means_no_serial_port_is_opened(self) -> None:
        gripper, servo, _ = make_gripper([])
        self.assertEqual(gripper.port, "FAKE")
        servo.initialize.assert_not_called()


class GripperMotionTests(unittest.TestCase):
    def test_open_uses_the_configured_open_position(self) -> None:
        gripper, servo, slept = make_gripper([300])

        report = gripper.open_jaws()

        servo.set_position.assert_called_once_with(300, 1000)
        self.assertEqual((report.command, report.target, report.position), (OPEN_COMMAND, 300, 300))
        self.assertTrue(report.reached)
        self.assertEqual(slept, [1.2])

    def test_close_uses_the_configured_closed_position(self) -> None:
        gripper, servo, _ = make_gripper([500])

        report = gripper.close_jaws()

        servo.set_position.assert_called_once_with(500, 1000)
        self.assertEqual(report.command, CLOSE_COMMAND)
        self.assertTrue(report.reached)

    def test_a_blocked_close_is_reported_as_not_reached_and_that_is_not_an_error(self) -> None:
        """爪口被物料挡住时读回值到不了闭合位——这正是"夹到了东西"的信号。"""
        gripper, _, _ = make_gripper([438])

        report = gripper.close_jaws()

        self.assertFalse(report.reached)
        self.assertEqual(report.error, -62)
        self.assertIn("未到位", report.summary())

    def test_the_tolerance_is_inclusive_at_its_boundary(self) -> None:
        gripper, _, _ = make_gripper([500 - DEFAULT_POSITION_TOLERANCE])
        self.assertTrue(gripper.close_jaws().reached)

        gripper, _, _ = make_gripper([500 - DEFAULT_POSITION_TOLERANCE - 1])
        self.assertFalse(gripper.close_jaws().reached)

    def test_wait_false_sends_the_command_without_sleeping_or_claiming_arrival(self) -> None:
        gripper, servo, slept = make_gripper([412])

        report = gripper.close_jaws(wait=False)

        servo.set_position.assert_called_once_with(500, 1000)
        self.assertEqual(slept, [])
        self.assertFalse(report.reached)
        self.assertEqual(report.position, 412)

    def test_set_position_drives_a_raw_value_for_calibration(self) -> None:
        gripper, servo, _ = make_gripper([420])

        report = gripper.set_position(420)

        servo.set_position.assert_called_once_with(420, 1000)
        self.assertEqual(report.command, "set")
        self.assertTrue(report.reached)

    def test_read_stop_and_torque_are_passed_through(self) -> None:
        gripper, servo, _ = make_gripper([377])

        self.assertEqual(gripper.read_position(), 377)
        gripper.stop()
        gripper.set_torque(False)

        servo.read_position.assert_called_once_with()
        servo.stop.assert_called_once_with()
        servo.set_torque.assert_called_once_with(False)


class GripperValidationTests(unittest.TestCase):
    def test_negative_tolerance_and_settle_are_rejected(self) -> None:
        for overrides in ({"position_tolerance": -1}, {"settle_seconds": -0.1}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_gripper([], **overrides)

    def test_the_settle_margin_is_added_on_top_of_the_configured_duration(self) -> None:
        gripper, _, slept = make_gripper([300], settle_seconds=0.5)
        gripper.open_jaws()
        self.assertEqual(slept, [1.5])


if __name__ == "__main__":
    unittest.main()

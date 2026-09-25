"""Verify the high-level chassis facade and coordinated position commands."""

from __future__ import annotations

import unittest

from chassis import MecanumChassis
from mecanum import MecanumDrive, WheelPosition
from test_mecanum import FakeSerial


class ChassisFacadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.drive = MecanumDrive("COM14", serial_port=self.serial)
        self.chassis = MecanumChassis(self.drive)

    def test_forward_delegates_to_four_wheel_drive(self) -> None:
        self.chassis.forward(rpm=30)
        self.assertEqual(
            self.serial.writes,
            [
                bytes.fromhex("01 F6 00 00 1E 0A 00 6B"),
                bytes.fromhex("02 F6 00 00 1E 0A 00 6B"),
                bytes.fromhex("03 F6 00 00 1E 0A 00 6B"),
                bytes.fromhex("04 F6 00 00 1E 0A 00 6B"),
            ],
        )

    def test_relative_position_moves_all_wheels_in_logical_order(self) -> None:
        self.chassis.move_wheels_relative(fl=90, fr=-90, rl=180, rr=-180, rpm=120)
        self.assertEqual([frame[0] for frame in self.serial.writes[:4]], [1, 2, 3, 4])
        self.assertEqual([frame[1] for frame in self.serial.writes[:4]], [0xFD] * 4)
        self.assertEqual(self.serial.writes[4], bytes.fromhex("00 FF 66 6B"))

    def test_facade_exposes_single_wheel_access_for_service_operations(self) -> None:
        self.assertIs(self.chassis.wheel(WheelPosition.FL), self.drive.get_wheel(WheelPosition.FL))


if __name__ == "__main__":
    unittest.main()

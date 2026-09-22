"""Verify command frames without requiring a physical serial device."""

from __future__ import annotations

import unittest

from motor_drivers.bus_servo import BusServo
from motor_drivers.emm42 import Direction, Emm42Driver, HomeMode


class FakeSerial:
    def __init__(self, replies: list[bytes] | None = None) -> None:
        self._receive_buffer = bytearray(b"".join(replies or []))
        self.writes: list[bytes] = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return data

    def close(self) -> None:
        self.is_open = False


class BusServoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.servo = BusServo("/dev/ttyUSB0", servo_id=1, serial_port=self.serial)

    def test_initialize_keeps_injected_serial_port_open(self) -> None:
        self.assertIs(self.servo.initialize(), self.servo)
        self.assertTrue(self.serial.is_open)

    def test_set_position_encodes_little_endian_position_and_duration(self) -> None:
        self.servo.set_position(position=500, duration_ms=1000)
        self.assertEqual(self.serial.writes, [bytes.fromhex("55 55 01 07 01 F4 01 E8 03 16")])

    def test_set_speed_encodes_signed_little_endian_motor_speed(self) -> None:
        self.servo.set_speed(-250)
        self.assertEqual(self.serial.writes, [bytes.fromhex("55 55 01 07 1D 01 00 06 FF D4")])

    def test_stop_writes_stop_command(self) -> None:
        self.servo.stop()
        self.assertEqual(self.serial.writes, [bytes.fromhex("55 55 01 03 0C EF")])

    def test_set_torque_writes_load_command(self) -> None:
        self.servo.set_torque(True)
        self.assertEqual(self.serial.writes, [bytes.fromhex("55 55 01 04 1F 01 DA")])

    def test_read_position_decodes_signed_little_endian_reply(self) -> None:
        self.serial._receive_buffer.extend(bytes.fromhex("55 55 01 05 1C FE FF E0"))
        self.assertEqual(self.servo.read_position(), -2)
        self.assertEqual(self.serial.writes, [bytes.fromhex("55 55 01 03 1C DF")])

    def test_set_position_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            self.servo.set_position(position=1001, duration_ms=1000)


class Emm42DriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=self.serial)

    def test_initialize_keeps_injected_serial_port_open(self) -> None:
        self.assertIs(self.motor.initialize(), self.motor)
        self.assertTrue(self.serial.is_open)

    def test_enable_writes_enable_command(self) -> None:
        self.motor.enable(True)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_set_speed_encodes_direction_speed_and_acceleration(self) -> None:
        self.motor.set_speed(Direction.CCW, rpm=1500, acceleration=10)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 F6 01 05 DC 0A 00 6B")])

    def test_move_relative_converts_degrees_to_pulses(self) -> None:
        self.motor.move_relative(
            degrees=-360.0,
            rpm=1500,
            acceleration=0,
        )
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 01 05 DC 00 00 00 0C 80 00 00 6B")],
        )

    def test_move_absolute_encodes_absolute_mode(self) -> None:
        self.motor.move_absolute(
            degrees=360.0,
            rpm=1000,
            acceleration=8,
        )
        self.assertEqual(   
            self.serial.writes,
            [bytes.fromhex("01 FD 00 03 E8 08 00 00 0C 80 01 00 6B")],
        )

    def test_stop_writes_immediate_stop_command(self) -> None:
        self.motor.stop()
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 FE 98 00 6B")])

    def test_home_writes_requested_home_mode(self) -> None:
        self.motor.home(HomeMode.MULTI_TURN_COLLISION)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 9A 02 00 6B")])

    def test_set_speed_rejects_invalid_rpm(self) -> None:
        with self.assertRaises(ValueError):
            self.motor.set_speed(Direction.CW, rpm=3001, acceleration=0)


if __name__ == "__main__":
    unittest.main()

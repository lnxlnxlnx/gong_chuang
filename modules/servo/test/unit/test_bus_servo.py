"""Hiwonder bus-servo command frames, verified without a physical serial device.

The one-file driver test of the old layout covered the Emm42 driver and the bus
servo together; the two halves now live next to the driver they test.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from modules.servo.drive.bus_servo import BusServo


class FakeSerial:
    """Records writes and replays queued replies."""

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

    def test_custom_baudrate_is_used_when_opening_serial(self) -> None:
        servo = BusServo("/dev/ttyUSB0", servo_id=2, baudrate=9600)
        with patch("serial.Serial") as serial_type:
            servo.initialize()
        serial_type.assert_called_once_with("/dev/ttyUSB0", baudrate=9600, timeout=0.1)

    def test_rejects_non_positive_baudrate(self) -> None:
        with self.assertRaises(ValueError):
            BusServo("/dev/ttyUSB0", servo_id=1, baudrate=0)

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

    def test_read_position_rejects_another_servos_reply(self) -> None:
        self.serial._receive_buffer.extend(bytes.fromhex("55 55 02 05 1C F4 01 E7"))
        with self.assertRaisesRegex(RuntimeError, "unexpected servo position response"):
            self.servo.read_position()

    def test_read_position_rejects_wrong_response_length(self) -> None:
        self.serial._receive_buffer.extend(bytes.fromhex("55 55 01 03 1C DF"))
        with self.assertRaisesRegex(RuntimeError, "unexpected servo position response"):
            self.servo.read_position()

    def test_set_position_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            self.servo.set_position(position=1001, duration_ms=1000)


if __name__ == "__main__":
    unittest.main()

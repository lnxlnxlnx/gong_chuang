"""Emm42 V5.0 command frames, verified without a physical serial device."""

from __future__ import annotations

import unittest

from modules.motor.drive.emm42 import Direction, Emm42Driver, HomeMode


class FakeSerial:
    """Records writes and can acknowledge Emm42 commands like the driver does."""

    def __init__(self, replies: list[bytes] | None = None, *, acknowledge_emm42: bool = False) -> None:
        self._receive_buffer = bytearray(b"".join(replies or []))
        self._acknowledge_emm42 = acknowledge_emm42
        self.writes: list[bytes] = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        if self._acknowledge_emm42 and len(frame) >= 3 and frame[-1] == 0x6B:
            self._receive_buffer.extend(bytes((frame[0], frame[1], 0x02, 0x6B)))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return data

    def close(self) -> None:
        self.is_open = False


class Emm42DriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial(acknowledge_emm42=True)
        self.motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=self.serial)

    def test_initialize_keeps_injected_serial_port_open(self) -> None:
        self.assertIs(self.motor.initialize(), self.motor)
        self.assertTrue(self.serial.is_open)

    def test_enable_writes_enable_command(self) -> None:
        self.motor.enable(True)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_enable_rejects_missing_driver_ack(self) -> None:
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=FakeSerial())

        with self.assertRaisesRegex(RuntimeError, "no ACK"):
            motor.enable(True)

    def test_enable_can_skip_ack_when_driver_response_is_none(self) -> None:
        serial = FakeSerial()
        motor = Emm42Driver(
            "/dev/ttyUSB1",
            motor_id=1,
            serial_port=serial,
            expect_ack=False,
        )

        motor.enable(True)

        self.assertEqual(serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

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

    def test_move_relative_can_be_buffered_for_synchronized_start(self) -> None:
        self.motor.move_relative(degrees=90.0, rpm=120, acceleration=5, synchronized=True)
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 00 00 78 05 00 00 03 20 00 01 6B")],
        )

    def test_trigger_sync_motion_broadcasts_start_frame(self) -> None:
        self.motor.trigger_sync_motion()
        self.assertEqual(self.serial.writes, [bytes.fromhex("00 FF 66 6B")])

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

    def test_stop_accepts_already_stopped_status(self) -> None:
        serial = FakeSerial(replies=[bytes.fromhex("01 FE E2 6B")])
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.stop()

        self.assertEqual(serial.writes, [bytes.fromhex("01 FE 98 00 6B")])

    def test_home_writes_requested_home_mode(self) -> None:
        self.motor.home(HomeMode.MULTI_TURN_COLLISION)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 9A 02 00 6B")])

    def test_set_speed_rejects_invalid_rpm(self) -> None:
        with self.assertRaises(ValueError):
            self.motor.set_speed(Direction.CW, rpm=3001, acceleration=0)


if __name__ == "__main__":
    unittest.main()

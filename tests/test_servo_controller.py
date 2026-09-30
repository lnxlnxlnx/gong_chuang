"""Check the 9600 baud multi-servo controller protocol without hardware."""

import unittest

from motor_drivers.servo_controller import ServoController


class FakeSerial:
    def __init__(self, reply: bytes = b"") -> None:
        self.reply = reply
        self.writes = []

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, count: int) -> bytes:
        data, self.reply = self.reply[:count], self.reply[count:]
        return data

    def close(self) -> None:
        pass


class ServoControllerTests(unittest.TestCase):
    def test_reads_one_servo_position(self) -> None:
        link = FakeSerial(bytes.fromhex("55 55 06 15 01 01 f4 01"))
        controller = ServoController("COM30", serial_port=link)
        self.assertEqual(controller.read_position(1), 500)
        self.assertEqual(link.writes, [bytes.fromhex("55 55 04 15 01 01")])

    def test_zero_count_reports_missing_servo(self) -> None:
        controller = ServoController("COM30", serial_port=FakeSerial(bytes.fromhex("55 55 03 15 00")))
        with self.assertRaisesRegex(RuntimeError, "舵机 ID 1 未返回位置"):
            controller.read_position(1)

    def test_move_frame_matches_old_controller(self) -> None:
        link = FakeSerial()
        controller = ServoController("COM30", serial_port=link)
        controller.set_position(1, position=500, duration_ms=400)
        self.assertEqual(link.writes, [bytes.fromhex("55 55 08 03 01 90 01 01 f4 01")])


if __name__ == "__main__":
    unittest.main()

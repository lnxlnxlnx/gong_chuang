"""STP-23L protocol tests using documented 195-byte measurement frames."""

import unittest

from modules.laser_scan.drive.stp23l import FRAME_SIZE, Stp23lReader, parse_measurement_frame


def measurement_frame(distances):
    points = bytearray()
    for distance in distances:
        points.extend(int(distance).to_bytes(2, "little", signed=True))
        points.extend(b"\x00\x00")  # noise
        points.extend(b"\x00\x00\x00\x00")  # peak
        points.extend(b"\x64")  # confidence
        points.extend(b"\x00\x00\x00\x00")  # integration
        points.extend(b"\x00\x00")  # reference ToF
    body = b"\x00\x02\x00\x00\xb8\x00" + points + b"\x00\x00\x00\x00"
    return b"\xaa" * 4 + body + bytes((sum(body) & 0xFF,))


class FakeSerial:
    def __init__(self, stream):
        self.stream = bytearray(stream)

    def read(self, size):
        chunk = bytes(self.stream[:size])
        del self.stream[:size]
        return chunk

    def reset_input_buffer(self):
        pass


class Stp23lTests(unittest.TestCase):
    def test_decodes_twelve_little_endian_signed_distances(self):
        frame = measurement_frame([300, -1, 1200] + [700] * 9)

        self.assertEqual(len(frame), FRAME_SIZE)
        self.assertEqual(parse_measurement_frame(frame)[:3], [300, -1, 1200])

    def test_rejects_bad_checksum_and_resynchronizes_to_next_frame(self):
        corrupted = bytearray(measurement_frame([200] * 12))
        corrupted[-1] ^= 0x01
        good = measurement_frame([450] * 12)
        reader = Stp23lReader(FakeSerial(b"junk" + corrupted + good))

        self.assertEqual(reader.read_distances(timeout_seconds=0.2), [450] * 12)

    def test_rejects_non_measurement_packet(self):
        frame = bytearray(measurement_frame([300] * 12))
        frame[5] = 0x10
        frame[-1] = sum(frame[4:-1]) & 0xFF

        with self.assertRaises(ValueError):
            parse_measurement_frame(bytes(frame))


if __name__ == "__main__":
    unittest.main()

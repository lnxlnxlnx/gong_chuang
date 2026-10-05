"""COIN-D6 frame decoding checks, driven by bytes built by hand."""

import unittest

from lidar import protocol
from lidar.protocol import (
    ACK_SIZE,
    Ack,
    DeviceInfo,
    PacketReader,
    PointCloudPacket,
    decode_angle,
    decode_sensor_point,
    interpolate_angle,
    parse_ack,
    parse_device_info,
    parse_point_packet,
    point_checksum,
    upload_checksum,
)
from lidar.simulate import encode_device_info, encode_point_packet, encode_sample, encode_upload


# One sample that encodes distance 1500 mm, intensity 100 and the高反 flag.
SAMPLE_DISTANCE_1500 = bytes((0x91, 0x71, 0x17))


class FakeSerial:
    """Just enough of a serial port for the packet reader."""

    def __init__(self, stream: bytes = b"") -> None:
        self.stream = bytearray(stream)

    def read(self, size: int) -> bytes:
        chunk = bytes(self.stream[:size])
        del self.stream[:size]
        return chunk

    def write(self, data: bytes) -> int:
        return len(data)

    def reset_input_buffer(self) -> None:
        self.stream.clear()


def packet_bytes(samples, start_raw=0x0001, end_raw=0x0081, control=0x14) -> bytes:
    """Assemble one point-cloud packet with an explicit checksum placeholder."""
    header = (
        protocol.POINT_SYNC
        + bytes((control, len(samples)))
        + start_raw.to_bytes(2, "little")
        + end_raw.to_bytes(2, "little")
    )
    return header + b"\x00\x00" + b"".join(samples)


class ChecksumTests(unittest.TestCase):
    def test_vendor_grouping_matches_a_hand_computed_value(self):
        # 0x55AA ^ (0x14|0x01<<8) ^ 0x0001 ^ 0x0081 ^ 0x00 ^ (0x00|0x01<<8) == 0x553E
        header = packet_bytes([bytes((0x00, 0x00, 0x01))])[:8]
        self.assertEqual(point_checksum(header, bytes((0x00, 0x00, 0x01)), "vendor"), 0x553E)

    def test_manual_grouping_matches_a_hand_computed_value(self):
        # 0x55AA ^ (0x14|0x01<<8) ^ 0x0001 ^ 0x0081 ^ (0x00|0x00<<8) ^ 0x01 == 0x543F
        header = packet_bytes([bytes((0x00, 0x00, 0x01))])[:8]
        self.assertEqual(point_checksum(header, bytes((0x00, 0x00, 0x01)), "manual"), 0x543F)

    def test_rejects_unknown_variant_and_ragged_input(self):
        header = packet_bytes([bytes(3)])[:8]
        with self.assertRaises(ValueError):
            point_checksum(header, bytes(3), "guess")
        with self.assertRaises(ValueError):
            point_checksum(header, bytes(4), "vendor")

    def test_upload_frame_checksum_matches_the_manual_example(self):
        model = bytes.fromhex("434F494E2D4434000000000000000000000001")
        frame = b"\xa5\x5a\x14\x00\xe3\x02\x01" + model

        self.assertEqual(upload_checksum(frame), 0x02E3)


class DecodeTests(unittest.TestCase):
    def test_sample_packs_distance_intensity_and_reflection_flag(self):
        point = decode_sensor_point(SAMPLE_DISTANCE_1500, angle_degrees=12.5)

        self.assertEqual(point.distance_mm, 1500)
        self.assertEqual(point.intensity, 100)
        self.assertTrue(point.high_reflection)
        self.assertEqual(point.angle_degrees, 12.5)

    def test_sample_without_the_reflection_flag(self):
        sample = bytes((0x90, 0x71, 0x17))
        self.assertFalse(decode_sensor_point(sample).high_reflection)

    def test_sample_must_be_three_bytes(self):
        with self.assertRaises(ValueError):
            decode_sensor_point(b"\x00\x00")

    def test_angle_drops_the_low_check_bit(self):
        self.assertAlmostEqual(decode_angle(0x0081), 1.0)
        self.assertAlmostEqual(decode_angle(0x0080), 1.0)
        self.assertAlmostEqual(decode_angle(0x0001), 0.0)

    def test_middle_angles_interpolate_and_wrap_forward(self):
        self.assertAlmostEqual(interpolate_angle(350.0, 10.0, 3, 0), 350.0)
        self.assertAlmostEqual(interpolate_angle(350.0, 10.0, 3, 1), 0.0)
        self.assertAlmostEqual(interpolate_angle(350.0, 10.0, 3, 2), 10.0)
        self.assertAlmostEqual(interpolate_angle(350.0, 10.0, 1, 0), 350.0)


class PointPacketTests(unittest.TestCase):
    def test_decodes_a_packet_built_by_hand(self):
        packet = packet_bytes([SAMPLE_DISTANCE_1500, SAMPLE_DISTANCE_1500])
        payload = packet[:8] + point_checksum(packet[:8], packet[10:], "vendor").to_bytes(2, "little")
        payload += packet[10:]

        decoded = parse_point_packet(payload)

        self.assertEqual(len(decoded.points), 2)
        self.assertEqual(decoded.speed_hz, 10)
        self.assertFalse(decoded.is_start)
        self.assertAlmostEqual(decoded.start_angle_degrees, 0.0)
        self.assertAlmostEqual(decoded.end_angle_degrees, 1.0)
        self.assertTrue(decoded.checksum_valid)
        self.assertEqual(decoded.checksum_variant, "vendor")
        self.assertEqual(decoded.points[0].distance_mm, 1500)

    def test_accepts_the_manual_grouping_too(self):
        packet = packet_bytes([SAMPLE_DISTANCE_1500])
        payload = packet[:8] + point_checksum(packet[:8], packet[10:], "manual").to_bytes(2, "little")
        payload += packet[10:]

        decoded = parse_point_packet(payload)

        self.assertEqual(decoded.checksum_variant, "manual")
        self.assertTrue(decoded.checksum_valid)

    def test_rejects_a_damaged_checksum_but_reports_it_when_not_verified(self):
        packet = bytearray(packet_bytes([SAMPLE_DISTANCE_1500]))
        packet[8:10] = (0x1234).to_bytes(2, "little")

        with self.assertRaises(ValueError):
            parse_point_packet(bytes(packet))
        lenient = parse_point_packet(bytes(packet), verify_checksum=False)
        self.assertFalse(lenient.checksum_valid)
        self.assertIsNone(lenient.checksum_variant)

    def test_rejects_a_wrong_length_or_header(self):
        packet = packet_bytes([SAMPLE_DISTANCE_1500])
        with self.assertRaises(ValueError):
            parse_point_packet(packet[:-1])
        with self.assertRaises(ValueError):
            parse_point_packet(b"\xaa\x00" + packet[2:])

    def test_marks_the_ring_start_packet(self):
        packet = encode_point_packet([encode_sample(500)], 0.0, 0.0, is_start=True)
        decoded = parse_point_packet(packet)

        self.assertTrue(decoded.is_start)
        self.assertEqual(len(decoded.points), 1)


class DeviceInfoTests(unittest.TestCase):
    def test_parses_the_documented_fields(self):
        payload = (
            b"COIN-D6".ljust(12, b"\x00")
            + b"\x00\x00"
            + b"\x00"
            + b"\x01"
            + b"\x00\x00\x00"
            + b"\x03"
        )

        info = parse_device_info(payload)

        self.assertEqual(info.model, "COIN-D6")
        self.assertEqual(info.zero_offset_raw, 0)
        self.assertTrue(info.clockwise)
        self.assertTrue(info.needs_angle_correction)
        self.assertEqual(info.software_revision, 3)

    def test_rejects_a_short_description(self):
        with self.assertRaises(ValueError):
            parse_device_info(bytes(19))

    def test_reads_the_upload_wrapper(self):
        reader = PacketReader(FakeSerial(encode_device_info("COIN-D6")))
        info = reader.read_packet(timeout_seconds=0.2)

        self.assertIsInstance(info, DeviceInfo)
        self.assertEqual(info.model, "COIN-D6")


class AckTests(unittest.TestCase):
    def test_parses_the_documented_start_acknowledgement(self):
        ack = parse_ack(bytes.fromhex("A55A500700000000000000A8"))

        self.assertIsInstance(ack, Ack)
        self.assertEqual(ack.status, protocol.ACK_STATUS_START)
        self.assertTrue(ack.is_start)
        self.assertTrue(ack.checksum_valid)
        self.assertEqual(len(ack.raw), ACK_SIZE)

    def test_parses_the_documented_stop_acknowledgement(self):
        ack = parse_ack(bytes.fromhex("A55A550700000000000000AD"))

        self.assertEqual(ack.status, protocol.ACK_STATUS_STOP)
        self.assertFalse(ack.is_start)
        self.assertTrue(ack.checksum_valid)

    def test_rejects_a_frame_with_the_wrong_header(self):
        with self.assertRaises(ValueError):
            parse_ack(b"\x00" * ACK_SIZE)


class PacketReaderTests(unittest.TestCase):
    def test_reads_bare_point_packets_in_order(self):
        stream = encode_point_packet([encode_sample(400)], 0.0, 0.0, is_start=True)
        stream += encode_point_packet([encode_sample(500), encode_sample(600)], 2.0, 4.0)
        reader = PacketReader(FakeSerial(stream))

        first = reader.read_packet(0.2)
        second = reader.read_packet(0.2)

        self.assertTrue(first.is_start)
        self.assertEqual([point.distance_mm for point in second.points], [500, 600])

    def test_reads_enveloped_point_packets(self):
        stream = encode_point_packet(
            [encode_sample(700)], 1.0, 1.0, envelope=True
        )
        reader = PacketReader(FakeSerial(stream))

        packet = reader.read_packet(0.2)

        self.assertIsInstance(packet, PointCloudPacket)
        self.assertEqual(packet.points[0].distance_mm, 700)

    def test_skips_noise_and_a_corrupted_packet(self):
        good = encode_point_packet([encode_sample(800)], 0.0, 0.0)
        damaged = bytearray(encode_point_packet([encode_sample(900)], 0.0, 0.0))
        damaged[10] ^= 0xFF
        reader = PacketReader(FakeSerial(b"\xfe\xff" + bytes(damaged) + good))

        packet = reader.read_packet(0.2)

        self.assertEqual(packet.points[0].distance_mm, 800)
        self.assertGreater(reader.skipped_bytes + reader.rejected_packets, 0)

    def test_lenient_mode_keeps_a_packet_with_a_bad_checksum(self):
        damaged = bytearray(encode_point_packet([encode_sample(900)], 0.0, 0.0))
        damaged[10] ^= 0xFF
        reader = PacketReader(FakeSerial(bytes(damaged)), strict_checksum=False)

        packet = reader.read_packet(0.2)

        self.assertFalse(packet.checksum_valid)
        self.assertEqual(packet.points[0].distance_mm, 900 & 0x3FFF)

    def test_reports_a_timeout_on_a_silent_port(self):
        reader = PacketReader(FakeSerial(b""))

        with self.assertRaises(TimeoutError):
            reader.read_packet(0.05)

    def test_resynchronises_after_a_truncated_packet(self):
        truncated = encode_point_packet([encode_sample(300)] * 4, 0.0, 3.0)[:-5]
        good = encode_point_packet([encode_sample(1234)], 5.0, 5.0)
        reader = PacketReader(FakeSerial(truncated + good))

        packet = reader.read_packet(0.2)

        self.assertEqual(packet.points[0].distance_mm, 1234)

    def test_feed_accepts_bytes_read_elsewhere(self):
        reader = PacketReader(FakeSerial(b""))
        reader.feed(encode_point_packet([encode_sample(321)], 0.0, 0.0))

        packet = reader.next_packet()

        self.assertEqual(packet.points[0].distance_mm, 321)

    def test_reset_drops_buffered_bytes(self):
        reader = PacketReader(FakeSerial(encode_point_packet([encode_sample(500)], 0.0, 0.0)))
        reader.read_packet(0.2)
        reader.reset()

        with self.assertRaises(TimeoutError):
            reader.read_packet(0.02)


if __name__ == "__main__":
    unittest.main()

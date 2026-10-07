"""COIN-D6 session checks against the built-in simulated sensor."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.lidar import CoinD6, ScanConfig, correct_angle
from modules.lidar.drive.coin_d6 import LidarAckError, LidarError, LidarTimeoutError
from modules.lidar.drive.protocol import START_COMMAND, STOP_COMMAND
from modules.lidar.app.simulate import SimulatedLidar, encode_ack, encode_point_packet, encode_sample
from modules.lidar.test.unit.test_lidar_protocol import FakeSerial


def scan_config(**overrides) -> ScanConfig:
    values = dict(
        revolutions=1,
        angle_min_degrees=-180.0,
        angle_max_degrees=180.0,
        min_distance_mm=30,
        max_distance_mm=12000,
        clockwise=True,
        angle_offset_degrees=0.0,
        max_points=40000,
        timeout_seconds=5.0,
    )
    values.update(overrides)
    return ScanConfig(**values)


def simulated(**kwargs) -> SimulatedLidar:
    return SimulatedLidar(**kwargs)


class AckSerial(FakeSerial):
    """A fake port that answers each documented command with an acknowledgement."""

    def __init__(self, *, start_status=0x50, stop_status=0x55, payload=b"") -> None:
        super().__init__(b"")
        self.written = []
        self.payload = payload
        self.statuses = {START_COMMAND: start_status, STOP_COMMAND: stop_status}

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        status = self.statuses.get(bytes(data))
        if status is not None:
            self.stream.extend(encode_ack(status))
            self.stream.extend(self.payload)
        return len(data)


class CommandTests(unittest.TestCase):
    def test_start_sends_the_documented_command_and_checks_the_ack(self):
        fake = AckSerial()
        lidar = CoinD6("COM1", serial_port=fake).initialize()

        ack = lidar.start()

        self.assertEqual(fake.written, [START_COMMAND])
        self.assertTrue(ack.is_start)

    def test_stop_sends_the_documented_command(self):
        fake = AckSerial()
        lidar = CoinD6("COM1", serial_port=fake).initialize()

        ack = lidar.stop()

        self.assertEqual(fake.written, [STOP_COMMAND])
        self.assertFalse(ack.is_start)

    def test_a_rejected_command_raises(self):
        lidar = CoinD6("COM1", serial_port=AckSerial(start_status=0x55)).initialize()

        with self.assertRaisesRegex(LidarAckError, "0x55"):
            lidar.start()

    def test_a_silent_sensor_raises_unless_acks_are_optional(self):
        strict = CoinD6("COM1", serial_port=FakeSerial(b""), ack_timeout=0.05).initialize()
        with self.assertRaises(LidarAckError):
            strict.start()

        lenient = CoinD6(
            "COM1", serial_port=FakeSerial(b""), ack_timeout=0.05, require_ack=False
        ).initialize()
        self.assertIsNone(lenient.start())

    def test_writing_before_initialize_is_rejected(self):
        lidar = CoinD6("COM1", serial_port=FakeSerial(b""))

        with self.assertRaises(LidarError):
            lidar.start()
        with self.assertRaises(LidarError):
            lidar.collect_scan(scan_config())

    def test_a_port_that_cannot_be_opened_raises_a_lidar_error(self):
        class BrokenSerial:
            def __init__(self, *args, **kwargs):
                raise OSError("port is busy")

        with patch.dict(sys.modules, {"serial": SimpleNamespace(Serial=BrokenSerial)}):
            with self.assertRaisesRegex(LidarError, "无法打开串口 COM99"):
                CoinD6("COM99").initialize()

    def test_a_silent_port_reports_lidar_timeouts(self):
        lidar = CoinD6("COM1", serial_port=FakeSerial(b"")).initialize()

        with self.assertRaises(LidarTimeoutError):
            lidar.read_packet(0.05)
        with self.assertRaises(LidarTimeoutError):
            lidar.read_device_info(0.05)

    def test_close_forgets_the_serial_port(self):
        lidar = CoinD6("COM1", serial_port=simulated())
        lidar.initialize()
        lidar.close()

        with self.assertRaises(LidarError):
            lidar.read_packet(0.01)


class ScanTests(unittest.TestCase):
    def test_scan_collects_one_full_revolution_and_stops_the_sensor(self):
        device = simulated(points_per_revolution=144)
        lidar = CoinD6("COM1", serial_port=device)

        cloud = lidar.scan(scan_config())

        self.assertEqual(len(cloud), 144)
        self.assertEqual(cloud.revolutions, 1)
        self.assertEqual(device.commands, [START_COMMAND, STOP_COMMAND])

    def test_the_device_description_is_read_before_scanning(self):
        device = simulated(points_per_revolution=48)
        lidar = CoinD6("COM1", serial_port=device).initialize()

        info = lidar.read_device_info(0.5)
        cloud = lidar.scan(scan_config())

        self.assertEqual(info.model, "COIN-D6")
        self.assertTrue(info.clockwise)
        self.assertEqual(len(cloud), 48)

    def test_scan_collects_the_configured_number_of_revolutions(self):
        device = simulated(points_per_revolution=48, points_per_packet=12)
        lidar = CoinD6("COM1", serial_port=device)

        cloud = lidar.scan(scan_config(revolutions=3))

        self.assertEqual(len(cloud), 144)
        self.assertEqual(cloud.revolutions, 3)

    def test_revolutions_are_reported_as_they_complete(self):
        device = simulated(points_per_revolution=48)
        lidar = CoinD6("COM1", serial_port=device)
        seen = []

        lidar.scan(scan_config(revolutions=2), on_revolution=seen.append)

        self.assertEqual(seen, [1, 2])

    def test_angle_window_keeps_only_the_requested_sector(self):
        device = simulated(points_per_revolution=72, points_per_packet=12)
        lidar = CoinD6("COM1", serial_port=device)

        cloud = lidar.scan(scan_config(angle_min_degrees=0.0, angle_max_degrees=90.0))

        self.assertTrue(cloud.points)
        self.assertTrue(all(0.0 <= point.angle_degrees <= 90.0 for point in cloud))

    def test_distance_window_drops_near_points(self):
        device = simulated(points_per_revolution=72, points_per_packet=12)
        lidar = CoinD6("COM1", serial_port=device)

        cloud = lidar.scan(scan_config(min_distance_mm=2000))

        self.assertTrue(cloud.points)
        self.assertTrue(all(point.distance_mm >= 2000 for point in cloud))
        self.assertGreater(cloud.nearest_mm, 1900)

    def test_max_points_caps_the_cloud(self):
        device = simulated(points_per_revolution=144)
        lidar = CoinD6("COM1", serial_port=device)

        cloud = lidar.scan(scan_config(revolutions=5, max_points=10))

        self.assertEqual(len(cloud), 10)

    def test_enveloped_streams_scan_too(self):
        device = simulated(envelope=True, points_per_revolution=48)
        lidar = CoinD6("COM1", serial_port=device)

        cloud = lidar.scan(scan_config())

        self.assertEqual(len(cloud), 48)

    def test_lenient_checksum_accepts_a_damaged_stream(self):
        packet = bytearray(encode_point_packet([encode_sample(700)], 0.0, 0.0))
        packet[12] ^= 0x01  # the high distance byte: 700 becomes 764, checksum no longer matches
        lidar = CoinD6("COM1", serial_port=AckSerial(payload=bytes(packet)),
                       strict_checksum=False, ack_timeout=0.1)

        cloud = lidar.scan(scan_config(revolutions=1, timeout_seconds=0.3))

        self.assertEqual(len(cloud), 1)
        self.assertEqual(cloud.points[0].distance_mm, 764)

    def test_strict_checksum_drops_the_same_damaged_stream(self):
        packet = bytearray(encode_point_packet([encode_sample(700)], 0.0, 0.0))
        packet[12] ^= 0x01
        lidar = CoinD6("COM1", serial_port=AckSerial(payload=bytes(packet)), ack_timeout=0.1)

        with self.assertRaises(LidarTimeoutError):
            lidar.scan(scan_config(revolutions=1, timeout_seconds=0.2))

    def test_a_silent_stream_raises_a_timeout_with_diagnostics(self):
        lidar = CoinD6("COM1", serial_port=FakeSerial(b""), ack_timeout=0.05, require_ack=False)

        with self.assertRaisesRegex(LidarTimeoutError, "点云"):
            lidar.scan(scan_config(timeout_seconds=0.1))

    def test_a_scan_with_no_usable_points_raises(self):
        device = simulated(points_per_revolution=48)
        lidar = CoinD6("COM1", serial_port=device)

        with self.assertRaises(LidarTimeoutError):
            lidar.scan(scan_config(
                min_distance_mm=15000, max_distance_mm=16000, timeout_seconds=0.2
            ))

    def test_a_session_can_prepare_itself_and_then_collect(self):
        device = simulated(points_per_revolution=48)
        lidar = CoinD6("COM1", serial_port=device).initialize()
        lidar.start()

        cloud = lidar.collect_scan(scan_config())

        self.assertEqual(len(cloud), 48)


class AngleCorrectionTests(unittest.TestCase):
    def test_clockwise_sweeps_are_mirrored(self):
        self.assertAlmostEqual(correct_angle(90.0, scan_config()), -90.0)

    def test_counter_clockwise_sweeps_are_kept(self):
        self.assertAlmostEqual(correct_angle(90.0, scan_config(clockwise=False)), 90.0)

    def test_offsets_are_applied_and_wrapped(self):
        config = scan_config(angle_offset_degrees=180.0)

        self.assertAlmostEqual(correct_angle(90.0, config), 90.0)
        self.assertAlmostEqual(correct_angle(0.0, config), -180.0)

    def test_the_result_stays_inside_minus_180_to_180(self):
        for angle in range(0, 360, 7):
            corrected = correct_angle(float(angle), scan_config())
            self.assertGreaterEqual(corrected, -180.0)
            self.assertLess(corrected, 180.0)


if __name__ == "__main__":
    unittest.main()

"""Bounded sector scan tests without real serial ports or motor movement."""

import unittest
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from laser_scanner import ScanConfig, load_scan_config, save_scan, scan_sector
from motor_drivers.emm42 import Emm42Driver


class FakeMotor:
    def __init__(self):
        self.commands = []

    def enable(self, enabled):
        self.commands.append(("enable", enabled))

    def move_relative(self, degrees, rpm, acceleration):
        self.commands.append(("move", degrees, rpm, acceleration))

    def stop(self):
        self.commands.append(("stop",))


class FakeLidar:
    def __init__(self, fail_after=None):
        self.samples = 0
        self.fail_after = fail_after

    def sample_distance(self, minimum_mm, maximum_mm, timeout_seconds):
        if self.samples == self.fail_after:
            raise TimeoutError("no fresh lidar frame")
        self.samples += 1
        return 500 if self.samples == 1 else 1500


def config(step_pulses=40):
    return ScanConfig(
        motor_id=8,
        motor_port="COM28",
        motor_expect_ack=False,
        lidar_port="COM29",
        pulses_per_revolution=3200,
        rpm=10,
        acceleration=5,
        max_ccw_pulses=800,
        step_pulses=step_pulses,
        sweep_degrees=90,
        settle_seconds=0.5,
        sample_timeout_seconds=2,
        min_distance_mm=70,
        max_distance_mm=7500,
        obstacle_distance_mm=1000,
    )


class LaserScannerTests(unittest.TestCase):
    def test_project_config_uses_confirmed_ports_and_bounded_sweep(self):
        settings = load_scan_config()

        self.assertEqual(settings.motor_id, 8)
        self.assertEqual(settings.motor_port, "COM28")
        self.assertEqual(settings.lidar_port, "COM26")
        self.assertEqual(settings.max_ccw_pulses, 800)
        self.assertEqual(settings.step_pulses, 40)
        self.assertEqual(settings.sweep_degrees, 90)

    def test_scans_to_endpoint_then_returns_with_position_commands(self):
        motor = FakeMotor()
        sleeps = []

        samples = scan_sector(motor, FakeLidar(), config(), sleep=sleeps.append)

        moves = [command for command in motor.commands if command[0] == "move"]
        self.assertEqual(len(samples), 21)
        self.assertEqual(samples[0].angle_degrees, 0)
        self.assertEqual(samples[-1].angle_degrees, 90)
        self.assertTrue(samples[0].obstacle)
        self.assertFalse(samples[1].obstacle)
        self.assertEqual([move[1] for move in moves], [-4.5] * 20 + [90])
        self.assertEqual(motor.commands[-2:], [("stop",), ("enable", False)])
        self.assertEqual(len(sleeps), 21)

    def test_final_step_uses_only_remaining_twenty_pulses(self):
        motor = FakeMotor()

        samples = scan_sector(motor, FakeLidar(), config(step_pulses=30), sleep=lambda _: None)

        moves = [command for command in motor.commands if command[0] == "move"]
        self.assertEqual(samples[-1].pulses, 800)
        self.assertEqual(len(moves), 28)
        self.assertEqual(moves[-2][1], -20 * 360 / 3200)
        self.assertEqual(moves[-1][1], 800 * 360 / 3200)

    def test_sensor_failure_stops_without_unverified_return(self):
        motor = FakeMotor()

        with self.assertRaises(TimeoutError):
            scan_sector(motor, FakeLidar(fail_after=1), config(), sleep=lambda _: None)

        moves = [command for command in motor.commands if command[0] == "move"]
        self.assertEqual(len(moves), 1)
        self.assertLess(moves[0][1], 0)
        self.assertEqual(motor.commands[-2:], [("stop",), ("enable", False)])

    def test_enable_error_is_preserved_when_cleanup_also_fails(self):
        class NoAckMotor(FakeMotor):
            def enable(self, enabled):
                super().enable(enabled)
                raise RuntimeError("enable no ACK" if enabled else "disable no ACK")

            def stop(self):
                super().stop()
                raise RuntimeError("stop no ACK")

        motor = NoAckMotor()

        with self.assertRaisesRegex(RuntimeError, "enable no ACK"):
            scan_sector(motor, FakeLidar(), config(), sleep=lambda _: None)

        self.assertEqual(motor.commands, [("enable", True), ("stop",), ("enable", False)])

    def test_emm42_commands_are_relative_position_mode_only(self):
        class MotorSerial:
            def __init__(self):
                self.writes = []

            def write(self, data):
                self.writes.append(bytes(data))
                return len(data)

            def flush(self):
                pass

        serial_port = MotorSerial()
        motor = Emm42Driver("COM28", 8, expect_ack=False, serial_port=serial_port)

        scan_sector(motor, FakeLidar(), config(), sleep=lambda _: None)

        positions = [frame for frame in serial_port.writes if frame[1] == 0xFD]
        self.assertEqual(len(positions), 21)
        self.assertTrue(all(frame[2] == 1 and frame[10:12] == b"\x00\x00"
                            for frame in positions[:-1]))
        self.assertEqual(positions[-1][2], 0)
        self.assertEqual(int.from_bytes(positions[-1][6:10], "big"), 800)
        self.assertNotIn(0xF6, [frame[1] for frame in serial_port.writes])

    def test_exports_csv_and_parseable_radar_svg(self):
        points = scan_sector(FakeMotor(), FakeLidar(), config(), sleep=lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "sweep"

            csv_path, svg_path = save_scan(points, config(), prefix)

            self.assertIn("0,0.0,500,True", csv_path.read_text(encoding="utf-8"))
            self.assertEqual(ET.parse(svg_path).getroot().tag, "{http://www.w3.org/2000/svg}svg")


if __name__ == "__main__":
    unittest.main()

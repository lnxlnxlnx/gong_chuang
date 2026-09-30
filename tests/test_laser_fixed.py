"""Fixed-head laser range display checks without serial hardware."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_laser_scanner import config
from tests.test_stp23l import measurement_frame


class FakeReader:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read_distances(self, timeout_seconds):
        try:
            return next(self.frames)
        except StopIteration:
            raise KeyboardInterrupt


class FixedLaserTests(unittest.TestCase):
    def test_reports_distance_obstacle_and_invalid_frame(self):
        from tests.manual.laser_fixed import display_distances

        reader = FakeReader(([500] * 11 + [10000], [1500] * 12, [-1] * 12))
        lines = []

        with self.assertRaises(KeyboardInterrupt):
            display_distances(reader, config(), output=lines.append)

        self.assertEqual(lines, [
            "距离: 500 mm | 有效点: 11/12 | 障碍物: 是",
            "距离: 1500 mm | 有效点: 12/12 | 障碍物: 否",
            "距离: 无有效数据 | 有效点: 0/12 | 障碍物: 未知",
        ])

    def test_main_opens_only_lidar_port_and_closes_on_interrupt(self):
        from tests.manual.laser_fixed import main

        class FakeSerial:
            def __init__(self, port, baudrate, timeout):
                opened.append((port, baudrate, timeout))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                closed.append(True)

            def read(self, size):
                if frames:
                    return frames.pop(0)
                raise KeyboardInterrupt

        opened = []
        closed = []
        frames = [measurement_frame([700] * 12)]
        with patch.dict(sys.modules, {"serial": SimpleNamespace(Serial=FakeSerial)}):
            with patch.object(sys, "argv", [str(Path("laser_fixed.py"))]):
                main()

        self.assertEqual(opened, [("COM26", 230400, 0.1)])
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()

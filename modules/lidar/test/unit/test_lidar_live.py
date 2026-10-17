"""Live acquisition model and operator-window checks, all headless."""

import time
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from modules.lidar import CoinD6, load_lidar_config
from modules.lidar.app.live import LiveCloud
from modules.lidar.drive.protocol import STOP_COMMAND
from modules.lidar.app.simulate import SimulatedLidar
from modules.lidar.test.manual import coin_d6_live
from tests.scratch import scratch_directory
from modules.lidar.test.unit.test_lidar_protocol import FakeSerial


def wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll until the background reader has produced something."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def live_reader(**simulate_kwargs) -> LiveCloud:
    config = load_lidar_config()
    device = SimulatedLidar(points_per_revolution=72, points_per_packet=12, **simulate_kwargs)
    lidar = CoinD6(config.device.port, serial_port=device)
    reader = LiveCloud(lidar, config.scan, revolutions_kept=2)
    reader.start()
    wait_for(lambda: reader.stats.points > 0)
    return reader


class LiveCloudTests(unittest.TestCase):
    def tearDown(self):
        reader = getattr(self, "reader", None)
        if reader is not None:
            reader.stop()

    def test_collects_points_in_the_background(self):
        self.reader = live_reader()

        stats = self.reader.stats
        cloud = self.reader.snapshot()

        self.assertGreater(stats.points, 0)
        self.assertTrue(stats.running)
        self.assertTrue(stats.healthy)
        self.assertFalse(cloud.empty)
        self.assertEqual(len(cloud), stats.points)

    def test_distance_window_filters_points_already_buffered(self):
        self.reader = live_reader()
        buffered = self.reader.stats.points

        self.reader.set_filters(min_distance_mm=100000)
        empty = self.reader.snapshot()
        self.reader.set_filters(min_distance_mm=1, max_distance_mm=16383)
        restored = self.reader.snapshot()

        self.assertTrue(empty.empty)
        self.assertGreaterEqual(len(restored), buffered)

    def test_angle_window_filters_points_already_buffered(self):
        self.reader = live_reader()

        self.reader.set_filters(angle_min_degrees=0.0, angle_max_degrees=30.0)
        cloud = self.reader.snapshot()

        self.assertTrue(cloud.points)
        self.assertTrue(all(0.0 <= point.angle_degrees <= 30.0 for point in cloud))

    def test_swapped_windows_are_put_back_in_order(self):
        self.reader = live_reader()

        self.reader.set_filters(min_distance_mm=3000, max_distance_mm=1000)
        self.reader.set_filters(angle_min_degrees=90.0, angle_max_degrees=-90.0)

        self.assertEqual(self.reader.scan.min_distance_mm, 1000)
        self.assertEqual(self.reader.scan.max_distance_mm, 3000)
        self.assertEqual(self.reader.scan.angle_min_degrees, -90.0)
        self.assertEqual(self.reader.scan.angle_max_degrees, 90.0)

    def test_kept_revolutions_bound_the_buffer(self):
        self.reader = live_reader()

        self.reader.set_revolutions_kept(1)
        wait_for(lambda: self.reader.stats.revolutions == 1)
        one = self.reader.stats.points
        self.reader.set_revolutions_kept(3)
        wait_for(lambda: self.reader.stats.revolutions == 3)
        three = self.reader.stats.points

        self.assertLessEqual(self.reader.stats.revolutions, 3)
        self.assertGreater(three, one)

    def test_clear_drops_every_point(self):
        self.reader = live_reader()

        self.reader.clear()

        self.assertEqual(self.reader.stats.points, 0)
        self.assertTrue(self.reader.snapshot().empty)

    def test_stop_stops_the_sensor_and_releases_the_port(self):
        self.reader = live_reader()
        device = self.reader.lidar._serial

        self.reader.stop()

        self.assertFalse(self.reader.running)
        self.assertIn(STOP_COMMAND, device.commands)
        self.assertFalse(device.is_open)

    def test_a_failing_port_is_reported_instead_of_crashing(self):
        class FailingSerial(FakeSerial):
            def read(self, size):
                raise OSError("device disconnected")

        config = load_lidar_config()
        lidar = CoinD6(config.device.port, serial_port=FailingSerial(b""), ack_timeout=0.05,
                       require_ack=False)
        self.reader = LiveCloud(lidar, config.scan)
        self.reader.start()

        self.assertTrue(wait_for(lambda: self.reader.stats.error is not None))
        self.assertIn("device disconnected", self.reader.stats.error)
        self.assertFalse(self.reader.running)

    def test_revolutions_kept_must_be_positive(self):
        config = load_lidar_config()
        lidar = CoinD6(config.device.port, serial_port=SimulatedLidar())
        reader = LiveCloud(lidar, config.scan)

        with self.assertRaises(ValueError):
            reader.set_revolutions_kept(0)
        with self.assertRaises(ValueError):
            LiveCloud(lidar, config.scan, revolutions_kept=0)


class LiveWindowTests(unittest.TestCase):
    def setUp(self):
        self.config = load_lidar_config()
        self.device = SimulatedLidar(points_per_revolution=72, points_per_packet=12)
        self.reader = LiveCloud(
            CoinD6(self.config.device.port, serial_port=self.device), self.config.scan
        )
        self.directory = None
        self.app = None

    def build(self):
        self.directory = scratch_directory()
        root = self.directory.__enter__()
        self.app = coin_d6_live.LiveCloudApp(
            self.reader, self.config, output=lambda line: None, output_root=root,
        )
        self.reader.start()
        wait_for(lambda: self.reader.stats.points > 0)
        return self.app

    def tearDown(self):
        if self.app is not None:
            self.app.close()
            plt.close(self.app.figure)
        self.reader.stop()
        if self.directory is not None:
            self.directory.__exit__(None, None, None)

    def test_menu_entry_is_declared(self):
        self.assertEqual(coin_d6_live.MENU_ENTRY, (17, "D6 点云实时显示"))

    def test_window_draws_points_and_a_status_line(self):
        app = self.build()

        app.step()

        offsets = app.scatter.get_offsets()
        self.assertGreater(len(offsets), 0)
        self.assertIn("点 ", app.status.get_text())
        self.assertIn("显示窗口", app.status.get_text())

    def test_distance_slider_refilters_immediately(self):
        app = self.build()

        app.distance_max.set_val(1500.0)
        app.step()

        offsets = app.scatter.get_offsets()
        radii = [max(abs(x), abs(y)) for x, y in offsets]
        self.assertTrue(radii)
        self.assertLessEqual(max(radii), 1500.0)
        self.assertIn("1500", app.status.get_text())

    def test_angle_slider_refilters_immediately(self):
        app = self.build()

        app.angle_min.set_val(0.0)
        app.angle_max.set_val(90.0)
        app.step()

        self.assertIn("0~90", app.status.get_text())
        self.assertGreater(len(app.scatter.get_offsets()), 0)

    def test_typed_value_is_clamped_into_range(self):
        app = self.build()

        app._submit(app.distance_min, "999999")
        self.assertEqual(app.distance_min.val, app.distance_min.valmax)
        self.assertEqual(app._boxes[app.distance_min].text, f"{app.distance_min.valmax:g}")

        before = app.distance_min.val
        app._submit(app.distance_min, "not a number")
        self.assertEqual(app.distance_min.val, before)

    def test_pause_freezes_the_display_but_not_the_reader(self):
        app = self.build()
        app.step()
        frozen = len(app.scatter.get_offsets())
        app._on_pause(None)
        before = self.reader.stats.points

        app.step()

        self.assertEqual(len(app.scatter.get_offsets()), frozen)
        self.assertIn("[已暂停]", app.status.get_text())
        self.assertGreaterEqual(self.reader.stats.points, before)

    def test_save_snapshot_writes_csv_and_png(self):
        app = self.build()
        app.step()

        png_path = app.save_snapshot()

        self.assertTrue(png_path.exists())
        self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue(png_path.with_suffix(".csv").exists())

    def test_save_without_data_reports_instead_of_writing(self):
        app = self.build()
        self.reader.clear()
        app.step()

        self.assertIsNone(app.save_snapshot())

    def test_close_stops_the_reader(self):
        app = self.build()

        app.close()

        self.assertFalse(self.reader.running)
        self.assertIn(STOP_COMMAND, self.device.commands)

    def test_closing_the_window_stops_the_reader(self):
        from matplotlib.backend_bases import CloseEvent

        app = self.build()

        app.figure.canvas.callbacks.process(
            "close_event", CloseEvent("close_event", app.figure.canvas)
        )

        self.assertFalse(self.reader.running)
        self.assertIn(STOP_COMMAND, self.device.commands)

    def test_close_is_idempotent(self):
        app = self.build()

        app.close()
        app.close()

        self.assertFalse(self.reader.running)


if __name__ == "__main__":
    unittest.main()

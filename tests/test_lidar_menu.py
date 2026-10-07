"""Manual COIN-D6 menu entry: discovery, capture and saved artifacts."""

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from modules.lidar import CoinD6, load_lidar_config
from modules.lidar.app.simulate import SimulatedLidar
from tests.manual import coin_d6
from tests.scratch import scratch_directory
from tests.test_lidar_protocol import FakeSerial


class MenuEntryTests(unittest.TestCase):
    def test_the_menu_entry_is_declared_for_the_test_launcher(self):
        self.assertEqual(coin_d6.MENU_ENTRY, (16, "D6 激光雷达点云"))

    def test_the_output_prefix_is_timestamped_under_output(self):
        prefix = coin_d6.scan_output_prefix(datetime(2026, 10, 2, 15, 4, 5, 123456))

        self.assertEqual(prefix.parent, coin_d6.OUTPUT_ROOT)
        self.assertEqual(prefix.name, "cloud-20261002-150405-123456")

    def test_simulate_wires_the_fake_sensor_into_the_driver(self):
        config = load_lidar_config()
        lidar = coin_d6.create_lidar(config, simulate=True)

        self.assertIsInstance(lidar, CoinD6)
        self.assertIsInstance(lidar._serial, SimulatedLidar)

    def test_real_mode_uses_the_configured_link_settings(self):
        config = load_lidar_config()
        lidar = coin_d6.create_lidar(config, simulate=False)

        self.assertEqual(lidar.port, config.device.port)
        self.assertEqual(lidar.baudrate, config.device.baudrate)
        self.assertEqual(lidar.require_ack, config.device.require_ack)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.config = load_lidar_config()
        self.lidar = CoinD6(
            self.config.device.port,
            serial_port=SimulatedLidar(points_per_revolution=72, points_per_packet=12),
        )

    def test_capture_reports_the_device_and_keeps_it_on_the_cloud(self):
        lines = []

        cloud = coin_d6.capture(self.lidar, self.config, output=lines.append)

        self.assertEqual(cloud.device.model, "COIN-D6")
        self.assertEqual(len(cloud), 72 * self.config.scan.revolutions)
        self.assertTrue(any("设备信息" in line for line in lines))
        self.assertTrue(any("已采集完整" in line for line in lines))

    def test_capture_survives_a_missing_device_info(self):
        lidar = CoinD6(
            self.config.device.port,
            serial_port=SimulatedLidar(points_per_revolution=48, device_info=False),
        )
        lines = []

        cloud = coin_d6.capture(lidar, self.config, output=lines.append)

        self.assertIsNone(cloud.device)
        self.assertEqual(len(cloud), 48 * self.config.scan.revolutions)
        self.assertTrue(any("未读到" in line for line in lines))

    def test_capture_closes_the_serial_port(self):
        device = self.lidar._serial

        coin_d6.capture(self.lidar, self.config, output=lambda line: None)

        self.assertFalse(device.is_open)
        self.assertIsNone(self.lidar._serial)

    def test_report_warns_about_a_short_capture(self):
        cloud = coin_d6.capture(self.lidar, self.config, output=lambda line: None)
        short = cloud.__class__(points=cloud.points, revolutions=0, device=cloud.device)
        lines = []

        coin_d6.report(short, self.config, output=lines.append)

        self.assertTrue(any("注意" in line for line in lines))
        self.assertTrue(any(f"{len(cloud)} 个点" in line for line in lines))


class SaveTests(unittest.TestCase):
    def setUp(self):
        self.config = load_lidar_config()

    def test_save_writes_csv_and_png_without_a_window(self):
        cloud = coin_d6.capture(
            CoinD6(self.config.device.port, serial_port=SimulatedLidar(points_per_revolution=48)),
            self.config,
            output=lambda line: None,
        )
        with scratch_directory() as directory:
            csv_path, png_path = coin_d6.save_cloud(
                cloud, self.config, Path(directory) / "cloud",
                output=lambda line: None, show_window=False,
            )

            self.assertTrue(csv_path.exists())
            self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_main_simulate_writes_both_artifacts(self):
        with scratch_directory() as directory:
            prefix = Path(directory) / "cloud"
            argv = ["coin_d6.py", "--simulate", "--no-window", "--output-prefix", str(prefix)]
            with patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()) as printed:
                    coin_d6.main()

            self.assertTrue(prefix.with_suffix(".csv").exists())
            self.assertTrue(prefix.with_suffix(".png").exists())
        self.assertIn("点云图片", printed.getvalue())

    def test_main_refuses_to_overwrite_existing_output(self):
        with scratch_directory() as directory:
            prefix = Path(directory) / "cloud"
            prefix.with_suffix(".csv").write_text("taken", encoding="utf-8")
            argv = ["coin_d6.py", "--simulate", "--no-window", "--output-prefix", str(prefix)]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        coin_d6.main()


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.config = load_lidar_config()

    def test_diagnose_reports_ack_info_and_points(self):
        lines = []

        counts = coin_d6.diagnose(
            SimulatedLidar(points_per_revolution=48),
            self.config,
            output=lines.append,
            seconds=0.2,
        )

        self.assertEqual(counts["ack"], 1)
        self.assertEqual(counts["info"], 1)
        self.assertGreater(counts["point"], 0)
        self.assertTrue(any("已发送启动指令" in line for line in lines))
        self.assertTrue(any("收到第一个点云包" in line for line in lines))
        self.assertTrue(any("统计：" in line for line in lines))

    def test_diagnose_explains_a_silent_port(self):
        lines = []

        counts = coin_d6.diagnose(FakeSerial(b""), self.config, output=lines.append, seconds=0.05)

        self.assertEqual(counts["point"], 0)
        self.assertEqual(counts["ack"], 0)
        self.assertTrue(any("波特率" in line for line in lines))


if __name__ == "__main__":
    unittest.main()

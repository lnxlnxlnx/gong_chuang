"""Point-cloud figure checks; rendering only, never a window."""

import unittest

import matplotlib

matplotlib.use("Agg")

from lidar.cloud import CloudPoint, PointCloud
from lidar.config import CloudView
from lidar.visualize import build_figure, render_point_cloud, save_figure
from tests.scratch import scratch_directory


def view(**overrides) -> CloudView:
    values = dict(range_mm=4000.0, color_by="distance", point_size=6.0,
                  colormap="turbo", show_window=False)
    values.update(overrides)
    return CloudView(**values)


def sample_cloud() -> PointCloud:
    return PointCloud(points=(
        CloudPoint(0.0, 1000, 120, False),
        CloudPoint(45.0, 1500, 200, True),
        CloudPoint(-120.0, 2500, 60, False),
    ), revolutions=1)


class FigureTests(unittest.TestCase):
    def test_figure_mentions_the_sensor_and_the_point_count(self):
        figure = build_figure(sample_cloud(), view())

        self.assertIn("COIN-D6", figure.axes[0].get_title())
        self.assertIn("3", figure.axes[0].get_title())

    def test_every_colour_mode_builds(self):
        for color_by in ("distance", "intensity", "angle"):
            figure = build_figure(sample_cloud(), view(color_by=color_by))
            self.assertIsNotNone(figure)
            matplotlib.pyplot.close(figure)

    def test_an_empty_cloud_still_builds_a_figure(self):
        figure = build_figure(PointCloud(), view())

        self.assertIsNotNone(figure)
        matplotlib.pyplot.close(figure)

    def test_save_writes_a_png_and_creates_the_directory(self):
        with scratch_directory() as directory:
            path = save_figure(build_figure(sample_cloud(), view()), directory / "nested" / "cloud.png")
            head = path.read_bytes()[:8]

        self.assertEqual(head, b"\x89PNG\r\n\x1a\n")

    def test_render_saves_and_returns_the_path(self):
        with scratch_directory() as directory:
            path = render_point_cloud(sample_cloud(), view(), directory / "cloud.png")

            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

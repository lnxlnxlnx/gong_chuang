"""Point-cloud value-object checks: geometry, statistics and CSV round trip."""

import unittest

from lidar.cloud import CloudPoint, PointCloud
from lidar.protocol import DeviceInfo
from tests.scratch import scratch_directory


def cloud(points=(CloudPoint(0.0, 1000, 120, False), CloudPoint(90.0, 2000, 90, True))) -> PointCloud:
    return PointCloud(points=tuple(points), revolutions=1, device=DeviceInfo("COIN-D6", 0, True, True, 1))


class CloudPointTests(unittest.TestCase):
    def test_forward_angle_lands_on_the_x_axis(self):
        point = CloudPoint(0.0, 1000, 10, False)

        self.assertAlmostEqual(point.x_mm, 1000.0)
        self.assertAlmostEqual(point.y_mm, 0.0)

    def test_left_angle_lands_on_the_y_axis(self):
        point = CloudPoint(90.0, 500, 10, False)

        self.assertAlmostEqual(point.x_mm, 0.0, places=6)
        self.assertAlmostEqual(point.y_mm, 500.0)

    def test_negative_angle_is_mirrored(self):
        point = CloudPoint(-90.0, 500, 10, False)

        self.assertAlmostEqual(point.y_mm, -500.0)


class PointCloudTests(unittest.TestCase):
    def test_length_iteration_and_extremes(self):
        points = cloud()

        self.assertEqual(len(points), 2)
        self.assertEqual([point.distance_mm for point in points], [1000, 2000])
        self.assertEqual(points.nearest_mm, 1000)
        self.assertEqual(points.farthest_mm, 2000)
        self.assertFalse(points.empty)
        self.assertEqual(points.device.model, "COIN-D6")

    def test_an_empty_cloud_reports_nothing(self):
        empty = PointCloud()

        self.assertTrue(empty.empty)
        self.assertEqual(len(empty), 0)
        self.assertIsNone(empty.nearest_mm)
        self.assertIsNone(empty.farthest_mm)

    def test_csv_round_trip(self):
        with scratch_directory() as directory:
            path = cloud().save_csv(directory / "cloud.csv")
            restored = PointCloud.load_csv(path)

        self.assertEqual(
            [(point.angle_degrees, point.distance_mm, point.intensity, point.high_reflection)
             for point in restored],
            [(0.0, 1000, 120, False), (90.0, 2000, 90, True)],
        )

    def test_csv_writes_a_header_and_refuses_to_overwrite(self):
        with scratch_directory() as directory:
            path = cloud().save_csv(directory / "cloud.csv")
            with self.assertRaises(FileExistsError):
                cloud().save_csv(path)
            header = path.read_text(encoding="utf-8").splitlines()[0]

        self.assertEqual(header, "angle_degrees,distance_mm,intensity,high_reflection")

    def test_csv_creates_missing_directories(self):
        with scratch_directory() as directory:
            path = cloud().save_csv(directory / "nested" / "cloud.csv")

            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()

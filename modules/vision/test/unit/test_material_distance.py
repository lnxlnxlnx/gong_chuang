"""Verify the material distance / offset maths offline."""

from __future__ import annotations

import unittest

from modules.vision.drive.hough_lock import LockedRegion
from modules.vision.drive.material_distance import MaterialDistanceEstimator
from modules.vision.drive.yolov8 import Detection


def detection(name="red", *, score=0.9, x1=260.0, y1=180.0, x2=380.0, y2=300.0, class_id=0):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id, name=name)


class CalibratedDistanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = MaterialDistanceEstimator(
            focal_px=560.0,
            real_widths_mm={"red": 60.0, "blue": 50.0},
            image_size=(640, 480),
        )
        self.region = LockedRegion(center_x=320.0, center_y=240.0, radius=80.0)

    def test_pinhole_distance_matches_the_documented_formula(self) -> None:
        # 560 * 60 / 120 = 280 mm
        result = self.estimator.estimate(detection(), region=self.region, frame_shape=(480, 640, 3))
        self.assertAlmostEqual(result.distance_mm, 280.0)
        self.assertAlmostEqual(result.distance_cm, 28.0)
        self.assertTrue(result.calibrated)
        self.assertEqual(result.box_width_px, 120.0)
        self.assertEqual(result.box_height_px, 120.0)

    def test_offsets_are_relative_to_the_locked_ring(self) -> None:
        result = self.estimator.estimate(detection(), region=self.region, frame_shape=(480, 640, 3))
        self.assertEqual(result.reference, "ring")
        self.assertAlmostEqual(result.offset_x_px, 0.0)
        self.assertAlmostEqual(result.offset_y_px, 0.0)
        self.assertAlmostEqual(result.region_distance_px, 0.0)

    def test_offsets_and_relative_coordinates_for_an_off_centre_box(self) -> None:
        off = detection(x1=330.0, y1=240.0, x2=390.0, y2=300.0)
        result = self.estimator.estimate(off, region=self.region, frame_shape=(480, 640, 3))

        self.assertAlmostEqual(result.offset_x_px, 40.0)
        self.assertAlmostEqual(result.offset_y_px, 30.0)
        self.assertAlmostEqual(result.region_distance_px, 50.0)
        self.assertAlmostEqual(result.relative_x, 40.0 / 320.0)
        self.assertAlmostEqual(result.relative_y, 30.0 / 240.0)

    def test_estimate_all_sorts_by_distance_to_the_ring(self) -> None:
        near = detection("blue", x1=330.0, y1=240.0, x2=390.0, y2=300.0, class_id=2)
        results = self.estimator.estimate_all([detection(), near], region=self.region, frame_shape=(480, 640))
        self.assertEqual([item.name for item in results], ["red", "blue"])

    def test_frame_shape_overrides_the_configured_image_size(self) -> None:
        result = self.estimator.estimate(detection(), frame_shape=(600, 800, 3))
        # 框中心 (320, 240) 在 800x600 画面里：相对中心 (400, 300) 偏移 (-80, -60)
        self.assertAlmostEqual(result.relative_x, -0.2)
        self.assertAlmostEqual(result.relative_y, -0.2)
        self.assertAlmostEqual(result.offset_x_px, -80.0)
        self.assertAlmostEqual(result.offset_y_px, -60.0)

    def test_invalid_frame_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.estimator.estimate(detection(), frame_shape=(480,))


class UncalibratedDistanceTests(unittest.TestCase):
    def test_without_real_widths_only_pixel_quantities_are_reported(self) -> None:
        estimator = MaterialDistanceEstimator(focal_px=560.0, image_size=(640, 480))
        result = estimator.estimate(detection(), frame_shape=(480, 640, 3))

        self.assertIsNone(result.distance_mm)
        self.assertIsNone(result.distance_cm)
        self.assertFalse(result.calibrated)
        self.assertEqual(result.reference, "image-center")
        self.assertIsNone(result.region_distance_px)
        self.assertAlmostEqual(result.offset_x_px, 0.0)
        self.assertAlmostEqual(result.offset_y_px, 0.0)

    def test_unknown_class_stays_uncalibrated_even_with_other_widths(self) -> None:
        estimator = MaterialDistanceEstimator(focal_px=560.0, real_widths_mm={"red": 60.0})
        result = estimator.estimate(detection("green", class_id=1), frame_shape=(480, 640))
        self.assertIsNone(result.distance_mm)

    def test_configured_image_size_is_used_when_no_frame_shape_is_given(self) -> None:
        estimator = MaterialDistanceEstimator(focal_px=560.0, image_size=(320, 240))
        result = estimator.estimate(detection(x1=60.0, y1=60.0, x2=100.0, y2=100.0))
        # 框中心 (80, 80)，画面中心 (160, 120)
        self.assertAlmostEqual(result.offset_x_px, -80.0)
        self.assertAlmostEqual(result.offset_y_px, -40.0)
        self.assertAlmostEqual(result.relative_x, -0.5)
        self.assertAlmostEqual(result.relative_y, -1.0 / 3.0)

    def test_estimate_all_without_a_region_sorts_by_absolute_offset(self) -> None:
        estimator = MaterialDistanceEstimator(focal_px=560.0)
        near = detection(x1=300.0, y1=220.0, x2=340.0, y2=260.0)
        far = detection(x1=0.0, y1=0.0, x2=40.0, y2=40.0)
        results = estimator.estimate_all([far, near], frame_shape=(480, 640))
        self.assertAlmostEqual(results[0].offset_x_px, 0.0)
        self.assertAlmostEqual(results[1].offset_x_px, -300.0)

    def test_rejects_invalid_focal_length(self) -> None:
        with self.assertRaises(ValueError):
            MaterialDistanceEstimator(focal_px=0.0)


if __name__ == "__main__":
    unittest.main()

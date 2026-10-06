"""Checks for the circle-first material report."""

from __future__ import annotations

import unittest
import numpy as np

from vision.hough_lock import synthetic_rings
from vision.material_locate import MaterialLocator
from vision.yolov8 import Detection


class FakeYOLO:
    def __init__(self, hits=()):
        self.hits = tuple(hits)

    def detect(self, _frame, *, materials_only=False):
        return list(self.hits)

    def close(self):
        pass


def hit(name, score, x):
    return Detection(x1=x - 20, y1=220, x2=x + 20, y2=260,
                     score=score, class_id=0, name=name)


class MaterialLocatorTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_rings()
        self.locator = MaterialLocator(
            yolo=FakeYOLO([hit("obj-red", 0.9, 160), hit("obj-blue", 0.7, 480)]),
            focal_px=300,
        )
        self.report = self.locator.locate(self.frame)

    def test_each_circle_is_a_material_even_when_yolo_misses_one(self):
        self.assertEqual(len(self.report.regions), 3)
        self.assertTrue(self.report.present)
        self.assertEqual([region.name for region in self.report.regions],
                         ["obj-red", None, "obj-blue"])
        self.assertTrue(all(region.present for region in self.report.regions))

    def test_distance_uses_circle_diameter(self):
        for region in self.report.regions:
            self.assertAlmostEqual(region.distance_mm,
                                   300 * 30 / (2 * region.region.radius))

    def test_best_prefers_the_highest_classification_score(self):
        self.assertEqual(self.report.best.name, "obj-red")

    def test_summary_keeps_the_unknown_category_and_distance(self):
        summary = self.report.summary()
        self.assertIn("obj-red", summary)
        self.assertIn("blue", summary)
        self.assertIn("mm", summary)
        self.assertIn("3", summary)

    def test_draw_marks_circles_without_changing_input(self):
        before = self.frame.copy()
        marked = self.locator.draw(self.frame, self.report)
        self.assertTrue(np.array_equal(self.frame, before))
        self.assertFalse(np.array_equal(marked, before))

    def test_yolo_miss_keeps_all_circles(self):
        report = MaterialLocator(yolo=FakeYOLO(), focal_px=300).locate(self.frame)
        self.assertEqual(len(report.regions), 3)
        self.assertTrue(all(region.name is None for region in report.regions))

    def test_no_circle_means_no_material_even_with_a_yolo_box(self):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        report = self.locator.locate(blank)
        self.assertFalse(report.present)
        self.assertEqual(report.regions, ())
        self.assertIsNone(report.best)

    def test_invalid_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            MaterialLocator(yolo=FakeYOLO(), focal_px=0)
        with self.assertRaises(ValueError):
            MaterialLocator(yolo=FakeYOLO(), top_diameter_mm=0)


if __name__ == "__main__":
    unittest.main()

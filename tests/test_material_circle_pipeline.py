"""Regression checks for locating a material by its 30 mm top face."""

from pathlib import Path
import unittest

import cv2
import numpy as np

from modules.vision.drive.hough_lock import HoughRingLocator, LockedRegion, RingLock
from modules.vision.drive.material_locate import MaterialLocator
from modules.vision.drive.yolov8 import Detection


FIXTURE = Path(__file__).parent / "fixtures" / "material_top.jpg"


class FakeYOLO:
    def __init__(self, hits=()):
        self.hits = tuple(hits)

    def detect(self, _frame, *, materials_only=False):
        return list(self.hits)

    def close(self):
        pass


def material_frame():
    frame = cv2.imread(str(FIXTURE))
    if frame is None:
        raise AssertionError(f"Missing test fixture: {FIXTURE}")
    return frame


class MaterialCirclePipelineTests(unittest.TestCase):
    def test_hough_finds_the_inner_top_face_in_the_camera_photo(self):
        lock = HoughRingLocator(expected=None).locate(material_frame())

        self.assertEqual(len(lock.regions), 1)
        top = lock.regions[0]
        self.assertAlmostEqual(top.center_x, 160, delta=5)
        self.assertAlmostEqual(top.center_y, 160, delta=5)
        self.assertAlmostEqual(top.radius, 85, delta=6)

    def test_yolo_miss_does_not_erase_circle_presence_or_distance(self):
        report = MaterialLocator(yolo=FakeYOLO(), focal_px=300).locate(material_frame())

        self.assertTrue(report.present)
        self.assertEqual(len(report.regions), 1)
        self.assertIsNone(report.regions[0].name)
        self.assertAlmostEqual(
            report.regions[0].distance_mm,
            300 * 30 / (2 * report.regions[0].region.radius),
        )

    def test_yolo_classifies_but_its_box_width_does_not_set_distance(self):
        hit = Detection(
            x1=150, y1=150, x2=170, y2=170,
            score=0.9, class_id=2, name="obj-blue",
        )
        report = MaterialLocator(yolo=FakeYOLO([hit]), focal_px=300).locate(material_frame())

        self.assertTrue(report.present)
        self.assertEqual(report.regions[0].name, "obj-blue")
        self.assertAlmostEqual(
            report.regions[0].distance_mm,
            300 * 30 / (2 * report.regions[0].region.radius),
        )

    def test_yolo_box_without_a_circle_is_not_a_material(self):
        hit = Detection(
            x1=100, y1=100, x2=200, y2=200,
            score=0.9, class_id=2, name="obj-blue",
        )
        frame = np.full((320, 320, 3), 128, dtype=np.uint8)
        report = MaterialLocator(yolo=FakeYOLO([hit]), focal_px=300).locate(frame)

        self.assertFalse(report.present)
        self.assertEqual(report.regions, ())

    def test_measured_calibration_scales_with_16_by_9_frame_width(self):
        class FixedCircle:
            def __init__(self, radius):
                self.radius = radius

            def locate(self, _frame):
                return RingLock((LockedRegion(320, 180, self.radius),))

        large = MaterialLocator(locator=FixedCircle(105), yolo=FakeYOLO()).locate(
            np.zeros((720, 1280, 3), dtype=np.uint8)
        )
        small = MaterialLocator(locator=FixedCircle(52.5), yolo=FakeYOLO()).locate(
            np.zeros((360, 640, 3), dtype=np.uint8)
        )
        self.assertAlmostEqual(large.regions[0].distance_mm, 1012 * 30 / 210)
        self.assertAlmostEqual(small.regions[0].distance_mm, large.regions[0].distance_mm)

    def test_unmatched_camera_crop_requires_explicit_calibration(self):
        with self.assertRaises(ValueError):
            MaterialLocator(yolo=FakeYOLO()).locate(material_frame())


if __name__ == "__main__":
    unittest.main()

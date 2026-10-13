"""颜色块定位：夹爪与台面不该进掩膜，同色噪点不该当物料。"""

from __future__ import annotations

import unittest

import numpy as np

from modules.vision.drive.hough_lock import synthetic_rings
from modules.vision.drive.material_blob import ColourBlobLocator, synthetic_scene
from modules.vision.drive.material_locate import MaterialLocator, parse_bgr

from modules.vision.test.unit.test_material_locate import FakeYOLO


MATERIAL_BGR = (114.0, 252.0, 165.0)


class ColourBlobLocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_scene()
        self.locator = ColourBlobLocator(MATERIAL_BGR)

    def test_only_the_material_coloured_blob_survives(self) -> None:
        lock = self.locator.locate(self.frame)
        self.assertEqual(len(lock.regions), 1)
        region = lock.regions[0]
        self.assertAlmostEqual(region.center_x, 300.0, delta=4)
        self.assertAlmostEqual(region.center_y, 240.0, delta=4)
        self.assertAlmostEqual(region.radius, 90.0, delta=4)

    def test_the_jaw_and_the_table_never_enter_the_mask(self) -> None:
        mask = self.locator.mask(self.frame)
        self.assertLess(int((mask > 0).sum()), 1.2 * np.pi * 90**2)

    def test_radius_window_filters_by_equivalent_radius(self) -> None:
        self.assertEqual(len(self.locator.locate(self.frame, min_radius=150, max_radius=300)), 0)
        self.assertEqual(len(self.locator.locate(self.frame, min_radius=60, max_radius=120)), 1)

    def test_a_frame_without_the_colour_gives_nothing(self) -> None:
        self.assertEqual(len(self.locator.locate(synthetic_rings())), 0)

    def test_locate_regions_matches_the_hough_interface(self) -> None:
        self.assertEqual(len(self.locator.locate_regions(self.frame)), 1)

    def test_debug_image_is_three_channel(self) -> None:
        self.assertEqual(self.locator.debug_image(self.frame).shape, self.frame.shape)

    def test_invalid_configuration_is_rejected(self) -> None:
        for kwargs in ({"reference_bgr": (1.0, 2.0)}, {"tolerance": 0.0},
                       {"min_area_px": 0}, {"min_radius": 0},
                       {"min_radius": 50, "max_radius": 50}):
            with self.subTest(kwargs=kwargs):
                settings = dict(reference_bgr=MATERIAL_BGR)
                settings.update(kwargs)
                with self.assertRaises(ValueError):
                    ColourBlobLocator(**settings)

    def test_empty_or_grey_frames_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.locator.mask(np.zeros((0, 0, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.locator.mask(np.zeros((10, 10), dtype=np.uint8))


class ColourFirstMaterialLocatorTests(unittest.TestCase):
    """配了颜色就不该再让霍夫的形状判据否决物料（12 cm 的顶面是带纹理的）。"""

    def test_colour_locator_is_chosen_when_a_reference_colour_is_given(self) -> None:
        locator = MaterialLocator(yolo=FakeYOLO(), focal_px=500.0, material_bgr=MATERIAL_BGR)
        self.assertTrue(locator.colour_first)
        self.assertIsInstance(locator.locator, ColourBlobLocator)
        report = locator.locate(synthetic_scene())
        self.assertEqual(len(report.faces), 1)
        self.assertEqual(len(report.rejected), 0)
        self.assertEqual(report.best.region.radius, report.faces[0].region.radius)

    def test_without_a_colour_the_hough_path_is_untouched(self) -> None:
        from modules.vision.drive.hough_lock import HoughRingLocator

        locator = MaterialLocator(yolo=FakeYOLO(), focal_px=500.0)
        self.assertFalse(locator.colour_first)
        self.assertIsInstance(locator.locator, HoughRingLocator)
        report = locator.locate(synthetic_scene())
        self.assertGreaterEqual(len(report.regions), 1)
        self.assertTrue(all(item.accepted == item.face.looks_like_face for item in report.regions))

    def test_colour_radius_window_is_applied_to_the_blob(self) -> None:
        locator = MaterialLocator(
            yolo=FakeYOLO(), focal_px=500.0, material_bgr=MATERIAL_BGR,
            depth_range_mm=(200.0, 260.0),   # 90 px 的等效半径不在这个窗口里
        )
        report = locator.locate(synthetic_scene())
        self.assertEqual(report.regions, ())
        self.assertIsNone(report.best)

    def test_best_takes_the_largest_accepted_face(self) -> None:
        """同一个顶面被凑出大小两个同心圆时，取大的那个（才是物体边界）。"""
        import cv2

        frame = np.full((480, 640, 3), (40, 70, 55), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 120, MATERIAL_BGR, -1)
        cv2.circle(frame, (320, 240), 60, MATERIAL_BGR, -1)
        locator = MaterialLocator(
            locator=ColourBlobLocator(MATERIAL_BGR, min_area_px=1),
            yolo=FakeYOLO(), focal_px=500.0, material_bgr=MATERIAL_BGR,
        )
        report = locator.locate(frame)
        self.assertEqual(len(report.faces), 1)
        self.assertAlmostEqual(report.best.region.radius, 120.0, delta=3)

    def test_rejected_candidates_are_marked_grey_in_the_drawing(self) -> None:
        import cv2

        frame = synthetic_scene()
        locator = MaterialLocator(yolo=FakeYOLO(), focal_px=500.0, material_bgr=MATERIAL_BGR)
        report = locator.locate(frame)
        before = frame.copy()
        marked = locator.draw(frame, report)
        self.assertTrue(np.array_equal(frame, before))
        self.assertFalse(np.array_equal(marked, before))


class ParseColourTests(unittest.TestCase):
    def test_bgr_text_is_parsed(self) -> None:
        self.assertEqual(parse_bgr("165,252,114"), (165.0, 252.0, 114.0))
        self.assertEqual(parse_bgr("165，252，114"), (165.0, 252.0, 114.0))
        self.assertEqual(parse_bgr([165, 252, 114]), (165.0, 252.0, 114.0))
        self.assertIsNone(parse_bgr(None))
        self.assertIsNone(parse_bgr(""))

    def test_invalid_colour_text_is_rejected(self) -> None:
        for bad in ("165,252", "a,b,c", "300,0,0"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_bgr(bad)


if __name__ == "__main__":
    unittest.main()

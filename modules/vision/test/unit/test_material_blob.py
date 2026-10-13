"""颜色块定位：夹爪与台面不该进掩膜，同色噪点不该当物料。"""

from __future__ import annotations

import unittest

import numpy as np

from modules.vision.drive.hough_lock import synthetic_rings
from modules.vision.drive.material_blob import (
    ColourBlobLocator,
    normalize_references,
    sample_reference,
    synthetic_scene,
)
from modules.vision.drive.material_locate import MaterialLocator, parse_bgr, parse_palette

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
        """白色夹爪和深色台面与参考色差得远，掩膜里不该有它们。"""
        mask = self.locator.mask(self.frame)
        self.assertGreater(int(mask[240, 300]), 0)      # 物料顶面在掩膜里
        self.assertEqual(int(mask[140, 300]), 0)        # 夹爪不在
        self.assertEqual(int(mask[460, 10]), 0)         # 台面不在
        self.assertLess(int((mask > 0).sum()), 0.5 * mask.size)  # 掩膜远小于整幅

    def test_radius_window_filters_by_equivalent_radius(self) -> None:
        self.assertEqual(len(self.locator.locate(self.frame, min_radius=150, max_radius=300)), 0)
        self.assertEqual(len(self.locator.locate(self.frame, min_radius=60, max_radius=120)), 1)

    def test_a_same_coloured_bracket_is_not_a_material(self) -> None:
        """调色板里有蓝色时，同色的 3D 打印件也会命中颜色；靠形状把它挡掉。"""
        loose = ColourBlobLocator(MATERIAL_BGR, max_boundary_variation=None, min_fill_ratio=None)
        self.assertEqual(len(self.locator.locate(self.frame)), 1)
        self.assertGreater(len(loose.locate(self.frame)), 1)

    def test_a_clipped_same_coloured_bracket_is_dropped_by_the_fill_ratio(self) -> None:
        """被画面切掉的打印件量不出边界起伏，要靠填充率兜住。"""
        import cv2

        frame = np.full((480, 640, 3), (40, 70, 55), dtype=np.uint8)
        cv2.rectangle(frame, (420, -60), (620, 40), MATERIAL_BGR, -1)   # 顶部被切掉
        locator = ColourBlobLocator(MATERIAL_BGR)
        self.assertEqual(locator.boundary_variation(
            locator.mask_for(frame, MATERIAL_BGR),
            cv2.connectedComponentsWithStats(locator.mask_for(frame, MATERIAL_BGR))[2][1],
            cv2.connectedComponentsWithStats(locator.mask_for(frame, MATERIAL_BGR))[3][1],
        ), None)
        self.assertEqual(len(locator.locate(frame)), 0)

    def test_a_blob_touching_the_frame_edge_is_kept(self) -> None:
        """贴边的色块起伏测不准，不能因此把物料丢掉。"""
        import cv2

        frame = np.full((200, 200, 3), (40, 70, 55), dtype=np.uint8)
        cv2.rectangle(frame, (0, 60), (60, 140), MATERIAL_BGR, -1)
        self.assertEqual(len(self.locator.locate(frame)), 1)

    def test_palette_keeps_one_region_per_object(self) -> None:
        palette = ColourBlobLocator({"green": MATERIAL_BGR, "blue": (248.0, 122.0, 41.0)})
        self.assertEqual(len(palette.locate(self.frame)), 1)

    def test_normalize_references_accepts_one_colour_a_list_or_a_palette(self) -> None:
        self.assertEqual(normalize_references(MATERIAL_BGR), (MATERIAL_BGR,))
        self.assertEqual(len(normalize_references([MATERIAL_BGR, (1.0, 2.0, 3.0)])), 2)
        self.assertEqual(len(normalize_references({"green": MATERIAL_BGR})), 1)
        for bad in ({}, [], "green", ((1.0, 2.0),)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    normalize_references(bad)

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

    def test_palette_text_is_parsed(self) -> None:
        self.assertEqual(
            parse_palette("green:165,252,114;blue:248,122,41"),
            {"green": (165.0, 252.0, 114.0), "blue": (248.0, 122.0, 41.0)},
        )
        self.assertEqual(
            parse_palette({"green": (165.0, 252.0, 114.0)}),
            {"green": (165.0, 252.0, 114.0)},
        )
        self.assertIsNone(parse_palette(None))
        self.assertIsNone(parse_palette(""))

    def test_invalid_palette_text_is_rejected(self) -> None:
        for bad in ("green", "green:", ":165,252,114", "green:1,2"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_palette(bad)


class SampledColourTests(unittest.TestCase):
    """调色板没配的颜色：从模型框里现取参考色，换料不必先量颜色。"""

    @staticmethod
    def red_frame():
        import cv2

        frame = np.full((480, 640, 3), (40, 70, 55), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 90, (30, 40, 210), -1)     # 红料，调色板里没有
        return frame

    def test_a_colour_outside_the_palette_is_learned_from_the_model_box(self) -> None:
        from modules.vision.drive.yolov8 import Detection

        frame = self.red_frame()
        hit = Detection(x1=230.0, y1=150.0, x2=410.0, y2=330.0, score=0.9, class_id=0, name="red")
        locator = MaterialLocator(
            yolo=FakeYOLO([hit]), focal_px=500.0,
            material_palette={"green": MATERIAL_BGR},
        )
        report = locator.locate(frame)

        self.assertEqual(report.colour_source, "model")
        self.assertFalse(report.colour_fallback)
        self.assertEqual(len(report.faces), 1)
        self.assertEqual(report.best.name, "red")
        self.assertAlmostEqual(report.best.region.radius, 90.0, delta=4)
        self.assertIn("参考色是从模型框里现取的", report.summary())

    def test_sample_reference_reads_the_centre_of_the_box(self) -> None:
        frame = self.red_frame()
        colour = sample_reference(frame, (230.0, 150.0, 410.0, 330.0))
        self.assertIsNotNone(colour)
        self.assertGreater(colour[2], colour[0])          # 红通道最大
        self.assertIsNone(sample_reference(frame, (0.0, 0.0, 3.0, 3.0)))

    def test_without_a_model_box_it_still_falls_back_to_hough(self) -> None:
        report = MaterialLocator(
            yolo=FakeYOLO(), focal_px=500.0, material_palette={"green": MATERIAL_BGR}
        ).locate(self.red_frame())
        self.assertIsNone(report.colour_source)
        self.assertEqual(report.regions, ())


class PaletteFallbackTests(unittest.TestCase):
    """调色板里没有这块料时，退回霍夫并在报告里说清楚，而不是"什么都看不见"。"""

    def test_a_palette_miss_falls_back_to_the_shape_judge(self) -> None:
        locator = MaterialLocator(
            yolo=FakeYOLO(), focal_px=500.0,
            material_palette={"blue": (248.0, 122.0, 41.0)},   # 画面里其实是绿料
        )
        report = locator.locate(synthetic_scene())
        self.assertTrue(locator.colour_first)
        self.assertTrue(report.colour_fallback)
        self.assertIn("调色板里没有这块料", report.summary())

    def test_a_palette_hit_does_not_fall_back(self) -> None:
        locator = MaterialLocator(
            yolo=FakeYOLO(), focal_px=500.0,
            material_palette={"green": MATERIAL_BGR},
        )
        report = locator.locate(synthetic_scene())
        self.assertFalse(report.colour_fallback)
        self.assertEqual(len(report.faces), 1)

    def test_the_palette_wins_over_a_single_colour(self) -> None:
        locator = MaterialLocator(
            yolo=FakeYOLO(), focal_px=500.0,
            material_bgr=(1.0, 2.0, 3.0),
            material_palette={"green": MATERIAL_BGR},
        )
        self.assertEqual(locator.locator.references, (MATERIAL_BGR,))


if __name__ == "__main__":
    unittest.main()

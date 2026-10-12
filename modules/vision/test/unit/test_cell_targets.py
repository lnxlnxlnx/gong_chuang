"""同一帧里按颜色区分夹爪靶标与物料，并各自按已知直径换算。"""

from __future__ import annotations

import unittest

import numpy as np

from modules.vision.config import TargetSettings
from modules.vision.drive.cell_targets import (
    MARKER_ROLE,
    MATERIAL_ROLE,
    CellTargetDetector,
)
from modules.vision.drive.pinhole import PinholeCamera

FOCAL_PX = 1012.0
MATERIAL_FILL = (200, 120, 60)
MARKER_FILL = (245, 245, 245)


def synthetic_frame() -> np.ndarray:
    """蓝色物料圆（半径 105 px）+ 白色靶标圆（半径 60 px）。

    画成「实心填充 + 深色外圈」：填充给出可采样的颜色，外圈给出霍夫需要的边缘。
    纯实心色块在默认霍夫阈值下检不出来。
    """
    import cv2

    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    for center, radius, fill, rim in (
        ((500, 400), 105, MATERIAL_FILL, (120, 70, 30)),
        ((900, 300), 60, MARKER_FILL, (150, 150, 150)),
    ):
        cv2.circle(frame, center, radius, fill, -1)
        cv2.circle(frame, center, radius, rim, 8)
    return frame


def make_detector(targets: TargetSettings) -> CellTargetDetector:
    return CellTargetDetector(
        camera=PinholeCamera(focal_px=FOCAL_PX, image_width=1280, image_height=720),
        targets=targets,
    )


def split_targets() -> TargetSettings:
    return TargetSettings(
        material_diameter_mm=30.0,
        marker_diameter_mm=20.0,
        material_bgr=tuple(float(c) for c in MATERIAL_FILL),
        marker_bgr=tuple(float(c) for c in MARKER_FILL),
        color_tolerance=60.0,
    )


class CellTargetSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_frame()

    def test_both_roles_are_found_in_one_frame(self) -> None:
        found = make_detector(split_targets()).detect(self.frame)
        self.assertEqual(len(found.materials), 1)
        self.assertEqual(len(found.markers), 1)
        self.assertEqual(found.unmatched, ())
        self.assertTrue(found.present)
        self.assertEqual(found.marker.role, MARKER_ROLE)
        self.assertEqual(found.material.role, MATERIAL_ROLE)

    def test_each_role_is_converted_with_its_own_known_diameter(self) -> None:
        found = make_detector(split_targets()).detect(self.frame)
        material = found.material
        marker = found.marker
        self.assertAlmostEqual(
            material.fix.depth_mm, FOCAL_PX * 30.0 / (2 * material.region.radius), places=9
        )
        self.assertAlmostEqual(
            marker.fix.depth_mm, FOCAL_PX * 20.0 / (2 * marker.region.radius), places=9
        )

    def test_screen_position_signs_are_preserved(self) -> None:
        found = make_detector(split_targets()).detect(self.frame)
        # 物料在画面中心左侧、下方；靶标在右侧、上方。
        self.assertLess(found.material.fix.x_mm, 0.0)
        self.assertGreater(found.material.fix.y_mm, 0.0)
        self.assertGreater(found.marker.fix.x_mm, 0.0)
        self.assertLess(found.marker.fix.y_mm, 0.0)

    def test_without_colour_references_every_circle_is_a_material(self) -> None:
        found = make_detector(TargetSettings(material_diameter_mm=30.0)).detect(self.frame)
        self.assertIsNone(found.marker)
        self.assertEqual(len(found.materials), 2)

    def test_circles_matching_no_reference_are_left_unmatched(self) -> None:
        targets = TargetSettings(
            material_diameter_mm=30.0,
            marker_diameter_mm=20.0,
            material_bgr=(0.0, 0.0, 0.0),
            marker_bgr=(0.0, 0.0, 0.0),
            color_tolerance=5.0,
        )
        found = make_detector(targets).detect(self.frame)
        self.assertIsNone(found.marker)
        self.assertFalse(found.materials)
        self.assertEqual(len(found.unmatched), 2)

    def test_a_marker_colour_without_a_diameter_is_not_reported_as_a_marker(self) -> None:
        """认出了颜色但没有真实直径就换算不出三维坐标，只能进 unmatched。"""
        targets = TargetSettings(
            material_diameter_mm=30.0,
            marker_diameter_mm=None,
            material_bgr=tuple(float(c) for c in MATERIAL_FILL),
            marker_bgr=tuple(float(c) for c in MARKER_FILL),
            color_tolerance=60.0,
        )
        found = make_detector(targets).detect(self.frame)
        self.assertIsNone(found.marker)
        self.assertEqual(len(found.materials), 1)
        self.assertEqual(len(found.unmatched), 1)


class CellTargetClassificationTests(unittest.TestCase):
    def test_classify_picks_the_nearest_reference_colour(self) -> None:
        detector = make_detector(split_targets())
        self.assertEqual(detector.classify((200.0, 120.0, 60.0)), MATERIAL_ROLE)
        self.assertEqual(detector.classify((245.0, 245.0, 245.0)), MARKER_ROLE)

    def test_classify_returns_none_beyond_the_tolerance(self) -> None:
        detector = make_detector(split_targets())
        self.assertIsNone(detector.classify((0.0, 255.0, 0.0)))

    def test_without_references_classify_defaults_to_material(self) -> None:
        detector = make_detector(TargetSettings(material_diameter_mm=30.0))
        self.assertEqual(detector.classify((0.0, 0.0, 0.0)), MATERIAL_ROLE)


class CellTargetSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_frame()

    def test_sampled_colour_is_the_fill_not_the_rim(self) -> None:
        detector = make_detector(split_targets())
        found = detector.detect(self.frame)
        material = found.material
        sampled = detector.sample_color(self.frame, material.region)
        for actual, expected in zip(sampled, MATERIAL_FILL):
            self.assertAlmostEqual(actual, float(expected), delta=1.0)

    def test_sample_fraction_must_be_a_valid_ratio(self) -> None:
        for fraction in (0.0, -0.5, 1.5):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    CellTargetDetector(
                        camera=PinholeCamera(focal_px=FOCAL_PX, image_width=1280, image_height=720),
                        targets=split_targets(),
                        color_sample_fraction=fraction,
                    )

    def test_empty_frame_is_rejected(self) -> None:
        detector = make_detector(split_targets())
        with self.assertRaisesRegex(ValueError, "empty"):
            detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_blank_frame_reports_nothing_rather_than_guessing(self) -> None:
        detector = make_detector(split_targets())
        found = detector.detect(np.full((720, 1280, 3), 40, dtype=np.uint8))
        self.assertIsNone(found.marker)
        self.assertIsNone(found.material)
        self.assertFalse(found.present)


if __name__ == "__main__":
    unittest.main()

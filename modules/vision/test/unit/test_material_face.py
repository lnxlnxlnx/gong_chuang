"""顶面判据：内部均匀度与边界支撑（不需要摄像头）。"""

from __future__ import annotations

import unittest

import numpy as np

from modules.vision.drive.hough_lock import LockedRegion
from modules.vision.drive.material_face import (
    DEFAULT_EDGE_SUPPORT_MIN,
    DEFAULT_INTERIOR_STD_MAX,
    FaceQuality,
    edge_support,
    interior_std,
    measure_face,
    to_grayscale,
)


def scene() -> tuple[np.ndarray, LockedRegion, LockedRegion, LockedRegion]:
    """单色亮圆面 + 内部不匀的干扰圆 + 平坦背景。"""
    frame = np.full((480, 640), 60.0)
    ys, xs = np.ogrid[:480, :640]
    frame[240:250, 120:520] = 250.0
    frame[(xs - 320) ** 2 + (ys - 240) ** 2 <= 70**2] = 200.0
    return (
        frame,
        LockedRegion(center_x=320.0, center_y=240.0, radius=70.0),
        LockedRegion(center_x=320.0, center_y=245.0, radius=140.0),
        LockedRegion(center_x=560.0, center_y=400.0, radius=60.0),
    )


class GrayscaleTests(unittest.TestCase):
    def test_bgr_gray_and_single_channel_inputs_are_accepted(self) -> None:
        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr[:, :, 1] = 200
        self.assertAlmostEqual(float(to_grayscale(bgr).mean()), 200 / 3, places=6)
        self.assertAlmostEqual(float(to_grayscale(np.full((4, 4), 7.0)).mean()), 7.0, places=9)
        self.assertEqual(to_grayscale(np.zeros((2, 2, 4), dtype=np.uint8)).shape, (2, 2))

    def test_empty_or_malformed_frames_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            to_grayscale(np.zeros((0, 0), dtype=np.uint8))
        with self.assertRaises(ValueError):
            to_grayscale(np.zeros((2, 2, 2, 2), dtype=np.uint8))
        # 空的 float64 灰度图也要照样拒绝，别被"原样返回"那条捷径漏过去
        with self.assertRaises(ValueError):
            to_grayscale(np.zeros((0, 0), dtype=np.float64))

    def test_an_already_float_gray_frame_is_returned_without_copying(self) -> None:
        """一帧 1280×720 的 astype(float64) 要 1.0–1.2 ms，而每个候选圆都要用同一张。"""
        gray = np.full((4, 4), 7.0)
        self.assertIs(to_grayscale(gray), gray)

    def test_other_dtypes_are_still_converted_to_float64(self) -> None:
        # float32 与 float64 的累加末位可能不同，下游要拿这些数比阈值，不能悄悄换精度。
        for array in (np.full((4, 4), 7, dtype=np.uint8),
                      np.full((4, 4), 7.0, dtype=np.float32)):
            with self.subTest(dtype=array.dtype):
                converted = to_grayscale(array)
                self.assertEqual(converted.dtype, np.float64)
                self.assertIsNot(converted, array)
                np.testing.assert_array_equal(converted, np.full((4, 4), 7.0))


class FaceQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame, self.material, self.clutter, self.flat = scene()

    def test_uniform_disc_looks_like_a_face(self) -> None:
        quality = measure_face(self.frame, self.material)
        self.assertLess(quality.interior_std, DEFAULT_INTERIOR_STD_MAX)
        self.assertGreaterEqual(quality.edge_support, DEFAULT_EDGE_SUPPORT_MIN)
        self.assertTrue(quality.looks_like_face)

    def test_cluttered_interior_does_not_look_like_a_face(self) -> None:
        quality = measure_face(self.frame, self.clutter)
        self.assertGreater(quality.interior_std, DEFAULT_INTERIOR_STD_MAX)
        self.assertFalse(quality.looks_like_face)

    def test_flat_background_has_no_boundary_so_it_is_not_a_face(self) -> None:
        quality = measure_face(self.frame, self.flat)
        self.assertLess(quality.interior_std, DEFAULT_INTERIOR_STD_MAX)
        self.assertLess(quality.edge_support, DEFAULT_EDGE_SUPPORT_MIN)
        self.assertFalse(quality.looks_like_face)

    def test_material_beats_the_clutter_on_both_metrics(self) -> None:
        material = measure_face(self.frame, self.material)
        clutter = measure_face(self.frame, self.clutter)
        self.assertLess(material.interior_std, clutter.interior_std)
        self.assertGreater(material.edge_support, clutter.edge_support)

    def test_bgr_frames_are_accepted_too(self) -> None:
        import cv2

        gray = self.frame.astype(np.uint8)
        quality = measure_face(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), self.material)
        self.assertTrue(quality.looks_like_face)

    def test_the_same_region_is_measured_the_same_way(self) -> None:
        first = measure_face(self.frame, self.material)
        second = measure_face(self.frame, self.material)
        self.assertEqual(first, second)

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs in ({"rays": 4}, {"interior_ratio": 0.0}, {"interior_ratio": 1.5},
                       {"outer_ratio": 1.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    measure_face(self.frame, self.material, **kwargs)

    def test_a_circle_hanging_over_the_frame_edge_is_measured_not_crashing(self) -> None:
        quality = measure_face(self.frame, LockedRegion(center_x=3.0, center_y=3.0, radius=40.0))
        self.assertGreaterEqual(quality.edge_support, 0.0)
        self.assertLessEqual(quality.edge_support, 1.0)

    def test_fully_offscreen_circle_has_no_support(self) -> None:
        quality = measure_face(self.frame, LockedRegion(center_x=-500.0, center_y=-500.0, radius=30.0))
        self.assertEqual(quality.edge_support, 0.0)
        self.assertEqual(quality.interior_std, float("inf"))

    def test_defaults_describe_the_measured_separation(self) -> None:
        """阈值要留在"物料约 4 / 干扰约 16–20"这段空隙里。"""
        self.assertGreater(DEFAULT_INTERIOR_STD_MAX, 4.0)
        self.assertLess(DEFAULT_INTERIOR_STD_MAX, 16.0)
        self.assertGreater(DEFAULT_EDGE_SUPPORT_MIN, 0.2)
        self.assertLess(DEFAULT_EDGE_SUPPORT_MIN, 0.9)

    def test_summary_names_both_metrics(self) -> None:
        summary = FaceQuality(interior_std=4.2, edge_support=0.8, sampled_rays=72).summary()
        self.assertIn("4.2", summary)
        self.assertIn("80%", summary)


class MetricTests(unittest.TestCase):
    def test_interior_std_ignores_the_edge_ring(self) -> None:
        frame = np.full((120, 120), 50.0)
        ys, xs = np.ogrid[:120, :120]
        frame[(xs - 60) ** 2 + (ys - 60) ** 2 <= 40**2] = 200.0
        region = LockedRegion(center_x=60.0, center_y=60.0, radius=40.0)
        self.assertAlmostEqual(interior_std(frame, region), 0.0, places=6)
        self.assertGreater(interior_std(frame, region, ratio=1.2), 10.0)

    def test_edge_support_is_one_for_a_clean_disc(self) -> None:
        frame = np.full((200, 200), 40.0)
        ys, xs = np.ogrid[:200, :200]
        frame[(xs - 100) ** 2 + (ys - 100) ** 2 <= 50**2] = 220.0
        self.assertAlmostEqual(
            edge_support(frame, LockedRegion(100.0, 100.0, 50.0)), 1.0, places=6
        )

    def test_edge_support_is_zero_on_a_flat_patch(self) -> None:
        frame = np.full((200, 200), 123.0)
        self.assertEqual(edge_support(frame, LockedRegion(100.0, 100.0, 50.0)), 0.0)


if __name__ == "__main__":
    unittest.main()

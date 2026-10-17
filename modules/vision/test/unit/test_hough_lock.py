"""Verify the Hough ring locator offline (no camera, no model).

The synthetic scene is the same one the module's ``__main__`` self-test uses:
three bright rings on a dark, unevenly lit background.
"""

from __future__ import annotations

import unittest

import numpy as np

from modules.vision.drive.hough_lock import (
    RELAXED_PARAM1,
    HoughRingLocator,
    LockedRegion,
    RingLock,
    synthetic_rings,
)


def soft_disc(width: int = 640, height: int = 480, radius: int = 90, sigma: float = 8.0):
    """A bright disc with a deliberately defocused edge (no camera needed)."""
    import cv2
    import numpy as np

    frame = np.full((height, width), 70, dtype=np.float32)
    ys, xs = np.ogrid[:height, :width]
    frame[(xs - width // 2) ** 2 + (ys - height // 2) ** 2 <= radius * radius] = 210
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_GRAY2BGR)


class LockedRegionTests(unittest.TestCase):
    def test_geometry_helpers(self) -> None:
        region = LockedRegion(center_x=100.0, center_y=50.0, radius=20.0)
        self.assertEqual(region.center, (100.0, 50.0))
        self.assertEqual(region.bbox, (80, 30, 120, 70))
        self.assertTrue(region.contains(110.0, 50.0))
        self.assertFalse(region.contains(130.0, 50.0))
        # margin 把判定半径放大
        self.assertTrue(region.contains(130.0, 50.0, margin=10.0))
        self.assertEqual(region.offset_from_center(90.0, 60.0), (-10.0, 10.0))
        self.assertEqual(region.scaled(2.0).radius, 40.0)
        self.assertEqual(region.expanded(5.0).radius, 25.0)
        self.assertEqual(region.expanded(-5.0).radius, 15.0)

    def test_expand_never_collapses_the_circle(self) -> None:
        self.assertGreaterEqual(LockedRegion(0.0, 0.0, 2.0).expanded(-10.0).radius, 1.0)

    def test_rejects_non_positive_scale(self) -> None:
        with self.assertRaises(ValueError):
            LockedRegion(0.0, 0.0, 5.0).scaled(0.0)


class RingLockTests(unittest.TestCase):
    def test_complete_flag_and_centers(self) -> None:
        regions = (LockedRegion(10.0, 0.0, 5.0), LockedRegion(20.0, 0.0, 5.0))
        self.assertTrue(RingLock(regions, expected=2).complete)
        self.assertFalse(RingLock(regions, expected=3).complete)
        self.assertTrue(RingLock(regions, expected=None).complete)
        self.assertEqual(RingLock(regions, expected=2).centers, ((10.0, 0.0), (20.0, 0.0)))
        self.assertEqual(len(RingLock(regions, expected=2)), 2)
        self.assertEqual([region.center_x for region in RingLock(regions, expected=2)], [10.0, 20.0])


class HoughRingLocatorTests(unittest.TestCase):
    def test_locks_three_synthetic_rings_left_to_right(self) -> None:
        lock = HoughRingLocator().locate(synthetic_rings())

        self.assertTrue(lock.complete)
        self.assertEqual(len(lock), 3)
        centers = [region.center for region in lock.regions]
        for (center_x, center_y), expected_x in zip(centers, (160, 320, 480)):
            self.assertAlmostEqual(center_x, expected_x, delta=3)
            self.assertAlmostEqual(center_y, 240, delta=3)
        for region in lock.regions:
            # 真实半径 40：合并同心命中后应当接近它
            self.assertAlmostEqual(region.radius, 40.0, delta=8)

    def test_reports_incomplete_when_expected_count_does_not_match(self) -> None:
        lock = HoughRingLocator(expected=4).locate(synthetic_rings())
        self.assertEqual(len(lock), 3)
        self.assertFalse(lock.complete)

    def test_no_rings_found_on_a_blank_frame(self) -> None:
        lock = HoughRingLocator().locate(np.full((480, 640, 3), 128, dtype=np.uint8))
        self.assertEqual(len(lock), 0)
        self.assertTrue(lock.complete)

    def test_radius_window_filters_out_the_wrong_size(self) -> None:
        # 合成环半径 40；把窗口限制到 15..25 就应当一个都不留
        lock = HoughRingLocator(min_radius=15, max_radius=25).locate(synthetic_rings())
        self.assertEqual(len(lock), 0)

    def test_merge_tolerance_keeps_distinct_rings_apart(self) -> None:
        # 三个环中心相距 160 px，远大于默认合并半径 25 px
        lock = HoughRingLocator().locate(synthetic_rings())
        self.assertEqual(len({round(region.center_x) for region in lock.regions}), 3)

    def test_preprocess_returns_a_smoothed_single_channel_edge_image(self) -> None:
        binary = HoughRingLocator().preprocess(synthetic_rings())
        self.assertEqual(binary.shape, (480, 640))
        self.assertEqual(binary.ndim, 2)
        self.assertEqual(binary.dtype, np.uint8)
        # 保留灰度和边缘，不再做二值化。
        self.assertGreater(int(binary.max()), 0)
        self.assertLess(int(binary.min()), int(binary.max()))
        self.assertGreater(int((binary > 127).sum()), 100)

    def test_debug_image_is_three_channel(self) -> None:
        debug = HoughRingLocator().debug_image(synthetic_rings())
        self.assertEqual(debug.shape, (480, 640, 3))

    def test_draw_returns_a_copy_and_keeps_the_original(self) -> None:
        locator = HoughRingLocator()
        frame = synthetic_rings()
        before = frame.copy()
        annotated = locator.draw(frame, locator.locate(frame))
        self.assertEqual(frame.tolist(), before.tolist())
        self.assertNotEqual(annotated.tolist(), before.tolist())

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            HoughRingLocator(expected=0)
        with self.assertRaises(ValueError):
            HoughRingLocator(min_radius=50, max_radius=50)
        with self.assertRaises(ValueError):
            HoughRingLocator(min_radius=0)
        with self.assertRaises(ValueError):
            HoughRingLocator(min_distance=0)

    def test_rejects_an_empty_frame(self) -> None:
        with self.assertRaises(ValueError):
            HoughRingLocator().preprocess(np.zeros((0, 0, 3), dtype=np.uint8))


class PerCallWindowTests(unittest.TestCase):
    """半径窗口可以逐帧给：调用方知道工作距离时用它代替"按画幅猜"。"""

    def test_call_window_replaces_the_instance_window(self) -> None:
        locator = HoughRingLocator()
        self.assertEqual(len(locator.locate(synthetic_rings(), min_radius=15, max_radius=25)), 0)
        lock = locator.locate(synthetic_rings(), min_radius=35, max_radius=50)
        self.assertEqual(len(lock), 3)
        for region in lock.regions:
            self.assertAlmostEqual(region.radius, 40.0, delta=8)

    def test_instance_window_is_untouched_after_a_call_override(self) -> None:
        locator = HoughRingLocator(min_radius=35, max_radius=50)
        locator.locate(synthetic_rings(), min_radius=15, max_radius=25)
        self.assertEqual(len(locator.locate(synthetic_rings())), 3)

    def test_invalid_call_window_is_rejected(self) -> None:
        locator = HoughRingLocator()
        for kwargs in ({"min_radius": 0}, {"max_radius": 0},
                       {"min_radius": 50, "max_radius": 50}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    locator.locate(synthetic_rings(), **kwargs)


class RelaxedLocatorTests(unittest.TestCase):
    """离焦画面：严格 Canny 阈值找不到的圆，放松后要能找到。"""

    def test_relaxed_lowers_the_canny_threshold_and_keeps_the_rest(self) -> None:
        locator = HoughRingLocator(param1=100.0, param2=30.0, dp=1.5,
                                   min_radius=10, max_radius=200, min_distance=20)
        relaxed = locator.relaxed()
        self.assertEqual(relaxed.param1, RELAXED_PARAM1)
        self.assertLess(relaxed.param1, locator.param1)
        self.assertEqual(
            (relaxed.param2, relaxed.dp, relaxed.min_radius, relaxed.max_radius,
             relaxed.min_distance),
            (locator.param2, locator.dp, locator.min_radius, locator.max_radius,
             locator.min_distance),
        )

    def test_relaxed_never_raises_the_threshold(self) -> None:
        locator = HoughRingLocator(param1=RELAXED_PARAM1 / 2)
        self.assertEqual(locator.relaxed().param1, locator.param1)

    def test_a_soft_disc_needs_the_relaxed_threshold(self) -> None:
        frame = soft_disc()
        strict = HoughRingLocator(param1=100.0, min_radius=60, max_radius=130, min_distance=100)
        self.assertEqual(len(strict.locate(frame)), 0)
        relaxed = strict.relaxed().locate(frame)
        self.assertEqual(len(relaxed), 1)
        self.assertAlmostEqual(relaxed.regions[0].radius, 90.0, delta=6)
        self.assertAlmostEqual(relaxed.regions[0].center_x, 320.0, delta=4)


class SyntheticSceneTests(unittest.TestCase):
    def test_scene_shape_and_ring_positions(self) -> None:
        frame = synthetic_rings()
        self.assertEqual(frame.shape, (480, 640, 3))
        # 环心处比底色亮（环是画上去的亮色）
        self.assertGreater(int(frame[240, 200, 0]), int(frame[10, 10, 0]))

    def test_custom_centers(self) -> None:
        frame = synthetic_rings(width=320, height=240, radius=20, thickness=6, centers=((60, 120),))
        lock = HoughRingLocator(min_radius=10, max_radius=40, expected=1).locate(frame)
        self.assertTrue(lock.complete)
        self.assertAlmostEqual(lock.regions[0].center_x, 60, delta=4)


if __name__ == "__main__":
    unittest.main()

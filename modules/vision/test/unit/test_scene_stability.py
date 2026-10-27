"""画面级停稳：**每一对**相邻帧都要够小，任何一对超限都算还在动。

这一组守的是抓取前的最后一道判断：物块在转盘上，对准的那一刻它可能还在跟着走。
判据必须回答"整幅画面有没有在动"，而不是"检测到的物料坐标稳不稳"——后者在目标被
爪子挡住、或已经转出画面时根本没有输入。
"""

from __future__ import annotations

import unittest

import numpy as np

from modules.vision.drive.scene_stability import (
    DEFAULT_MAX_CHANGED_RATIO,
    DEFAULT_MAX_MEAN_DIFF,
    DEFAULT_PIXEL_DIFF_THRESHOLD,
    DEFAULT_STABLE_FRAMES,
    FrameDifference,
    evaluate_stability,
    frame_difference,
)


def blank(value: int = 0) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def with_square(x: int, *, value: int = 220, y: int = 16) -> np.ndarray:
    frame = blank()
    frame[y:y + 16, x:x + 16] = value
    return frame


class FrameDifferenceTests(unittest.TestCase):
    def test_identical_frames_have_no_difference(self) -> None:
        difference = frame_difference(blank(), blank())
        self.assertEqual(difference.mean_abs_diff, 0.0)
        self.assertEqual(difference.changed_ratio, 0.0)

    def test_a_changed_pixel_changes_the_ratio(self) -> None:
        current = blank()
        current[0, 0] = 255
        difference = frame_difference(blank(), current)
        self.assertAlmostEqual(
            difference.changed_ratio, 1.0 / (48 * 64), places=12
        )
        self.assertAlmostEqual(
            difference.mean_abs_diff, 255.0 / (48 * 64), places=12
        )

    def test_a_difference_at_the_threshold_is_not_counted_as_changed(self) -> None:
        """判据是"超过门槛"，不是"达到门槛"——阈值附近的噪声不该算运动。"""
        current = blank()
        current[0, 0] = DEFAULT_PIXEL_DIFF_THRESHOLD
        difference = frame_difference(blank(), current)
        self.assertEqual(difference.changed_ratio, 0.0)
        self.assertGreater(difference.mean_abs_diff, 0.0)

    def test_gray_and_bgr_frames_are_both_accepted(self) -> None:
        first = blank()
        second = blank(10)
        bgr = frame_difference(first, second)
        gray = frame_difference(first[:, :, 0], second[:, :, 0])
        self.assertAlmostEqual(bgr.mean_abs_diff, gray.mean_abs_diff, places=9)

    def test_different_frame_sizes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "尺寸不一致"):
            frame_difference(blank(), np.zeros((24, 32, 3), dtype=np.uint8))

    def test_the_threshold_must_be_a_byte(self) -> None:
        for bad in (-1, 256, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    frame_difference(blank(), blank(), pixel_threshold=bad)


class EvaluateStabilityTests(unittest.TestCase):
    def test_identical_frames_are_stable(self) -> None:
        result = evaluate_stability([blank()] * DEFAULT_STABLE_FRAMES)
        self.assertTrue(result.stable)
        self.assertEqual(result.frames, DEFAULT_STABLE_FRAMES)
        self.assertEqual(len(result.pairs), DEFAULT_STABLE_FRAMES - 1)
        self.assertEqual(result.max_mean_abs_diff, 0.0)
        self.assertEqual(result.max_changed_ratio, 0.0)
        self.assertIn("静止", result.summary())

    def test_a_moving_block_is_not_stable(self) -> None:
        result = evaluate_stability([with_square(x) for x in (4, 12, 20, 28)])
        self.assertFalse(result.stable)
        self.assertGreater(result.max_changed_ratio, DEFAULT_MAX_CHANGED_RATIO)
        self.assertGreater(result.max_mean_abs_diff, DEFAULT_MAX_MEAN_DIFF)
        self.assertIn("还在动", result.summary())

    def test_every_adjacent_pair_must_pass(self) -> None:
        """只比首尾两帧会被"转过去又转回来"骗过去，所以中间那一对也要看。"""
        result = evaluate_stability([blank(), blank(), with_square(24), blank()])
        self.assertFalse(result.stable)
        self.assertTrue(result.pairs[0].changed_ratio == 0.0)
        self.assertGreater(result.pairs[1].changed_ratio, 0.0)
        self.assertEqual(len(result.pairs), 3)

    def test_thresholds_are_honoured(self) -> None:
        moving = [with_square(x) for x in (4, 12)]
        strict = evaluate_stability(moving)
        relaxed = evaluate_stability(
            moving, max_changed_ratio=1.0, max_mean_diff=255.0
        )
        self.assertFalse(strict.stable)
        self.assertTrue(relaxed.stable)

    def test_fewer_than_two_frames_is_rejected(self) -> None:
        for frames in ([], [blank()]):
            with self.subTest(frames=len(frames)):
                with self.assertRaisesRegex(ValueError, "至少需要两帧"):
                    evaluate_stability(frames)

    def test_invalid_thresholds_are_rejected(self) -> None:
        for kwargs in (
            {"pixel_threshold": -1},
            {"pixel_threshold": 256},
            {"max_mean_diff": -1.0},
            {"max_mean_diff": float("nan")},
            {"max_changed_ratio": -0.1},
            {"max_changed_ratio": 1.1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    evaluate_stability([blank()] * 2, **kwargs)

    def test_the_summary_prints_the_measured_values(self) -> None:
        result = evaluate_stability([with_square(4), with_square(20)])
        text = result.summary()
        self.assertIn("灰阶", text)
        self.assertIn("%", text)


class FrameDifferenceShapeTests(unittest.TestCase):
    """纯值对象的边界：``FrameDifference`` 只描述一对帧，不承担判定。"""

    def test_it_formats_both_metrics(self) -> None:
        text = FrameDifference(mean_abs_diff=1.25, changed_ratio=0.01).summary()
        self.assertIn("1.25", text)
        self.assertIn("1.000%", text)


if __name__ == "__main__":
    unittest.main()

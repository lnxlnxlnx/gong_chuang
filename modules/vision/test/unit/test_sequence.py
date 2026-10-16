"""多帧融合：静止剔野值、运动报速度，两者不能被同一套平均糊过去。

模块自测（``python -m modules.vision.drive.sequence``）跑的是同一批不变量；
这里从单元测试的角度再钉一遍，并补上自测里没有的边界。
"""

from __future__ import annotations

import unittest

from modules.vision.drive.sequence import (
    DEFAULT_MIN_SAMPLES,
    Fusion,
    TimedPoint,
    fuse,
)


def point(stamp: float, x: float = 0.0, y: float = 0.0, z: float = 100.0, **rest) -> TimedPoint:
    return TimedPoint(stamp=stamp, x_mm=x, y_mm=y, z_mm=z, **rest)


def still(count: int = 6, *, value: float = 10.0, step: float = 0.2):
    """一串静止读数：围绕 ``value`` 对称抖动。"""
    jitter = (0.0, 0.1, -0.1, 0.1, -0.1, 0.0)
    return [point(i * step, value + jitter[i % len(jitter)]) for i in range(count)]


def moving(count: int = 6, *, speed: float = 5.0, step: float = 0.2):
    """一串匀速读数：沿 x 以 ``speed`` 毫米/秒走。"""
    return [point(i * step, speed * i * step) for i in range(count)]


class StillnessTests(unittest.TestCase):
    def test_a_still_sequence_reports_the_value_and_not_moving(self) -> None:
        fused = fuse(still())
        self.assertIsInstance(fused, Fusion)
        self.assertFalse(fused.moving)
        self.assertTrue(fused.confident)
        self.assertAlmostEqual(fused.point.x_mm, 10.0, delta=0.05)
        self.assertAlmostEqual(fused.speed_mm_s, 0.0, delta=0.5)

    def test_an_outlier_is_dropped_and_the_value_holds(self) -> None:
        samples = still()
        samples.insert(3, point(0.6, 90.0))
        fused = fuse(samples)
        self.assertEqual(fused.rejected, 1)
        self.assertEqual(fused.samples, 6)
        self.assertAlmostEqual(fused.point.x_mm, 10.0, delta=0.05)

    def test_trembling_below_the_still_threshold_is_not_motion(self) -> None:
        trembling = [point(i * 0.2, 0.1 if i % 2 else -0.1) for i in range(6)]
        fused = fuse(trembling, still_mm=0.4)
        self.assertFalse(fused.moving)

    def test_a_creeping_drift_below_the_threshold_is_not_motion(self) -> None:
        creeping = [point(i * 0.2, 0.1 * i * 0.2) for i in range(6)]
        fused = fuse(creeping, still_mm=0.4)
        self.assertFalse(fused.moving)


class MotionTests(unittest.TestCase):
    def test_a_moving_sequence_reports_the_velocity(self) -> None:
        fused = fuse(moving(speed=5.0))
        self.assertTrue(fused.moving)
        self.assertFalse(fused.confident)
        self.assertAlmostEqual(fused.velocity_mm_s[0], 5.0, delta=1e-9)
        # 参考时刻默认取最后一帧：值是外推"到现在"的位置，不是整段的平均。
        self.assertAlmostEqual(fused.point.x_mm, 5.0, delta=1e-6)

    def test_motion_does_not_get_rejected_as_outliers(self) -> None:
        """这是"用平均 + 离中心太远"做不到的那一条：运动本身不该被剔掉。"""
        fused = fuse(moving())
        self.assertEqual(fused.rejected, 0)
        self.assertEqual(fused.samples, 6)

    def test_an_outlier_inside_a_moving_sequence_is_still_dropped(self) -> None:
        samples = moving()
        samples.insert(2, point(0.3, 77.0))
        fused = fuse(samples)
        self.assertEqual(fused.rejected, 1)
        self.assertAlmostEqual(fused.velocity_mm_s[0], 5.0, delta=0.05)

    def test_extrapolating_to_an_earlier_stamp_gives_that_moment(self) -> None:
        fused = fuse(moving(), reference_stamp=0.0)
        self.assertAlmostEqual(fused.point.x_mm, 0.0, delta=1e-6)
        self.assertAlmostEqual(fused.point.stamp, 0.0)

    def test_motion_is_measured_across_all_three_axes(self) -> None:
        samples = [point(i * 0.2, 0.0, 0.0, 100.0 + 2.0 * i * 0.2) for i in range(6)]
        fused = fuse(samples, still_mm=0.4)
        self.assertTrue(fused.moving, "只有 z 在动也算在动")
        self.assertAlmostEqual(fused.velocity_mm_s[2], 2.0, delta=1e-9)


class MissesAndQualityTests(unittest.TestCase):
    def test_missing_frames_are_counted_but_do_not_take_part_in_the_fit(self) -> None:
        samples = [still()[0], None, still()[1], None, still()[2], still()[3], still()[4], still()[5]]
        fused = fuse(samples)
        self.assertEqual(fused.misses, 2)
        self.assertEqual(fused.frames, 8)
        self.assertAlmostEqual(fused.point.x_mm, 10.0, delta=0.05)

    def test_seen_ratio_counts_dropped_frames_as_seen(self) -> None:
        samples = still(6)
        samples.insert(3, point(0.6, 90.0))
        fused = fuse(samples)
        self.assertAlmostEqual(fused.seen_ratio, 6 / 7, delta=1e-9)

    def test_a_majority_of_outliers_is_not_confident(self) -> None:
        """野值占多数说明这一串根本不是一个目标，不该算可信。"""
        samples = still(3) + [point(0.6 + i * 0.2, 90.0 + i) for i in range(4)]
        fused = fuse(samples)
        self.assertIsNotNone(fused)
        self.assertFalse(fused.confident)

    def test_colors_are_recorded_so_a_switched_target_is_visible(self) -> None:
        samples = [point(0.0, 1.0, color="green"), point(0.2, 1.0, color="green"),
                   point(0.4, 1.0, color="red"), point(0.6, 1.0, color="green")]
        fused = fuse(samples)
        self.assertEqual(set(fused.colors), {"green", "red"})


class DegenerateInputTests(unittest.TestCase):
    def test_too_few_valid_readings_give_none(self) -> None:
        for samples in ([], [None] * 4, [point(0.0, 1.0), None, None],
                        [point(0.0, 1.0), point(0.2, 1.0)]):
            with self.subTest(samples=len(samples)):
                self.assertIsNone(fuse(samples))

    def test_identical_timestamps_degrade_to_the_median(self) -> None:
        """时间全相同时不能除以 0：应当退化成"没有趋势"的中位数。"""
        samples = [point(1.0, 3.0), point(1.0, 3.1), point(1.0, 2.9), point(1.0, 3.05)]
        fused = fuse(samples)
        self.assertIsNotNone(fused)
        self.assertFalse(fused.moving)
        self.assertAlmostEqual(fused.point.x_mm, 3.0, delta=0.06)
        self.assertEqual(fused.span_seconds, 0.0)

    def test_duplicate_timestamps_among_others_do_not_break_the_fit(self) -> None:
        samples = still()
        samples.append(point(samples[2].stamp, 10.05))
        fused = fuse(samples)
        self.assertIsNotNone(fused)
        self.assertAlmostEqual(fused.point.x_mm, 10.0, delta=0.1)

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs in ({"reject_k": 0.0}, {"floor_mm": 0.0}, {"still_mm": -0.1},
                       {"min_samples": 2}, {"min_samples": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    fuse(still(), **kwargs)

    def test_the_default_minimum_is_three(self) -> None:
        """少于 3 个有效读数时"剔野值"没有意义——剔掉一个就只剩猜。"""
        self.assertEqual(DEFAULT_MIN_SAMPLES, 3)
        self.assertIsNone(fuse([point(0.0, 1.0), point(0.2, 1.0)]))


if __name__ == "__main__":
    unittest.main()

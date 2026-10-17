"""圆拟合：顶面被挡掉一块时，圆心与半径不能被"可见的那一半"带偏。

这一组是**回归测试**：改动之前，颜色块路径用「连通域质心 + 面积等效半径」，
顶面被夹爪挡掉 30% 就把 104 mm 报成 121 mm（实测真机帧），而且照样接受这个候选。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from modules.vision.drive.circle_fit import (
    CircleFit,
    angular_coverage,
    as_points,
    fit_circle,
    fit_circle_kasa,
)


def circle_points(center_x, center_y, radius, count=720, *, keep=None):
    """一圈点；``keep`` 给定时只保留满足它的那些。"""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.column_stack(
        [center_x + radius * np.cos(angles), center_y + radius * np.sin(angles)]
    )
    if keep is not None:
        points = points[keep(points)]
    return points


def covered_disc(radius=140.0, coverage=0.4, center=(320.0, 240.0)):
    """一个圆面被左侧竖直遮挡盖掉 ``coverage`` 比例的直径后留下的轮廓点。

    返回 ``(轮廓点, 面积等效半径)``：面积等效半径用多边形面积算，
    它就是被替换掉的那个估计。
    """
    cut = center[0] - radius + 2.0 * radius * coverage
    points = circle_points(center[0], center[1], radius, count=720)
    arc = points[points[:, 0] >= cut]
    # 遮挡留下的直边（弦）：两个交点之间的一段。
    half = math.sqrt(max(radius**2 - (cut - center[0]) ** 2, 0.0))
    chord = np.column_stack([
        np.full(400, cut),
        np.linspace(center[1] - half, center[1] + half, 400),
    ])
    outline = np.vstack([arc, chord])
    angles = np.linspace(0.0, 2.0 * np.pi, 1441)
    polygon = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])
    inside = polygon[polygon[:, 0] >= cut]
    area = 0.5 * abs(
        np.dot(inside[:, 0], np.roll(inside[:, 1], -1))
        - np.dot(inside[:, 1], np.roll(inside[:, 0], -1))
    )
    return outline, math.sqrt(area / math.pi)


class KasaFitTests(unittest.TestCase):
    def test_a_clean_ring_is_recovered_exactly(self) -> None:
        solved = fit_circle_kasa(circle_points(300.0, 200.0, 120.0))
        self.assertIsNotNone(solved)
        center_x, center_y, radius = solved
        self.assertAlmostEqual(center_x, 300.0, delta=0.01)
        self.assertAlmostEqual(center_y, 200.0, delta=0.01)
        self.assertAlmostEqual(radius, 120.0, delta=0.01)

    def test_fewer_than_three_points_give_nothing(self) -> None:
        self.assertIsNone(fit_circle_kasa(np.zeros((2, 2))))
        self.assertIsNone(fit_circle_kasa(np.zeros((0, 2))))

    def test_collinear_points_give_a_useless_solution(self) -> None:
        """共线时**裸 Kåsa 照样给一个解**（半径几百像素的"圆"），这一点要写下来。

        它正是 :func:`fit_circle` 存在的理由：单靠这个线性解挡不住退化解，
        要靠"内点绕圆心铺开的角度"和调用方给的半径域。
        """
        line = np.column_stack([np.linspace(0.0, 500.0, 40), np.full(40, 100.0)])
        solved = fit_circle_kasa(line)
        self.assertIsNotNone(solved)
        self.assertGreater(solved[2], 200.0)    # 半径已经明显不合理
        self.assertIsNone(fit_circle(line))     # 带保护的那一层必须拒绝


class FitCircleTests(unittest.TestCase):
    def test_the_whole_contour_is_used_when_it_is_complete(self) -> None:
        fit = fit_circle(circle_points(300.0, 200.0, 120.0))
        self.assertIsInstance(fit, CircleFit)
        self.assertAlmostEqual(fit.radius, 120.0, delta=0.2)
        self.assertAlmostEqual(fit.center_x, 300.0, delta=0.2)
        self.assertGreater(fit.coverage, 0.95)

    def test_a_partly_covered_face_keeps_its_radius_while_the_area_estimate_does_not(self) -> None:
        """**这一条就是这次改动的理由。**

        遮挡 40% 时面积等效半径只有真值的八成左右（距离因此报远十几毫米），
        而拟合圆仍然贴着真值——因为那条直边在 RANSAC 里是外点。
        """
        outline, equivalent = covered_disc(radius=140.0, coverage=0.4)
        fit = fit_circle(outline)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.radius, 140.0, delta=2.0)
        self.assertAlmostEqual(fit.center_x, 320.0, delta=2.0)
        self.assertAlmostEqual(fit.center_y, 240.0, delta=2.0)
        # 被替换掉的那个估计在同一批点上有多少偏：这条断言保证上面那两行有对照。
        self.assertLess(equivalent, 125.0)
        self.assertLess(fit.inlier_ratio, 1.0)      # 弦上的点确实是外点

    def test_more_occlusion_is_still_better_than_the_area_estimate(self) -> None:
        outline, equivalent = covered_disc(radius=140.0, coverage=0.55)
        fit = fit_circle(outline)
        self.assertIsNotNone(fit)
        self.assertLess(abs(fit.radius - 140.0), abs(equivalent - 140.0))

    def test_the_radius_domain_rejects_a_solution_outside_it(self) -> None:
        """半径域是挡退化解最有效的一道闸：真解在域外时必须拒绝，而不是硬给。"""
        points = circle_points(300.0, 200.0, 120.0)
        self.assertIsNone(fit_circle(points, min_radius=40.0, max_radius=100.0))
        self.assertIsNone(fit_circle(points, min_radius=140.0, max_radius=400.0))
        self.assertIsNotNone(fit_circle(points, min_radius=100.0, max_radius=140.0))

    def test_a_short_arc_is_not_enough_to_pin_a_circle(self) -> None:
        """很短的弧也能"拟合"出一个大圆，但那条解没有约束力，靠角度覆盖挡住。"""
        short = circle_points(300.0, 200.0, 120.0, keep=lambda p: p[:, 1] < 200.0 - 0.97 * 120.0)
        self.assertIsNone(fit_circle(short))

    def test_degenerate_input_gives_nothing_instead_of_a_bad_circle(self) -> None:
        self.assertIsNone(fit_circle(np.zeros((0, 2))))
        self.assertIsNone(fit_circle(np.zeros((4, 2))))
        line = np.column_stack([np.linspace(0.0, 500.0, 60), np.full(60, 100.0)])
        self.assertIsNone(fit_circle(line))

    def test_every_contour_shape_is_accepted(self) -> None:
        raw = circle_points(300.0, 200.0, 120.0, count=120)
        for name, value in (("(N,1,2)", raw.reshape(-1, 1, 2)),
                            ("list", raw.tolist()),
                            ("(N,2)", raw)):
            with self.subTest(shape=name):
                fit = fit_circle(value)
                self.assertIsNotNone(fit)
                self.assertAlmostEqual(fit.radius, 120.0, delta=0.5)

    def test_non_finite_points_are_dropped_rather_than_poisoning_the_fit(self) -> None:
        raw = circle_points(300.0, 200.0, 120.0, count=120)
        dirty = np.vstack([raw, [[np.nan, 10.0], [np.inf, 20.0]]])
        self.assertEqual(len(as_points(dirty)), len(raw))
        fit = fit_circle(dirty)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.radius, 120.0, delta=0.5)

    def test_the_result_does_not_depend_on_the_random_seed_in_practice(self) -> None:
        outline, _equivalent = covered_disc(radius=140.0, coverage=0.4)
        radii = [fit_circle(outline, seed=seed).radius for seed in range(6)]
        self.assertLess(max(radii) - min(radii), 1.0)

    def test_invalid_parameters_are_rejected(self) -> None:
        points = circle_points(300.0, 200.0, 120.0)
        for kwargs in ({"band_px": 0.0}, {"iterations": 0}, {"min_radius": 0.0},
                       {"max_radius": -1.0}, {"min_inlier_ratio": 0.0},
                       {"min_inlier_ratio": 1.5}, {"min_coverage": 1.0},
                       {"min_radius": 200.0, "max_radius": 100.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    fit_circle(points, **kwargs)


class AngularCoverageTests(unittest.TestCase):
    def test_a_closed_ring_covers_the_whole_turn(self) -> None:
        self.assertGreater(angular_coverage(circle_points(0.0, 0.0, 100.0), 0.0, 0.0), 0.98)

    def test_a_half_ring_covers_about_half(self) -> None:
        half = circle_points(0.0, 0.0, 100.0, keep=lambda p: p[:, 1] <= 0.0)
        self.assertAlmostEqual(angular_coverage(half, 0.0, 0.0), 0.5, delta=0.05)

    def test_too_few_points_cover_nothing(self) -> None:
        self.assertEqual(angular_coverage(np.zeros((2, 2)), 0.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()

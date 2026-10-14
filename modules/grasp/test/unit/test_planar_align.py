"""平面精确解的不变量。

:mod:`modules.grasp.drive.align` 第一版连着错了三次（Cramer 符号、约束里把 Δr 又旋转
了一次、以及一个把弧度当度用的调用），而每一次"看起来"都像正常动作。所以这里的
判据刻意只有一条硬标准：**把解出来的两个量施加到位姿上，爪子中心必须落在物料上。**

## 为什么用手算的公式来判，而不是复用模块里的对比函数

``planar_error_mm`` 和被测代码会一起错。这里用显式的
``hypot(r·cosθ − r_t·cosθ_t, r·sinθ − r_t·sinθ_t)`` 独立算一遍，
虽然它和 :func:`~modules.arm.drive.kinematics.forward_cylindrical` 是同一个数学，
但写法独立、并且在下文里单独验证过一次，能挡住"函数本身被改坏"这类回归。
"""

from __future__ import annotations

import math
import unittest

from modules.arm.drive.kinematics import (
    CylindricalPose,
    TravelLimit,
    check_reachable,
    forward_cylindrical,
    require_reachable,
    shortest_angle_delta_degrees,
)
from modules.grasp.drive.align import (
    AlignmentError,
    alignment_to_pose,
    describe_planar_step,
    is_aligned,
    planar_error_mm,
    pose_after,
    solve_planar_alignment,
)


def independent_planar_error(first: CylindricalPose, second: CylindricalPose) -> float:
    """不用被测代码，按定义算两个位姿在水平面上的距离。"""
    first_x = first.radial_mm * math.cos(math.radians(first.base_degrees))
    first_y = first.radial_mm * math.sin(math.radians(first.base_degrees))
    second_x = second.radial_mm * math.cos(math.radians(second.base_degrees))
    second_y = second.radial_mm * math.sin(math.radians(second.base_degrees))
    return math.hypot(first_x - second_x, first_y - second_y)


#: 覆盖各种相对位置的位姿对：(当前 θ, 当前 r, 物料 θ, 物料 r)。
POSE_PAIRS = (
    (0.0, 80.0, 30.0, 90.0),
    (0.0, 80.0, 0.0, 90.0),
    (30.0, 90.0, 30.0, 90.0),
    (30.0, 90.0, 40.0, 60.0),
    (-170.0, 100.0, 170.0, 100.0),
    (0.0, 100.0, 180.0, 100.0),
    (10.0, 100.0, -170.0, 40.0),
    (90.0, 5.0, -90.0, 120.0),
    (-179.9, 120.0, 179.9, 120.0),
    (123.456, 42.0, -67.89, 155.0),
    (0.0, 1.0, 90.0, 1.0),
    (45.0, 200.0, -135.0, 30.0),
)


class PlanarErrorTests(unittest.TestCase):
    def test_matches_the_independent_formula(self) -> None:
        for theta, radius, theta_t, radius_t in POSE_PAIRS:
            with self.subTest(pair=(theta, radius, theta_t, radius_t)):
                first = CylindricalPose(theta, radius, 10.0)
                second = CylindricalPose(theta_t, radius_t, 20.0)
                self.assertAlmostEqual(
                    planar_error_mm(first, second),
                    independent_planar_error(first, second),
                    places=9,
                )

    def test_ignores_height(self) -> None:
        """判据是**平面**误差：高度差再大也不该影响它。"""
        low = CylindricalPose(10.0, 50.0, 0.0)
        high = CylindricalPose(10.0, 50.0, 999.0)
        self.assertEqual(planar_error_mm(low, high), 0.0)

    def test_identical_poses_are_zero(self) -> None:
        pose = CylindricalPose(37.0, 88.0, 12.0)
        self.assertEqual(planar_error_mm(pose, pose), 0.0)


class SolvePlanarAlignmentTests(unittest.TestCase):
    def test_solution_lands_the_gripper_on_the_material(self) -> None:
        """核心不变量：施加解出来的 (Δθ, Δr) 之后，爪子中心正好在物料上。"""
        for theta, radius, theta_t, radius_t in POSE_PAIRS:
            with self.subTest(pair=(theta, radius, theta_t, radius_t)):
                current = CylindricalPose(theta, radius, 50.0)
                target = CylindricalPose(theta_t, radius_t, 20.0)
                alignment = solve_planar_alignment(current, target)

                reached = pose_after(
                    current, alignment.delta_base_degrees, alignment.delta_radial_mm
                )
                self.assertLess(
                    independent_planar_error(reached, target), 1e-9, alignment
                )
                # 模块自己报的残差也要与独立算法一致。
                self.assertAlmostEqual(
                    alignment.residual_mm,
                    independent_planar_error(reached, target),
                    places=9,
                )
                self.assertAlmostEqual(
                    alignment.error_before_mm,
                    independent_planar_error(current, target),
                    places=9,
                )

    def test_rotates_to_the_material_shorter_way(self) -> None:
        """转角就是"从当前方向到物料方向"的最短角，不多也不少。"""
        for theta, radius, theta_t, radius_t in POSE_PAIRS:
            with self.subTest(pair=(theta, radius, theta_t, radius_t)):
                current = CylindricalPose(theta, radius, 50.0)
                target = CylindricalPose(theta_t, radius_t, 20.0)
                alignment = solve_planar_alignment(current, target)
                self.assertAlmostEqual(
                    alignment.delta_base_degrees,
                    shortest_angle_delta_degrees(theta, theta_t),
                    places=9,
                )

    def test_extends_to_the_material_radius(self) -> None:
        """伸缩量是"从当前半径到物料半径"的差，符号必须对。"""
        for theta, radius, theta_t, radius_t in POSE_PAIRS:
            with self.subTest(pair=(theta, radius, theta_t, radius_t)):
                current = CylindricalPose(theta, radius, 50.0)
                target = CylindricalPose(theta_t, radius_t, 20.0)
                alignment = solve_planar_alignment(current, target)
                self.assertAlmostEqual(alignment.delta_radial_mm, radius_t - radius, places=9)
                self.assertAlmostEqual(alignment.radial_after_mm, radius_t, places=9)
                self.assertGreaterEqual(alignment.radial_after_mm, 0.0)

    def test_angle_stays_inside_the_atan2_domain(self) -> None:
        """累计角必须落在 (-180, 180]，否则行程检查和最短转角都会算错。"""
        for theta, radius, theta_t, radius_t in POSE_PAIRS:
            with self.subTest(pair=(theta, radius, theta_t, radius_t)):
                pose = alignment_to_pose(
                    CylindricalPose(theta, radius, 50.0),
                    CylindricalPose(theta_t, radius_t, 20.0),
                )
                self.assertGreater(pose.base_degrees, -180.0)
                self.assertLessEqual(pose.base_degrees, 180.0)

    def test_already_aligned_produces_no_motion(self) -> None:
        alignment = solve_planar_alignment(
            CylindricalPose(30.0, 90.0, 50.0), CylindricalPose(30.0, 90.0, 20.0)
        )
        self.assertAlmostEqual(alignment.delta_base_degrees, 0.0, places=12)
        self.assertAlmostEqual(alignment.delta_radial_mm, 0.0, places=12)
        self.assertAlmostEqual(alignment.error_before_mm, 0.0, places=12)
        self.assertTrue(alignment.rotation_only)
        self.assertTrue(alignment.rotation_free)

    def test_rotation_only_and_extension_only_classification(self) -> None:
        extend = solve_planar_alignment(
            CylindricalPose(0.0, 80.0, 50.0), CylindricalPose(0.0, 95.0, 20.0)
        )
        self.assertTrue(extend.rotation_free)
        self.assertFalse(extend.rotation_only)
        self.assertAlmostEqual(extend.delta_radial_mm, 15.0, places=9)

        turn = solve_planar_alignment(
            CylindricalPose(0.0, 80.0, 50.0), CylindricalPose(25.0, 80.0, 20.0)
        )
        self.assertTrue(turn.rotation_only)
        self.assertFalse(turn.rotation_free)
        self.assertAlmostEqual(turn.delta_base_degrees, 25.0, places=9)

    def test_crossing_the_pm_180_boundary_takes_the_short_way(self) -> None:
        """−170° → 170° 是 **−20°**（不是 +20°，也不是 +340°）。

        符号很容易记反：从 −170° 往正方向要走 340°，往负方向只要 20°，
        所以最短转角是负的。``shortest_angle_delta_degrees`` 的返回值直接给出结论。
        """
        crossing = solve_planar_alignment(
            CylindricalPose(-170.0, 100.0, 50.0), CylindricalPose(170.0, 100.0, 20.0)
        )
        self.assertAlmostEqual(crossing.delta_base_degrees, -20.0, places=9)
        self.assertAlmostEqual(
            crossing.delta_base_degrees,
            shortest_angle_delta_degrees(-170.0, 170.0),
            places=9,
        )

        # −150° → 153.43°：穿过 ±180 只要 −56.57°，直接跳却是 +303.43°。
        around = solve_planar_alignment(
            CylindricalPose(-150.0, 120.0, 50.0), CylindricalPose(153.434948, 22.36, 20.0)
        )
        self.assertAlmostEqual(around.delta_base_degrees, -56.565052, places=5)

    def test_target_on_the_rotation_axis_is_legal(self) -> None:
        """径向 0 是合法解（机械上未必做得到，但几何上没有矛盾）。"""
        alignment = solve_planar_alignment(
            CylindricalPose(0.0, 100.0, 50.0), CylindricalPose(180.0, 0.0, 20.0)
        )
        self.assertAlmostEqual(alignment.radial_after_mm, 0.0, places=9)
        self.assertAlmostEqual(alignment.delta_radial_mm, -100.0, places=9)

    def test_negative_target_radius_is_rejected(self) -> None:
        with self.assertRaisesRegex(AlignmentError, "非负"):
            solve_planar_alignment(
                CylindricalPose(0.0, 100.0, 50.0), CylindricalPose(180.0, -5.0, 20.0)
            )

    def test_negative_current_radius_is_rejected(self) -> None:
        with self.assertRaisesRegex(AlignmentError, "非负"):
            solve_planar_alignment(
                CylindricalPose(0.0, -1.0, 50.0), CylindricalPose(0.0, 10.0, 20.0)
            )

    def test_solver_does_not_check_travel_limits(self) -> None:
        """解算器只管平面几何；越界是调用方的事（否则错误会在错误的地方抛出）。

        这一条把一个**刻意的**职责划分钉下来：越界必须由 ``require_reachable``
        在发帧之前拦住，而不是指望解算器。
        """
        current = CylindricalPose(0.0, 80.0, 50.0)
        target = CylindricalPose(10.0, 500.0, 20.0)
        solved = alignment_to_pose(current, target)  # 不抛
        self.assertAlmostEqual(solved.radial_mm, 500.0, places=9)
        with self.assertRaisesRegex(ValueError, "行程"):
            require_reachable(solved, radial_limit=TravelLimit(40.0, 160.0))

    def test_reachability_check_sees_the_limits(self) -> None:
        """``check_reachable`` 返回越界说明；**不传的轴不检查**（没标定就没有保护）。"""
        target = CylindricalPose(0.0, 200.0, 20.0)
        self.assertEqual(check_reachable(target), (), "没有 limits 时不该报任何问题")
        self.assertEqual(len(check_reachable(target, radial_limit=TravelLimit(40.0, 160.0))), 1)
        self.assertEqual(check_reachable(target, radial_limit=TravelLimit(40.0, 260.0)), ())
        problems = check_reachable(
            target,
            # 0° 不在 100..200 里，200 mm 也不在 240..300 里：两个轴都要报。
            base_limit=TravelLimit(100.0, 200.0),
            radial_limit=TravelLimit(240.0, 300.0),
        )
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(any("base" in problem for problem in problems), problems)
        self.assertTrue(any("extension" in problem for problem in problems), problems)


class PoseAfterTests(unittest.TestCase):
    def test_advances_and_normalizes_the_angle(self) -> None:
        pose = pose_after(CylindricalPose(170.0, 50.0, 30.0), 20.0, -5.0)
        self.assertAlmostEqual(pose.base_degrees, -170.0, places=9)
        self.assertAlmostEqual(pose.radial_mm, 45.0, places=9)
        self.assertAlmostEqual(pose.height_mm, 30.0, places=9)

    def test_keeps_height(self) -> None:
        pose = pose_after(CylindricalPose(0.0, 50.0, 77.0), 10.0, 10.0)
        self.assertEqual(pose.height_mm, 77.0)


class AlignmentToPoseTests(unittest.TestCase):
    def test_matches_solve_then_pose_after(self) -> None:
        current = CylindricalPose(12.0, 70.0, 45.0)
        target = CylindricalPose(-140.0, 130.0, 20.0)
        direct = alignment_to_pose(current, target)
        alignment = solve_planar_alignment(current, target)
        stepwise = pose_after(
            current, alignment.delta_base_degrees, alignment.delta_radial_mm
        )
        self.assertEqual(direct, stepwise)

    def test_reaches_the_target_on_the_plane(self) -> None:
        for theta, radius, theta_t, radius_t in POSE_PAIRS:
            with self.subTest(pair=(theta, radius, theta_t, radius_t)):
                pose = alignment_to_pose(
                    CylindricalPose(theta, radius, 45.0),
                    CylindricalPose(theta_t, radius_t, 20.0),
                )
                self.assertLess(independent_planar_error(pose, CylindricalPose(theta_t, radius_t, 20.0)), 1e-9)
                # forward_cylindrical 也要给出同一结论（两条独立路径）。
                pose_x, pose_y, _ = forward_cylindrical(pose)
                target_pose = CylindricalPose(theta_t, radius_t, 20.0)
                target_x, target_y, _ = forward_cylindrical(target_pose)
                self.assertLess(math.hypot(pose_x - target_x, pose_y - target_y), 1e-9)


class IsAlignedTests(unittest.TestCase):
    def test_tolerance_is_a_plain_upper_bound_on_the_planar_error(self) -> None:
        """判据就是"平面距离 ≤ 容差"，不要再加点别的余量。

        刻意不用"误差正好等于容差"这种例子：那条边界落在浮点尾数上
        （``hypot`` 算出来可能是 1.0000000000000002），断它会变成在测浮点实现，
        而不是测这条判据。
        """
        current = CylindricalPose(30.0, 90.0, 50.0)
        self.assertTrue(is_aligned(current, CylindricalPose(30.0, 90.5, 20.0), tolerance_mm=1.0))
        self.assertFalse(is_aligned(current, CylindricalPose(30.0, 92.0, 20.0), tolerance_mm=1.0))
        # 同一个误差，容差一放开就必须算对准。
        self.assertTrue(is_aligned(current, CylindricalPose(30.0, 92.0, 20.0), tolerance_mm=3.0))

    def test_non_positive_tolerance_is_rejected(self) -> None:
        pose = CylindricalPose(0.0, 10.0, 0.0)
        for tolerance in (0.0, -1.0):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    is_aligned(pose, pose, tolerance_mm=tolerance)


class DescribePlanarStepTests(unittest.TestCase):
    def test_reports_aligned_targets_without_solving(self) -> None:
        pose = CylindricalPose(30.0, 90.0, 50.0)
        self.assertIn("已经对准", describe_planar_step(pose, pose))

    def test_reports_the_step_for_a_misaligned_target(self) -> None:
        text = describe_planar_step(
            CylindricalPose(0.0, 80.0, 50.0), CylindricalPose(30.0, 90.0, 20.0)
        )
        self.assertIn("转", text)
        self.assertIn("伸缩", text)


if __name__ == "__main__":
    unittest.main()

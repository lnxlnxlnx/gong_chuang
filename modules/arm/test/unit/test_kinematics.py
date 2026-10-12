"""圆柱运动学与单位换算：逆解、最短转角、深度→高度、可达性。"""

from __future__ import annotations

import math
import unittest

from modules.arm.drive.kinematics import (
    CylindricalPose,
    TravelLimit,
    base_degrees_for_pulses,
    base_pulses_for_degrees,
    check_reachable,
    forward_cylindrical,
    height_above_table,
    inverse_cylindrical,
    linear_pulses_for_mm,
    marker_height_for_grasp,
    mm_for_linear_pulses,
    normalize_degrees,
    require_reachable,
    shortest_angle_delta_degrees,
)

PULSES_PER_REVOLUTION = 3200
BASE_REDUCTION = 5
MM_PER_MOTOR_REV = 40.0


class BaseAxisConversionTests(unittest.TestCase):
    def test_one_output_turn_is_a_full_reduction_worth_of_pulses(self) -> None:
        self.assertEqual(
            base_pulses_for_degrees(
                360.0, pulses_per_revolution=PULSES_PER_REVOLUTION, reduction_ratio=BASE_REDUCTION
            ),
            PULSES_PER_REVOLUTION * BASE_REDUCTION,
        )

    def test_g_five_makes_1000_pulses_22_5_output_degrees(self) -> None:
        """config.json 里底座减速比是 5，所以文档里的 18.75° 作废，应为 22.5°。"""
        self.assertAlmostEqual(
            base_degrees_for_pulses(
                1000, pulses_per_revolution=PULSES_PER_REVOLUTION, reduction_ratio=BASE_REDUCTION
            ),
            22.5,
            places=9,
        )

    def test_conversion_round_trips_and_keeps_the_sign(self) -> None:
        for degrees in (0.0, 22.5, -90.0, 180.0):
            with self.subTest(degrees=degrees):
                pulses = base_pulses_for_degrees(
                    degrees,
                    pulses_per_revolution=PULSES_PER_REVOLUTION,
                    reduction_ratio=BASE_REDUCTION,
                )
                self.assertEqual(
                    base_degrees_for_pulses(
                        pulses,
                        pulses_per_revolution=PULSES_PER_REVOLUTION,
                        reduction_ratio=BASE_REDUCTION,
                    ),
                    degrees,
                )

    def test_half_pulses_round_away_from_zero_on_both_signs(self) -> None:
        """银行家舍入会让 ±0.5 脉冲都变成 0，对称的位移就不对称了。"""
        half = 0.5 * 360.0 / (PULSES_PER_REVOLUTION * BASE_REDUCTION)
        self.assertEqual(
            base_pulses_for_degrees(
                half, pulses_per_revolution=PULSES_PER_REVOLUTION, reduction_ratio=BASE_REDUCTION
            ),
            1,
        )
        self.assertEqual(
            base_pulses_for_degrees(
                -half, pulses_per_revolution=PULSES_PER_REVOLUTION, reduction_ratio=BASE_REDUCTION
            ),
            -1,
        )

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            base_degrees_for_pulses(100, pulses_per_revolution=0, reduction_ratio=5)
        with self.assertRaises(ValueError):
            base_degrees_for_pulses(100, pulses_per_revolution=3200, reduction_ratio=0)
        with self.assertRaises(ValueError):
            base_pulses_for_degrees(float("nan"), pulses_per_revolution=3200, reduction_ratio=5)


class LinearAxisConversionTests(unittest.TestCase):
    def test_mm_per_motor_rev_is_the_only_constant_needed(self) -> None:
        self.assertEqual(
            linear_pulses_for_mm(
                40.0, pulses_per_revolution=PULSES_PER_REVOLUTION, mm_per_motor_rev=MM_PER_MOTOR_REV
            ),
            PULSES_PER_REVOLUTION,
        )
        self.assertAlmostEqual(
            mm_for_linear_pulses(
                1600, pulses_per_revolution=PULSES_PER_REVOLUTION, mm_per_motor_rev=MM_PER_MOTOR_REV
            ),
            20.0,
            places=9,
        )

    def test_sub_pulse_moves_round_to_zero_so_callers_can_treat_them_as_arrived(self) -> None:
        self.assertEqual(
            linear_pulses_for_mm(
                0.001, pulses_per_revolution=PULSES_PER_REVOLUTION, mm_per_motor_rev=MM_PER_MOTOR_REV
            ),
            0,
        )

    def test_an_uncalibrated_axis_refuses_millimetres_in_both_directions(self) -> None:
        for call in (
            lambda: linear_pulses_for_mm(
                10.0, pulses_per_revolution=PULSES_PER_REVOLUTION, mm_per_motor_rev=None
            ),
            lambda: mm_for_linear_pulses(
                100, pulses_per_revolution=PULSES_PER_REVOLUTION, mm_per_motor_rev=None
            ),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "标定"):
                    call()

    def test_non_positive_calibration_values_are_rejected(self) -> None:
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    mm_for_linear_pulses(
                        100, pulses_per_revolution=PULSES_PER_REVOLUTION, mm_per_motor_rev=value
                    )


class CylindricalKinematicsTests(unittest.TestCase):
    def test_forward_and_inverse_are_exact_inverses(self) -> None:
        for plane_x, plane_y in ((100.0, 0.0), (0.0, 100.0), (-30.0, -40.0), (0.0, 0.0)):
            with self.subTest(point=(plane_x, plane_y)):
                pose = inverse_cylindrical(plane_x, plane_y, 17.0)
                back = forward_cylindrical(pose)
                self.assertAlmostEqual(back[0], plane_x, places=9)
                self.assertAlmostEqual(back[1], plane_y, places=9)
                self.assertAlmostEqual(back[2], 17.0, places=9)

    def test_radial_distance_is_never_negative(self) -> None:
        pose = inverse_cylindrical(-3.0, -4.0, 0.0)
        self.assertAlmostEqual(pose.radial_mm, 5.0, places=9)
        self.assertAlmostEqual(pose.base_degrees, math.degrees(math.atan2(-4.0, -3.0)), places=9)

    def test_signs_match_the_documented_plane_axes(self) -> None:
        self.assertAlmostEqual(inverse_cylindrical(10.0, 10.0, 0.0).base_degrees, 45.0, places=9)
        self.assertAlmostEqual(inverse_cylindrical(-10.0, 10.0, 0.0).base_degrees, 135.0, places=9)
        self.assertAlmostEqual(inverse_cylindrical(0.0, -10.0, 0.0).base_degrees, -90.0, places=9)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inverse_cylindrical(float("inf"), 0.0, 0.0)
        with self.assertRaises(ValueError):
            forward_cylindrical(CylindricalPose(0.0, -1.0, 0.0))


class ShortestAngleTests(unittest.TestCase):
    def test_it_takes_the_short_way_around_the_wrap_point(self) -> None:
        """170° 到 -170° 是 20°，不是 340°；底座扫一大圈会撞到东西。"""
        self.assertAlmostEqual(shortest_angle_delta_degrees(170.0, -170.0), 20.0, places=9)
        self.assertAlmostEqual(shortest_angle_delta_degrees(-170.0, 170.0), -20.0, places=9)

    def test_a_half_turn_is_reported_as_positive_180(self) -> None:
        self.assertAlmostEqual(shortest_angle_delta_degrees(0.0, 180.0), 180.0, places=9)
        self.assertAlmostEqual(shortest_angle_delta_degrees(0.0, -180.0), 180.0, places=9)

    def test_the_result_always_lands_in_the_documented_range(self) -> None:
        for start in range(-180, 181, 30):
            for target in range(-180, 181, 30):
                delta = shortest_angle_delta_degrees(float(start), float(target))
                self.assertGreater(delta, -180.0)
                self.assertLessEqual(delta, 180.0)

    def test_applying_the_delta_lands_on_the_target_modulo_a_full_turn(self) -> None:
        for start, target in ((175.0, -175.0), (-175.0, 175.0), (0.0, 90.0), (90.0, 0.0)):
            with self.subTest(start=start, target=target):
                current = start + shortest_angle_delta_degrees(start, target)
                self.assertAlmostEqual((current - target) % 360.0, 0.0, places=9)
                # 而且必须是短边：走过的角度不超过半圈。
                self.assertLessEqual(abs(current - start), 180.0)


class NormalizeDegreesTests(unittest.TestCase):
    def test_the_range_matches_atan2(self) -> None:
        """相机只能给 atan2 的输出，预测位姿必须落在同一个区间里。"""
        for value in (-540.0, -180.0, -170.0, 0.0, 170.0, 180.0, 190.0, 540.0, 720.0):
            with self.subTest(value=value):
                wrapped = normalize_degrees(value)
                self.assertGreater(wrapped, -180.0)
                self.assertLessEqual(wrapped, 180.0)
                self.assertAlmostEqual((wrapped - value) % 360.0, 0.0, places=9)

    def test_turning_20_degrees_past_170_lands_on_the_measured_representation(self) -> None:
        """170° + 20° 物理上就是 -170°；不归一就会记成 190° 而通不过 ±180 的行程检查。"""
        self.assertAlmostEqual(normalize_degrees(170.0 + 20.0), -170.0, places=9)

    def test_minus_180_and_180_collapse_to_the_same_representation(self) -> None:
        self.assertAlmostEqual(normalize_degrees(-180.0), 180.0, places=9)
        self.assertAlmostEqual(normalize_degrees(180.0), 180.0, places=9)

    def test_non_finite_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_degrees(float("nan"))


class HeightTests(unittest.TestCase):
    def test_table_depth_minus_point_depth_is_the_height(self) -> None:
        self.assertAlmostEqual(
            height_above_table(table_depth_mm=300.0, point_depth_mm=285.0), 15.0, places=9
        )

    def test_a_point_on_the_table_has_zero_height(self) -> None:
        self.assertAlmostEqual(
            height_above_table(table_depth_mm=300.0, point_depth_mm=300.0), 0.0, places=9
        )

    def test_side_grasp_height_follows_the_material_height(self) -> None:
        # 物料高 20 mm，夹半高 -> 夹持点高 10 mm；靶标在夹持点上方 12 mm -> 靶标要到 22 mm。
        self.assertAlmostEqual(
            marker_height_for_grasp(
                material_height_mm=20.0, grip_ratio=0.5, marker_to_grasp_mm=12.0
            ),
            22.0,
            places=9,
        )
        self.assertAlmostEqual(
            marker_height_for_grasp(
                material_height_mm=20.0, grip_ratio=1.0, marker_to_grasp_mm=0.0
            ),
            20.0,
            places=9,
        )

    def test_invalid_grasp_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            marker_height_for_grasp(
                material_height_mm=20.0, grip_ratio=1.5, marker_to_grasp_mm=0.0
            )
        with self.assertRaises(ValueError):
            marker_height_for_grasp(
                material_height_mm=-1.0, grip_ratio=0.5, marker_to_grasp_mm=0.0
            )


class TravelLimitTests(unittest.TestCase):
    def test_minimum_must_not_exceed_maximum(self) -> None:
        with self.assertRaises(ValueError):
            TravelLimit(10.0, 5.0)

    def test_contains_is_inclusive_on_both_ends(self) -> None:
        limit = TravelLimit(-5.0, 5.0)
        self.assertTrue(limit.contains(-5.0))
        self.assertTrue(limit.contains(5.0))
        self.assertFalse(limit.contains(5.001))


class ReachabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_limit = TravelLimit(-110.0, 110.0)
        self.radial_limit = TravelLimit(40.0, 160.0)
        self.height_limit = TravelLimit(-5.0, 120.0)

    def check(self, pose: CylindricalPose):
        return check_reachable(
            pose,
            base_limit=self.base_limit,
            radial_limit=self.radial_limit,
            height_limit=self.height_limit,
        )

    def test_a_pose_inside_every_limit_has_no_problems(self) -> None:
        self.assertEqual(self.check(CylindricalPose(90.0, 80.0, 10.0)), ())

    def test_each_violated_axis_is_named_in_its_own_message(self) -> None:
        problems = self.check(CylindricalPose(150.0, 200.0, 200.0))
        self.assertEqual(len(problems), 3)
        self.assertTrue(any("base" in item for item in problems))
        self.assertTrue(any("extension" in item for item in problems))
        self.assertTrue(any("lift" in item for item in problems))

    def test_an_omitted_limit_is_skipped_and_that_is_the_point(self) -> None:
        """没标定行程就是没有软件保护；这里如实跳过，不假装检查过。"""
        self.assertEqual(check_reachable(CylindricalPose(9999.0, 9999.0, 9999.0)), ())

    def test_require_reachable_raises_with_every_reason_joined(self) -> None:
        with self.assertRaises(ValueError) as caught:
            require_reachable(
                CylindricalPose(150.0, 200.0, 10.0),
                base_limit=self.base_limit,
                radial_limit=self.radial_limit,
            )
        message = str(caught.exception)
        self.assertIn("150.0", message)
        self.assertIn("200.0", message)

    def test_require_reachable_returns_nothing_when_the_pose_is_fine(self) -> None:
        require_reachable(
            CylindricalPose(0.0, 100.0, 0.0),
            base_limit=self.base_limit,
            radial_limit=self.radial_limit,
        )


if __name__ == "__main__":
    unittest.main()

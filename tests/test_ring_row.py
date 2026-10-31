"""三环成排这条直线的三个用法：外推环心、转车对垂直、一帧算完三个落点。

全都不碰真机：假视觉按当前"倾角"造三个环心，假底盘按"每脉冲转多少度"改它。
守四件容易静默错掉的事：

- 中间环**实测**、两端按间距**外推**（错一格就是 150 mm，动作看起来完全正常）；
- 一帧里三个号各取一个（不能只挑分数最高的那个）；
- 转车那个循环先试方向、再按实测响应比一步到位，车没转就报错；
- v3 只拍一帧、解算一次，然后按 ``orders`` 把三趟放完。
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import app.gripper as app_gripper
from modules.arm.config import load_arm_config
from modules.grasp.drive.ring_plan import RingAlignResult
from modules.vision.drive.numbered_ring import (
    RingTarget,
    RingTargetLocator,
    derive_ring_row,
)


VISIBLE_MIDDLE_RING = RingTarget(
    center_px=(640.0, 360.0), radius_px=200.0, digit=2, score=0.95,
    clipped=False, x_mm=0.0, y_mm=0.0,
)


class DeriveRingRowTests(unittest.TestCase):
    def test_three_collinear_centers_keep_their_places(self) -> None:
        row = derive_ring_row(
            {1: (-150.0, 0.0), 2: (0.0, 0.0), 3: (150.0, 0.0)}, spacing_mm=150.0
        )

        self.assertAlmostEqual(row.yaw_deg, 0.0)
        self.assertAlmostEqual(row.measured_span_mm, 300.0)
        for digit in (1, 2, 3):
            self.assertAlmostEqual(row.centers_mm[digit][0], (digit - 2) * 150.0)
            self.assertAlmostEqual(row.centers_mm[digit][1], 0.0)

    def test_the_ends_are_extrapolated_from_the_measured_middle(self) -> None:
        """倾角 30°：两端 = 中间 ∓ 150 mm 沿直线——不是照抄（更不准的）两端框心。"""
        unit = (math.cos(math.radians(30.0)), math.sin(math.radians(30.0)))
        row = derive_ring_row(
            {
                1: (-300.0 * unit[0], -300.0 * unit[1]),   # 故意放远：外推不读这个值
                2: (0.0, 0.0),
                3: (300.0 * unit[0], 300.0 * unit[1]),
            },
            spacing_mm=150.0,
        )

        self.assertAlmostEqual(row.yaw_deg, 30.0)
        self.assertAlmostEqual(row.centers_mm[1][0], -150.0 * unit[0])
        self.assertAlmostEqual(row.centers_mm[1][1], -150.0 * unit[1])
        self.assertAlmostEqual(row.centers_mm[3][0], 150.0 * unit[0])
        self.assertAlmostEqual(row.centers_mm[3][1], 150.0 * unit[1])

    def test_one_visible_neighbour_is_enough(self) -> None:
        row = derive_ring_row({2: (0.0, 0.0), 3: (0.0, 150.0)}, spacing_mm=150.0)

        self.assertIsNone(row.measured_span_mm)
        self.assertAlmostEqual(row.centers_mm[1][1], -150.0)
        self.assertAlmostEqual(row.centers_mm[3][1], 150.0)
        # 直线没有方向：90° 与 -90° 是同一条线，折进 (-90, 90] 就是 -90。
        self.assertAlmostEqual(row.yaw_deg, -90.0)

    def test_a_line_without_a_middle_ring_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 号环"):
            derive_ring_row({1: (0.0, 0.0), 3: (150.0, 0.0)}, spacing_mm=150.0)

    def test_the_middle_ring_alone_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "定不出"):
            derive_ring_row({2: (0.0, 0.0)}, spacing_mm=150.0)

    def test_a_negative_spacing_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            derive_ring_row({2: (0.0, 0.0), 3: (150.0, 0.0)}, spacing_mm=0.0)


class _FakeDetector:
    """按给定框返回 ``Detection``：一帧几个框，正是一帧 → 三个环的接口。"""

    def __init__(self, hits) -> None:
        self.hits = list(hits)
        self.class_names = ["tg1", "tg2", "tg3"]
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        return list(self.hits)


class DetectAllTests(unittest.TestCase):
    def test_one_model_call_gives_every_ring_number(self) -> None:
        from modules.vision.config import RingTargetSettings
        from modules.vision.drive.yolov8 import Detection

        detector = _FakeDetector(
            [
                Detection(x1=100.0, y1=100.0, x2=200.0, y2=200.0, score=0.4,
                          class_id=4, name="tg1"),
                Detection(x1=102.0, y1=102.0, x2=202.0, y2=202.0, score=0.9,
                          class_id=4, name="tg1"),
                Detection(x1=400.0, y1=100.0, x2=500.0, y2=200.0, score=0.8,
                          class_id=6, name="tg3"),
            ]
        )
        locator = RingTargetLocator(
            detector,
            settings=RingTargetSettings(refine_center=False, outer_diameter_mm=95.0),
        )

        found = locator.detect_all(np.zeros((720, 1280, 3), dtype=np.uint8))

        self.assertEqual(sorted(found), [1, 3])
        self.assertEqual(detector.calls, 1)
        self.assertGreater(found[1].score, found[3].score)  # 同一个号取最好的那个
        self.assertEqual(found[1].scale_source, "ring box")


class RowPlant:
    """假车 + 假视觉：自转按 ``deg_per_pulse`` 改倾角，视觉每帧按当前倾角造一个环排。"""

    def __init__(self, yaw_deg: float, deg_per_pulse: float = 0.04) -> None:
        self.yaw_deg = yaw_deg
        self.deg_per_pulse = deg_per_pulse
        self.moves: list[int] = []  # 带符号脉冲：正 = 顺时针

    def rotate_cw(self, *, rpm=None, pulses=0, **_kwargs):
        self.moves.append(int(pulses))
        self.yaw_deg -= pulses * self.deg_per_pulse

    def rotate_ccw(self, *, rpm=None, pulses=0, **_kwargs):
        self.moves.append(-int(pulses))
        self.yaw_deg += pulses * self.deg_per_pulse


class RowVision:
    ring_center_spacing_mm = 150.0

    def __init__(self, plant: RowPlant, *, blanks: int = 0) -> None:
        self.plant = plant
        self.blanks = blanks

    def locate_rings(self):
        if self.blanks > 0:
            self.blanks -= 1
            return {}
        yaw = math.radians(self.plant.yaw_deg)
        unit = (math.cos(yaw), math.sin(yaw))
        return {
            digit: RingTarget(
                center_px=(
                    640.0 + (digit - 2) * 300.0 * unit[0],
                    360.0 + (digit - 2) * 300.0 * unit[1],
                ),
                radius_px=200.0,
                digit=digit,
                score=0.9,
                clipped=False,
                x_mm=(digit - 2) * 150.0 * unit[0],
                y_mm=(digit - 2) * 150.0 * unit[1],
                scale_mm_per_px=0.5,
                scale_source="ring box",
            )
            for digit in (1, 2, 3)
        }


def make_row_robot(plant: RowPlant, **kwargs) -> Mock:
    robot = Mock()
    robot.chassis = plant
    robot.vision = RowVision(plant, **kwargs)
    return robot


class SquareChassisToRingRowTests(unittest.TestCase):
    def run_it(self, plant: RowPlant, **kwargs) -> float:
        robot = make_row_robot(plant)
        # 目标偏置是现场标定量（会被改），用例里固定成 0，跟那个数无关。
        kwargs.setdefault("yaw_offset_deg", 0.0)
        with patch("time.sleep"):
            return app_gripper.square_chassis_to_ring_row(robot, **kwargs)

    def test_it_squares_a_tilted_row_in_two_turns(self) -> None:
        plant = RowPlant(yaw_deg=10.0)

        residual = self.run_it(plant, tolerance_deg=0.5)

        self.assertLessEqual(abs(residual), 0.5)
        # 小步试转后按实测响应一次补齐，避免大步跨过目标再倒回来。
        self.assertEqual(len(plant.moves), 2)
        self.assertAlmostEqual(plant.yaw_deg, 0.0, delta=0.5)

    def test_line_angle_wrap_uses_the_shortest_residual(self) -> None:
        plant = RowPlant(yaw_deg=-89.0)

        residual = self.run_it(plant, yaw_offset_deg=89.0, tolerance_deg=0.5)

        self.assertLessEqual(abs(residual), 0.5)
        self.assertLessEqual(sum(abs(step) for step in plant.moves), 300)

    def test_it_turns_the_other_way_for_a_negative_angle(self) -> None:
        plant = RowPlant(yaw_deg=-10.0)

        residual = self.run_it(plant, tolerance_deg=1.0)

        self.assertLessEqual(abs(residual), 1.0)
        self.assertLess(plant.moves[0], 0)

    def test_the_measured_response_determines_turn_direction(self) -> None:
        plant = RowPlant(yaw_deg=10.0, deg_per_pulse=-0.04)

        residual = self.run_it(plant, tolerance_deg=0.5)

        self.assertLessEqual(abs(residual), 0.5)
        self.assertAlmostEqual(plant.yaw_deg, 0.0, delta=0.5)
        self.assertLess(plant.moves[1], 0)

    def test_box_scale_changes_do_not_tilt_a_horizontal_pixel_row(self) -> None:
        plant = RowPlant(yaw_deg=0.0)
        robot = make_row_robot(plant)
        targets = robot.vision.locate_rings()
        for digit, scale in ((1, 0.3), (2, 0.5), (3, 0.8)):
            target = targets[digit]
            u, v = target.center_px[0], target.center_px[1] + 100.0
            targets[digit] = replace(
                target, center_px=(u, v), scale_mm_per_px=scale,
                x_mm=(u - 640.0) * scale, y_mm=(v - 360.0) * scale,
            )
        robot.vision = Mock()
        robot.vision.locate_rings.return_value = targets
        robot.vision.ring_center_spacing_mm = 150.0

        with patch("time.sleep"):
            residual = app_gripper.square_chassis_to_ring_row(
                robot, yaw_offset_deg=0.0
            )

        self.assertAlmostEqual(residual, 0.0)
        self.assertEqual(plant.moves, [])

    def test_an_already_square_row_moves_nothing(self) -> None:
        plant = RowPlant(yaw_deg=0.0)

        self.run_it(plant)

        self.assertEqual(plant.moves, [])

    def test_the_mounting_offset_is_the_target_angle(self) -> None:
        """相机与车头差 4°：车已经摆正（量到 4°）时一步都不该转。"""
        plant = RowPlant(yaw_deg=4.0)

        residual = self.run_it(plant, yaw_offset_deg=4.0)

        self.assertAlmostEqual(residual, 0.0, places=6)
        self.assertEqual(plant.moves, [])

    def test_it_squares_to_the_offset_instead_of_zero(self) -> None:
        plant = RowPlant(yaw_deg=10.0)

        residual = self.run_it(plant, yaw_offset_deg=8.0, tolerance_deg=0.5)

        self.assertLessEqual(abs(residual), 0.5)
        self.assertAlmostEqual(plant.yaw_deg, 8.0, delta=0.5)

    def test_a_chassis_that_does_not_turn_is_reported(self) -> None:
        plant = RowPlant(yaw_deg=10.0, deg_per_pulse=0.0)

        with self.assertRaisesRegex(RuntimeError, "一点没变"):
            self.run_it(plant, max_iterations=3)

    def test_a_blurred_frame_is_retried_before_giving_up(self) -> None:
        plant = RowPlant(yaw_deg=4.0)
        robot = make_row_robot(plant, blanks=2)

        with patch("time.sleep"):
            residual = app_gripper.square_chassis_to_ring_row(robot, yaw_offset_deg=0.0)

        self.assertLessEqual(abs(residual), app_gripper.RING_ROW_SQUARE_TOLERANCE_DEG)
        self.assertGreaterEqual(len(plant.moves), 1)

    def test_no_usable_row_raises_before_any_wheel_turns(self) -> None:
        plant = RowPlant(yaw_deg=0.0)
        robot = Mock()
        robot.vision.locate_rings.return_value = {}

        with patch("time.sleep"), self.assertRaisesRegex(RuntimeError, "三环直线"):
            app_gripper.square_chassis_to_ring_row(robot)
        self.assertEqual(plant.moves, [])


class RowStepPulsesTests(unittest.TestCase):
    def robot(self, *, lead=61.0) -> Mock:
        robot = Mock()
        robot.arm.pulses_per_revolution = 3200
        robot.arm.config.base_reduction = 5
        robot.arm.config.axes = {"extension": SimpleNamespace(mm_per_motor_rev=lead)}
        return robot

    def test_one_step_is_a_chord_not_an_arc(self) -> None:
        """转底座 + 多伸一截：atan(150/270)=29.05° → 1291 脉冲；多伸 38.9 mm → 2039 脉冲。"""
        base_pulses, extension_pulses = app_gripper._row_step_pulses(
            self.robot(), spacing_mm=150.0, radius_mm=270.0, middle_yaw_deg=0.0
        )

        self.assertEqual(base_pulses, 1291)
        self.assertEqual(extension_pulses, 2039)

    def test_a_missing_extension_lead_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "mm_per_motor_rev"):
            app_gripper._row_step_pulses(
                self.robot(lead=None), spacing_mm=150.0, radius_mm=270.0,
                middle_yaw_deg=0.0,
            )


class PlaceMaterialFromCenterRingTests(unittest.TestCase):
    def test_translated_chassis_keeps_the_row_in_the_chassis_frame(self) -> None:
        """车身方向不变而停车位置平移：落点必须沿固定法线的垂线展开。"""
        normal_motor_degrees = -17.0
        for forward_mm in (245.0, 280.0, 310.0):
            for lateral_mm in (-60.0, 0.0, 55.0):
                with self.subTest(forward=forward_mm, lateral=lateral_mm):
                    radius = math.hypot(forward_mm, lateral_mm)
                    base_degrees = normal_motor_degrees + 5.0 * math.degrees(
                        math.atan2(lateral_mm, forward_mm)
                    )
                    extension_degrees = 270.0 + (radius - 235.0) * 360.0 / 61.0
                    _robot, targets, _grab = self.run_it(
                        (1, 2, 3),
                        base_pulses=round(base_degrees * 3200.0 / 360.0),
                        extension_pulses=round(extension_degrees * 3200.0 / 360.0),
                        row_normal_base_motor_degrees=normal_motor_degrees,
                        row_yaw_correction_deg=0.0,
                    )
                    self.assert_row_geometry(
                        targets, normal_motor_degrees=normal_motor_degrees
                    )

    def test_calibration_returns_to_the_middle_alignment_pose(self) -> None:
        """960° 观察姿态只供转车，环心对准必须回到落料补偿使用的 600°。"""
        robot = Mock()
        robot.vision.ring_center_spacing_mm = 150.0
        robot.arm.config = load_arm_config()
        robot.arm.pulses_per_revolution = 3200
        current = {}

        def move_axes(_robot, pose):
            current.update(pose)
            return Mock()

        def move_axis(axis, degrees):
            current[axis] = degrees

        def square(_robot):
            self.assertEqual(current, app_gripper.RING_ROW_OBSERVE_POSE)

        def align(_robot, **_kwargs):
            self.assertEqual(current, app_gripper.RING2_APPROACH_POSE)
            return -151, 4800

        robot.arm.move_axis_to_motor_degrees.side_effect = move_axis
        with patch.object(app_gripper, "move_axes_to", side_effect=move_axes), \
                patch.object(app_gripper, "square_chassis_to_ring_row", side_effect=square), \
                patch.object(app_gripper, "_align_floor_ring", side_effect=align), \
                patch.object(app_gripper, "_grab_material_from_car"), \
                patch("time.sleep"):
            app_gripper.place_material_from_repo_to_floor_from_center_ring(
                robot, orders=(2,), need_calibrate=True
            )

    def test_failed_chassis_calibration_retracts_before_any_pickup(self) -> None:
        robot = Mock()
        with patch.object(app_gripper, "move_axes_to") as move, \
                patch.object(app_gripper, "square_chassis_to_ring_row",
                             side_effect=RuntimeError("车身未垂直")), \
                patch.object(app_gripper, "_align_floor_ring") as align, \
                patch.object(app_gripper, "_grab_material_from_car") as grab, \
                patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "车身未垂直"):
                app_gripper.place_material_from_repo_to_floor_from_center_ring(
                    robot, need_calibrate=True
                )
        self.assertEqual(move.call_args.args[1], app_gripper.SAFE_RETRACT)
        align.assert_not_called()
        grab.assert_not_called()
        robot.gripper.open_jaws.assert_not_called()

    def test_default_yaw_is_added_to_the_measured_middle_angle(self) -> None:
        _robot, targets, _grab = self.run_it(
            (1, 2, 3), base_pulses=-419, extension_pulses=3645,
            row_normal_base_motor_degrees=-17.0,
        )
        self.assert_row_geometry(targets, normal_motor_degrees=-17.0 - 8.5 * 5.0)

    def test_invalid_row_normal_is_rejected_before_motion(self) -> None:
        robot = Mock()
        for angle in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                app_gripper.place_material_from_repo_to_floor_from_center_ring(
                    robot, row_normal_base_motor_degrees=angle
                )
        robot.arm.move_axis_to_motor_degrees.assert_not_called()

    def test_unaligned_middle_uses_the_best_position_for_placement(self) -> None:
        robot = Mock()
        robot.vision.locate_ring.return_value = VISIBLE_MIDDLE_RING
        robot.vision.ring_center_spacing_mm = 150.0
        robot.arm.config = load_arm_config()
        robot.arm.pulses_per_revolution = 3200
        robot.arm.turns_state.of.side_effect = lambda name: SimpleNamespace(
            pulses={"base": 1000, "extension": 4800}[name]
        )
        robot.grasp.align_ring.return_value = RingAlignResult(
            aligned=False, final_error_mm=20.0, note="走满步数"
        )
        with patch.object(app_gripper, "move_axes_to") as move, \
                patch.object(app_gripper, "_grab_material_from_car") as grab, \
                patch("time.sleep"):
            locations = app_gripper.place_material_from_repo_to_floor_from_center_ring(
                robot, orders=(2,)
            )

        self.assertEqual(locations[2], {"base": 112.5, "extension": 540.0})
        grab.assert_called_once_with(robot)
        robot.gripper.open_jaws.assert_called_once_with()
        self.assertEqual(move.call_args.args[1], app_gripper.SAFE_RETRACT)

    def test_alignment_error_is_propagated_after_retract(self) -> None:
        robot = Mock()
        robot.vision.locate_ring.return_value = VISIBLE_MIDDLE_RING
        robot.grasp.align_ring.side_effect = RuntimeError("环心读数不可用")
        with patch.object(app_gripper, "move_axes_to") as move, patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "环心读数不可用"):
                app_gripper.place_material_from_repo_to_floor_from_center_ring(robot)

        robot.gripper.open_jaws.assert_not_called()
        self.assertEqual(move.call_args.args[1], app_gripper.SAFE_RETRACT)

    def run_it(
        self, orders, *, extension_pulses=4800, base_pulses=1000,
        use_default_offsets=False, **kwargs,
    ):
        robot = Mock()
        robot.vision.locate_ring.return_value = VISIBLE_MIDDLE_RING
        robot.vision.ring_center_spacing_mm = 150.0
        robot.arm = RowStepPulsesTests().robot().arm
        robot.arm.config = load_arm_config()
        robot.arm.turns_state = Mock()
        current = {"base": base_pulses, "extension": extension_pulses}
        robot.arm.turns_state.of.side_effect = lambda name: SimpleNamespace(
            pulses=current[name]
        )

        def align_ring(*, require_digit, **_kwargs):
            if require_digit == 2:
                current.update(base=base_pulses, extension=extension_pulses)
            return RingAlignResult(aligned=True, final_error_mm=0.5)

        def move_axes(_robot, pose):
            for axis in ("base", "extension"):
                if axis in pose:
                    current[axis] = round(pose[axis] * 3200 / 360)
            return Mock()

        robot.grasp.align_ring.side_effect = align_ring

        with patch.object(app_gripper, "move_axes_to") as move, \
                patch.object(app_gripper, "_grab_material_from_car") as grab, \
                patch("time.sleep"):
            move.side_effect = move_axes
            # 对称用例显式把中环放在法线上；平移用例会传入独立的固定法线。
            kwargs.setdefault("row_normal_base_motor_degrees", base_pulses * 360.0 / 3200.0)
            if not use_default_offsets:
                kwargs.setdefault("ring3_right_offset_mm", 0.0)
                kwargs.setdefault("ring3_down_offset_mm", 0.0)
            app_gripper.place_material_from_repo_to_floor_from_center_ring(
                robot, orders=orders, **kwargs
            )
        targets = [call.args[1] for call in move.call_args_list]
        placed = [pose for pose in targets if set(pose) == {"base", "extension"}]
        return robot, [targets[0], *placed, targets[-1]], grab

    def test_ring_three_down_offset_is_independent_of_right_offset(self) -> None:
        for yaw in (0.0, -6.5):
            for down in (10.0, -4.0, 0.0, None):
                with self.subTest(yaw=yaw, down=down):
                    offsets = {} if down is None else {"ring3_down_offset_mm": down}
                    robot, targets, _grab = self.run_it(
                        (3, 2, 1), base_pulses=-738, extension_pulses=3294,
                        row_yaw_correction_deg=yaw, use_default_offsets=True,
                        **offsets,
                    )
                    _robot, original, _grab = self.run_it(
                        (3, 2, 1), base_pulses=-738, extension_pulses=3294,
                        row_yaw_correction_deg=yaw,
                    )
                    self.assertEqual(targets[2:4], original[2:4])
                    positions = []
                    for pose in (original[1], targets[1]):
                        radius = 235.0 + (pose["extension"] - 270.0) * 61.0 / 360.0
                        theta = math.radians((pose["base"] + 83.025) / 5.0 + yaw)
                        positions.append((radius * math.cos(theta), radius * math.sin(theta)))
                    before, after = positions
                    expected_down = 7.6 if down is None else down
                    self.assertAlmostEqual(after[0] - before[0], -expected_down, delta=0.15)
                    self.assertAlmostEqual(after[1] - before[1], 10.3, delta=0.15)
                    self.assertEqual(
                        [call.kwargs["require_digit"] for call in robot.grasp.align_ring.call_args_list],
                        [2],
                    )

    def test_invalid_ring_three_down_offset_is_rejected_before_motion(self) -> None:
        robot = Mock()
        for offset in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                app_gripper.place_material_from_repo_to_floor_from_center_ring(
                    robot, ring3_down_offset_mm=offset
                )
        robot.arm.move_axis_to_motor_degrees.assert_not_called()

    def test_ring_three_offset_moves_only_ring_three_along_the_row(self) -> None:
        for yaw in (0.0, -6.5):
            for offset in (15.0, 8.0, -5.0):
                with self.subTest(yaw=yaw, offset=offset):
                    robot, targets, _grab = self.run_it(
                        (3, 2, 1), base_pulses=-738, extension_pulses=3294,
                        row_yaw_correction_deg=yaw, ring3_right_offset_mm=offset,
                    )
                    _robot, original, _grab = self.run_it(
                        (3, 2, 1), base_pulses=-738, extension_pulses=3294,
                        row_yaw_correction_deg=yaw, ring3_right_offset_mm=0.0,
                    )
                    self.assertEqual(targets[2:4], original[2:4])  # 2/1 环不变
                    positions = []
                    for pose in (original[1], targets[1]):
                        radius = 235.0 + (pose["extension"] - 270.0) * 61.0 / 360.0
                        theta = math.radians((pose["base"] + 83.025) / 5.0 + yaw)
                        positions.append((radius * math.cos(theta), radius * math.sin(theta)))
                    before, after = positions
                    self.assertAlmostEqual(after[0], before[0], delta=0.15)
                    self.assertAlmostEqual(after[1] - before[1], offset, delta=0.15)
                    self.assertEqual(
                        [call.kwargs["require_digit"] for call in robot.grasp.align_ring.call_args_list],
                        [2],
                    )

    def test_invalid_ring_three_offset_is_rejected_before_motion(self) -> None:
        robot = Mock()
        for offset in (float("nan"), float("inf")):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                app_gripper.place_material_from_repo_to_floor_from_center_ring(
                    robot, ring3_right_offset_mm=offset
                )
        robot.arm.move_axis_to_motor_degrees.assert_not_called()

    def test_tunable_angle_changes_the_row_direction_with_only_middle_alignment(self) -> None:
        for angle in (0.0, 8.5, -6.5, -13.205):
            with self.subTest(angle=angle):
                robot, targets, grab = self.run_it(
                    (1, 2, 3), base_pulses=-738, extension_pulses=3294,
                    row_yaw_correction_deg=angle, need_calibrate=False,
                )
                self.assertEqual(
                    [call.kwargs["require_digit"] for call in robot.grasp.align_ring.call_args_list],
                    [2],
                )
                self.assertEqual(grab.call_count, 3)
                self.assertAlmostEqual(targets[2]["base"], -83.025)
                self.assertAlmostEqual(targets[2]["extension"], 370.575)
                self.assert_row_geometry(
                    targets, normal_motor_degrees=-83.025 - angle * 5.0
                )

    def test_invalid_angle_is_rejected_before_motion(self) -> None:
        robot = Mock()
        for angle in (float("nan"), float("inf")):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                app_gripper.place_material_from_repo_to_floor_from_center_ring(
                    robot, row_yaw_correction_deg=angle
                )
        robot.arm.move_axis_to_motor_degrees.assert_not_called()

    def test_ring_two_is_aligned_then_the_three_poses_are_on_one_line(self) -> None:
        robot, targets, _grab = self.run_it((1, 2, 3), row_yaw_correction_deg=0.0)

        self.assertEqual(
            [call.kwargs["require_digit"] for call in robot.grasp.align_ring.call_args_list],
            [2],
        )
        self.assertIn("require_digit", robot.grasp.align_ring.call_args.kwargs)
        self.assertEqual(targets[0], app_gripper.RING2_APPROACH_POSE)
        self.assertEqual(targets[-1], app_gripper.SAFE_RETRACT)

        ring1, ring2, ring3 = targets[1:-1]
        # 只有中环恰好在法线上时，两端的伸缩相同，底座对称地转开。
        self.assertAlmostEqual(ring1["extension"], ring3["extension"])
        self.assertGreater(ring1["extension"], ring2["extension"])
        self.assertLess(ring1["base"], ring2["base"])
        self.assertLess(ring2["base"], ring3["base"])
        self.assertAlmostEqual(ring2["base"], 1000 * 360.0 / 3200.0)
        self.assertAlmostEqual(ring2["extension"], 4800 * 360.0 / 3200.0)
        self.assert_row_geometry(targets, normal_motor_degrees=112.5)

    def assert_row_geometry(self, targets, *, normal_motor_degrees=112.5):
        """正向还原末端坐标，检查共线、间距以及相对实测法线的方向。"""
        positions = []
        for pose in targets[1:-1]:
            radius = 235.0 + (pose["extension"] - 270.0) * 61.0 / 360.0
            theta = math.radians((pose["base"] - normal_motor_degrees) / 5.0)
            positions.append((radius * math.cos(theta), radius * math.sin(theta)))
        first, middle, last = positions
        tangent = (last[0] - first[0], last[1] - first[1])
        span = math.hypot(*tangent)
        line_error = abs(
            tangent[0] * (middle[1] - first[1])
            - tangent[1] * (middle[0] - first[0])
        ) / span
        self.assertLess(line_error, 0.1, "三个落点没有共线")
        for side in (first, last):
            self.assertAlmostEqual(math.dist(side, middle), 150.0, delta=0.1)
            self.assertAlmostEqual(side[0], middle[0], delta=0.1)

    def test_row_stays_straight_at_different_aligned_extensions(self) -> None:
        for pulses in (2400, 4800, 6400):
            with self.subTest(extension_pulses=pulses):
                _robot, targets, _grab = self.run_it(
                    (1, 2, 3), extension_pulses=pulses, row_yaw_correction_deg=0.0
                )
                self.assert_row_geometry(targets)

    def test_logged_middle_pose_keeps_both_ends_on_the_actual_ring_row(self) -> None:
        """现场中环不在环排法线上：落点仍须沿实测 -17° 电机角的垂线展开。"""
        _robot, targets, _grab = self.run_it(
            (1, 2, 3), base_pulses=-781, extension_pulses=3566,
            row_normal_base_motor_degrees=-17.0,
            row_yaw_correction_deg=0.0,
        )
        positions = []
        for pose in targets[1:-1]:
            radius = 235.0 + (pose["extension"] - 270.0) * 61.0 / 360.0
            theta = math.radians((pose["base"] - (-17.0)) / 5.0)
            positions.append((radius * math.cos(theta), radius * math.sin(theta)))
        first, middle, last = positions
        self.assertAlmostEqual(first[0], middle[0], delta=0.1)
        self.assertAlmostEqual(last[0], middle[0], delta=0.1)
        self.assertAlmostEqual(first[1], middle[1] - 150.0, delta=0.1)
        self.assertAlmostEqual(last[1], middle[1] + 150.0, delta=0.1)

    def test_the_placements_follow_the_order(self) -> None:
        _robot, targets, grab = self.run_it((3, 1, 2))

        placed = targets[1:-1]
        self.assertEqual(len(placed), 3)
        self.assertGreater(placed[0]["base"], placed[2]["base"])  # 3 号先放
        self.assertGreater(placed[2]["base"], placed[1]["base"])  # 1 号第二趟
        self.assertEqual(grab.call_count, 3)

    def test_a_bad_order_is_rejected_before_anything_moves(self) -> None:
        robot = Mock()
        with patch.object(app_gripper, "move_axes_to") as move:
            with self.assertRaises(ValueError):
                app_gripper.place_material_from_repo_to_floor_from_center_ring(
                    robot, orders=(0, 4)
                )
        move.assert_not_called()


class PlaceAndGrabMaterialTests(unittest.TestCase):
    """使用真实放料、回抓流程，仅替换硬件动作，核对松爪与闭爪的落点。"""

    def setUp(self) -> None:
        self.robot = Mock()
        self.robot.vision.locate_ring.return_value = VISIBLE_MIDDLE_RING
        self.robot.vision.ring_center_spacing_mm = 150.0
        self.robot.arm.config = load_arm_config()
        self.robot.arm.pulses_per_revolution = 3200
        self.robot.arm.turns_state.of.side_effect = lambda name: SimpleNamespace(
            pulses={"base": -738, "extension": 3294}[name]
        )
        self.robot.grasp.align_ring.return_value = RingAlignResult(
            aligned=True, final_error_mm=0.5
        )
        current = {}
        self.placed = []
        self.picked = []

        def move_axes(_robot, pose):
            current.update(pose)
            return Mock()

        def position():
            return {axis: current[axis] for axis in ("base", "extension")}

        self.robot.gripper.open_jaws.side_effect = lambda: self.placed.append(position())
        self.robot.gripper.close_jaws.side_effect = lambda: self.picked.append(position())
        for patcher in (
            patch.object(app_gripper, "move_axes_to", side_effect=move_axes),
            patch.object(app_gripper, "square_chassis_to_ring_row"),
            patch.object(app_gripper, "_grab_material_from_car"),
            patch.object(app_gripper, "_place_material_to_repo"),
            patch("time.sleep"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_returned_locations_are_motor_degrees_keyed_by_placed_ring(self) -> None:
        orders = (3, 2)
        locations = app_gripper.place_material_from_repo_to_floor_from_center_ring(
            self.robot, orders=orders
        )

        self.assertEqual(set(locations), {2, 3})
        self.assertEqual([locations[digit] for digit in orders], self.placed)
        self.assertAlmostEqual(locations[2]["base"], -83.025)
        self.assertAlmostEqual(locations[2]["extension"], 370.575)

    def test_default_grab_order_reuses_all_placed_locations_without_xy_alignment(self) -> None:
        app_gripper.place_and_grab_material(self.robot)

        self.assertEqual(len(self.placed), 3)
        self.assertEqual(self.picked, self.placed)
        self.robot.grasp.align_xy.assert_not_called()
        self.assertEqual(
            [call.kwargs["require_digit"] for call in self.robot.grasp.align_ring.call_args_list],
            [2],
        )

    def test_different_grab_orders_follow_the_matching_ring_locations(self) -> None:
        place_orders = (3, 1, 2)
        for grab_orders in ((1, 2, 3), (2, 3, 1)):
            with self.subTest(grab_orders=grab_orders):
                self.placed.clear()
                self.picked.clear()
                app_gripper.place_and_grab_material(
                    self.robot, place_orders=place_orders, grab_orders=grab_orders
                )

                locations = dict(zip(place_orders, self.placed))
                self.assertEqual(self.picked, [locations[digit] for digit in grab_orders])

    def test_unaligned_middle_continues_placement_and_pickup_at_best_locations(self) -> None:
        self.robot.grasp.align_ring.return_value = RingAlignResult(
            aligned=False, final_error_mm=20.0
        )

        app_gripper.place_and_grab_material(self.robot)

        self.assertEqual(len(self.placed), 3)
        self.assertEqual(self.picked, self.placed)
        self.robot.grasp.align_xy.assert_not_called()


if __name__ == "__main__":
    unittest.main()

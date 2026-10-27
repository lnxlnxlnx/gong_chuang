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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import app.gripper as app_gripper
from modules.vision.drive.numbered_ring import (
    RingTarget,
    RingTargetLocator,
    derive_ring_row,
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
                center_px=(640.0, 360.0),
                radius_px=200.0,
                digit=digit,
                score=0.9,
                clipped=False,
                x_mm=(digit - 2) * 150.0 * unit[0],
                y_mm=(digit - 2) * 150.0 * unit[1],
                scale_mm_per_px=0.2,
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
        # 第一步试转 300 脉冲（10° → -2°），第二步按学到的 -0.04°/脉冲退回 50。
        self.assertEqual(plant.moves, [app_gripper.ROW_ROTATE_TRIAL_PULSES, -50])

    def test_it_turns_the_other_way_for_a_negative_angle(self) -> None:
        plant = RowPlant(yaw_deg=-10.0)

        residual = self.run_it(plant, tolerance_deg=1.0)

        self.assertLessEqual(abs(residual), 1.0)
        self.assertLess(plant.moves[0], 0)

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


class PlaceBoxFromRepoToFloorV3Tests(unittest.TestCase):
    def run_it(self, orders):
        robot = Mock()
        robot.vision.ring_center_spacing_mm = 150.0
        robot.grasp.align_ring.return_value = SimpleNamespace(
            aligned=True,
            final_reading=(0.0, 0.0),
            final_error_mm=0.5,
            scale_source="ring box",
            response=None,
            note="",
        )
        robot.grasp.plan_ring_readings.return_value = {
            digit: SimpleNamespace(
                base=10.0 * digit,
                extension=20.0 * digit,
                summary=lambda digit=digit: f"plan {digit}",
            )
            for digit in (1, 2, 3)
        }

        with patch.object(app_gripper, "_move_axes_to") as move, \
                patch.object(app_gripper, "_grab_box_from_car") as grab, \
                patch("time.sleep"):
            move.return_value.wait.return_value = None
            app_gripper.place_box_from_repo_to_floor_v3(robot, orders=orders)
        return robot, [call.args[1] for call in move.call_args_list], grab

    def test_aligning_ring_two_then_the_three_placements_in_order(self) -> None:
        robot, targets, grab = self.run_it((3, 1, 2))

        robot.grasp.align_ring.assert_called_once()
        self.assertIn("require_digit", robot.grasp.align_ring.call_args.kwargs)
        robot.grasp.plan_ring_readings.assert_called_once_with(
            {
                1: (-150.0, 0.0),
                2: (0.0, 0.0),
                3: (150.0, 0.0),
            },
            scale_source="ring box",
            offset_digit=2,
        )
        self.assertEqual(
            targets,
            [
                app_gripper.RING2_APPROACH_POSE,
                {"extension": 60.0, "base": 30.0},
                {"extension": 20.0, "base": 10.0},
                {"extension": 40.0, "base": 20.0},
                app_gripper.SAFE_RETRACT,
            ],
        )
        self.assertEqual(grab.call_count, 3)
        self.assertEqual(robot.gripper.open_jaws.call_count, 3)

    def test_a_bad_order_is_rejected_before_anything_moves(self) -> None:
        robot = Mock()
        with patch.object(app_gripper, "_move_axes_to") as move:
            with self.assertRaises(ValueError):
                app_gripper.place_box_from_repo_to_floor_v3(robot, orders=(0, 4))
        move.assert_not_called()


if __name__ == "__main__":
    unittest.main()

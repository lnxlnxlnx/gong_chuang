"""回放被画面裁切的检测框，验证底盘取景后才允许机械臂对准。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app import gripper
from modules.grasp.drive.ring_plan import RingAlignResult
from modules.vision.config import RingTargetSettings, StationGeometrySettings
from modules.vision.drive.numbered_ring import RingTargetLocator
from modules.vision.drive.yolov8 import Detection


class ViewPlant:
    """底盘位移改变真实框位置，检测器只返回画面内的可见框。"""

    class_names = ["tg1", "tg2", "tg3"]

    def __init__(self, x=320.0, y=240.0, *, response=0.1, digit=2):
        self.x, self.y = x, y
        self.response = response
        self.digit = digit
        self.moves = []
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.locator = RingTargetLocator(
            self, settings=RingTargetSettings(refine_center=False),
        )

    def detect(self, _frame):
        return [Detection(
            x1=max(0.0, self.x - 80), y1=max(0.0, self.y - 80),
            x2=min(640.0, self.x + 80), y2=min(480.0, self.y + 80),
            score=0.95, class_id=self.digit - 1, name=f"tg{self.digit}",
        )]

    def locate_ring(self, *, digit=None):
        return self.locator.detect(self.frame, digit=digit)

    def _move(self, name, dx, dy, pulses):
        self.moves.append((name, pulses))
        self.x += dx * pulses * self.response
        self.y += dy * pulses * self.response

    def strafe_left(self, *, rpm, pulses):
        self._move("left", 1, 0, pulses)

    def strafe_right(self, *, rpm, pulses):
        self._move("right", -1, 0, pulses)

    def forward(self, *, rpm, pulses):
        self._move("forward", 0, 1, pulses)

    def backward(self, *, rpm, pulses):
        self._move("backward", 0, -1, pulses)

    def robot(self):
        return SimpleNamespace(chassis=self, vision=self)


class FloorRingViewTests(unittest.TestCase):
    def test_each_border_is_cleared_by_the_chassis(self):
        for x, y, direction in (
            (40, 240, "left"), (600, 240, "right"),
            (320, 40, "forward"), (320, 440, "backward"),
        ):
            with self.subTest(direction=direction), patch("time.sleep"):
                plant = ViewPlant(x, y)
                self.assertIsNone(plant.locate_ring(digit=2).reading())
                gripper._ensure_floor_ring_in_view(plant.robot(), digit=2)
                self.assertIsNotNone(plant.locate_ring(digit=2).reading())
                self.assertTrue(plant.moves)
                self.assertTrue(all(name == direction for name, _ in plant.moves))
                self.assertTrue(all(pulses == 300 for _, pulses in plant.moves))

    def test_a_corner_uses_both_translation_axes(self):
        plant = ViewPlant(600, 40)
        with patch("time.sleep"):
            gripper._ensure_floor_ring_in_view(plant.robot(), digit=2)
        self.assertIsNotNone(plant.locate_ring(digit=2).reading())
        self.assertEqual({name for name, _ in plant.moves}, {"right", "forward"})

    def test_a_visible_off_center_ring_does_not_move_the_chassis(self):
        plant = ViewPlant(470, 330)
        gripper._ensure_floor_ring_in_view(plant.robot(), digit=2)
        self.assertEqual(plant.moves, [])

    def test_the_wrong_digit_never_moves_the_chassis(self):
        plant = ViewPlant(40, 240, digit=3)
        with patch("time.sleep"), self.assertRaisesRegex(RuntimeError, "2 号环"):
            gripper._ensure_floor_ring_in_view(plant.robot(), digit=2)
        self.assertEqual(plant.moves, [])

    def test_losing_the_locked_ring_stops_after_the_first_move(self):
        plant = ViewPlant(40, 240)
        original = plant.strafe_left

        def switch_ring(**kwargs):
            original(**kwargs)
            plant.digit = 3

        plant.strafe_left = switch_ring
        with patch("time.sleep"), self.assertRaisesRegex(RuntimeError, "2 号环"):
            gripper._ensure_floor_ring_in_view(plant.robot(), digit=2)
        self.assertEqual(len(plant.moves), 1)

    def test_no_response_or_reversed_response_stops_after_one_move(self):
        for response in (0.0, -0.02):
            with self.subTest(response=response), patch("time.sleep"):
                plant = ViewPlant(40, 240, response=response)
                with self.assertRaisesRegex(RuntimeError, "响应"):
                    gripper._ensure_floor_ring_in_view(plant.robot(), digit=2)
                self.assertEqual(len(plant.moves), 1)

    def test_the_last_allowed_move_is_observed(self):
        plant = ViewPlant(60, 240)
        with patch("time.sleep"):
            gripper._ensure_floor_ring_in_view(plant.robot(), digit=2, max_iterations=1)
        self.assertEqual(len(plant.moves), 1)
        self.assertIsNotNone(plant.locate_ring(digit=2).reading())

    def test_exhausting_the_move_budget_is_an_error(self):
        plant = ViewPlant(10, 240, response=0.01)
        with patch("time.sleep"), self.assertRaisesRegex(RuntimeError, "仍.*贴边"):
            gripper._ensure_floor_ring_in_view(plant.robot(), digit=2, max_iterations=2)
        self.assertEqual(len(plant.moves), 2)

    def test_a_ring_larger_than_the_frame_never_moves_the_chassis(self):
        plant = ViewPlant()
        plant.detect = lambda _frame: [Detection(
            x1=0, y1=120, x2=640, y2=360,
            score=0.95, class_id=1, name="tg2",
        )]
        with self.assertRaisesRegex(RuntimeError, "尺寸超出视野"):
            gripper._ensure_floor_ring_in_view(plant.robot(), digit=2)
        self.assertEqual(plant.moves, [])

    def test_a_visible_ring_without_scale_is_not_allowed_to_align(self):
        plant = ViewPlant()
        plant.locator = RingTargetLocator(
            plant, settings=RingTargetSettings(
                refine_center=False, outer_diameter_mm=None,
            ), geometry=StationGeometrySettings(mm_per_px=None),
        )
        with self.assertRaisesRegex(RuntimeError, "比例尺"):
            gripper._ensure_floor_ring_in_view(plant.robot(), digit=2)
        self.assertEqual(plant.moves, [])

    def test_failed_view_adjustment_prevents_pickup_and_retracts_the_arm(self):
        plant = ViewPlant(40, 240, response=0.0)
        robot = plant.robot()
        robot.arm = Mock()
        robot.grasp = Mock()
        robot.gripper = Mock()
        with patch("time.sleep"), patch.object(gripper, "move_axes_to") as move, \
                patch.object(gripper, "_grab_material_from_car") as pickup:
            with self.assertRaisesRegex(RuntimeError, "响应"):
                gripper.place_material_from_repo_to_floor_from_center_ring(robot)
        pickup.assert_not_called()
        robot.grasp.align_ring.assert_not_called()
        robot.gripper.open_jaws.assert_not_called()
        self.assertEqual(move.call_args.args[1], gripper.SAFE_RETRACT)

    def test_alignment_starts_only_after_the_ring_is_fully_visible(self):
        plant = ViewPlant(600, 40)
        robot = plant.robot()
        robot.arm = Mock()
        robot.arm.turns_state.of.side_effect = lambda axis: SimpleNamespace(
            pulses={"base": 100, "extension": 200}[axis],
        )
        robot.grasp = Mock()

        def align(**_kwargs):
            self.assertIsNotNone(plant.locate_ring(digit=2).reading())
            self.assertTrue(plant.moves)
            return RingAlignResult(aligned=True, final_error_mm=0.5)

        robot.grasp.align_ring.side_effect = align
        with patch("time.sleep"):
            result = gripper._align_floor_ring(robot, digit=2, tolerance_mm=4)
        self.assertEqual(result, (100, 200))


if __name__ == "__main__":
    unittest.main()

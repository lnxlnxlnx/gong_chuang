"""环排不可见时主动调整机械臂，并保持车头校准的角度基准。"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import gripper
from modules.arm.config import load_arm_config
from modules.arm.drive.kinematics import TravelLimit
from tests.test_ring_row import RowPlant, RowVision


class RingRowViewTests(unittest.TestCase):
    def test_iteration_limit_returns_residual_and_continues_placement(self):
        robot, pose, moves, move = self.make_scene(
            yaw=14.0, visible=lambda pose: True,
        )
        robot.arm.pulses_per_revolution = 3200
        robot.gripper = Mock()
        square = gripper.square_chassis_to_ring_row
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move), \
                patch.object(gripper, "square_chassis_to_ring_row", side_effect=lambda robot: square(
                    robot, yaw_offset_deg=4.0, max_iterations=1,
                )), \
                patch.object(gripper, "_align_floor_ring", return_value=(-151, 4800)), \
                patch.object(gripper, "_grab_material_from_car"), \
                patch.object(gripper, "_row_step_pulses", return_value=(10, 20)) as step:
            locations = gripper.place_material_from_repo_to_floor_from_center_ring(
                robot, orders=(1, 2, 3), need_calibrate=True,
                row_yaw_correction_deg=6.0, row_normal_base_motor_degrees=-32.0,
            )
        self.assertEqual(set(locations), {1, 2, 3})
        self.assertEqual(len(robot.chassis.moves), 1)
        expected_yaw = (-151 * 360.0 / 3200 + 32.0) / robot.arm.config.base_reduction + 6.0
        self.assertEqual(step.call_count, 2)
        for call in step.call_args_list:
            self.assertAlmostEqual(call.kwargs["middle_yaw_deg"], expected_yaw)

    def make_scene(self, yaw=4.0, visible=lambda pose: False):
        plant = RowPlant(yaw_deg=yaw)
        pose = {"extension": 550.0, "base": -32.0}
        moves = []
        robot = SimpleNamespace(chassis=plant, arm=Mock(), vision=Mock())
        robot.arm.config = load_arm_config()
        robot.arm.read_abs_state.side_effect = lambda axis: SimpleNamespace(
            motor_degrees=pose[axis],
        )
        robot.vision.ring_center_spacing_mm = 150.0

        def locate():
            camera_yaw = plant.yaw_deg - (pose["base"] + 32.0) / 5.0
            targets = RowVision(RowPlant(camera_yaw)).locate_rings()
            return targets if visible(pose) else {2: targets[2]}

        def move(_robot, target):
            pose.update(target)
            moves.append(dict(target))
            return Mock()

        robot.vision.locate_rings.side_effect = locate
        return robot, pose, moves, move

    def test_extension_adjustment_recovers_the_row(self):
        robot, pose, moves, move = self.make_scene(
            visible=lambda pose: pose["extension"] < 550,
        )
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            residual = gripper.square_chassis_to_ring_row(robot, yaw_offset_deg=4.0)
        self.assertAlmostEqual(residual, 0.0)
        self.assertLess(pose["extension"], 550)
        self.assertTrue(moves)
        self.assertEqual(robot.chassis.moves, [])

    def test_base_adjustment_does_not_rotate_an_already_square_chassis(self):
        robot, pose, moves, move = self.make_scene(
            visible=lambda pose: pose["base"] != -32,
        )
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            residual = gripper.square_chassis_to_ring_row(robot, yaw_offset_deg=4.0)
        self.assertAlmostEqual(residual, 0.0)
        self.assertNotEqual(pose["base"], -32)
        self.assertEqual(robot.chassis.moves, [])

    def test_extension_search_extends_before_trying_retraction(self):
        robot, pose, moves, move = self.make_scene(
            visible=lambda pose: pose["extension"] >= 730.0,
        )
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            residual = gripper.square_chassis_to_ring_row(robot, yaw_offset_deg=4.0)
        self.assertAlmostEqual(residual, 0.0)
        self.assertEqual([target["extension"] for target in moves], [640.0, 730.0])
        self.assertEqual(robot.chassis.moves, [])

    def test_combined_search_uses_extension_increase(self):
        robot, pose, moves, move = self.make_scene(
            visible=lambda pose: (
                pose["extension"] > 550.0 and pose["base"] != -32.0
            ),
        )
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            residual = gripper.square_chassis_to_ring_row(robot, yaw_offset_deg=4.0)
        self.assertAlmostEqual(residual, 0.0)
        self.assertGreater(pose["extension"], 550.0)
        self.assertNotEqual(pose["base"], -32.0)
        self.assertEqual(robot.chassis.moves, [])

    def test_chassis_feedback_uses_the_original_angle_after_base_adjustment(self):
        robot, pose, moves, move = self.make_scene(
            yaw=14.0, visible=lambda pose: pose["base"] != -32,
        )
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            residual = gripper.square_chassis_to_ring_row(robot, yaw_offset_deg=4.0)
        self.assertLessEqual(abs(residual), 1.0)
        self.assertAlmostEqual(robot.chassis.yaw_deg, 4.0, delta=1.0)

    def test_exhausted_search_restores_the_original_observation_pose(self):
        robot, pose, moves, move = self.make_scene()
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            with self.assertRaisesRegex(RuntimeError, "三环直线"):
                gripper.square_chassis_to_ring_row(robot)
        self.assertTrue(moves)
        self.assertEqual(pose, {"extension": 550.0, "base": -32.0})
        self.assertEqual(robot.chassis.moves, [])

    def test_missing_scale_is_not_treated_as_a_view_problem(self):
        robot, pose, moves, move = self.make_scene(visible=lambda pose: True)
        targets = robot.vision.locate_rings()
        targets[2] = replace(targets[2], scale_mm_per_px=None)
        robot.vision.locate_rings.side_effect = None
        robot.vision.locate_rings.return_value = targets
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            with self.assertRaisesRegex(RuntimeError, "比例尺"):
                gripper.square_chassis_to_ring_row(robot)
        self.assertEqual(moves, [])

    def test_reframing_during_chassis_rotation_preserves_feedback(self):
        robot, pose, moves, move = self.make_scene(
            yaw=14.0,
            visible=lambda pose: (
                robot.chassis.yaw_deg > 8.0 or pose["base"] != -32
            ),
        )
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            residual = gripper.square_chassis_to_ring_row(robot, yaw_offset_deg=4.0)
        self.assertLessEqual(abs(residual), 1.0)
        self.assertAlmostEqual(robot.chassis.yaw_deg, 4.0, delta=1.0)
        self.assertTrue(moves)

    def test_search_does_not_command_positions_outside_hardware_limits(self):
        robot, pose, moves, move = self.make_scene(
            visible=lambda pose: pose["base"] != -32,
        )
        robot.arm.config = replace(
            robot.arm.config, radial_limit=TravelLimit(280.0, 285.0),
        )
        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move):
            gripper.square_chassis_to_ring_row(robot, yaw_offset_deg=4.0)
        self.assertTrue(moves)
        self.assertTrue(all(target["extension"] == 550.0 for target in moves))

    def test_placement_returns_to_the_calibrated_middle_alignment_pose(self):
        robot, pose, moves, move = self.make_scene(
            visible=lambda pose: pose["base"] != -32,
        )
        robot.arm.pulses_per_revolution = 3200
        robot.gripper = Mock()

        def align(_robot, **_kwargs):
            for axis, degrees in gripper.RING2_APPROACH_POSE.items():
                self.assertEqual(pose[axis], degrees)
            return -151, 4800

        with patch("time.sleep"), \
                patch.object(gripper, "move_axes_to", side_effect=move), \
                patch.object(gripper, "_align_floor_ring", side_effect=align), \
                patch.object(gripper, "_grab_material_from_car"):
            locations = gripper.place_material_from_repo_to_floor_from_center_ring(
                robot, orders=(2,), need_calibrate=True,
            )
        self.assertEqual(set(locations), {2})
        self.assertTrue(any(target["base"] != -32 for target in moves))


if __name__ == "__main__":
    unittest.main()

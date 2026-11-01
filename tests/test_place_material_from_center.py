"""验证中心物料叠放的视觉次数、平面几何与搬运顺序。"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import gripper
from modules.arm.config import load_arm_config


class PlaceMaterialFromCenterTests(unittest.TestCase):
    def run_flow(self, orders=(3, 1, 2), *, base_degrees=-32.0):
        robot = Mock()
        robot.arm.config = config = load_arm_config()
        robot.arm.pulses_per_revolution = ppr = config.pulses_per_revolution
        state = {"base": 0.0, "extension": 270.0, "lift": 95.0}
        released = []
        events = []

        def move_axis(axis, degrees):
            state[axis] = degrees

        def move_axes(_robot, pose):
            if "base" in pose:
                self.assertEqual(state["lift"], 950)
            state.update(pose)
            return Mock()

        def align(**kwargs):
            self.assertEqual(state["lift"], 480)
            state.update(base=base_degrees, extension=540.0)
            events.append("align")
            return Mock()

        def grab(_robot):
            self.assertEqual(state["lift"], 950)
            events.append("grab")
            # 取料改变关节，返回落点必须使用对准时保存的绝对坐标。
            state.update(base=475.0, extension=400.0, lift=760.0)

        def release():
            self.assertEqual(state["lift"], 480)
            events.append("release")
            released.append(dict(state))

        robot.arm.move_axis_to_motor_degrees.side_effect = move_axis
        robot.arm.turns_state.of.side_effect = lambda axis: SimpleNamespace(
            pulses=round(state[axis] * ppr / 360)
        )
        robot.grasp.align_xy.side_effect = align
        robot.gripper.open_jaws.side_effect = release
        with patch.object(gripper, "move_axes_to", side_effect=move_axes), \
                patch.object(gripper, "_grab_material_from_car", side_effect=grab):
            poses = gripper.place_material_on_material_from_center(
                robot, orders=orders, row_yaw_correction_deg=0.0,
            )
        robot.grasp.align_xy.assert_called_once_with(
            tolerance_mm=2.0, max_iterations=4, return_best=True,
        )
        robot.grasp.align_ring.assert_not_called()
        self.assertEqual(events, ["align"] + ["grab", "release"] * len(orders))
        self.assertEqual(list(poses), list(orders))
        for digit, pose in zip(orders, released):
            self.assertEqual(poses[digit], {k: pose[k] for k in ("base", "extension")})
        self.assertEqual(state["lift"], 950)
        return robot, poses

    def test_only_center_is_seen_and_side_targets_are_150mm_apart(self):
        for base in (-82.0, 110.0):
            with self.subTest(base=base):
                robot, poses = self.run_flow(base_degrees=base)
                config = robot.arm.config
                points = {}
                for digit, pose in poses.items():
                    radius = config.extension_radius_mm(round(
                        pose["extension"] * config.pulses_per_revolution / 360
                    ))
                    yaw = math.radians((pose["base"] + 17.0) / config.base_reduction)
                    points[digit] = radius * math.cos(yaw), radius * math.sin(yaw)
                for digit in (1, 3):
                    self.assertAlmostEqual(points[digit][0], points[2][0], delta=0.1)
                    self.assertAlmostEqual(
                        points[digit][1] - points[2][1], (digit - 2) * 150,
                        delta=0.1,
                    )

    def test_center_is_aligned_even_when_only_a_side_is_placed(self):
        self.run_flow(orders=(1,))

    def test_alignment_failure_stops_before_pickup(self):
        robot = Mock()
        robot.grasp.align_xy.side_effect = RuntimeError("无法识别中心")
        with patch.object(gripper, "move_axes_to") as move, \
                patch.object(gripper, "_grab_material_from_car") as grab:
            with self.assertRaisesRegex(RuntimeError, "无法识别中心"):
                gripper.place_material_on_material_from_center(robot)
        grab.assert_not_called()
        robot.gripper.open_jaws.assert_not_called()
        self.assertEqual(move.call_args.args[1], gripper.SAFE_RETRACT)

    def test_invalid_arguments_do_not_move_hardware(self):
        for kwargs in ({"orders": (0, 4)},
                       {"row_yaw_correction_deg": float("nan")},
                       {"row_normal_base_motor_degrees": float("inf")}):
            with self.subTest(kwargs=kwargs):
                robot = Mock()
                with self.assertRaises(ValueError):
                    gripper.place_material_on_material_from_center(robot, **kwargs)
                robot.arm.move_axis_to_motor_degrees.assert_not_called()


if __name__ == "__main__":
    unittest.main()

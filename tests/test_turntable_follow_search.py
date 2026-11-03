"""转盘抓料区、丢料回中与检测抖动的行为验证。"""

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app.gripper as app_gripper
from modules.grasp import Grasp

from modules.grasp.drive import eye_in_hand as algorithms
from modules.grasp.drive.visual_servo_model import (
    MaterialSelector, VisualServoConfig, VisualServoError, VisualServoTarget,
)
from modules.grasp.test.unit.test_continuous_follow import JACOBIAN, VelocityArm, WorldLink, follow
from modules.grasp.test.unit.test_eye_in_hand import FollowClock


class GrabZoneArm(VelocityArm):
    def __init__(self, clock, base=420):
        super().__init__(clock)
        self.position[0] = base * self.pulses_per_revolution / 360
        self.recenters = []

    def read_abs_state(self, axis):
        self._advance()
        index = 0 if axis == "base" else 1
        return SimpleNamespace(motor_degrees=self.position[index] * 360 / self.pulses_per_revolution)

    def move_axis_to_motor_degrees(self, axis, degrees):
        self._advance()
        self.recenters.append((axis, degrees))
        self.position[0 if axis == "base" else 1] = degrees * self.pulses_per_revolution / 360


class GrabZoneLink(WorldLink):
    def material_point(self, stamp):
        x, y = super().material_point(stamp)
        angle = -math.radians(420 / 5)
        return x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)


class TurntableFollowSearchTests(unittest.TestCase):
    def test_cancelled_grab_resumes_at_current_position_after_temporary_loss(self):
        """重试首帧丢失不能把机械臂带回默认区域，目标再现后应原地接续。"""
        clock = FollowClock()
        arm = GrabZoneArm(clock, base=600)
        link = GrabZoneLink(arm, stop_at=0)
        radius = link.origin_radius
        angle = -math.radians(600 / 5)
        link.material_point = lambda stamp: (radius * math.cos(angle), radius * math.sin(angle))
        grasp = Grasp(
            config=VisualServoConfig(inter_frame_seconds=0.01),
            selector=MaterialSelector(), target=VisualServoTarget(0, 0),
            jacobian=JACOBIAN, lift_scale=None,
        )
        grasp._arm, grasp._source = arm, link
        robot = SimpleNamespace(arm=arm, grasp=grasp, gripper=Mock())
        attempts = []

        def descend(_robot, color, **_kwargs):
            attempts.append((color, clock.now))
            if len(attempts) == 1:
                link.missing = ((clock.now, clock.now + 0.24),)
                return False
            return True

        with patch.object(algorithms, "time", clock), patch.object(app_gripper, "time", clock):
            with patch.object(app_gripper, "_descend_and_grab_from_turntable_monitored", side_effect=descend):
                with patch.object(app_gripper, "_stow_material_to_repo") as stow:
                    with patch.object(app_gripper, "move_axes_to"):
                        app_gripper.grab_material_from_turntable_to_repo_follow_v2(
                            robot, ("green",), home=None, follow_seconds=4, stable_seconds=0.1,
                        )
        self.assertEqual([color for color, _ in attempts], ["green", "green"])
        stow.assert_called_once()
        retry_bases = [
            sample.base_pulses * 360 / arm.pulses_per_revolution
            for sample in arm.samples if sample.stamp >= attempts[0][1]
        ]
        self.assertTrue(retry_bases)
        self.assertTrue(all(abs(base - 600) <= 1 for base in retry_bases), retry_bases)

    def run_follow(self, arm, link, **kwargs):
        with patch.object(algorithms, "time", arm.clock):
            return follow(
                arm, link, min_follow_seconds=0, base_motor_limits=(220, 680),
                search_base_degrees=420, **kwargs,
            )

    def test_outside_start_returns_to_420_before_starting_bounded_follow(self):
        clock = FollowClock()
        arm = GrabZoneArm(clock, base=900)
        link = GrabZoneLink(arm, stop_at=0)
        self.assertTrue(self.run_follow(arm, link, timeout=3).aligned)
        self.assertEqual(arm.recenters, [("base", 420)])
        self.assertEqual(arm.options["motor_limits"]["base"], (220, 680))

    def test_missing_material_returns_to_center_and_waits_there(self):
        clock = FollowClock()
        arm = GrabZoneArm(clock, base=650)
        link = GrabZoneLink(arm, stop_at=0, missing=((0, 10),))
        with self.assertRaises(VisualServoError):
            self.run_follow(arm, link, timeout=4)
        self.assertAlmostEqual(arm.read_abs_state("base").motor_degrees, 420, delta=1)
        self.assertTrue(all(e == 0 for _, _, e in arm.commands))
        bases = [p.base_pulses * 360 / arm.pulses_per_revolution for p in arm.samples]
        self.assertGreaterEqual(min(bases), 220)
        self.assertLessEqual(max(bases), 680)

    def test_target_seen_again_near_center_can_be_grabbed(self):
        clock = FollowClock()
        arm = GrabZoneArm(clock, base=650)
        link = GrabZoneLink(arm, stop_at=0, missing=((0, 0.7),))
        self.assertTrue(self.run_follow(arm, link, timeout=4).aligned)
        self.assertLess(arm.read_abs_state("base").motor_degrees, 650)

    def test_visible_target_beyond_grab_zone_is_not_chased(self):
        clock = FollowClock()
        arm = GrabZoneArm(clock)
        link = GrabZoneLink(arm, stop_at=0)
        radius = link.origin_radius
        angle = -math.radians(850 / 5)
        link.material_point = lambda stamp: (radius * math.cos(angle), radius * math.sin(angle))
        with self.assertRaises(VisualServoError):
            self.run_follow(arm, link, timeout=1)
        self.assertAlmostEqual(arm.read_abs_state("base").motor_degrees, 420, delta=1)
        self.assertTrue(all(b == 0 for _, b, _ in arm.commands))

    def test_small_horizontal_jitter_does_not_reverse_base_while_extension_aligns(self):
        clock = FollowClock()
        arm = VelocityArm(clock)
        link = WorldLink(arm, stop_at=0)
        x, y = link.material_point(0)
        link.material_point = lambda stamp: (x + 35, y)
        original = link.frame_fast

        def jittered(**kwargs):
            result = original(**kwargs)
            return (replace(result[0], x_mm=result[0].x_mm + 1.99 * (-1) ** link.reads),)

        link.frame_fast = jittered
        with patch.object(algorithms, "time", clock):
            # 严格停稳判据可以拒绝这组噪声；不能把“不确定”变成底座往返运动。
            with self.assertRaises(VisualServoError):
                follow(arm, link, timeout=3, min_follow_seconds=0)
        self.assertTrue(any(e != 0 for _, _, e in arm.commands))
        self.assertTrue(all(b == 0 for _, b, _ in arm.commands))

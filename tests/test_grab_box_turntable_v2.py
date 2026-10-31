"""``grab_material_from_turntable_to_repo_follow``：跟着料走，停住了才抓。

守的是这一版与旧版唯一的不同：抓之前是**一直跟**（发完动作就取下一帧、不等动作走完），
不是站在原地等。料在动的时候也可能一帧都看不见（转出去/被爪影挡），那不算失败；只有跟满
时限还没停住才报错。所有动作都是 Mock，不开串口也不开相机。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app.gripper as app_gripper


def tracking_result():
    """``Grasp.follow_material`` 的返回值：料停住那一刻的读数（打印用得到）。"""
    return SimpleNamespace(readings=((0.4, -0.5),), error_mm=0.64, iterations=7)


def make_robot() -> Mock:
    robot = Mock()
    robot.vision.scene_stable.return_value = True
    robot.grasp.follow_material.return_value = tracking_result()
    # 真实会话里这一项是 ``MotorTurns``；Mock 会返回 Mock，和 460 比大小会炸。
    robot.arm.read_abs_state.return_value.motor_degrees = 760.0
    return robot


class FollowUntilStableTests(unittest.TestCase):
    def test_it_delegates_to_the_tracking_loop(self) -> None:
        """跟踪循环在会话里（发完就走、不等动作），这里只确认参数原样交下去。"""
        robot = make_robot()

        app_gripper._follow_material_until_stable(robot, "red", timeout=3.0)

        robot.grasp.follow_material.assert_called_once_with(
            "red", timeout=3.0, min_follow_seconds=4.0, stable_seconds=0.5,
        )
        robot.vision.scene_stable.assert_not_called()

    def test_a_tracking_failure_stops_the_run(self) -> None:
        robot = make_robot()
        robot.grasp.follow_material.side_effect = RuntimeError("画面里没有物块")

        with self.assertRaisesRegex(RuntimeError, "没有物块"):
            app_gripper._follow_material_until_stable(robot, "red", timeout=0.05)


class FollowGrabGateTests(unittest.TestCase):
    def test_stopped_material_is_grabbed_without_realigning_or_scene_check(self):
        from modules.grasp.drive import eye_in_hand as algorithms
        from modules.grasp.test.unit.test_continuous_follow import (
            VelocityArm, WorldLink, follow,
        )
        from modules.grasp.test.unit.test_eye_in_hand import FollowClock

        robot = make_robot()
        clock = FollowClock()
        arm = VelocityArm(clock)
        link = WorldLink(arm, stop_at=4.2)
        robot.grasp.follow_material.side_effect = (
            lambda _color, **kw: follow(arm, link, **kw)
        )
        robot.vision.scene_stable.return_value = False
        with (
            patch.object(algorithms, "time", clock),
            patch.object(app_gripper, "_stow_material_to_repo"),
            patch.object(app_gripper, "_go_home"),
            patch.object(app_gripper, "move_axes_to"),
            patch("builtins.input", return_value=""),
        ):
            app_gripper.grab_material_from_turntable_to_repo_follow(
                robot, ("red",), home=None,
            )
        robot.gripper.close_jaws.assert_called_once()
        robot.grasp.align_xy_for_turntable.assert_not_called()
        robot.vision.scene_stable.assert_not_called()
        self.assertGreaterEqual(clock.now, 4.7)
        self.assertLess(clock.now, 4.95)


class PlaceAfterFollowTests(unittest.TestCase):
    def run_it(self, want_colors=("red",)):
        robot = make_robot()
        with patch.object(app_gripper, "_follow_material_until_stable") as follow, \
                patch.object(app_gripper, "_grab_material_from_turntable") as grab, \
                patch.object(app_gripper, "_stow_material_to_repo") as stow, \
                patch.object(app_gripper, "_go_home"), \
                patch.object(app_gripper, "move_axes_to"), \
                patch("builtins.input", return_value=""):
            app_gripper.grab_material_from_turntable_to_repo_follow(
                robot, want_colors, home=None
            )
        return robot, follow, grab, stow

    def test_each_colour_is_followed_then_grabbed_then_stowed(self) -> None:
        robot, follow, grab, stow = self.run_it(("red", "green"))

        self.assertEqual(follow.call_count, 2)
        self.assertEqual(
            [call.args[1] for call in follow.call_args_list], ["red", "green"],
        )
        grab.assert_not_called()
        self.assertEqual(robot.gripper.close_jaws.call_count, 2)
        self.assertEqual(stow.call_count, 2)

    def test_follow_grab_and_stow_stay_in_order_with_adjustable_stability(self):
        robot = make_robot()
        events = []
        gates = []

        def follow(*_args, **kwargs):
            events.append("follow")
            gates.append(kwargs["stable_seconds"])

        robot.gripper.close_jaws.side_effect = lambda: events.append("grab")

        with patch.object(
            app_gripper, "_follow_material_until_stable", side_effect=follow,
        ), \
                patch.object(
                    app_gripper, "_stow_material_to_repo",
                    side_effect=lambda *_: events.append("stow"),
                ), \
                patch.object(app_gripper, "_go_home"), \
                patch.object(app_gripper, "move_axes_to"), \
                patch("builtins.input", return_value=""):
            app_gripper.grab_material_from_turntable_to_repo_follow(
                robot, ("red", "green"), home=None, stable_seconds=0.7,
            )
        self.assertEqual(events, ["follow", "grab", "stow", "follow", "grab", "stow"])
        self.assertEqual(gates, [0.7, 0.7])

    def test_a_follow_timeout_stops_the_whole_run(self) -> None:
        robot = make_robot()
        with patch.object(
            app_gripper, "_follow_material_until_stable",
            side_effect=RuntimeError("没停稳"),
        ), patch.object(app_gripper, "_grab_material_from_turntable") as grab, \
                patch.object(app_gripper, "_go_home"), \
                patch.object(app_gripper, "move_axes_to") as move, \
                patch("builtins.input", return_value=""):
            app_gripper.grab_material_from_turntable_to_repo_follow(robot, home=None)

        grab.assert_not_called()
        robot.gripper.close_jaws.assert_not_called()
        move.assert_called_once_with(robot, app_gripper.SAFE_RETRACT)   # 收尾回缩


if __name__ == "__main__":
    unittest.main()

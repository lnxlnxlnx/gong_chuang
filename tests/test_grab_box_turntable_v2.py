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

        robot.grasp.follow_material.assert_called_once_with("red", timeout=3.0)
        robot.vision.scene_stable.assert_not_called()

    def test_a_tracking_failure_stops_the_run(self) -> None:
        robot = make_robot()
        robot.grasp.follow_material.side_effect = RuntimeError("画面里没有物块")

        with self.assertRaisesRegex(RuntimeError, "没有物块"):
            app_gripper._follow_material_until_stable(robot, "red", timeout=0.05)


class PlaceAfterFollowTests(unittest.TestCase):
    def run_it(self, want_colors=("red",)):
        robot = make_robot()
        with patch.object(app_gripper, "_follow_material_until_stable") as follow, \
                patch.object(app_gripper, "_grab_material_from_turntable") as grab, \
                patch.object(app_gripper, "_stow_material_to_repo") as stow, \
                patch.object(app_gripper, "_go_home"), \
                patch.object(app_gripper, "move_axes_to"), \
                patch("builtins.input", return_value=""):
            # 真实函数没有返回值（重试循环靠 `if res:` 判）；Mock 默认真值会让 while 死循环。
            grab.return_value = None
            app_gripper.grab_material_from_turntable_to_repo_follow(
                robot, want_colors, home=None
            )
        return follow, grab, stow

    def test_each_colour_is_followed_then_grabbed_then_stowed(self) -> None:
        follow, grab, stow = self.run_it(("red", "green"))

        self.assertEqual(follow.call_count, 2)
        self.assertEqual([call.args[1] for call in follow.call_args_list], ["red", "green"])
        self.assertEqual(grab.call_count, 2)
        self.assertEqual(grab.call_args_list[1].kwargs, {"want_color": "green"})
        self.assertEqual(stow.call_count, 2)

    def test_a_follow_timeout_stops_the_whole_run(self) -> None:
        robot = make_robot()
        with patch.object(
            app_gripper, "_follow_material_until_stable", side_effect=RuntimeError("没停稳")
        ), patch.object(app_gripper, "_grab_material_from_turntable") as grab, \
                patch.object(app_gripper, "_go_home"), \
                patch.object(app_gripper, "move_axes_to") as move, \
                patch("builtins.input", return_value=""):
            app_gripper.grab_material_from_turntable_to_repo_follow(robot, home=None)

        grab.assert_not_called()
        move.assert_called_once_with(robot, app_gripper.SAFE_RETRACT)   # 收尾回缩


if __name__ == "__main__":
    unittest.main()

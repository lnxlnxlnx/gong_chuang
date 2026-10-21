"""``app.gripper.grab_box_from_turntable``：下降之前必须确认**整幅画面**已经停稳。

这是用户现场遇到的那条：物块在转盘上，x/y 对准之后它仍可能跟着转盘走。所以对准成功
不等于可以下降——下降前要再看一眼画面；还在动就拒绝下降，而不是"落下去扑空"。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

import app.gripper as app_gripper


class GrabBoxFromTurntableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = Mock()
        self.robot.vision.wait_scene_stable.return_value = True

    def test_it_checks_the_scene_after_aligning_and_before_descending(self) -> None:
        app_gripper.grab_box_from_turntable(self.robot, want_color="red")

        self.robot.grasp.align_xy.assert_called_once_with(want="red")
        self.robot.vision.wait_scene_stable.assert_called_once_with(
            timeout=app_gripper.SCENE_STABLE_TIMEOUT_SECONDS
        )
        self.robot.arm.move_axis_to_motor_degrees.assert_called_once_with("lift", 530)
        self.robot.gripper.close_jaws.assert_called_once_with()

    def test_it_forwards_a_custom_scene_timeout(self) -> None:
        app_gripper.grab_box_from_turntable(
            self.robot, want_color="red", scene_stable_timeout=20.0
        )

        self.robot.vision.wait_scene_stable.assert_called_once_with(timeout=20.0)
        self.robot.gripper.close_jaws.assert_called_once_with()

    def test_it_refuses_to_descend_while_the_scene_is_moving(self) -> None:
        self.robot.vision.wait_scene_stable.return_value = False
        self.robot.vision.scene_stability.return_value.summary.return_value = "还在动"

        with self.assertRaisesRegex(RuntimeError, "不下降"):
            app_gripper.grab_box_from_turntable(self.robot)

        self.robot.arm.move_axis_to_motor_degrees.assert_not_called()
        self.robot.gripper.close_jaws.assert_not_called()

    def test_an_align_failure_does_not_run_the_scene_check(self) -> None:
        self.robot.grasp.align_xy.side_effect = RuntimeError("对准失败")

        app_gripper.grab_box_from_turntable(self.robot)

        self.robot.vision.wait_scene_stable.assert_not_called()
        self.robot.gripper.close_jaws.assert_not_called()


if __name__ == "__main__":
    unittest.main()

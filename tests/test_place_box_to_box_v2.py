"""``place_box_to_box_v2`` 的循环骨架：按 ``orders`` 依次去三个位置、放完再回来。

只守**顺序与落点**这两件事（真实帧序列由 ``tests/test_place_box_sequence.py`` 管）：
每个位置上应该是"先去位置 → 对准 → 去仓库抓料 → **回到对准后的角度** → 松开"，
且 ``orders`` 写什么就按什么顺序走。离开这台设备也能跑。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app.gripper as app_gripper


def make_robot() -> Mock:
    """一个只会记账的 Robot：三个轴的角度写死，动作全部是 Mock。"""
    robot = Mock()
    robot.arm.turns_state.of.side_effect = lambda axis: SimpleNamespace(
        motor_degrees={"base": 111.0, "extension": 222.0}[axis]
    )
    return robot


class PlaceBoxToBoxV2Tests(unittest.TestCase):
    def run_it(self, orders):
        robot = make_robot()
        with patch.object(app_gripper, "_move_axes_to") as move, \
                patch.object(app_gripper, "_go_home"), \
                patch.object(app_gripper, "_grab_box_from_car") as grab, \
                patch("builtins.input", return_value=""):
            move.return_value.wait.return_value = None
            app_gripper.place_box_to_box_v2(robot, orders=orders, home=None)
        return robot, [call.args[1] for call in move.call_args_list], grab

    def test_orders_walk_the_three_positions_in_that_order(self) -> None:
        _robot, targets, grab = self.run_it((3, 1, 2))

        # 每趟两次 _move_axes_to（先去位置，抓完料再回到对准后的角度），最后收尾回缩一次。
        self.assertEqual(len(targets), 7)
        self.assertEqual(targets[-1], app_gripper.SAFE_RETRACT)
        self.assertEqual(
            [targets[index] for index in (0, 2, 4)],
            [
                {"lift": 480, "extension": 690, "base": 120},
                {"lift": 480, "extension": 690, "base": -190},
                {"lift": 480, "extension": 540, "base": -32},
            ],
        )
        for target in (targets[1], targets[3], targets[5]):
            self.assertEqual(target, {"lift": 480, "extension": 222.0, "base": 111.0})
        self.assertEqual(grab.call_count, 3)

    def test_each_placement_aligns_then_grabs_then_releases(self) -> None:
        robot, _targets, _grab = self.run_it((1,))

        robot.grasp.align_xy.assert_called_once_with(tolerance_mm=3.0)
        robot.gripper.open_jaws.assert_called_once()
        robot.arm.move_axis_to_motor_degrees.assert_called_once_with("lift", 695)

    def test_a_bad_order_is_rejected_before_anything_moves(self) -> None:
        robot = make_robot()
        with patch.object(app_gripper, "_move_axes_to") as move:
            with self.assertRaises(ValueError):
                app_gripper.place_box_to_box_v2(robot, orders=(0, 4))
        move.assert_not_called()


if __name__ == "__main__":
    unittest.main()

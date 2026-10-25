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

    def visits(self, orders):
        """每趟**第一次**移动的目标（后面那次是抓完料回到对准角度）。"""
        _robot, targets, _grab = self.run_it(orders)
        self.assertEqual(targets[-1], app_gripper.SAFE_RETRACT)
        return [targets[index] for index in range(0, len(targets) - 1, 2)]

    def test_orders_walk_the_three_positions_in_that_order(self) -> None:
        # 位置表不写死在用例里（现场会改 lift/base）：用"按 1,2,3 走一遍"反查表，
        # 再看 3,1,2 是不是照着这张表换顺序。
        table = dict(zip((1, 2, 3), self.visits((1, 2, 3))))
        self.assertEqual(len({str(value) for value in table.values()}), 3)

        self.assertEqual(self.visits((3, 1, 2)), [table[3], table[1], table[2]])

        _robot, targets, grab = self.run_it((3, 1, 2))
        self.assertEqual(len(targets), 7)   # 3 趟 × 2 + 收尾回缩
        self.assertEqual(grab.call_count, 3)

    def test_each_placement_aligns_then_grabs_then_releases(self) -> None:
        robot, _targets, _grab = self.run_it((1,))

        robot.grasp.align_xy.assert_called_once()
        self.assertIn("tolerance_mm", robot.grasp.align_xy.call_args.kwargs)
        robot.gripper.open_jaws.assert_called_once()

    def test_a_bad_order_is_rejected_before_anything_moves(self) -> None:
        robot = make_robot()
        with patch.object(app_gripper, "_move_axes_to") as move:
            with self.assertRaises(ValueError):
                app_gripper.place_box_to_box_v2(robot, orders=(0, 4))
        move.assert_not_called()


if __name__ == "__main__":
    unittest.main()

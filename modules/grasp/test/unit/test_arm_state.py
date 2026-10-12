"""机械臂状态：来源标记、共享语义，以及"未知"保护。

这一层**只记状态、不做判断**，所以测试的重点不是策略，而是"事实有没有被如实记下来"：
来源是什么、未知时会不会被拒绝、同一个实例是不是同一份状态。
"""

from __future__ import annotations

import unittest

from modules.arm.drive.kinematics import CylindricalPose
from modules.grasp.drive.arm_state import (
    DEAD_RECKONED,
    HOMED,
    MEASURED,
    UNKNOWN,
    ArmState,
)

HOME_POSE = CylindricalPose(0.0, 40.0, 60.0)
NEXT_POSE = CylindricalPose(10.0, 50.0, 55.0)


class ArmStateFreshTests(unittest.TestCase):
    def test_a_new_state_starts_unknown(self) -> None:
        """状态只在内存里，掉电即失效：新实例绝不能假装自己知道在哪。"""
        state = ArmState()
        self.assertEqual(state.source, UNKNOWN)
        self.assertFalse(state.is_known)
        self.assertFalse(state.is_absolute)
        self.assertIsNone(state.pose)

    def test_an_unknown_state_refuses_to_hand_out_a_pose(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "未知"):
            ArmState().require_pose()

    def test_moves_recorded_before_any_reference_do_not_invent_a_pose(self) -> None:
        state = ArmState()

        state.record_move(NEXT_POSE)

        self.assertFalse(state.is_known, state)
        self.assertEqual(state.source, UNKNOWN)

    def test_marking_unknown_clears_the_pose(self) -> None:
        state = ArmState()
        state.record_home(HOME_POSE)

        state.mark_unknown()

        self.assertFalse(state.is_known)
        self.assertIsNone(state.pose)
        self.assertEqual(state.source, UNKNOWN)

    def test_marking_unknown_keeps_what_the_jaws_already_did(self) -> None:
        """夹爪状态是"已经发生的事"，不该被位姿失效一起抹掉。"""
        state = ArmState(pose=HOME_POSE, source=HOMED)
        state.record_gripper(position=430, reached=False)

        state.mark_unknown()

        self.assertTrue(state.gripper_holds)
        self.assertEqual(state.gripper_position, 430)


class ArmStateSourceTests(unittest.TestCase):
    def test_homing_is_an_absolute_reference(self) -> None:
        state = ArmState()

        state.record_home(HOME_POSE)

        self.assertEqual(state.source, HOMED)
        self.assertTrue(state.is_absolute)
        self.assertEqual(state.require_pose(), HOME_POSE)

    def test_measuring_is_also_absolute(self) -> None:
        state = ArmState()

        state.record_measurement(NEXT_POSE)

        self.assertTrue(state.is_absolute)
        self.assertEqual(state.source, MEASURED)

    def test_one_move_after_an_absolute_reference_becomes_dead_reckoned(self) -> None:
        state = ArmState()
        state.record_home(HOME_POSE)

        state.record_move(NEXT_POSE)

        self.assertEqual(state.source, DEAD_RECKONED)
        self.assertFalse(state.is_absolute)
        self.assertTrue(state.is_known)
        self.assertEqual(state.require_pose(), NEXT_POSE)

    def test_a_measurement_restores_the_absolute_reference(self) -> None:
        state = ArmState()
        state.record_home(HOME_POSE)
        state.record_move(NEXT_POSE)

        state.record_measurement(NEXT_POSE)

        self.assertTrue(state.is_absolute)
        self.assertEqual(state.source, MEASURED)


class ArmStateSharingTests(unittest.TestCase):
    def test_two_references_to_the_same_instance_share_one_state(self) -> None:
        """这就是"唯一所有者"的意思：电机在别处被驱动，状态仍然只有一份。"""
        state = ArmState()
        elsewhere = state

        state.record_home(HOME_POSE)

        self.assertEqual(elsewhere.pose, HOME_POSE)
        self.assertEqual(elsewhere.source, HOMED)

    def test_two_instances_do_not_leak_into_each_other(self) -> None:
        first = ArmState()
        second = ArmState()

        first.record_home(HOME_POSE)

        self.assertFalse(second.is_known)


class ArmStateGripperTests(unittest.TestCase):
    def test_a_blocked_close_means_the_jaws_are_holding_something(self) -> None:
        state = ArmState()

        state.record_gripper(position=430, reached=False)

        self.assertTrue(state.gripper_holds)
        self.assertEqual(state.gripper_position, 430)

    def test_reaching_the_closed_position_means_an_empty_grasp(self) -> None:
        state = ArmState()

        state.record_gripper(position=500, reached=True)

        self.assertFalse(state.gripper_holds)

    def test_the_summary_reports_both_pose_and_jaws(self) -> None:
        state = ArmState()
        self.assertIn("位姿未知", state.summary())

        state.record_home(HOME_POSE)
        state.record_gripper(position=430, reached=False)

        described = state.summary()
        self.assertIn(HOMED, described)
        self.assertIn("夹着东西", described)

    def test_the_summary_says_so_when_the_jaws_have_not_moved(self) -> None:
        state = ArmState(pose=HOME_POSE, source=MEASURED)
        self.assertIn("夹爪未动作", state.summary())


if __name__ == "__main__":
    unittest.main()

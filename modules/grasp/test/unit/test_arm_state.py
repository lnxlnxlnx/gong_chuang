"""机械臂状态：来源跟踪、对账策略，以及"未知"保护。

这一层的价值全在"位姿是怎么来的"上——回零给绝对参考、相机给对账、命令量累加会漂。
所以测试的重点不是数值，而是**来源与由此推出的策略**。
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
        self.assertFalse(state.needs_verification)

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
        self.assertEqual(state.source, UNKNOWN)
        self.assertEqual(state.moves_since_verification, 0)


class ArmStateSourceTests(unittest.TestCase):
    def test_homing_is_an_absolute_reference_and_needs_no_reconciliation(self) -> None:
        state = ArmState()

        state.record_home(HOME_POSE)

        self.assertEqual(state.source, HOMED)
        self.assertTrue(state.is_absolute)
        self.assertFalse(state.needs_verification)
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
        self.assertEqual(state.require_pose(), NEXT_POSE)

    def test_reconciling_resets_the_source_and_the_counter(self) -> None:
        state = ArmState()
        state.record_home(HOME_POSE)
        state.record_move(NEXT_POSE)

        state.record_measurement(NEXT_POSE)

        self.assertEqual(state.source, MEASURED)
        self.assertEqual(state.moves_since_verification, 0)
        self.assertFalse(state.needs_verification)


class ArmStateVerificationPolicyTests(unittest.TestCase):
    def test_with_the_default_interval_every_dead_reckoned_move_asks_for_a_check(self) -> None:
        state = ArmState(pose=HOME_POSE, source=HOMED)
        state.record_move(NEXT_POSE)
        self.assertTrue(state.needs_verification)

    def test_a_larger_interval_trades_drift_for_speed(self) -> None:
        state = ArmState(pose=HOME_POSE, source=HOMED, verification_interval=3)

        for _ in range(2):
            state.record_move(HOME_POSE)
        self.assertFalse(state.needs_verification)

        state.record_move(HOME_POSE)
        self.assertTrue(state.needs_verification)

    def test_an_unknown_state_never_asks_for_verification(self) -> None:
        """未知状态该做的是回零或测量，而不是"对账"——没有东西可以对。"""
        self.assertFalse(ArmState().needs_verification)


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
        self.assertIn("homed", described)
        self.assertIn("夹着东西", described)

    def test_the_summary_says_so_when_the_jaws_have_not_moved(self) -> None:
        state = ArmState(pose=HOME_POSE, source=MEASURED)
        self.assertIn("夹爪未动作", state.summary())


if __name__ == "__main__":
    unittest.main()

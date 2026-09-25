"""Verify the arm hardware-test sequence without a connected serial device."""

from __future__ import annotations

import unittest
import os
from io import StringIO
from contextlib import redirect_stderr
from unittest.mock import patch

from tests.manual import arm as position_motion
from mecanum_config import load_profile
from robot_arm import MotionReport
from robot_arm_config import ArmConfig, AxisConfig


class FakeRobotArm:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def initialize(self) -> "FakeRobotArm":
        self.calls.append(("initialize",))
        return self

    def enable_all(self) -> None:
        self.calls.append(("enable_all",))

    def disable_all(self) -> None:
        self.calls.append(("disable_all",))

    def stop_all(self) -> None:
        self.calls.append(("stop_all",))

    def close(self) -> None:
        self.calls.append(("close",))

    def rotate_cw(self, pulses: int) -> MotionReport:
        self.calls.append(("rotate_cw", pulses))
        return MotionReport(5, pulses, 11.25, 1.875)

    def rotate_ccw(self, pulses: int) -> MotionReport:
        self.calls.append(("rotate_ccw", pulses))
        return MotionReport(5, pulses, 11.25, 1.875)

    def move_up(self, pulses: int) -> MotionReport:
        self.calls.append(("move_up", pulses))
        return MotionReport(6, pulses, 11.25, None)

    def move_down(self, pulses: int) -> MotionReport:
        self.calls.append(("move_down", pulses))
        return MotionReport(6, pulses, 11.25, None)

    def move_forward(self, pulses: int) -> MotionReport:
        self.calls.append(("move_forward", pulses))
        return MotionReport(7, pulses, 11.25, None)

    def move_backward(self, pulses: int) -> MotionReport:
        self.calls.append(("move_backward", pulses))
        return MotionReport(7, pulses, 11.25, None)


class ArmPositionMotionScriptTests(unittest.TestCase):
    def test_parse_args_uses_the_configured_default_and_limit(self) -> None:
        config = ArmConfig(
            pulses_per_revolution=3200,
            rpm=10,
            acceleration=5,
            default_pulses=1200,
            max_pulses=5000,
            axes={
                "base": AxisConfig(motor_id=5, reduction_ratio=6),
                "lift": AxisConfig(motor_id=6),
                "extension": AxisConfig(motor_id=7),
            },
        )

        with patch("sys.argv", ["arm.py"]):
            args = position_motion.parse_args(config)

        self.assertEqual(args.pulses, 1200)
        self.assertEqual(args.port, load_profile("windows" if os.name == "nt" else "jetson").port)
        error_output = StringIO()
        with patch("sys.argv", ["position_motion.py", "--port", "COM29", "--pulses", "5001"]):
            with redirect_stderr(error_output), self.assertRaises(SystemExit):
                position_motion.parse_args(config)
        self.assertIn("must be in 1..5000", error_output.getvalue())
        self.assertNotIn("choose from", error_output.getvalue())

    def test_sequence_sends_each_confirmed_axis_direction_with_one_safe_pulse_count(self) -> None:
        arm = FakeRobotArm()

        position_motion.run_sequence(arm, pulses=100, prompt=lambda _: "")

        self.assertEqual(
            arm.calls,
            [
                ("rotate_cw", 100),
                ("rotate_ccw", 100),
                ("move_up", 100),
                ("move_down", 100),
                ("move_forward", 100),
                ("move_backward", 100),
            ],
        )

    def test_sequence_waits_for_confirmation_after_the_final_backward_motion(self) -> None:
        arm = FakeRobotArm()
        prompts: list[str] = []

        position_motion.run_sequence(
            arm,
            pulses=100,
            prompt=lambda message: prompts.append(message) or "",
        )

        self.assertEqual(len(prompts), 7)
        self.assertIn("确认机械臂后缩完成", prompts[-1])
        self.assertEqual(arm.calls[-1], ("move_backward", 100))

    def test_hardware_run_stops_disables_and_closes_after_an_interruption(self) -> None:
        class InterruptedArm(FakeRobotArm):
            def rotate_cw(self, pulses: int) -> MotionReport:
                super().rotate_cw(pulses)
                raise KeyboardInterrupt

        arm = InterruptedArm()

        with self.assertRaises(KeyboardInterrupt):
            position_motion.run_hardware_test(arm, pulses=100, prompt=lambda _: "")

        self.assertEqual(
            arm.calls,
            [
                ("initialize",),
                ("enable_all",),
                ("rotate_cw", 100),
                ("stop_all",),
                ("disable_all",),
                ("close",),
            ],
        )

    def test_hardware_run_preserves_the_enable_error_when_cleanup_also_fails(self) -> None:
        class NoAckArm(FakeRobotArm):
            def enable_all(self) -> None:
                self.calls.append(("enable_all",))
                raise RuntimeError("Emm42 motor 5 command 0xF3 received no ACK")

            def stop_all(self) -> None:
                self.calls.append(("stop_all",))
                raise RuntimeError("Emm42 motor 5 command 0xFE received no ACK")

            def disable_all(self) -> None:
                self.calls.append(("disable_all",))
                raise RuntimeError("Emm42 motor 5 command 0xF3 received no ACK")

        arm = NoAckArm()

        with self.assertRaisesRegex(RuntimeError, "command 0xF3 received no ACK"):
            position_motion.run_hardware_test(arm, pulses=100, prompt=lambda _: "")

        self.assertEqual(
            arm.calls,
            [
                ("initialize",),
                ("enable_all",),
                ("stop_all",),
                ("disable_all",),
                ("close",),
            ],
        )


if __name__ == "__main__":
    unittest.main()

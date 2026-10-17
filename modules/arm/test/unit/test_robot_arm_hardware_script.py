"""Verify the arm hardware-test sequence without a connected serial device."""

from __future__ import annotations

import unittest
import os
from io import StringIO
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from modules.arm.test.manual import arm as position_motion
from modules.motor.config import load_bus_config
from modules.arm.runtime import MotionReport
from modules.arm.config import ArmConfig, AxisConfig


class FakeRobotArm:
    """假臂：只实现脚本真正用到的动作原语。

    脚本**不再**自己收尾（没有 ``initialize`` / ``_cleanup``）：正常关闭由 ``main`` 里的
    ``with RobotArm._from_settings(...)`` 负责，真正的中断照原样向上抛。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

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


class ContextRobotArm(FakeRobotArm):
    """加上上下文协议的假臂：``main`` 的 ``with`` 会在这里收尾一次。"""

    def __enter__(self) -> "ContextRobotArm":
        self.calls.append(("enter",))
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.calls.append(("close",))
        return False


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
        self.assertEqual(args.port, load_bus_config().port)
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

    def test_an_interruption_propagates_without_the_script_cleaning_up(self) -> None:
        """中断照旧原样抛出；收尾是 ``main`` 里 ``with`` 的事，脚本不再自己 stop/disable。"""
        class InterruptedArm(FakeRobotArm):
            def rotate_cw(self, pulses: int) -> MotionReport:
                super().rotate_cw(pulses)
                raise KeyboardInterrupt

        arm = InterruptedArm()

        with self.assertRaises(KeyboardInterrupt):
            position_motion.run_hardware_test(arm, pulses=100, prompt=lambda _: "")

        self.assertEqual(arm.calls, [("rotate_cw", 100)])

    def test_a_driver_error_is_never_replaced_by_a_cleanup_error(self) -> None:
        """脚本没有自己的收尾，所以原始驱动错误一定原样抛出。

        收尾失败该怎么呈现由 ``RobotArm.__exit__`` 定义（见 arm 自己的单测）。
        """
        class NoAckArm(FakeRobotArm):
            def move_up(self, pulses: int) -> MotionReport:
                self.calls.append(("move_up", pulses))
                raise RuntimeError("Emm42 motor 6 command 0xFD received no ACK")

        arm = NoAckArm()

        with self.assertRaisesRegex(RuntimeError, "command 0xFD received no ACK"):
            position_motion.run_hardware_test(arm, pulses=100, prompt=lambda _: "")

        self.assertEqual(arm.calls[-1], ("move_up", 100))

    def test_main_lets_the_arm_context_own_the_lifecycle(self) -> None:
        """``main`` 用 ``with RobotArm._from_settings(...)``：装配一次、关闭一次。"""
        arm = ContextRobotArm()
        config = ArmConfig(
            pulses_per_revolution=3200,
            rpm=10,
            acceleration=5,
            default_pulses=800,
            max_pulses=5000,
            axes={
                "base": AxisConfig(motor_id=5, reduction_ratio=6),
                "lift": AxisConfig(motor_id=6),
                "extension": AxisConfig(motor_id=7),
            },
        )

        with patch.object(
            position_motion, "load_arm_config", return_value=config
        ), patch.object(
            position_motion,
            "parse_args",
            return_value=SimpleNamespace(port="COM29", pulses=100),
        ), patch.object(
            position_motion.RobotArm, "_from_settings", return_value=arm
        ) as build, patch.object(
            position_motion, "run_hardware_test"
        ) as run:
            position_motion.main()

        build.assert_called_once_with(config=config, port="COM29")
        run.assert_called_once_with(arm, pulses=100)
        self.assertEqual(arm.calls, [("enter",), ("close",)])


if __name__ == "__main__":
    unittest.main()

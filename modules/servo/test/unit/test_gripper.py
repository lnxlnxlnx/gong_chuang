"""The gripper's own settings, the merged manual flow, and its wiring."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from modules.servo.config import GripperConfig, load_gripper_config
from modules.servo.test.manual import gripper as gripper_manual
from modules.servo.test.manual.gripper import run_manual


class GripperConfigTests(unittest.TestCase):
    def test_project_config_loads_gripper_positions(self) -> None:
        config = load_gripper_config()
        # 80 / 10 是现场量出来的：爪子从 80 走到 10 正好闭合夹住物块。
        self.assertEqual(
            (config.servo_id, config.open_position, config.closed_position, config.duration_ms),
            (1, 80, 10, 1000),
        )

    def test_endpoint_outside_the_protocol_range_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "gripper": {
                    "servo_id": 1,
                    "open_position": 300,
                    "closed_position": 1001,
                    "duration_ms": 1000,
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closed_position"):
                load_gripper_config(path)

    def test_identical_open_and_closed_positions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "gripper": {
                    "servo_id": 1,
                    "open_position": 500,
                    "closed_position": 500,
                    "duration_ms": 1000,
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不能相同"):
                load_gripper_config(path)


class GripperManualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GripperConfig(
            servo_id=1, open_position=300, closed_position=500, duration_ms=1000
        )

    def test_manual_actions_require_individual_confirmation(self) -> None:
        servo = Mock()
        servo.read_position.return_value = 400
        answers = iter(["no", ""])
        run_manual(servo, self.config, prompt=lambda _: next(answers), sleep=Mock())
        # 只有第二项得到确认（回车）：第一次不会发送任何位置指令
        servo.set_position.assert_called_once_with(500, 1000)
        # 起始位置读一次，张开的确认被跳过所以没有读回，闭合后读回一次
        self.assertEqual(servo.read_position.call_count, 2)

    def test_pressing_enter_executes_instead_of_skipping(self) -> None:
        """这次改动的核心：以前空输入（回车）等于跳过，现在等于执行。"""
        servo = Mock()
        servo.read_position.return_value = 300
        answers = iter(["", "no"])
        run_manual(servo, self.config, prompt=lambda _: next(answers), sleep=Mock())
        servo.set_position.assert_called_once_with(300, 1000)

    def test_q_stops_the_remaining_actions(self) -> None:
        servo = Mock()
        servo.read_position.return_value = 400
        answers = iter(["", "q"])
        with redirect_stdout(io.StringIO()) as output:
            run_manual(servo, self.config, prompt=lambda _: next(answers), sleep=Mock())
        servo.set_position.assert_called_once_with(300, 1000)
        self.assertIn("已结束后续夹爪动作", output.getvalue())

    def test_confirmed_actions_send_both_positions_in_order(self) -> None:
        servo = Mock()
        servo.read_position.side_effect = [506, 300, 499]
        sleep = Mock()
        run_manual(servo, self.config, prompt=lambda _: "", sleep=sleep)
        self.assertEqual(
            [call.args for call in servo.set_position.call_args_list],
            [(300, 1000), (500, 1000)],
        )
        self.assertEqual(sleep.call_count, 2)

    def test_patched_time_sleep_is_used_when_no_sleep_is_injected(self) -> None:
        """A default bound at definition time would silently ignore this patch."""
        servo = Mock()
        servo.read_position.return_value = 400
        with patch.object(gripper_manual.time, "sleep") as sleep:
            run_manual(servo, self.config, prompt=lambda _: "")
        self.assertEqual(sleep.call_count, 2)

    def test_manual_does_not_move_when_servo_does_not_answer(self) -> None:
        servo = Mock()
        servo.read_position.side_effect = TimeoutError("no response")
        with self.assertRaises(TimeoutError):
            run_manual(servo, self.config, prompt=lambda _: "", sleep=Mock())
        servo.set_position.assert_not_called()


class GripperWiringTests(unittest.TestCase):
    def test_main_opens_the_gripper_id_on_the_configured_bus_port(self) -> None:
        config = GripperConfig(
            servo_id=2, open_position=300, closed_position=500, duration_ms=1000
        )
        bus = Mock(port="COM14", baudrate=115200, servo_id=1)
        with patch.object(gripper_manual, "parse_port", return_value=None), \
                patch.object(gripper_manual, "load_gripper_config", return_value=config), \
                patch.object(gripper_manual, "load_bus_config", return_value=bus), \
                patch.object(gripper_manual, "run_manual"), \
                patch.object(gripper_manual, "BusServo") as bus_servo:
            gripper_manual.main()
        bus_servo.assert_called_once_with("COM14", 2, baudrate=115200, timeout=0.5)
        bus_servo.return_value.initialize.return_value.close.assert_called_once()

    def test_port_override_wins_over_the_configured_bus_port(self) -> None:
        config = GripperConfig(
            servo_id=1, open_position=300, closed_position=500, duration_ms=1000
        )
        bus = Mock(port="COM14", baudrate=115200, servo_id=1)
        with patch.object(gripper_manual, "parse_port", return_value="COM77"), \
                patch.object(gripper_manual, "load_gripper_config", return_value=config), \
                patch.object(gripper_manual, "load_bus_config", return_value=bus), \
                patch.object(gripper_manual, "run_manual"), \
                patch.object(gripper_manual, "BusServo") as bus_servo:
            gripper_manual.main()
        bus_servo.assert_called_once_with("COM77", 1, baudrate=115200, timeout=0.5)


if __name__ == "__main__":
    unittest.main()

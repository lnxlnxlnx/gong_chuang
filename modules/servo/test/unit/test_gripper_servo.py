"""夹爪上层：开合、等待、读回到位判据、"夹住东西"的检测，以及统一生命周期。

两个容易踩的点：

1. 注入 ``servo`` 时**不做任何串口 I/O**——``open`` 才开真实总线，``__enter__`` 不重复初始化；
2. ``close()`` 只管**连接**（幂等、置 ``closed``），``close_jaws()`` 才管**爪口**，两者不是一回事。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from modules.servo.config import GripperConfig
from modules.servo.runtime import (
    CLOSE_COMMAND,
    DEFAULT_POSITION_TOLERANCE,
    OPEN_COMMAND,
    GripperServo,
)

CONFIG = GripperConfig(servo_id=1, open_position=300, closed_position=500, duration_ms=1000)


def make_gripper(positions, **overrides) -> tuple[GripperServo, Mock, list]:
    servo = Mock()
    servo.read_position.side_effect = list(positions)
    slept: list = []
    settings = dict(
        config=CONFIG,
        port="FAKE",
        servo=servo,
        sleep=slept.append,
        settle_seconds=0.2,
    )
    settings.update(overrides)
    return GripperServo(**settings), servo, slept


class GripperLifecycleTests(unittest.TestCase):
    def test_construction_opens_nothing(self) -> None:
        """``__init__`` 只保存配置：不读文件、不开串口、不上扭矩。"""
        gripper, servo, _ = make_gripper([])

        servo.initialize.assert_not_called()
        servo.set_torque.assert_not_called()
        self.assertFalse(gripper.closed)

    def test_open_reads_one_config_initializes_once_and_holds_torque(self) -> None:
        servo = Mock()
        with patch(
            "modules.servo.runtime.load_gripper_config", return_value=CONFIG
        ) as load_gripper, patch(
            "modules.servo.runtime.load_bus_config"
        ) as load_bus, patch(
            "modules.servo.runtime.BusServo", return_value=servo
        ) as bus_servo:
            load_bus.return_value = Mock(port="COM7", baudrate=115200, servo_id=1)
            gripper = GripperServo.open()

        # 两个加载器读的是同一份配置路径。
        load_gripper.assert_called_once()
        load_bus.assert_called_once()
        self.assertEqual(
            load_gripper.call_args.args[0], load_bus.call_args.kwargs["config_path"]
        )
        bus_servo.assert_called_once_with("COM7", CONFIG.servo_id, baudrate=115200)
        servo.initialize.assert_called_once_with()
        servo.set_torque.assert_called_once_with(True)
        self.assertEqual(gripper.port, "COM7")

    def test_enter_does_not_initialize_again(self) -> None:
        """``open`` 已经初始化过；``__enter__`` 只校验并返回自身。"""
        gripper, servo, _ = make_gripper([])
        servo.initialize.reset_mock()

        with gripper as entered:
            self.assertIs(entered, gripper)

        servo.initialize.assert_not_called()
        # 借来的舵机不归本实例管：close 只置位失效。
        servo.close.assert_not_called()
        self.assertTrue(gripper.closed)

    def test_close_is_idempotent_and_only_closes_the_connection(self) -> None:
        gripper, servo, _ = make_gripper([])

        gripper.close()
        gripper.close()

        # 借来的总线由装配者负责关闭，本实例不擅自关它，也不动爪口。
        servo.close.assert_not_called()
        servo.set_position.assert_not_called()
        self.assertTrue(gripper.closed)

    def test_an_owned_bus_is_closed_exactly_once(self) -> None:
        servo = Mock()
        gripper = GripperServo(config=CONFIG, port="COM7", servo=servo)
        # 显式声明"这条总线归我"（``open`` 之外的构造路径）。
        gripper._borrowed = False

        gripper.close()
        gripper.close()

        servo.close.assert_called_once_with()

    def test_every_action_fails_after_close(self) -> None:
        gripper, _servo, _ = make_gripper([300, 300, 300, 300])
        gripper.close()

        with self.assertRaises(RuntimeError):
            gripper.open_jaws()
        with self.assertRaises(RuntimeError):
            gripper.close_jaws()
        with self.assertRaises(RuntimeError):
            gripper.read_position()
        with self.assertRaises(RuntimeError):
            gripper.stop()
        with self.assertRaises(RuntimeError):
            gripper.set_torque(False)
        with self.assertRaises(RuntimeError):
            gripper.set_position(300)
        with self.assertRaises(RuntimeError):
            gripper.initialize()

    def test_a_closed_session_cannot_be_entered_again(self) -> None:
        gripper, _servo, _ = make_gripper([])
        gripper.close()
        with self.assertRaises(RuntimeError):
            with gripper:
                pass  # pragma: no cover - 进不来

    def test_business_error_wins_over_cleanup_error(self) -> None:
        servo = Mock()
        servo.close.side_effect = RuntimeError("close broke")
        gripper = GripperServo(config=CONFIG, port="COM7", servo=servo)
        gripper._borrowed = False

        with self.assertRaises(ValueError) as caught:
            with gripper:
                raise ValueError("business broke")

        self.assertIn("business broke", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_injected_servo_means_no_serial_port_is_needed(self) -> None:
        gripper, servo, _ = make_gripper([])
        self.assertEqual(gripper.port, "FAKE")
        servo.initialize.assert_not_called()

    def test_a_real_session_without_a_port_is_rejected(self) -> None:
        """没有注入舵机时必须有串口名，否则会去开一个空设备名。"""
        with self.assertRaises(ValueError):
            GripperServo(config=CONFIG)


class GripperMotionTests(unittest.TestCase):
    def test_open_uses_the_configured_open_position(self) -> None:
        gripper, servo, slept = make_gripper([300])

        report = gripper.open_jaws()

        servo.set_position.assert_called_once_with(300, 1000)
        self.assertEqual((report.command, report.target, report.position), (OPEN_COMMAND, 300, 300))
        self.assertTrue(report.reached)
        self.assertEqual(slept, [1.2])

    def test_close_uses_the_configured_closed_position(self) -> None:
        gripper, servo, _ = make_gripper([500])

        report = gripper.close_jaws()

        servo.set_position.assert_called_once_with(500, 1000)
        self.assertEqual(report.command, CLOSE_COMMAND)
        self.assertTrue(report.reached)

    def test_a_blocked_close_is_reported_as_not_reached_and_that_is_not_an_error(self) -> None:
        """爪口被物料挡住时读回值到不了闭合位——这正是"夹到了东西"的信号。"""
        gripper, _, _ = make_gripper([438])

        report = gripper.close_jaws()

        self.assertFalse(report.reached)
        self.assertEqual(report.error, -62)
        self.assertIn("未到位", report.summary())

    def test_the_tolerance_is_inclusive_at_its_boundary(self) -> None:
        gripper, _, _ = make_gripper([500 - DEFAULT_POSITION_TOLERANCE])
        self.assertTrue(gripper.close_jaws().reached)

        gripper, _, _ = make_gripper([500 - DEFAULT_POSITION_TOLERANCE - 1])
        self.assertFalse(gripper.close_jaws().reached)

    def test_wait_false_sends_the_command_without_sleeping_or_claiming_arrival(self) -> None:
        gripper, servo, slept = make_gripper([412])

        report = gripper.close_jaws(wait=False)

        servo.set_position.assert_called_once_with(500, 1000)
        self.assertEqual(slept, [])
        self.assertFalse(report.reached)
        self.assertEqual(report.position, 412)

    def test_set_position_drives_a_raw_value_for_calibration(self) -> None:
        gripper, servo, _ = make_gripper([420])

        report = gripper.set_position(420)

        servo.set_position.assert_called_once_with(420, 1000)
        self.assertEqual(report.command, "set")
        self.assertTrue(report.reached)

    def test_read_stop_and_torque_are_passed_through(self) -> None:
        gripper, servo, _ = make_gripper([377])

        self.assertEqual(gripper.read_position(), 377)
        gripper.stop()
        gripper.set_torque(False)

        servo.read_position.assert_called_once_with()
        servo.stop.assert_called_once_with()
        servo.set_torque.assert_called_once_with(False)


class GripperValidationTests(unittest.TestCase):
    def test_negative_tolerance_and_settle_are_rejected(self) -> None:
        for overrides in ({"position_tolerance": -1}, {"settle_seconds": -0.1}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_gripper([], **overrides)

    def test_the_settle_margin_is_added_on_top_of_the_configured_duration(self) -> None:
        gripper, _, slept = make_gripper([300], settle_seconds=0.5)
        gripper.open_jaws()
        self.assertEqual(slept, [1.5])


if __name__ == "__main__":
    unittest.main()

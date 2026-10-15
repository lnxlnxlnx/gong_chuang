"""Verify the high-level chassis facade and coordinated position commands."""

from __future__ import annotations

import unittest

from modules.chassis.runtime import MecanumChassis
from modules.chassis.drive.mecanum import MecanumDrive, WheelPosition
from modules.chassis.test.unit.test_mecanum import FakeSerial


class ChassisFacadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.drive = MecanumDrive("COM14", serial_port=self.serial, sleep=lambda _seconds: None)
        self.chassis = MecanumChassis(self.drive)

    def test_forward_delegates_to_four_wheel_drive(self) -> None:
        self.chassis.forward(rpm=30)
        self.assertEqual(
            self.serial.writes,
            [
                bytes.fromhex("01 F6 00 00 1E 0A 00 6B"),
                bytes.fromhex("02 F6 00 00 1E 0A 00 6B"),
                bytes.fromhex("03 F6 00 00 1E 0A 00 6B"),
                bytes.fromhex("04 F6 00 00 1E 0A 00 6B"),
            ],
        )

    def test_forward_with_pulses_uses_the_position_mode(self) -> None:
        counts = self.chassis.forward(rpm=200, pulses=3200)

        self.assertEqual(counts, [3200, 3200, 3200, 3200])
        self.assertEqual([frame[1] for frame in self.serial.writes[:4]], [0xFD] * 4)
        self.assertEqual(self.serial.writes[4], bytes.fromhex("00 FF 66 6B"))

    def test_move_wheels_relative_pulses_passthrough(self) -> None:
        self.chassis.move_wheels_relative_pulses(0, 3200, 0, -3200, rpm=200)

        self.assertEqual([frame[0] for frame in self.serial.writes[:2]], [2, 4])
        self.assertEqual([frame[1] for frame in self.serial.writes[:2]], [0xFD, 0xFD])

    def test_relative_position_moves_all_wheels_in_logical_order(self) -> None:
        self.chassis.move_wheels_relative(fl=90, fr=-90, rl=180, rr=-180, rpm=120)
        self.assertEqual([frame[0] for frame in self.serial.writes[:4]], [1, 2, 3, 4])
        self.assertEqual([frame[1] for frame in self.serial.writes[:4]], [0xFD] * 4)
        self.assertEqual(self.serial.writes[4], bytes.fromhex("00 FF 66 6B"))

    def test_facade_exposes_single_wheel_access_for_service_operations(self) -> None:
        self.assertIs(self.chassis.wheel(WheelPosition.FL), self.drive.get_wheel(WheelPosition.FL))


class ChassisLifecycleTests(unittest.TestCase):
    """统一生命周期：关闭幂等、每项收尾独立尝试、关闭后动作失败。"""

    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.drive = MecanumDrive("COM14", serial_port=self.serial, sleep=lambda _seconds: None)
        self.chassis = MecanumChassis(self.drive)

    def test_close_stops_disables_and_releases_each_independently(self) -> None:
        calls = []
        self.drive.stop = lambda: calls.append("stop")
        self.drive.disable_all = lambda: calls.append("disable_all")
        original_close = self.drive.close

        def failing_ok():
            calls.append("drive.close")
            original_close()

        self.drive.close = failing_ok

        self.chassis.close()
        self.chassis.close()  # 幂等：不再发命令

        self.assertEqual(calls, ["stop", "disable_all", "drive.close"])
        self.assertTrue(self.chassis.closed)

    def test_a_failing_stop_does_not_skip_the_rest(self) -> None:
        calls = []
        self.drive.stop = lambda: (_ for _ in ()).throw(RuntimeError("stop broke"))
        self.drive.disable_all = lambda: calls.append("disable_all")
        self.drive.close = lambda: calls.append("drive.close")

        with self.assertRaises(RuntimeError) as caught:
            self.chassis.close()

        self.assertIn("stop broke", str(caught.exception))
        self.assertEqual(calls, ["disable_all", "drive.close"])

    def test_every_action_fails_after_close(self) -> None:
        self.chassis.close()
        with self.assertRaises(RuntimeError):
            self.chassis.forward(rpm=30)
        with self.assertRaises(RuntimeError):
            self.chassis.stop()
        with self.assertRaises(RuntimeError):
            self.chassis.wheel(WheelPosition.FL)

    def test_a_borrowed_bus_is_not_closed_by_the_chassis(self) -> None:
        """借来的步进连接不归底盘关：擅自关掉会切断 arm 的总线。"""
        self.assertFalse(self.drive.owns_serial)
        self.chassis.close()
        self.assertIsNone(self.drive._serial)
        self.assertTrue(self.serial.is_open, "借来的串口必须保持打开")

    def test_an_owned_bus_is_closed_exactly_once(self) -> None:
        """自己开的串口才物理关闭，而且只关一次。"""
        serial = FakeSerial()
        # 注入假串口建立驱动，再声明"这条总线归我"（真实路径上由构造时的默认值决定）。
        drive = MecanumDrive("COM14", serial_port=serial, sleep=lambda _seconds: None)
        drive._owns_serial = True
        chassis = MecanumChassis(drive)

        chassis.close()
        chassis.close()

        self.assertFalse(serial.is_open, "自有的串口必须关掉")


if __name__ == "__main__":
    unittest.main()

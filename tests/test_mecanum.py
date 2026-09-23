"""Verify mecanum kinematics and per-wheel passthrough without hardware."""

from __future__ import annotations

import unittest

from mecanum import MecanumDrive, WheelPosition, WHEEL_ORDER
from motor_drivers.emm42 import Direction, HomeMode


class FakeSerial:
    def __init__(self) -> None:
        self._receive_buffer = bytearray()
        self.writes: list[bytes] = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        self._receive_buffer.extend(bytes((frame[0], frame[1], 0x02, 0x6B)))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return data

    def close(self) -> None:
        self.is_open = False


def speed_frame(motor_id: int, direction: int, rpm: int, acceleration: int) -> str:
    """按 Emm42 速度模式 F6 帧 = [id F6 dir rpm_be accel sync 6B] 拼十六进制串。"""
    return f"{motor_id:02X} F6 {direction:02X} {rpm.to_bytes(2, 'big').hex(' ')} {acceleration:02X} 00 6B"


class MecanumDriveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.car = MecanumDrive("/dev/ttyUSB0", serial_port=self.serial)

    def test_default_motor_ids_use_standard_logical_order(self) -> None:
        self.assertEqual(
            self.car.motor_ids,
            {
                WheelPosition.FL: 1,
                WheelPosition.FR: 2,
                WheelPosition.RL: 3,
                WheelPosition.RR: 4,
            },
        )

    def assert_wheel_speeds(self, speeds: tuple[int, int, int, int], *, rpm: int, accel: int = 10) -> None:
        """断言四个轮（FL FR RL RR）各自的带符号轮速对应的帧序列。"""
        expected = []
        for wheel, signed in zip(WHEEL_ORDER, speeds):
            if signed == 0:
                expected.append(bytes.fromhex(f"{self.car.motor_ids[wheel]:02X} FE 98 00 6B"))
            else:
                direction = Direction.CW if signed > 0 else Direction.CCW
                expected.append(bytes.fromhex(speed_frame(self.car.motor_ids[wheel], int(direction), abs(signed) * rpm, accel)))
        self.assertEqual(self.serial.writes, expected)


class DriveKinematicsTests(MecanumDriveTests):
    def test_calibrated_matrix_sends_engine_signs(self) -> None:
        serial = FakeSerial()
        car = MecanumDrive("/dev/ttyUSB0", motion_transform="calibrated", serial_port=serial)

        def speeds_for(vx: float, vy: float, omega: float) -> list[bytes]:
            serial.writes.clear()
            car.drive(vx, vy, omega, rpm=60)
            return list(serial.writes)

        forward_frames = [
            bytes.fromhex("01 F6 01 00 3C 0A 00 6B"),
            bytes.fromhex("02 F6 01 00 3C 0A 00 6B"),
            bytes.fromhex("03 F6 00 00 3C 0A 00 6B"),
            bytes.fromhex("04 F6 01 00 3C 0A 00 6B"),
        ]
        backward_frames = [
            bytes.fromhex("01 F6 00 00 3C 0A 00 6B"),
            bytes.fromhex("02 F6 00 00 3C 0A 00 6B"),
            bytes.fromhex("03 F6 01 00 3C 0A 00 6B"),
            bytes.fromhex("04 F6 00 00 3C 0A 00 6B"),
        ]
        strafe_left_frames = [
            bytes.fromhex("01 F6 00 00 3C 0A 00 6B"),
            bytes.fromhex("02 F6 01 00 3C 0A 00 6B"),
            bytes.fromhex("03 F6 00 00 3C 0A 00 6B"),
            bytes.fromhex("04 F6 00 00 3C 0A 00 6B"),
        ]
        rotate_ccw_frames = [
            bytes.fromhex("01 F6 00 00 3C 0A 00 6B"),
            bytes.fromhex("02 F6 01 00 3C 0A 00 6B"),
            bytes.fromhex("03 F6 01 00 3C 0A 00 6B"),
            bytes.fromhex("04 F6 01 00 3C 0A 00 6B"),
        ]

        # 标定表 vx 列 = [-1,-1,+1,-1]：forward 引擎符号 [-, -, +, -]
        self.assertEqual(speeds_for(1.0, 0.0, 0.0), forward_frames)
        # backward 整体取反：[+, +, -, +]
        self.assertEqual(speeds_for(-1.0, 0.0, 0.0), backward_frames)
        # vy 列 = [+1,-1,+1,+1]：strafe_left -> 引擎 [-, +, +, -]（物理）
        self.assertEqual(speeds_for(0.0, 1.0, 0.0), strafe_left_frames)
        # ω 列 = [+1,-1,-1,-1]：rotate_ccw -> 引擎 [-, +, -, +]（物理，cw/ccw 已对调）
        self.assertEqual(speeds_for(0.0, 0.0, 1.0), rotate_ccw_frames)

    def test_forward_all_wheels_cw(self) -> None:
        self.car.drive(1.0, 0.0, 0.0, rpm=60)
        self.assert_wheel_speeds((1, 1, 1, 1), rpm=60)

    def test_backward_all_wheels_ccw(self) -> None:
        self.car.drive(-1.0, 0.0, 0.0, rpm=60)
        self.assert_wheel_speeds((-1, -1, -1, -1), rpm=60)

    def test_strafe_left(self) -> None:
        self.car.drive(0.0, 1.0, 0.0, rpm=50)
        self.assert_wheel_speeds((1, -1, 1, -1), rpm=50)

    def test_strafe_right(self) -> None:
        self.car.drive(0.0, -1.0, 0.0, rpm=50)
        self.assert_wheel_speeds((-1, 1, -1, 1), rpm=50)

    def test_rotate_ccw(self) -> None:
        self.car.drive(0.0, 0.0, 1.0, rpm=40)
        self.assert_wheel_speeds((1, -1, -1, 1), rpm=40)

    def test_rotate_cw(self) -> None:
        self.car.drive(0.0, 0.0, -1.0, rpm=40)
        self.assert_wheel_speeds((-1, 1, 1, -1), rpm=40)

    def test_diagonal_forward_left(self) -> None:
        # vx=1, vy=1: FL/RL 叠加到 +2 满载，FR/RR 抵消到 0 -> 左排轮前进、右排轮停
        self.car.drive(1.0, 1.0, 0.0, rpm=60)
        self.assert_wheel_speeds((1, 0, 1, 0), rpm=60)

    def test_zero_input_stops(self) -> None:
        self.car.drive(0.0, 0.0, 0.0, rpm=60)
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FE 98 00 6B"), bytes.fromhex("02 FE 98 00 6B"),
             bytes.fromhex("03 FE 98 00 6B"), bytes.fromhex("04 FE 98 00 6B")],
        )

    def test_rejects_out_of_range_rpm(self) -> None:
        with self.assertRaises(ValueError):
            self.car.drive(1.0, 0.0, 0.0, rpm=3001)

    def test_custom_motor_ids_mapping(self) -> None:
        serial = FakeSerial()
        car = MecanumDrive(
            "/dev/ttyUSB0",
            motor_ids={WheelPosition.RL: 4, WheelPosition.RR: 3, WheelPosition.FL: 1, WheelPosition.FR: 2},
            serial_port=serial,
        )
        car.drive(1.0, 0.0, 0.0, rpm=30, acceleration=5)
        self.assertEqual(
            serial.writes,
            [
                bytes.fromhex("01 F6 00 00 1E 05 00 6B"),
                bytes.fromhex("02 F6 00 00 1E 05 00 6B"),
                bytes.fromhex("04 F6 00 00 1E 05 00 6B"),
                bytes.fromhex("03 F6 00 00 1E 05 00 6B"),
            ],
        )


class WheelSignTests(MecanumDriveTests):
    def test_wheel_sign_flips_single_wheel_direction(self) -> None:
        self.car.wheel_signs[WheelPosition.FR] = -1
        self.car.drive(1.0, 0.0, 0.0, rpm=60)
        self.assert_wheel_speeds((1, -1, 1, 1), rpm=60)


class ConvenienceMethodTests(MecanumDriveTests):
    def test_forward_wraps_drive(self) -> None:
        self.car.forward(rpm=60)
        self.assert_wheel_speeds((1, 1, 1, 1), rpm=60)

    def test_backward_wraps_drive(self) -> None:
        self.car.backward(rpm=60)
        self.assert_wheel_speeds((-1, -1, -1, -1), rpm=60)

    def test_strafe_left_wraps_drive(self) -> None:
        self.car.strafe_left(rpm=50)
        self.assert_wheel_speeds((1, -1, 1, -1), rpm=50)

    def test_strafe_right_wraps_drive(self) -> None:
        self.car.strafe_right(rpm=50)
        self.assert_wheel_speeds((-1, 1, -1, 1), rpm=50)

    def test_rotate_cw_wraps_drive(self) -> None:
        self.car.rotate_cw(rpm=40)
        self.assert_wheel_speeds((-1, 1, 1, -1), rpm=40)

    def test_rotate_ccw_wraps_drive(self) -> None:
        self.car.rotate_ccw(rpm=40)
        self.assert_wheel_speeds((1, -1, -1, 1), rpm=40)

    def test_diagonal_methods_wrap_drive(self) -> None:
        self.car.forward_left(rpm=60)
        self.assert_wheel_speeds((1, 0, 1, 0), rpm=60)

        self.serial.writes.clear()
        self.car.forward_right(rpm=60)
        self.assert_wheel_speeds((0, 1, 0, 1), rpm=60)

        self.serial.writes.clear()
        self.car.backward_left(rpm=60)
        self.assert_wheel_speeds((0, -1, 0, -1), rpm=60)

        self.serial.writes.clear()
        self.car.backward_right(rpm=60)
        self.assert_wheel_speeds((-1, 0, -1, 0), rpm=60)

    def test_duration_stops_after_sleep(self) -> None:
        self.car.forward(rpm=30, duration=0.01)
        expected = self.serial.writes[:4] + [
            bytes.fromhex("01 FE 98 00 6B"), bytes.fromhex("02 FE 98 00 6B"),
            bytes.fromhex("03 FE 98 00 6B"), bytes.fromhex("04 FE 98 00 6B"),
        ]
        self.assertEqual(self.serial.writes, expected)


class LowLevelApiTests(MecanumDriveTests):
    def test_move_wheels_relative_sends_four_position_frames(self) -> None:
        self.car.move_wheels_relative(90, -90, 180, -180, rpm=120, acceleration=5)
        self.assertEqual([frame[0] for frame in self.serial.writes[:4]], [1, 2, 3, 4])
        self.assertEqual([frame[1] for frame in self.serial.writes[:4]], [0xFD] * 4)
        self.assertEqual([frame[-2] for frame in self.serial.writes[:4]], [1] * 4)
        self.assertEqual(self.serial.writes[4], bytes.fromhex("00 FF 66 6B"))

    def test_move_wheels_absolute_sends_four_position_frames(self) -> None:
        self.car.move_wheels_absolute(90, 90, 90, 90, rpm=120, acceleration=5)
        self.assertEqual([frame[0] for frame in self.serial.writes[:4]], [1, 2, 3, 4])
        self.assertEqual([frame[1] for frame in self.serial.writes[:4]], [0xFD] * 4)
        self.assertEqual([frame[-2] for frame in self.serial.writes[:4]], [1] * 4)
        self.assertEqual(self.serial.writes[4], bytes.fromhex("00 FF 66 6B"))

    def test_set_speeds_direct(self) -> None:
        self.car.set_speeds(10, -20, 30, -40)
        self.assertEqual(
            self.serial.writes,
            [
                bytes.fromhex("01 F6 00 00 0A 0A 00 6B"),
                bytes.fromhex("02 F6 01 00 14 0A 00 6B"),
                bytes.fromhex("03 F6 00 00 1E 0A 00 6B"),
                bytes.fromhex("04 F6 01 00 28 0A 00 6B"),
            ],
        )

    def test_enable_all_sends_four_enable_frames(self) -> None:
        self.car.enable_all(True)
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex(f"{i:02X} F3 AB 01 00 6B") for i in range(1, 5)],
        )

    def test_disable_all_sends_four_disable_frames(self) -> None:
        self.car.disable_all()
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex(f"{i:02X} F3 AB 00 00 6B") for i in range(1, 5)],
        )

    def test_stop_wheel_sends_single_estop(self) -> None:
        self.car.stop_wheel(WheelPosition.RL)
        self.assertEqual(self.serial.writes, [bytes.fromhex("03 FE 98 00 6B")])

    def test_move_wheel_relative_passthrough(self) -> None:
        self.car.move_wheel_relative(WheelPosition.FL, degrees=-360.0, rpm=1500, acceleration=0)
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 01 05 DC 00 00 00 0C 80 00 00 6B")],
        )

    def test_home_wheel_passthrough(self) -> None:
        self.car.home_wheel(WheelPosition.RR, HomeMode.MULTI_TURN_COLLISION)
        self.assertEqual(self.serial.writes, [bytes.fromhex("04 9A 02 00 6B")])

    def test_get_wheel_returns_underlying_driver(self) -> None:
        self.assertIs(self.car.get_wheel(WheelPosition.FL), self.car.wheels[WheelPosition.FL])


class PatternOTests(unittest.TestCase):
    def test_pattern_o_strafe_left(self) -> None:
        serial = FakeSerial()
        car = MecanumDrive("/dev/ttyUSB0", pattern="O", serial_port=serial)
        car.drive(0.0, 1.0, 0.0, rpm=50)
        # O 排布：vy 列取反 -> FL=-1, FR=+1, RL=-1, RR=+1
        self.assertEqual(
            serial.writes,
            [
                bytes.fromhex("01 F6 01 00 32 0A 00 6B"),
                bytes.fromhex("02 F6 00 00 32 0A 00 6B"),
                bytes.fromhex("03 F6 01 00 32 0A 00 6B"),
                bytes.fromhex("04 F6 00 00 32 0A 00 6B"),
            ],
        )


if __name__ == "__main__":
    unittest.main()

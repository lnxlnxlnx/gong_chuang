"""Emm42 V5.0 command frames, verified without a physical serial device."""

from __future__ import annotations

import unittest

from modules.motor.drive.emm42 import (
    Direction,
    Emm42Driver,
    HomeMode,
    MotorStatus,
)


class FakeSerial:
    """Records writes and can acknowledge Emm42 commands like the driver does."""

    def __init__(
        self,
        replies: list[bytes] | None = None,
        *,
        acknowledge_emm42: bool = False,
        read_replies: dict[int, list[bytes]] | None = None,
    ) -> None:
        self._receive_buffer = bytearray(b"".join(replies or []))
        self._acknowledge_emm42 = acknowledge_emm42
        #: 功能码 -> 依次返回的应答帧；最后一条会一直重复，用来表达"状态不再变化"。
        self._read_replies = {
            function: list(frames) for function, frames in (read_replies or {}).items()
        }
        self.writes: list[bytes] = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        # 读取命令是 3 字节帧 [地址][功能码][校验]，驱动器按功能码回一条定长数据。
        # 读命令不参与控制命令的 ACK 机制，所以两条分支互斥（真机上也是这样：
        # Response=None 仍然会回答读命令，见说明书 4.2 / 8.1）。
        sequence = self._read_replies.get(frame[1]) if len(frame) == 3 else None
        if sequence:
            self._receive_buffer.extend(
                sequence.pop(0) if len(sequence) > 1 else sequence[0]
            )
        elif self._acknowledge_emm42 and len(frame) >= 3 and frame[-1] == 0x6B:
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


class FakeClock:
    """A monotonic clock that only moves when the injected sleep function is called."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def no_ack_driver(
    read_replies: dict[int, list[bytes]] | None = None,
) -> tuple[Emm42Driver, FakeSerial]:
    """A driver shaped like the arm's axes: ``Response=None``, so no control ACKs."""
    serial = FakeSerial(read_replies=read_replies)
    motor = Emm42Driver(
        "/dev/ttyUSB1", motor_id=1, serial_port=serial, expect_ack=False
    )
    return motor, serial


class Emm42DriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial(acknowledge_emm42=True)
        self.motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=self.serial)

    def test_initialize_keeps_injected_serial_port_open(self) -> None:
        self.assertIs(self.motor.initialize(), self.motor)
        self.assertTrue(self.serial.is_open)

    def test_enable_writes_enable_command(self) -> None:
        self.motor.enable(True)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_enable_rejects_missing_driver_ack(self) -> None:
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=FakeSerial())

        with self.assertRaisesRegex(RuntimeError, "no ACK"):
            motor.enable(True)

    def test_enable_can_skip_ack_when_driver_response_is_none(self) -> None:
        serial = FakeSerial()
        motor = Emm42Driver(
            "/dev/ttyUSB1",
            motor_id=1,
            serial_port=serial,
            expect_ack=False,
        )

        motor.enable(True)

        self.assertEqual(serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_set_speed_encodes_direction_speed_and_acceleration(self) -> None:
        self.motor.set_speed(Direction.CCW, rpm=1500, acceleration=10)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 F6 01 05 DC 0A 00 6B")])

    def test_move_relative_converts_degrees_to_pulses(self) -> None:
        self.motor.move_relative(
            degrees=-360.0,
            rpm=1500,
            acceleration=0,
        )
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 01 05 DC 00 00 00 0C 80 00 00 6B")],
        )

    def test_move_relative_can_be_buffered_for_synchronized_start(self) -> None:
        self.motor.move_relative(degrees=90.0, rpm=120, acceleration=5, synchronized=True)
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 00 00 78 05 00 00 03 20 00 01 6B")],
        )

    def test_trigger_sync_motion_broadcasts_start_frame(self) -> None:
        self.motor.trigger_sync_motion()
        self.assertEqual(self.serial.writes, [bytes.fromhex("00 FF 66 6B")])

    def test_move_absolute_encodes_absolute_mode(self) -> None:
        self.motor.move_absolute(
            degrees=360.0,
            rpm=1000,
            acceleration=8,
        )
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 00 03 E8 08 00 00 0C 80 01 00 6B")],
        )

    def test_stop_writes_immediate_stop_command(self) -> None:
        self.motor.stop()
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 FE 98 00 6B")])

    def test_stop_accepts_already_stopped_status(self) -> None:
        serial = FakeSerial(replies=[bytes.fromhex("01 FE E2 6B")])
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.stop()

        self.assertEqual(serial.writes, [bytes.fromhex("01 FE 98 00 6B")])

    def test_home_writes_requested_home_mode(self) -> None:
        self.motor.home(HomeMode.MULTI_TURN_COLLISION)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 9A 02 00 6B")])

    def test_absolute_zero_is_allowed_because_it_means_go_to_the_origin(self) -> None:
        """绝对模式下 0 是"回原点"，是回零之后归位的唯一一条命令形式。"""
        self.motor.move_absolute_pulses(0, rpm=300, acceleration=10)

        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 00 01 2C 0A 00 00 00 00 01 00 6B")],
        )

    def test_relative_zero_is_still_rejected(self) -> None:
        """相对模式下 0 位移没有意义，照旧拒绝。"""
        with self.assertRaisesRegex(ValueError, "relative"):
            self.motor.move_relative_pulses(0, rpm=300, acceleration=10)

        self.assertEqual(self.serial.writes, [])

    def test_absolute_negative_position_still_encodes_ccw(self) -> None:
        self.motor.move_absolute_pulses(-3200, rpm=300, acceleration=10)

        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 01 01 2C 0A 00 00 0C 80 01 00 6B")],
        )

    def test_set_speed_rejects_invalid_rpm(self) -> None:
        with self.assertRaises(ValueError):
            self.motor.set_speed(Direction.CW, rpm=3001, acceleration=0)


class Emm42ReadCommandTests(unittest.TestCase):
    """读取类命令：帧格式、返回长度、符号解码与错误帧（说明书 6.3.2 / 6.3.4）。"""

    def test_position_reply_is_eight_bytes_because_of_the_sign_byte(self) -> None:
        """7 字节是错的：位置值前面还有一个符号字节，少算一个字节整帧就错位。"""
        self.assertEqual(Emm42Driver.REPLY_LENGTHS[0x36], 8)
        self.assertEqual(Emm42Driver.REPLY_LENGTHS[0x3A], 4)
        self.assertEqual(Emm42Driver.REPLY_LENGTHS[0x3B], 4)

    def test_read_position_matches_the_manual_example_and_decodes_negative(
        self,
    ) -> None:
        """说明书 6.3.4 的示例：01 36 01 00 01 00 00 6B 即 -0x00010000 = -360.0°。"""
        motor, serial = no_ack_driver(
            {0x36: [bytes.fromhex("01 36 01 00 01 00 00 6B")]}
        )

        self.assertEqual(motor.read_position(), -65536)
        self.assertEqual(serial.writes, [bytes.fromhex("01 36 6B")])

    def test_read_position_decodes_a_positive_payload(self) -> None:
        motor, _ = no_ack_driver({0x36: [bytes.fromhex("01 36 00 00 00 0C 80 6B")]})

        self.assertEqual(motor.read_position(), 3200)

    def test_read_position_rejects_an_unknown_sign_byte(self) -> None:
        """符号字节只有 00/01；别的值说明帧已经错位，不能猜出一个"看起来合理"的数。"""
        motor, _ = no_ack_driver({0x36: [bytes.fromhex("01 36 02 00 00 0C 80 6B")]})

        with self.assertRaisesRegex(RuntimeError, "sign byte"):
            motor.read_position()

    def test_read_encoder_returns_the_linearised_single_turn_value(self) -> None:
        """0x31 是单圈绝对值（说明书 6.3.4 示例 0x8D9E = 36254），区分不了圈数。"""
        motor, serial = no_ack_driver({0x31: [bytes.fromhex("01 31 8D 9E 6B")]})

        self.assertEqual(motor.read_encoder(), 36254)
        self.assertEqual(serial.writes, [bytes.fromhex("01 31 6B")])

    def test_read_status_decodes_every_documented_bit(self) -> None:
        motor, _ = no_ack_driver({0x3A: [bytes.fromhex("01 3A 0F 6B")]})

        status = motor.read_status()

        self.assertTrue(status.enabled)
        self.assertTrue(status.in_position)
        self.assertTrue(status.stalled)
        self.assertTrue(status.stall_protected)

    def test_read_status_clears_every_bit_on_zero(self) -> None:
        motor, _ = no_ack_driver({0x3A: [bytes.fromhex("01 3A 00 6B")]})

        self.assertEqual(
            motor.read_status(),
            MotorStatus(
                enabled=False, in_position=False, stalled=False, stall_protected=False
            ),
        )

    def test_read_homing_status_separates_failure_from_still_running(self) -> None:
        """0x0C = 正在回零(0x04) + 回零失败(0x08)：两个位是独立的，不能只看一个。"""
        motor, _ = no_ack_driver({0x3B: [bytes.fromhex("01 3B 0C 6B")]})

        status = motor.read_homing_status()

        self.assertTrue(status.homing)
        self.assertTrue(status.failed)
        self.assertFalse(status.encoder_ready)
        self.assertFalse(status.calibration_ready)

    def test_read_commands_ignore_the_control_command_ack_setting(self) -> None:
        """``Response`` 只管控制动作命令，读命令两种设置下都必须能读（4.2 / 8.1）。"""
        for expect_ack in (True, False):
            with self.subTest(expect_ack=expect_ack):
                serial = FakeSerial(read_replies={0x3A: [bytes.fromhex("01 3A 03 6B")]})
                motor = Emm42Driver(
                    "/dev/ttyUSB1",
                    motor_id=1,
                    serial_port=serial,
                    expect_ack=expect_ack,
                )

                self.assertTrue(motor.read_status().in_position)
                self.assertEqual(serial.writes, [bytes.fromhex("01 3A 6B")])

    def test_read_command_surfaces_the_driver_error_frame(self) -> None:
        motor, _ = no_ack_driver({0x36: [bytes.fromhex("01 00 EE 6B")]})

        with self.assertRaisesRegex(RuntimeError, "rejected read command"):
            motor.read_position()

    def test_read_command_rejects_a_short_reply(self) -> None:
        motor, _ = no_ack_driver({0x36: [bytes.fromhex("01 36 00 00 0C 80 6B")]})

        with self.assertRaisesRegex(RuntimeError, "unexpected reply"):
            motor.read_position()

    def test_read_command_reports_a_missing_reply(self) -> None:
        motor, _ = no_ack_driver()

        with self.assertRaisesRegex(RuntimeError, "no reply"):
            motor.read_position()

    def test_read_command_rejects_an_unknown_function_code_before_writing(self) -> None:
        motor, serial = no_ack_driver()

        with self.assertRaises(ValueError):
            motor._read(0x99)

        self.assertEqual(serial.writes, [])


class Emm42WaitForPositionTests(unittest.TestCase):
    """``0x3A`` 轮询：先等到位位离开，再等它回来；超时与堵转都要能被调用方看见。"""

    def poll(self, flags_sequence, **kwargs):
        serial = FakeSerial(
            read_replies={
                0x3A: [bytes((0x01, 0x3A, flags, 0x6B)) for flags in flags_sequence]
            }
        )
        motor = Emm42Driver(
            "/dev/ttyUSB1", motor_id=1, serial_port=serial, expect_ack=False
        )
        clock = FakeClock()
        status = motor.wait_until_in_position(
            kwargs.pop("timeout", 5.0),
            poll_interval=0.25,
            sleep=clock.sleep,
            clock=clock,
            **kwargs,
        )
        return status, clock

    def test_waits_for_the_stale_in_position_bit_to_clear_before_waiting_for_it_again(
        self,
    ) -> None:
        """刚发完命令时到位位还留着上一次动作的结果；直接信它会等于根本没等。"""
        status, clock = self.poll([0x03, 0x01, 0x01, 0x03])

        self.assertTrue(status.in_position)
        self.assertEqual(len(clock.sleeps), 3)

    def test_returns_the_last_status_when_the_move_never_settles(self) -> None:
        status, clock = self.poll([0x01], timeout=1.0)

        self.assertFalse(status.in_position)
        self.assertFalse(status.stalled)
        self.assertGreaterEqual(clock.now, 1.0)

    def test_gives_up_once_the_axis_reports_a_stall_instead_of_burning_the_timeout(
        self,
    ) -> None:
        """堵转不会再到位：继续等只是白等，而且会把"撞上了"拖成一次超时。"""
        status, clock = self.poll([0x01, 0x05], timeout=30.0)

        self.assertTrue(status.stalled)
        self.assertFalse(status.in_position)
        self.assertLess(clock.now, 30.0)

    def test_gives_up_once_stall_protection_has_tripped(self) -> None:
        status, clock = self.poll([0x09], timeout=30.0)

        self.assertTrue(status.stall_protected)
        self.assertEqual(clock.sleeps, [])

    def test_an_already_settled_move_returns_without_burning_the_timeout(self) -> None:
        status, clock = self.poll([0x03], timeout=30.0, departure_timeout=0.0)

        self.assertTrue(status.in_position)
        self.assertEqual(clock.sleeps, [])

    def test_rejects_invalid_timing_arguments(self) -> None:
        motor, _ = no_ack_driver({0x3A: [bytes.fromhex("01 3A 00 6B")]})

        with self.assertRaises(ValueError):
            motor.wait_until_in_position(-1.0)
        with self.assertRaises(ValueError):
            motor.wait_until_in_position(1.0, poll_interval=0.0)
        with self.assertRaises(ValueError):
            motor.wait_until_in_position(1.0, departure_timeout=-0.5)


if __name__ == "__main__":
    unittest.main()

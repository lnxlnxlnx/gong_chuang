"""Verify the conservative three-axis robot-arm position controller offline."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from modules.motor.drive.motion import estimate_move_seconds
from modules.motor.drive.emm42 import Direction
from modules.arm.runtime import WAIT_ESTIMATE, WAIT_IN_POSITION, RobotArm
from modules.arm.config import ArmConfig, AxisConfig


# Independent protocol fixtures: these literals intentionally do not reuse the
# controller implementation, so a wrong command field still fails a test.
POSITION_COMMAND = 0xFD
TORQUE_COMMAND = 0xF3
EMERGENCY_STOP_COMMAND = 0xFE
SAFE_TEST_RPM = 10
SAFE_TEST_ACCELERATION = 5
PULSES_PER_REVOLUTION = 3200
RELATIVE_POSITION = 0
IMMEDIATE_EXECUTION = 0
FRAME_TAIL = 0x6B


def fixture_config(**overrides) -> ArmConfig:
    """Explicit arm settings so the frame fixtures do not follow modules/arm/config.json.

    The editable JSON is allowed to change (the operator tunes rpm there), while
    these tests must keep checking the exact position frame layout.
    """
    settings = dict(
        pulses_per_revolution=PULSES_PER_REVOLUTION,
        rpm=SAFE_TEST_RPM,
        acceleration=SAFE_TEST_ACCELERATION,
        default_pulses=800,
        max_pulses=5000,
        axes={
            "base": AxisConfig(motor_id=5, reduction_ratio=6),
            "lift": AxisConfig(motor_id=6),
            "extension": AxisConfig(motor_id=7),
        },
    )
    settings.update(overrides)
    return ArmConfig(**settings)


class RecordingSleep:
    """Records requested waits instead of sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def relative_position_frame(
    *,
    motor_id: int,
    direction: Direction,
    pulses: int,
    rpm: int = SAFE_TEST_RPM,
    acceleration: int = SAFE_TEST_ACCELERATION,
) -> bytes:
    """Build the documented Emm42 relative-position frame used by arm tests."""
    return (
        bytes((motor_id, POSITION_COMMAND, int(direction)))
        + rpm.to_bytes(2, "big")
        + bytes((acceleration,))
        + pulses.to_bytes(4, "big")
        + bytes((RELATIVE_POSITION, IMMEDIATE_EXECUTION, FRAME_TAIL))
    )


def torque_frame(*, motor_id: int, enabled: bool) -> bytes:
    """Build a documented Emm42 torque enable or disable frame."""
    return bytes((motor_id, TORQUE_COMMAND, 0xAB, int(enabled), 0, FRAME_TAIL))


def emergency_stop_frame(*, motor_id: int) -> bytes:
    """Build a documented Emm42 immediate-stop frame."""
    return bytes((motor_id, EMERGENCY_STOP_COMMAND, 0x98, 0, FRAME_TAIL))


class FakeSerial:
    """In-memory Emm42 bus that ACKs every command frame."""

    def __init__(self) -> None:
        self._receive_buffer = bytearray()
        self.writes: list[bytes] = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        self._receive_buffer.extend(bytes((frame[0], frame[1], 0x02, 0x6B)))
        return len(frame)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return data

    def close(self) -> None:
        self.is_open = False


class ArmTestCase(unittest.TestCase):
    """Build an arm through the assembly seam, with waits recorded instead of slept.

    ``RobotArm._from_settings`` 是 ``open`` 与假设备测试共用的装配接缝：它自己建三个
    驱动器（真实协议 + 注入的内存串口）并使能，不做任何运动。
    """

    def make_arm(self, serial=None, config=None):
        self.sleep = RecordingSleep()
        self.serial = serial if serial is not None else FakeSerial()
        self.arm = RobotArm._from_settings(
            config=config if config is not None else fixture_config(),
            port="/dev/ttyUSB0",
            bus=self.serial,
            sleep=self.sleep,
        )
        # 装配时已经使能三个轴（3 条 0xF3 帧）；本类只看之后的动作帧，使能本身由
        # RobotArmConstructionTests 与 enable 用例单独断言。
        self.serial.writes.clear()
        return self.arm


class RobotArmBaseAxisTests(ArmTestCase):
    def test_rotate_cw_uses_position_mode_safe_limits_and_reports_output_angle(self) -> None:
        arm = self.make_arm()

        report = arm.rotate_cw(pulses=1000)

        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=5, direction=Direction.CW, pulses=1000)],
        )
        self.assertEqual(report.motor_id, 5)
        self.assertEqual(report.pulses, 1000)
        self.assertEqual(report.motor_degrees, 112.5)
        self.assertEqual(report.output_degrees, 112.5 / arm.config.base_reduction)

    def test_current_response_none_configuration_does_not_require_an_ack(self) -> None:
        class NoAckSerial(FakeSerial):
            def write(self, data: bytes) -> int:
                frame = bytes(data)
                self.writes.append(frame)
                return len(frame)

        arm = self.make_arm(serial=NoAckSerial())

        arm.enable_all()
        arm.rotate_cw(pulses=1)

        self.assertEqual(
            self.serial.writes,
            [
                torque_frame(motor_id=5, enabled=True),
                torque_frame(motor_id=6, enabled=True),
                torque_frame(motor_id=7, enabled=True),
                relative_position_frame(motor_id=5, direction=Direction.CW, pulses=1),
            ],
        )

    def test_rejects_unsafe_or_non_integer_pulse_counts_before_writing(self) -> None:
        """越界与非整数照旧在发帧之前拒绝；``pulses=0`` 不再是错误，见下一条。"""
        arm = self.make_arm()

        for pulses in (arm.config.max_pulses + 1, 1.5):
            with self.subTest(pulses=pulses):
                with self.assertRaises(ValueError):
                    arm.rotate_cw(pulses=pulses)  # type: ignore[arg-type]

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep.calls, [])

    def test_zero_pulses_is_a_no_op_rather_than_an_error_or_a_zero_pulse_frame(self) -> None:
        """新契约：位移不足半个脉冲就是"已经到位"——不发帧、不报错、不改变位置。

        旧控制器把 0 当成非法输入拒绝；``RobotArm._signed_move`` 现在把它定义成空操作
        （与 ``move_base_degrees(0.0)`` 同一条规矩），所以断言改成"什么都不发"。
        """
        arm = self.make_arm()

        report = arm.rotate_cw(pulses=0)

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep.calls, [])
        self.assertEqual(report.pulses, 0)
        self.assertEqual(report.estimated_seconds, 0.0)

    def test_uses_the_configured_motion_limit_and_speed(self) -> None:
        config = fixture_config(pulses_per_revolution=4000, default_pulses=1200)
        arm = self.make_arm(config=config)

        arm.rotate_cw(pulses=5000)

        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=5, direction=Direction.CW, pulses=5000)],
        )
        with self.assertRaises(ValueError):
            arm.rotate_cw(pulses=5001)


class RobotArmWaitTests(ArmTestCase):
    def test_move_waits_for_the_estimated_motion_time(self) -> None:
        arm = self.make_arm()

        report = arm.move_up(pulses=1000)

        expected = estimate_move_seconds(1000, SAFE_TEST_RPM, PULSES_PER_REVOLUTION, SAFE_TEST_ACCELERATION)
        self.assertEqual(self.sleep.calls, [expected])
        self.assertEqual(report.estimated_seconds, expected)
        # 10 RPM + 加速度档 5 + 1000 脉冲（1/3.2 圈）约 2.15 s：必须真的等，不能立刻发下一条。
        self.assertGreater(expected, 2.0)
        self.assertLess(expected, 2.5)

    def test_wait_false_sends_the_same_frame_without_blocking(self) -> None:
        arm = self.make_arm()

        report = arm.move_up(pulses=1000, wait=False)

        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=6, direction=Direction.CW, pulses=1000)],
        )
        self.assertEqual(self.sleep.calls, [])
        self.assertGreater(report.estimated_seconds, 0.0)

    def test_rpm_and_acceleration_can_be_overridden_per_move(self) -> None:
        arm = self.make_arm()

        arm.move_forward(pulses=400, rpm=20, acceleration=0)

        self.assertEqual(
            self.serial.writes,
            [
                relative_position_frame(
                    motor_id=7,
                    direction=Direction.CW,
                    pulses=400,
                    rpm=20,
                    acceleration=0,
                )
            ],
        )
        # 加速度档 0 = 无曲线，只等匀速段 + 到位余量。
        expected = estimate_move_seconds(400, 20, PULSES_PER_REVOLUTION, 0)
        self.assertEqual(self.sleep.calls, [expected])

    def test_zero_rpm_is_rejected_before_any_frame_is_sent(self) -> None:
        arm = self.make_arm()

        with self.assertRaises(ValueError):
            arm.rotate_cw(pulses=100, rpm=0)

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep.calls, [])


class FakeClock:
    """A monotonic clock that only advances when something calls its sleep."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class PositionStatusSerial(FakeSerial):
    """Records every frame and answers ``0x3A`` from a scripted flag sequence.

    控制帧**不回 ACK**：5/6/7 号轴是 ``Response=None``，而且真机上不回的应答会留在
    接收缓冲里，把紧跟着的读命令答非所问地顶掉——这台假串口照实模拟这一点。
    """

    def __init__(self, flags_sequence: list[int]) -> None:
        super().__init__()
        self._flag_sequence = list(flags_sequence)

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        # [地址][0x3A][校验] 是读取命令；其余都是控制帧，只记录不回包。
        if len(frame) == 3 and frame[-1] == FRAME_TAIL:
            flags = (
                self._flag_sequence.pop(0)
                if len(self._flag_sequence) > 1
                else self._flag_sequence[0]
            )
            self._receive_buffer.extend(bytes((frame[0], frame[1], flags, FRAME_TAIL)))
        return len(frame)


class RobotArmInPositionWaitTests(unittest.TestCase):
    """``WAIT_IN_POSITION``：等的是驱动器实测的到位位，不是加减速公式。"""

    def make_polling_arm(self, flags_sequence, config=None):
        self.clock = FakeClock()
        self.serial = PositionStatusSerial(flags_sequence)
        self.arm = RobotArm._from_settings(
            config=config if config is not None else fixture_config(),
            port="/dev/ttyUSB0",
            bus=self.serial,
            sleep=self.clock.sleep,
            clock=self.clock,
            wait_mode=WAIT_IN_POSITION,
        )
        self.serial.writes.clear()
        return self.arm

    def test_move_returns_as_soon_as_the_driver_reports_in_position(self) -> None:
        """到位位说到了就返回：不再无条件睡满估算的 2.15 s。"""
        arm = self.make_polling_arm([0x03, 0x01, 0x03])

        report = arm.move_up(pulses=1000)

        # 一条动作帧，然后三次状态读取：离开到位区、确认离开、等到位。
        self.assertEqual(
            self.serial.writes,
            [relative_position_frame(motor_id=6, direction=Direction.CW, pulses=1000)]
            + [bytes((6, 0x3A, FRAME_TAIL))] * 3,
        )
        self.assertIsNotNone(report.motor_status)
        assert report.motor_status is not None
        self.assertTrue(report.motor_status.in_position)
        self.assertLess(report.waited_seconds, report.estimated_seconds)

    def test_move_hands_a_stall_back_to_the_caller_instead_of_sleeping_through_it(
        self,
    ) -> None:
        """估算模式对堵转完全无感；轮询模式必须把它交回调用方。"""
        arm = self.make_polling_arm([0x01, 0x05])

        report = arm.move_up(pulses=1000)

        self.assertIsNotNone(report.motor_status)
        assert report.motor_status is not None
        self.assertTrue(report.motor_status.stalled)
        self.assertFalse(report.motor_status.in_position)
        self.assertLess(report.waited_seconds, report.estimated_seconds)

    def test_estimate_mode_still_reports_no_status_at_all(self) -> None:
        """默认路径不变：估算模式不读驱动器，所以也编不出一个状态来。"""
        self.sleep = RecordingSleep()
        arm = RobotArm._from_settings(
            config=fixture_config(),
            port="/dev/ttyUSB0",
            bus=FakeSerial(),
            sleep=self.sleep,
            wait_mode=WAIT_ESTIMATE,
        )

        report = arm.move_up(pulses=1000)

        self.assertIsNone(report.motor_status)
        self.assertEqual(report.waited_seconds, report.estimated_seconds)
        self.assertEqual(self.sleep.calls, [report.estimated_seconds])

    def test_unknown_wait_mode_is_rejected_before_the_serial_is_opened(self) -> None:
        with self.assertRaises(ValueError):
            RobotArm._from_settings(
                config=fixture_config(),
                port="/dev/ttyUSB0",
                bus=FakeSerial(),
                wait_mode="whatever",
            )


class RobotArmAxisMappingTests(ArmTestCase):
    def test_physical_convenience_methods_use_the_confirmed_addresses_and_directions(self) -> None:
        arm = self.make_arm()

        arm.rotate_ccw(pulses=1)
        arm.move_up(pulses=1)
        arm.move_down(pulses=1)
        arm.move_forward(pulses=1)
        arm.move_backward(pulses=1)

        self.assertEqual(
            self.serial.writes,
            [
                relative_position_frame(motor_id=5, direction=Direction.CCW, pulses=1),
                relative_position_frame(motor_id=6, direction=Direction.CW, pulses=1),
                relative_position_frame(motor_id=6, direction=Direction.CCW, pulses=1),
                relative_position_frame(motor_id=7, direction=Direction.CW, pulses=1),
                relative_position_frame(motor_id=7, direction=Direction.CCW, pulses=1),
            ],
        )

    def test_enable_stop_and_disable_all_axes_in_address_order(self) -> None:
        arm = self.make_arm()

        arm.enable_all()
        arm.stop()
        arm.disable_all()

        self.assertEqual(
            self.serial.writes,
            [
                torque_frame(motor_id=5, enabled=True),
                torque_frame(motor_id=6, enabled=True),
                torque_frame(motor_id=7, enabled=True),
                emergency_stop_frame(motor_id=5),
                emergency_stop_frame(motor_id=6),
                emergency_stop_frame(motor_id=7),
                torque_frame(motor_id=5, enabled=False),
                torque_frame(motor_id=6, enabled=False),
                torque_frame(motor_id=7, enabled=False),
            ],
        )


class RobotArmConstructionTests(unittest.TestCase):
    """装配中途失败时的资源归属。

    ``RobotArm._from_settings`` 会**先打开共享串口、再建三个驱动器并使能**。装配没返回
    时调用方拿不到对象，所以那个串口只能由装配方自己收尾——不收的话下一次装配会莫名其妙地
    打不开同一个口，而且现场看不出是谁占着。借来的串口**不归**本实例管。
    """

    def test_assembly_enables_every_axis_without_moving(self) -> None:
        """装配完就是"已使能、没动过"：三条使能帧，没有动作帧，位姿仍是未知。"""
        serial = FakeSerial()

        arm = RobotArm._from_settings(
            config=fixture_config(), port="/dev/ttyUSB0", bus=serial
        )

        self.assertEqual(
            serial.writes,
            [torque_frame(motor_id=motor_id, enabled=True) for motor_id in (5, 6, 7)],
        )
        self.assertIsNone(arm.pose, "装配不做任何运动，位姿必须未知")

    def test_motor_failure_closes_the_serial_the_assembly_opened(self) -> None:
        serial = FakeSerial()
        opened = []

        def fake_open(port):
            opened.append(port)
            return serial

        with patch.object(RobotArm, "_open_serial", fake_open):
            with patch(
                "modules.arm.runtime.Emm42Driver",
                side_effect=RuntimeError("motor 6 不存在"),
            ):
                with self.assertRaisesRegex(RuntimeError, "motor 6 不存在"):
                    RobotArm._from_settings(config=fixture_config(), port="/dev/ttyUSB0")
        self.assertEqual(opened, ["/dev/ttyUSB0"], "串口确实由本次装配打开")
        self.assertFalse(serial.is_open, "自己开的串口必须由自己关掉")

    def test_injected_serial_is_never_closed_by_a_failed_assembly(self) -> None:
        """借来的 ``bus`` **不归**本实例管，失败时不许擅自关掉。"""
        serial = FakeSerial()
        with patch(
            "modules.arm.runtime.Emm42Driver",
            side_effect=RuntimeError("motor 6 不存在"),
        ):
            with self.assertRaisesRegex(RuntimeError, "motor 6 不存在"):
                RobotArm._from_settings(
                    config=fixture_config(), port="/dev/ttyUSB0", bus=serial
                )
        self.assertTrue(serial.is_open, "借来的串口不该被失败的装配关掉")

    def test_successful_assembly_keeps_the_serial_open(self) -> None:
        """正常路径不能被收尾逻辑误伤：串口要保持打开，三个轴都建好。"""
        arm = ArmTestCase().make_arm()
        self.assertTrue(arm._bus.is_open)
        self.assertEqual(sorted(arm._motors), ["base", "extension", "lift"])

    def test_cleanup_failure_keeps_the_assembly_error_as_primary(self) -> None:
        """关串口也失败时，装配失败仍是主异常，关串口的错误挂在 ``__cause__``。"""
        serial = FakeSerial()

        def broken_close() -> None:
            raise RuntimeError("close broke")

        serial.close = broken_close
        with patch.object(RobotArm, "_open_serial", lambda port: serial):
            with patch(
                "modules.arm.runtime.Emm42Driver",
                side_effect=RuntimeError("motor 6 不存在"),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    RobotArm._from_settings(config=fixture_config(), port="/dev/ttyUSB0")
        self.assertIn("motor 6 不存在", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("close broke", str(caught.exception.__cause__))


if __name__ == "__main__":
    unittest.main()

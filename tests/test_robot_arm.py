"""Verify the conservative three-axis robot-arm position controller offline."""

from __future__ import annotations

import unittest

from motion import estimate_move_seconds
from motor_drivers.emm42 import Direction
from robot_arm import RobotArm
from robot_arm_config import ArmConfig, AxisConfig


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
    """Explicit arm settings so the frame fixtures do not follow config/robot_arm.json.

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
    """Build an arm whose waits are recorded instead of slept through."""

    def make_arm(self, serial=None, config=None):
        self.sleep = RecordingSleep()
        self.serial = serial if serial is not None else FakeSerial()
        self.arm = RobotArm(
            "/dev/ttyUSB0",
            config=config if config is not None else fixture_config(),
            serial_port=self.serial,
            sleep=self.sleep,
        )
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
        arm = self.make_arm()

        for pulses in (0, arm.config.max_pulses + 1, 1.5):
            with self.subTest(pulses=pulses):
                with self.assertRaises(ValueError):
                    arm.rotate_cw(pulses=pulses)  # type: ignore[arg-type]

        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep.calls, [])

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
        arm.stop_all()
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


if __name__ == "__main__":
    unittest.main()

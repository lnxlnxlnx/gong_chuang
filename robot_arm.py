"""Conservative position-mode controller for the three ZDT Emm42 arm axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from motor_drivers.emm42 import Direction, Emm42Driver, SerialPort


@dataclass(frozen=True)
class MotionReport:
    """Measured command quantity; linear axes have no millimeter calibration yet."""

    motor_id: int
    pulses: int
    motor_degrees: float
    output_degrees: Optional[float]


class RobotArm:
    """Control arm motors 5--7 using only bounded relative position commands."""

    BASE_ID = 5
    LIFT_ID = 6
    EXTENSION_ID = 7
    AXIS_IDS = (BASE_ID, LIFT_ID, EXTENSION_ID)
    TEST_RPM = 10
    TEST_ACCELERATION = 5
    MAX_TEST_PULSES = 1000
    BASE_REDUCTION = 6

    def __init__(
        self,
        port: str,
        *,
        pulses_per_revolution: int = 3200,
        serial_port: Optional[SerialPort] = None,
    ) -> None:
        if pulses_per_revolution <= 0:
            raise ValueError("pulses_per_revolution must be positive")
        self.pulses_per_revolution = pulses_per_revolution
        self.port = port
        self._serial = serial_port if serial_port is not None else self._open_shared_serial()
        self._motors = {
            motor_id: Emm42Driver(
                port,
                motor_id,
                pulses_per_revolution=pulses_per_revolution,
                # 已读取的 5/6/7 配置均为 Response=None，控制帧不回 ACK。
                expect_ack=False,
                serial_port=self._serial,
            )
            for motor_id in self.AXIS_IDS
        }

    def initialize(self) -> "RobotArm":
        """Return the controller after the shared bus has been prepared."""
        return self

    def close(self) -> None:
        """Close the one shared serial connection used by all three axes."""
        self._serial.close()

    def enable_all(self) -> None:
        """Enable torque on base, lift, and extension axes in address order."""
        for motor_id in self.AXIS_IDS:
            self._motors[motor_id].enable(True)

    def disable_all(self) -> None:
        """Release torque on every axis after a test completes or fails."""
        for motor_id in self.AXIS_IDS:
            self._motors[motor_id].enable(False)

    def stop_all(self) -> None:
        """Immediately brake all axes; this is valid during position motion."""
        for motor_id in self.AXIS_IDS:
            self._motors[motor_id].stop()

    def rotate_cw(self, pulses: int) -> MotionReport:
        """Rotate the base clockwise within the fixed safe position-mode limit."""
        return self._move(self.BASE_ID, Direction.CW, pulses)

    def rotate_ccw(self, pulses: int) -> MotionReport:
        """Rotate the base counterclockwise within the fixed safe limit."""
        return self._move(self.BASE_ID, Direction.CCW, pulses)

    def move_up(self, pulses: int) -> MotionReport:
        """Move the lift axis upward; CW is the mechanically confirmed direction."""
        return self._move(self.LIFT_ID, Direction.CW, pulses)

    def move_down(self, pulses: int) -> MotionReport:
        """Move the lift axis downward; CCW is the mechanically confirmed direction."""
        return self._move(self.LIFT_ID, Direction.CCW, pulses)

    def move_forward(self, pulses: int) -> MotionReport:
        """Extend the arm forward; CW is the mechanically confirmed direction."""
        return self._move(self.EXTENSION_ID, Direction.CW, pulses)

    def move_backward(self, pulses: int) -> MotionReport:
        """Retract the arm backward; CCW is the mechanically confirmed direction."""
        return self._move(self.EXTENSION_ID, Direction.CCW, pulses)

    def _move(self, motor_id: int, direction: Direction, pulses: int) -> MotionReport:
        self._require_safe_pulses(pulses)
        motor_degrees = pulses * 360 / self.pulses_per_revolution
        signed_degrees = motor_degrees if direction is Direction.CW else -motor_degrees
        self._motors[motor_id].move_relative(
            signed_degrees,
            rpm=self.TEST_RPM,
            acceleration=self.TEST_ACCELERATION,
        )
        return MotionReport(
            motor_id=motor_id,
            pulses=pulses,
            motor_degrees=motor_degrees,
            output_degrees=motor_degrees / self.BASE_REDUCTION if motor_id == self.BASE_ID else None,
        )

    def _open_shared_serial(self) -> SerialPort:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required on Jetson. Install it in the target Python environment first."
            ) from error
        return serial.Serial(self.port, baudrate=Emm42Driver.BAUDRATE, timeout=0.1)

    @classmethod
    def _require_safe_pulses(cls, pulses: int) -> None:
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        if not 1 <= pulses <= cls.MAX_TEST_PULSES:
            raise ValueError(f"pulses must be in 1..{cls.MAX_TEST_PULSES}")

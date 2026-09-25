"""Conservative position-mode controller for the three ZDT Emm42 arm axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from motor_drivers.emm42 import Direction, Emm42Driver, SerialPort
from robot_arm_config import AXIS_NAMES, ArmConfig, load_arm_config


@dataclass(frozen=True)
class MotionReport:
    """Measured command quantity; linear axes have no millimeter calibration yet."""

    motor_id: int
    pulses: int
    motor_degrees: float
    output_degrees: Optional[float]


class RobotArm:
    """Control arm motors 5--7 using only bounded relative position commands."""

    BASE_AXIS = "base"
    LIFT_AXIS = "lift"
    EXTENSION_AXIS = "extension"

    def __init__(
        self,
        port: str,
        *,
        config: Optional[ArmConfig] = None,
        serial_port: Optional[SerialPort] = None,
    ) -> None:
        self.config = config if config is not None else load_arm_config()
        self.pulses_per_revolution = self.config.pulses_per_revolution
        self.port = port
        self._serial = serial_port if serial_port is not None else self._open_shared_serial()
        self._motors = {
            axis_name: Emm42Driver(
                port,
                self.config.motor_id(axis_name),
                pulses_per_revolution=self.pulses_per_revolution,
                # 已读取的 5/6/7 配置均为 Response=None，控制帧不回 ACK。
                expect_ack=False,
                serial_port=self._serial,
            )
            for axis_name in AXIS_NAMES
        }

    def initialize(self) -> "RobotArm":
        """Return the controller after the shared bus has been prepared."""
        return self

    def close(self) -> None:
        """Close the one shared serial connection used by all three axes."""
        self._serial.close()

    def enable_all(self) -> None:
        """Enable torque on base, lift, and extension axes in address order."""
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].enable(True)

    def disable_all(self) -> None:
        """Release torque on every axis after a test completes or fails."""
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].enable(False)

    def stop_all(self) -> None:
        """Immediately brake all axes; this is valid during position motion."""
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].stop()

    def rotate_cw(self, pulses: int) -> MotionReport:
        """Rotate the base clockwise within the fixed safe position-mode limit."""
        return self._move(self.BASE_AXIS, Direction.CW, pulses)

    def rotate_ccw(self, pulses: int) -> MotionReport:
        """Rotate the base counterclockwise within the fixed safe limit."""
        return self._move(self.BASE_AXIS, Direction.CCW, pulses)

    def move_up(self, pulses: int) -> MotionReport:
        """Move the lift axis upward; CW is the mechanically confirmed direction."""
        return self._move(self.LIFT_AXIS, Direction.CW, pulses)

    def move_down(self, pulses: int) -> MotionReport:
        """Move the lift axis downward; CCW is the mechanically confirmed direction."""
        return self._move(self.LIFT_AXIS, Direction.CCW, pulses)

    def move_forward(self, pulses: int) -> MotionReport:
        """Extend the arm forward; CW is the mechanically confirmed direction."""
        return self._move(self.EXTENSION_AXIS, Direction.CW, pulses)

    def move_backward(self, pulses: int) -> MotionReport:
        """Retract the arm backward; CCW is the mechanically confirmed direction."""
        return self._move(self.EXTENSION_AXIS, Direction.CCW, pulses)

    def _move(self, axis_name: str, direction: Direction, pulses: int) -> MotionReport:
        self._require_safe_pulses(pulses)
        motor_degrees = pulses * 360 / self.pulses_per_revolution
        signed_degrees = motor_degrees if direction is Direction.CW else -motor_degrees
        self._motors[axis_name].move_relative(
            signed_degrees,
            rpm=self.config.rpm,
            acceleration=self.config.acceleration,
        )
        axis = self.config.axes[axis_name]
        return MotionReport(
            motor_id=axis.motor_id,
            pulses=pulses,
            motor_degrees=motor_degrees,
            output_degrees=(
                motor_degrees / axis.reduction_ratio
                if axis.reduction_ratio is not None
                else None
            ),
        )

    def _open_shared_serial(self) -> SerialPort:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required on Jetson. Install it in the target Python environment first."
            ) from error
        return serial.Serial(self.port, baudrate=Emm42Driver.BAUDRATE, timeout=0.1)

    def _require_safe_pulses(self, pulses: int) -> None:
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        if not 1 <= pulses <= self.config.max_pulses:
            raise ValueError(f"pulses must be in 1..{self.config.max_pulses}")

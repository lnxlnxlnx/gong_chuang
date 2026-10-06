"""Conservative position-mode controller for the three ZDT Emm42 arm axes.

参数约定（与 ``mecanum.MecanumDrive`` 完全一致，见 ``motion.py``）：

- ``pulses``：本次相对移动的脉冲数（必填，正数；方向由动作方法决定）；
- ``rpm`` / ``acceleration``：可选覆盖 ``config/robot_arm.json`` 中的默认值；
- ``wait``：``True``（默认）等到动作走完再返回，``False`` 只发命令。

三个轴的驱动器是 ``Response=None``：控制帧没有 ACK，也没有到位回包，所以
``wait`` 用 ``motion.estimate_move_seconds`` 按说明书公式估算动作耗时后等待。
不等待时连续调用会让后一条位置帧覆盖前一条的目标，看起来就是"上一个动作没执行完"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from motion import DEFAULT_SETTLE_SECONDS, estimate_move_seconds
from motor_drivers.emm42 import Direction, Emm42Driver, SerialPort
from robot_arm_config import AXIS_NAMES, ArmConfig, load_arm_config


@dataclass(frozen=True)
class MotionReport:
    """Measured command quantity; linear axes have no millimeter calibration yet."""

    motor_id: int
    pulses: int
    motor_degrees: float
    output_degrees: Optional[float]
    #: 本次动作的估算耗时（秒）；``wait=False`` 时调用方可以自己用它来等。
    estimated_seconds: float = 0.0


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
        sleep: Callable[[float], None] = time.sleep,
        settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    ) -> None:
        self.config = config if config is not None else load_arm_config()
        self.pulses_per_revolution = self.config.pulses_per_revolution
        self.port = port
        #: 到位余量（秒），估算耗时里额外留出的安全时间。
        self.settle_seconds = settle_seconds
        # 注入 sleep 便于离线单测；真机默认 time.sleep。
        self._sleep = sleep
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

    def rotate_cw(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Rotate the base clockwise within the fixed safe position-mode limit."""
        return self._move(self.BASE_AXIS, Direction.CW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def rotate_ccw(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Rotate the base counterclockwise within the fixed safe limit."""
        return self._move(self.BASE_AXIS, Direction.CCW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_up(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Move the lift axis upward; CW is the mechanically confirmed direction."""
        return self._move(self.LIFT_AXIS, Direction.CW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_down(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Move the lift axis downward; CCW is the mechanically confirmed direction."""
        return self._move(self.LIFT_AXIS, Direction.CCW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_forward(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Extend the arm forward; CW is the mechanically confirmed direction."""
        return self._move(self.EXTENSION_AXIS, Direction.CW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_backward(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Retract the arm backward; CCW is the mechanically confirmed direction."""
        return self._move(self.EXTENSION_AXIS, Direction.CCW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def _move(
        self,
        axis_name: str,
        direction: Direction,
        pulses: int,
        *,
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
    ) -> MotionReport:
        self._require_safe_pulses(pulses)
        speed = self.config.rpm if rpm is None else rpm
        accel = self.config.acceleration if acceleration is None else acceleration
        # 先算耗时再发命令：参数非法时抛出，串口里不会出现半条动作。
        seconds = estimate_move_seconds(
            pulses,
            speed,
            self.pulses_per_revolution,
            accel,
            self.settle_seconds,
        )
        signed_pulses = pulses if direction is Direction.CW else -pulses
        self._motors[axis_name].move_relative_pulses(signed_pulses, speed, accel)
        if wait:
            # 驱动器只收命令、不回 ACK / 到位包，只能按估算耗时等它走完。
            self._sleep(seconds)
        motor_degrees = pulses * 360 / self.pulses_per_revolution
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
            estimated_seconds=seconds,
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

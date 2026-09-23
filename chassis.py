"""High-level mecanum chassis API.

`MecanumDrive` owns motor protocol and kinematics. `MecanumChassis` is the
small application-facing facade for whole-vehicle actions.
"""

from __future__ import annotations

from mecanum import MecanumDrive, WheelPosition


class MecanumChassis:
    """面向上层应用的麦轮底盘 API。"""

    def __init__(self, drive: MecanumDrive) -> None:
        self.drive = drive

    def initialize(self) -> "MecanumChassis":
        self.drive.initialize()
        return self

    def close(self) -> None:
        self.drive.close()

    def enable(self) -> None:
        self.drive.enable_all()

    def enable_all(self) -> None:
        self.enable()

    def disable(self) -> None:
        self.drive.disable_all()

    def disable_all(self) -> None:
        self.disable()

    def stop(self) -> None:
        self.drive.stop()

    def wheel(self, wheel: WheelPosition):
        """返回底层单轮对象，供回零等维护操作使用。"""
        return self.drive.get_wheel(wheel)

    def set_wheel_speed(self, wheel: WheelPosition, signed_rpm: int, acceleration: int | None = None) -> None:
        self.drive.set_wheel_speed(wheel, signed_rpm, acceleration)

    def forward(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.forward(rpm, acceleration, duration)

    def backward(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.backward(rpm, acceleration, duration)

    def left(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.strafe_left(rpm, acceleration, duration)

    def right(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.strafe_right(rpm, acceleration, duration)

    def strafe_left(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.left(rpm, acceleration, duration)

    def strafe_right(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.right(rpm, acceleration, duration)

    def rotate_cw(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.rotate_cw(rpm, acceleration, duration)

    def rotate_ccw(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.rotate_ccw(rpm, acceleration, duration)

    def forward_left(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.forward_left(rpm, acceleration, duration)

    def forward_right(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.forward_right(rpm, acceleration, duration)

    def backward_left(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.backward_left(rpm, acceleration, duration)

    def backward_right(self, rpm: int, acceleration: int | None = None, duration: float | None = None) -> None:
        self.drive.backward_right(rpm, acceleration, duration)

    def move_wheels_relative(
        self,
        fl: float,
        fr: float,
        rl: float,
        rr: float,
        rpm: int,
        acceleration: int | None = None,
        synchronized: bool = True,
    ) -> None:
        self.drive.move_wheels_relative(fl, fr, rl, rr, rpm, acceleration, synchronized)

    def move_wheels_absolute(
        self,
        fl: float,
        fr: float,
        rl: float,
        rr: float,
        rpm: int,
        acceleration: int | None = None,
        synchronized: bool = True,
    ) -> None:
        self.drive.move_wheels_absolute(fl, fr, rl, rr, rpm, acceleration, synchronized)

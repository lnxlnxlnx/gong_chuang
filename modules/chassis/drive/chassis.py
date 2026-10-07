"""High-level mecanum chassis API.

`MecanumDrive` owns motor protocol and kinematics. `MecanumChassis` is the
small application-facing facade for whole-vehicle actions.

门面方法的三选一参数与机械臂一致（见 ``motion.py``）：

- ``pulses=N``  ：位置模式，按逆运动学比例走 N 个脉冲，``wait`` 默认等动作走完；
- ``duration=T``：速度模式，跑 T 秒后自动停止；
- 都不给        ：速度模式持续运动，直到 ``stop()``。
"""

from __future__ import annotations

from modules.chassis.drive.mecanum import MecanumDrive, WheelPosition


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

    def drive_pulses(
        self,
        vx: float,
        vy: float,
        omega: float,
        pulses: int,
        rpm: int,
        acceleration: int | None = None,
        *,
        wait: bool = True,
    ) -> list[int] | None:
        """逆运动学 + 脉冲数：任意方向走固定距离。"""
        return self.drive.drive_pulses(vx, vy, omega, pulses, rpm, acceleration, wait=wait)

    def forward(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.forward(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def backward(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.backward(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def left(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.strafe_left(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def right(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.strafe_right(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def strafe_left(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.left(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def strafe_right(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.right(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def rotate_cw(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.rotate_cw(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def rotate_ccw(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.rotate_ccw(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def forward_left(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.forward_left(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def forward_right(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.forward_right(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def backward_left(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.backward_left(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def backward_right(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        return self.drive.backward_right(rpm, acceleration, duration, pulses=pulses, wait=wait)

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

    def move_wheels_relative_pulses(
        self,
        fl: int,
        fr: int,
        rl: int,
        rr: int,
        rpm: int,
        acceleration: int | None = None,
        synchronized: bool = True,
    ) -> None:
        """四轮位置模式：直接给脉冲数，正数=该轮前进方向。"""
        self.drive.move_wheels_relative_pulses(
            fl, fr, rl, rr, rpm, acceleration, synchronized
        )

    def move_wheels_absolute_pulses(
        self,
        fl: int,
        fr: int,
        rl: int,
        rr: int,
        rpm: int,
        acceleration: int | None = None,
        synchronized: bool = True,
    ) -> None:
        """四轮位置模式：直接给绝对脉冲位置。"""
        self.drive.move_wheels_absolute_pulses(
            fl, fr, rl, rr, rpm, acceleration, synchronized
        )

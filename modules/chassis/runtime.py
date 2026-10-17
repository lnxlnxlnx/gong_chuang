"""底盘上层：麦轮整车动作与统一生命周期。

真实定义就在这里（原 ``modules/chassis/drive/chassis.py`` 已退出）。

```python
from modules.chassis import MecanumChassis

with MecanumChassis.open() as chassis:
    chassis.forward(rpm=30, duration=1)
```

## 配置装配也在这里

``load_profile`` 只负责"读并校验 ``config.json``"，:meth:`MecanumChassis.open` 负责
"按 profile 建驱动并打开（或借用）总线"。原来的 ``create_drive`` / ``create_chassis``
工厂已退出——它们会重新读文件、开第二条连接，是"重复打开设备"的源头之一。

## 共享一条步进连接

arm 与 chassis 用的是**同一根** Emm42 总线（``modules/motor/config.json`` 里的端口）。
所以 :meth:`MecanumChassis.open` 接受一个已经打开的 ``bus``：

```python
Robot 会话
├─ 一条 Emm42 步进连接   ← 由 Robot 打开并最终关闭
│  ├─ arm  借用（地址 5/6/7）
│  └─ chassis 借用（配置里的四轮地址）
```

借用者 ``close`` 只让自身失效，**不切断**别人的总线：只有创建者才物理关闭。

## 门面方法的三选一参数与机械臂一致

- ``pulses=N``  ：位置模式，按逆运动学比例走 N 个脉冲；
- ``duration=T``：速度模式，跑 T 秒后自动停止；
- 都不给        ：速度模式持续运动，直到 ``stop()``。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from modules.chassis.config import DEFAULT_CONFIG_PATH, HardwareProfile, load_profile
from modules.chassis.drive.mecanum import (
    MecanumDrive,
    SerialPort,
    WheelPosition,
)
from modules.motor.config import DEFAULT_CONFIG_PATH as MOTOR_CONFIG_PATH
from modules.motor.config import load_bus_config


class MecanumChassis:
    """面向上层应用的麦轮底盘 API。"""

    def __init__(self, drive: MecanumDrive) -> None:
        """只保存**已经装配好的**驱动；本构造器不做设备 I/O。"""
        self.drive = drive
        self._closed = False

    # ------------------------------------------------------------- 生命周期
    @classmethod
    def open(
        cls,
        *,
        config_path: Optional[Path] = None,
        motor_config_path: Optional[Path] = None,
        bus: Optional[SerialPort] = None,
        owns_bus: Optional[bool] = None,
        sleep=time.sleep,
        settle_seconds=None,
    ) -> "MecanumChassis":
        """读 chassis 配置 + 总线配置，建驱动并使能四轮。

        :param config_path: chassis 的硬件 profile 配置；``None`` 用模块默认路径。
        :param motor_config_path: 步进总线配置；``None`` 用 motor 模块的默认路径。
        :param bus: 由调用方**已经打开**的共享连接（``Robot`` 装配 arm+chassis 时用）。
            给了就借用，不再自己开第二条、也不会在 ``close`` 里关掉它。
        :param owns_bus: 显式声明这条 ``bus`` 归不归本底盘；默认"不是自己开的就是借的"。
        """
        profile = cls.load_hardware_profile(config_path)
        path = MOTOR_CONFIG_PATH if motor_config_path is None else Path(motor_config_path)
        # 注意 load_bus_config 的第一个位置参数是 platform，路径只能走关键字。
        port = load_bus_config(config_path=path).port
        return cls._from_settings(
            profile=profile, port=port, bus=bus, owns_bus=owns_bus,
            sleep=sleep, settle_seconds=settle_seconds,
        )

    @classmethod
    def load_hardware_profile(cls, config_path: Optional[Path] = None) -> HardwareProfile:
        """按当前平台读并校验 chassis 配置；``Robot`` 在开设备前用它做预校验。"""
        platform = "windows" if os.name == "nt" else "jetson"
        return load_profile(
            platform, DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        )

    @classmethod
    def _from_settings(
        cls,
        *,
        profile: HardwareProfile,
        port: str,
        bus: Optional[SerialPort] = None,
        owns_bus: Optional[bool] = None,
        sleep=time.sleep,
        settle_seconds=None,
    ) -> "MecanumChassis":
        """按**已经读好的配置快照**装配；``Robot`` 与独立 ``open`` 共用这一处。

        构造中途失败时由 :class:`MecanumDrive` 自己收尾（借来的串口不擅自关闭）。
        """
        kwargs = {}
        if settle_seconds is not None:
            kwargs["settle_seconds"] = settle_seconds
        borrowing = bus is not None
        owner = (not borrowing) if owns_bus is None else bool(owns_bus)
        drive = MecanumDrive(
            port,
            motor_ids=profile.motor_ids,
            wheel_signs=profile.wheel_signs,
            acceleration=profile.acceleration,
            pattern=profile.pattern,
            rotation_gain=profile.rotation_gain,
            motion_transform=profile.motion_transform,
            serial_port=bus,
            sleep=sleep,
            **kwargs,
        )
        if borrowing:
            # 借来的连接：无论调用方怎么说，都不许由本底盘物理关闭。
            drive._owns_serial = bool(owns_bus)
        session = cls(drive)
        try:
            drive.initialize()
            drive.enable_all(True)
        except BaseException as primary:
            try:
                session.close()
            except Exception as cleanup:
                raise primary.with_traceback(primary.__traceback__) from cleanup
            raise
        return session

    @property
    def closed(self) -> bool:
        """会话是否已经关闭；关闭后动作方法统一抛 ``RuntimeError``。"""
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MecanumChassis 已关闭")

    def close(self) -> None:
        """停住执行器并释放**自有**资源；借来的总线只失效自身。

        内部顺序是"停 → 失能 → 释放驱动/自有串口"，每项独立尝试。
        幂等：重复调用不再发命令。上层（``Robot``）**不**在它之前再执行一遍 stop/disable。
        """
        if self._closed:
            return
        self._closed = True
        errors = []
        for stage, action in (
            ("stop", self.drive.stop),
            ("disable_all", self.drive.disable_all),
            ("drive.close", self.drive.close),
        ):
            try:
                action()
            except Exception as error:
                errors.append((stage, error))
        if errors:
            details = "; ".join("{}: {}".format(stage, error) for stage, error in errors)
            raise RuntimeError("MecanumChassis 清理失败：" + details) from errors[0][1]

    def __enter__(self) -> "MecanumChassis":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """收尾失败不吞掉：没有业务异常就直接抛清理错误，有则把清理错误挂成 cause。"""
        try:
            self.close()
        except Exception as cleanup:
            if exc is not None:
                raise exc.with_traceback(traceback) from cleanup
            raise
        return False

    def stop(self) -> None:
        """停住四个轮子；不关连接、不额外失能。"""
        self._require_open()
        self.drive.stop()

    def enable(self) -> None:
        self._require_open()
        self.drive.enable_all()

    def enable_all(self) -> None:
        self.enable()

    def disable(self) -> None:
        self._require_open()
        self.drive.disable_all()

    def disable_all(self) -> None:
        self.disable()

    def wheel(self, wheel: WheelPosition):
        """返回底层单轮对象，供回零等维护操作使用。"""
        self._require_open()
        return self.drive.get_wheel(wheel)

    def set_wheel_speed(self, wheel: WheelPosition, signed_rpm: int, acceleration: int | None = None) -> None:
        self._require_open()
        self.drive.set_wheel_speed(wheel, signed_rpm, acceleration)

    # ------------------------------------------------------------- 整车动作
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
        self._require_open()
        return self.drive.drive_pulses(vx, vy, omega, pulses, rpm, acceleration, wait=wait)

    def _direction(self, name: str, rpm: int, acceleration, duration, *, pulses, wait):
        self._require_open()
        return getattr(self.drive, name)(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def forward(self, rpm: int, acceleration: int | None = None,
                duration: float | None = None, *, pulses: int | None = None,
                wait: bool = True) -> list[int] | None:
        return self._direction("forward", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def backward(self, rpm: int, acceleration: int | None = None,
                 duration: float | None = None, *, pulses: int | None = None,
                 wait: bool = True) -> list[int] | None:
        return self._direction("backward", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def left(self, rpm: int, acceleration: int | None = None,
             duration: float | None = None, *, pulses: int | None = None,
             wait: bool = True) -> list[int] | None:
        return self._direction("strafe_left", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def right(self, rpm: int, acceleration: int | None = None,
              duration: float | None = None, *, pulses: int | None = None,
              wait: bool = True) -> list[int] | None:
        return self._direction("strafe_right", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def strafe_left(self, rpm: int, acceleration: int | None = None,
                    duration: float | None = None, *, pulses: int | None = None,
                    wait: bool = True) -> list[int] | None:
        return self.left(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def strafe_right(self, rpm: int, acceleration: int | None = None,
                     duration: float | None = None, *, pulses: int | None = None,
                     wait: bool = True) -> list[int] | None:
        return self.right(rpm, acceleration, duration, pulses=pulses, wait=wait)

    def rotate_cw(self, rpm: int, acceleration: int | None = None,
                  duration: float | None = None, *, pulses: int | None = None,
                  wait: bool = True) -> list[int] | None:
        return self._direction("rotate_cw", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def rotate_ccw(self, rpm: int, acceleration: int | None = None,
                   duration: float | None = None, *, pulses: int | None = None,
                   wait: bool = True) -> list[int] | None:
        return self._direction("rotate_ccw", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def forward_left(self, rpm: int, acceleration: int | None = None,
                     duration: float | None = None, *, pulses: int | None = None,
                     wait: bool = True) -> list[int] | None:
        return self._direction("forward_left", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def forward_right(self, rpm: int, acceleration: int | None = None,
                      duration: float | None = None, *, pulses: int | None = None,
                      wait: bool = True) -> list[int] | None:
        return self._direction("forward_right", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def backward_left(self, rpm: int, acceleration: int | None = None,
                      duration: float | None = None, *, pulses: int | None = None,
                      wait: bool = True) -> list[int] | None:
        return self._direction("backward_left", rpm, acceleration, duration, pulses=pulses, wait=wait)

    def backward_right(self, rpm: int, acceleration: int | None = None,
                       duration: float | None = None, *, pulses: int | None = None,
                       wait: bool = True) -> list[int] | None:
        return self._direction("backward_right", rpm, acceleration, duration, pulses=pulses, wait=wait)

    # ------------------------------------------------------------- 四轮直控
    def move_wheels_relative(self, fl: float, fr: float, rl: float, rr: float,
                             rpm: int, acceleration: int | None = None,
                             synchronized: bool = True) -> None:
        self._require_open()
        self.drive.move_wheels_relative(fl, fr, rl, rr, rpm, acceleration, synchronized)

    def move_wheels_absolute(self, fl: float, fr: float, rl: float, rr: float,
                             rpm: int, acceleration: int | None = None,
                             synchronized: bool = True) -> None:
        self._require_open()
        self.drive.move_wheels_absolute(fl, fr, rl, rr, rpm, acceleration, synchronized)

    def move_wheels_relative_pulses(self, fl: int, fr: int, rl: int, rr: int,
                                    rpm: int, acceleration: int | None = None,
                                    synchronized: bool = True) -> None:
        """四轮位置模式：直接给脉冲数，正数=该轮前进方向。"""
        self._require_open()
        self.drive.move_wheels_relative_pulses(fl, fr, rl, rr, rpm, acceleration, synchronized)

    def move_wheels_absolute_pulses(self, fl: int, fr: int, rl: int, rr: int,
                                    rpm: int, acceleration: int | None = None,
                                    synchronized: bool = True) -> None:
        """四轮位置模式：直接给绝对脉冲位置。"""
        self._require_open()
        self.drive.move_wheels_absolute_pulses(fl, fr, rl, rr, rpm, acceleration, synchronized)

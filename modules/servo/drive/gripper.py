"""夹爪开合封装：把裸舵机位置值收成一个开合接口。

``BusServo`` 只知道"走到 0..1000 的某个位置"，不知道哪一端是张开、哪一端是夹紧。
这一层补上那个语义，并且利用总线舵机的**位置读回**做一件顺手的事：

> 命令走到夹紧位置，读回值却没到——说明爪口被东西挡住了，也就是**夹到物料了**。

所以 :meth:`GripperServo.close_jaws` 返回的 :class:`GripperReport` 里 ``reached`` 为
``False`` 往往是抓取成功而不是失败。抓取流程据此判断"真的夹住了"还是"空夹"，
而不必额外装压力传感器。

命名上刻意避开歧义：``open_jaws`` / ``close_jaws`` 管爪口，``close`` 只管串口
（与 ``BusServo.close`` 一致），两者不是一回事。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from modules.servo.config import GripperConfig, load_gripper_config
from modules.servo.drive.bus_servo import BusServo, BusServoConfig, load_bus_config

#: 读回位置与目标位置相差在这个范围内就算到位（舵机位置量程 0..1000）。
DEFAULT_POSITION_TOLERANCE = 30

#: 动作时长之外额外等待的余量（秒），让舵机真正停稳后再读回。
DEFAULT_SETTLE_SECONDS = 0.2

OPEN_COMMAND = "open"
CLOSE_COMMAND = "close"


@dataclass(frozen=True)
class GripperReport:
    """一次开合动作的结果。"""

    command: str
    target: int
    #: 动作结束后读回的实际位置。
    position: int
    reached: bool

    @property
    def error(self) -> int:
        """读回位置减目标位置（舵机量程单位）。"""
        return self.position - self.target

    def summary(self) -> str:
        state = "到位" if self.reached else f"未到位（差 {self.error:+d}）"
        return f"夹爪{self.command}：目标 {self.target}，读回 {self.position}，{state}"


class GripperServo:
    """夹爪伺服的开合、读回与串口生命周期。"""

    def __init__(
        self,
        port: str,
        *,
        config: Optional[GripperConfig] = None,
        bus_config: Optional[BusServoConfig] = None,
        servo: Optional[BusServo] = None,
        sleep: Callable[[float], None] = time.sleep,
        position_tolerance: int = DEFAULT_POSITION_TOLERANCE,
        settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    ) -> None:
        self.config = config if config is not None else load_gripper_config()
        bus = bus_config if bus_config is not None else load_bus_config()
        self.port = port or bus.port
        if position_tolerance < 0:
            raise ValueError("position_tolerance must not be negative")
        if settle_seconds < 0:
            raise ValueError("settle_seconds must not be negative")
        self.position_tolerance = position_tolerance
        self.settle_seconds = settle_seconds
        self._sleep = sleep
        self._servo = (
            servo
            if servo is not None
            else BusServo(
                self.port,
                self.config.servo_id,
                baudrate=bus.baudrate,
            )
        )

    # ------------------------------------------------------------- 生命周期
    def initialize(self) -> "GripperServo":
        """Open the bus and hold torque so the jaws keep their position."""
        self._servo.initialize()
        self._servo.set_torque(True)
        return self

    def close(self) -> None:
        """Close the bus only; the jaws stay wherever they are."""
        self._servo.close()

    def __enter__(self) -> "GripperServo":
        return self.initialize()

    def __exit__(self, *_exc_info) -> None:
        self.close()

    # ------------------------------------------------------------- 开合
    def open_jaws(self, *, wait: bool = True) -> GripperReport:
        """Move to the configured open position."""
        return self._go_to(OPEN_COMMAND, self.config.open_position, wait=wait)

    def close_jaws(self, *, wait: bool = True) -> GripperReport:
        """Move to the configured closed position.

        ``reached`` 为 ``False`` 说明爪口被挡住，通常是夹到了物料。
        """
        return self._go_to(CLOSE_COMMAND, self.config.closed_position, wait=wait)

    def set_position(self, position: int, *, wait: bool = True) -> GripperReport:
        """Drive a raw servo position; used while calibrating open/closed values."""
        return self._go_to("set", position, wait=wait)

    def read_position(self) -> int:
        """Read the servo's current position (absolute; no homing required)."""
        return self._servo.read_position()

    def stop(self) -> None:
        """Brake at the current position."""
        self._servo.stop()

    def set_torque(self, enabled: bool) -> None:
        """Enable or release holding torque."""
        self._servo.set_torque(enabled)

    def _go_to(self, command: str, target: int, *, wait: bool) -> GripperReport:
        self._servo.set_position(target, self.config.duration_ms)
        if not wait:
            # 只发命令：读回的是"命令前"的位置，所以不声称到位，如实标 False。
            return GripperReport(
                command=command,
                target=target,
                position=self.read_position(),
                reached=False,
            )
        self._sleep(self.config.duration_ms / 1000.0 + self.settle_seconds)
        position = self.read_position()
        return GripperReport(
            command=command,
            target=target,
            position=position,
            reached=abs(position - target) <= self.position_tolerance,
        )


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要硬件：用一个假舵机验证等待、读回与"夹住"判据。"""
    from modules.servo.config import GripperConfig
    from modules.servo.drive.bus_servo import BusServoConfig

    class FakeServo:
        """记录调用，并按脚本返回位置。"""

        def __init__(self, positions):
            self.calls = []
            self._positions = list(positions)

        def initialize(self):
            self.calls.append(("initialize",))
            return self

        def set_torque(self, enabled):
            self.calls.append(("set_torque", enabled))

        def set_position(self, position, duration_ms):
            self.calls.append(("set_position", position, duration_ms))

        def read_position(self):
            self.calls.append(("read_position",))
            return self._positions.pop(0) if self._positions else 0

        def stop(self):
            self.calls.append(("stop",))

        def close(self):
            self.calls.append(("close",))

    config = GripperConfig(
        servo_id=1, open_position=300, closed_position=500, duration_ms=1000
    )
    # 显式给出总线参数，自测不去读 modules/servo/config.json，避免依赖本机端口配置。
    bus = BusServoConfig(port="FAKE", baudrate=115200, servo_id=1)
    slept = []

    # 场景一：空夹，读回到位。
    servo = FakeServo([300, 500])
    gripper = GripperServo(
        "",
        config=config,
        servo=servo,
        sleep=slept.append,
        bus_config=bus,
    )
    gripper.initialize()
    opened = gripper.open_jaws()
    closed = gripper.close_jaws()
    assert opened.reached and opened.position == 300, opened
    assert closed.reached and closed.position == 500, closed
    assert slept == [1.2, 1.2], slept
    assert ("set_torque", True) in servo.calls

    # 场景二：夹到东西，读回没到目标 —— reached 必须是 False，且这不是错误。
    servo = FakeServo([438])
    gripper = GripperServo("", config=config, servo=servo, sleep=slept.append, bus_config=bus)
    gripped = gripper.close_jaws()
    assert not gripped.reached, gripped
    assert gripped.error == -62, gripped
    assert "未到位" in gripped.summary()

    # 场景三：wait=False 只发命令、不等待，也不声称到位。
    slept.clear()
    servo = FakeServo([412])
    gripper = GripperServo("", config=config, servo=servo, sleep=slept.append, bus_config=bus)
    report = gripper.close_jaws(wait=False)
    assert not report.reached and slept == [], (report, slept)

    print("夹爪封装自测通过：开合等待、读回到位判据、夹住检测都不依赖硬件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

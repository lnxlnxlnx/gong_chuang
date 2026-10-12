"""Shared motion parameters and Emm42 timing estimates.

``robot_arm.RobotArm``（机械臂三轴）和 ``modules.chassis.drive.mecanum.MecanumDrive``（麦轮四轮）挂的是
同一款 Emm42 V5.0 闭环步进驱动器，所以两个模块必须使用**同一套参数名和同一套
"一次动作要跑多久"的算法**。

## 统一参数约定（机械臂 / 底盘一致）

| 参数 | 含义 | 备注 |
|---|---|---|
| ``pulses`` | 位置模式：本次移动的脉冲数（正数，方向由动作方法决定） | 机械臂必填；底盘可选 |
| ``duration`` | 速度模式：一条速度命令持续多少秒 | 只有底盘支持 |
| ``rpm`` | 目标转速；不给就用配置里的默认值 | 机械臂配置在 ``modules/arm/config.json`` |
| ``acceleration`` | 曲线加减速档位 ``0..255``；不给就用配置里的默认值 | ``0`` = 无曲线 |
| ``wait`` | ``True``（默认）= 等到动作走完再返回；``False`` = 只发命令 | 见下 |

机械臂按安全约定**只有相对位置模式**（见 ``modules/arm/docs/机械臂Emm42三轴控制说明.md``），
因此只接受 ``pulses``，不接受 ``duration``。

## 为什么 ``wait`` 必须存在

Emm42 的位置模式命令是"发完就返回"：驱动器一收到帧就开始转，但上位机不会
自动知道它什么时候停。机械臂 5/6/7 号驱动器在菜单里是 ``Response=None``，
连四字节 ACK 都不回，更不会有到位回包。如果连续下发多条动作，下一条会在上
一条还没走完时改写目标，现象就是"上一个动作被跳过"。

``wait=True`` 时按说明书给出的曲线加减速公式估算本次动作耗时并等待：

```
每档加速度让转速 +1 RPM 约需 (256 - acceleration) * 50µs
加速段 = rpm * (256 - acceleration) * 50µs        （acceleration = 0 时无曲线）
匀速段 = pulses / (rpm * 脉冲每圈 / 60)
总耗时 ≈ 加速段 + 匀速段 + 到位余量
```

这是估算值（把驱动器减速段和串口往返都折进"到位余量"），不是编码器实测。
需要毫秒级精确时必须改用 ``0x3A`` 读运动状态标志轮询到位位。
"""

from __future__ import annotations

from typing import Final

from modules.motor.drive.emm42 import Emm42Driver

#: 说明书 1.3 节：曲线加减速的斜率，每让转速 +1 RPM 需要 (256 - 档位) * 50µs。
RAMP_SECONDS_PER_RPM_PER_GEAR: Final = 50e-6

#: 加速度档上限；``0`` 表示无曲线加减速，直接按设定速度运行。
MAX_ACCELERATION: Final = 255

#: 默认到位余量（秒）：补偿减速段、串口往返和驱动器内部处理时间。
DEFAULT_SETTLE_SECONDS: Final = 0.15

#: 位置模式一条命令能表达的最大脉冲数（协议为 4 字节大端无符号）。
MAX_PULSES: Final = 0xFFFFFFFF


def ramp_seconds(rpm: int, acceleration: int) -> float:
    """Return the datasheet ramp time from 0 to ``rpm`` for one acceleration gear."""
    _require_int("rpm", rpm, 0, Emm42Driver.MAX_RPM)
    _require_int("acceleration", acceleration, 0, MAX_ACCELERATION)
    if acceleration == 0 or rpm == 0:
        return 0.0
    return rpm * (MAX_ACCELERATION + 1 - acceleration) * RAMP_SECONDS_PER_RPM_PER_GEAR


def cruise_seconds(pulses: int, rpm: int, pulses_per_revolution: int) -> float:
    """Return the constant-speed time needed to cover ``pulses`` at ``rpm``."""
    _require_int("pulses", pulses, 1, MAX_PULSES)
    _require_int("rpm", rpm, 1, Emm42Driver.MAX_RPM)
    _require_int("pulses_per_revolution", pulses_per_revolution, 1, MAX_PULSES)
    revolutions = pulses / pulses_per_revolution
    return revolutions * 60.0 / rpm


def estimate_move_seconds(
    pulses: int,
    rpm: int,
    pulses_per_revolution: int,
    acceleration: int = 0,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
) -> float:
    """Estimate how long one relative position move takes.

    Used by both the arm and the chassis to make ``wait=True`` blocking.
    """
    _require_number("settle_seconds", settle_seconds, 0.0)
    return (
        ramp_seconds(rpm, acceleration)
        + cruise_seconds(pulses, rpm, pulses_per_revolution)
        + float(settle_seconds)
    )


def _require_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")


def _require_number(name: str, value: object, minimum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

"""Load the shared Emm42 stepper-bus **hardware** settings.

这个文件只读 ``hardware.json``：端口、波特率、细分（脉冲/圈）、串口超时。
运动调参（rpm / acceleration / 加速档）不在这里，见各模块的 ``config.json``。

## 为什么波特率和细分在硬件文件里，而不在代码里

它们不是协议不变量，而是**每台驱动器菜单里的状态**：

- 波特率是 Emm42 菜单项 ``uart_baud_option``（``only_for_test/read_driver_config.py`` 能读回来）；
- 细分 ``MStep`` 决定一圈多少脉冲，本项目按 16 细分 = ``3200``。

两者一旦和实物不一致，``mm_per_motor_rev`` 之类的换算会**静默错一个整数倍**。
所以它们必须是可核对、可修改的配置，而不是代码里的常量。

## 缺键即报错

``bus`` 段缺失时直接报错并提示迁移，而不是退回代码默认值：默认值会让"配置漏写"
看起来像"跑通了"。这条规则来自旧的 ``_mapping``/``_integer`` 那套校验，只是把
默认值去掉了。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


#: 硬件参数与加载器放在一起，因此与仓库深度、当前目录都无关。
DEFAULT_HARDWARE_PATH = Path(__file__).resolve().parent / "hardware.json"


@dataclass(frozen=True)
class MotorBusConfig:
    """The one stepper bus every Emm42 device in this project shares."""

    port: str
    baudrate: int
    pulses_per_revolution: int
    timeout_seconds: float


def load_bus_config(
    platform: str | None = None,
    hardware_path: Path = DEFAULT_HARDWARE_PATH,
) -> MotorBusConfig:
    """Read the stepper bus hardware block for one platform.

    没有任何一项有代码默认值：缺了就报出**是哪个文件的哪个键**。
    """
    platform = platform or ("windows" if os.name == "nt" else "jetson")
    document = _read_document(hardware_path)
    _reject_legacy_shape(document, hardware_path)

    profile = _mapping(document.get(platform), platform)
    port = profile.get("port")
    if not isinstance(port, str) or not port.strip():
        raise ValueError(f"{hardware_path} 的 {platform}.port 还没填")

    bus = _mapping(document.get("bus"), "bus")
    return MotorBusConfig(
        port=port.strip(),
        baudrate=_integer(bus, "baudrate", 1, 4_000_000, where=f"{hardware_path}: bus"),
        pulses_per_revolution=_integer(
            bus, "pulses_per_revolution", 1, 0xFFFFFFFF, where=f"{hardware_path}: bus"
        ),
        timeout_seconds=_positive(bus, "timeout_seconds", where=f"{hardware_path}: bus"),
    )


def load_pulses_per_revolution(hardware_path: Path = DEFAULT_HARDWARE_PATH) -> int:
    """Read only the microstep fact (脉冲/圈), without touching the port block.

    arm 与 chassis 的角度/毫米换算都需要它，而它们不该因为"端口还没填"就无法读配置。
    """
    document = _read_document(hardware_path)
    _reject_legacy_shape(document, hardware_path)
    bus = _mapping(document.get("bus"), "bus")
    return _integer(
        bus,
        "pulses_per_revolution",
        1,
        0xFFFFFFFF,
        where=f"{hardware_path}: bus",
    )


def _read_document(path: Path) -> Mapping[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取步进总线硬件配置：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"步进总线硬件配置不是合法 JSON：{path}") from error
    return _mapping(document, str(path))


def _reject_legacy_shape(document: Mapping[str, object], path: Path) -> None:
    """旧结构只有平台段；见到它就要求按新结构写，不猜。"""
    if "bus" in document:
        return
    raise ValueError(
        f"{path} 是旧结构（只有端口，没有 bus 段）。"
        "现在硬件参数统一放在 hardware.json，bus 段至少要写：\n"
        '  "bus": {"baudrate": 115200, "pulses_per_revolution": 3200, "timeout_seconds": 0.1}'
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是一个 JSON 对象")
    return value


def _integer(
    document: Mapping[str, object], key: str, minimum: int, maximum: int, *, where: str
) -> int:
    value = document.get(key)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{where}.{key} 必须是 {minimum}..{maximum} 的整数，实际 {value!r}")
    return value


def _positive(document: Mapping[str, object], key: str, *, where: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}.{key} 必须是数字，实际 {value!r}")
    if not float(value) > 0:
        raise ValueError(f"{where}.{key} 必须大于 0，实际 {value!r}")
    return float(value)

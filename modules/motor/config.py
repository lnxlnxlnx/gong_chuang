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

## ``port`` 可以写 ``auto``

两块 USB-TTL 都是 CH340、**都没有序列号**，Jetson 每次上电按驱动注册先后发 ``ttyUSB``
编号，插在同一个物理口上也可能换号（Windows 的 COM 号跟着设备实例走，不会变）。
所以"端口名"在 Jetson 上不是稳定身份。

``"port": "auto"`` 改成"打开前问一句"：哪条 USB 串口回了 Emm42 的**读状态**帧，哪条就是
步进总线。探测只发 :data:`PROBE_FUNCTION` 这一条只读命令，不会让任何电机动作；一条都
探不到就报错并列出试过的端口，不猜、也不退回某个 ``ttyUSB``。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Final, Mapping, Optional, Protocol, Sequence, Tuple


#: 硬件参数与加载器放在一起，因此与仓库深度、当前目录都无关。
DEFAULT_HARDWARE_PATH = Path(__file__).resolve().parent / "hardware.json"


@dataclass(frozen=True)
class MotorBusConfig:
    """The one stepper bus every Emm42 device in this project shares."""

    port: str
    baudrate: int
    pulses_per_revolution: int
    timeout_seconds: float


#: ``port`` 写这个值 = "总线上是谁，开之前自己探"，见模块开头。
AUTO_PORT: Final = "auto"

#: 探测时依次询问的地址：1~4 是底盘四个轮，5 是机械臂底座。
#: **6/7 不能写进来**：它们的驱动器 Response 菜单是关的（``expect_ack=false``），
#: 根本不回包，拿它们当判据会永远探不到总线。
PROBE_ADDRESSES: Final = (1, 2, 3, 4, 5)

#: 探测用的只读命令：``0x3A`` 读状态标志位，回复固定 ``地址 3A 状态 6B``。
PROBE_FUNCTION: Final = 0x3A

#: 回包长度与帧尾校验字节（与驱动器约定，全项目同一个值）。
REPLY_LENGTH: Final = 4
FRAME_TAIL: Final = 0x6B

#: 探测结果按 (波特率, 候选端口) 缓存：一次运行里 arm / chassis / grasp 都会调
#: :func:`load_bus_config`，不缓存就会重复探测；缓存也让它们必然拿到**同一条**总线。
_DETECTED_PORTS: Dict[Tuple[int, Tuple[str, ...]], str] = {}


class ProbedPort(Protocol):
    """探测一条串口只用这几个方法；测试给个假的就行，不必装 pyserial。"""

    def reset_input_buffer(self) -> None: ...

    def write(self, data: bytes) -> int: ...

    def flush(self) -> None: ...

    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...


def usb_serial_candidates() -> Tuple[str, ...]:
    """本机所有的 **USB** 串口，按名字排序——顺序固定，探测结果才可复现。

    只看 ``hwid`` 里带 ``USB`` 的口：Jetson 上还挂着一堆 ``ttyS*``/``ttyTHS*`` 板载串口，
    挨个打开又慢又没意义。
    """
    try:
        from serial.tools import list_ports
    except ImportError as error:
        raise RuntimeError(
            "auto 端口要先装 pyserial 才能枚举串口；装不了就把 hardware.json 里的 "
            "port 写成具体设备名"
        ) from error
    return tuple(
        sorted(
            port.device for port in list_ports.comports() if "USB" in (port.hwid or "").upper()
        )
    )


def _serial_opener(baudrate: int, timeout_seconds: float) -> Callable[[str], ProbedPort]:
    from modules.serial_link import open_serial

    def open_port(port: str) -> ProbedPort:
        return open_serial(port, baudrate=baudrate, timeout=timeout_seconds)

    return open_port


def resolve_bus_port(
    configured_port: str,
    *,
    baudrate: int,
    timeout_seconds: float,
    candidates: Optional[Sequence[str]] = None,
    opener: Optional[Callable[[str], ProbedPort]] = None,
    addresses: Sequence[int] = PROBE_ADDRESSES,
) -> str:
    """把配置里的端口解析成真正要打开的那一条。

    ``configured_port`` 不是 :data:`AUTO_PORT` 时原样返回，写死端口的老配置一点都不变。
    """
    if configured_port != AUTO_PORT:
        return configured_port

    ports = tuple(candidates) if candidates is not None else usb_serial_candidates()
    key = (baudrate, ports)
    if key in _DETECTED_PORTS:
        return _DETECTED_PORTS[key]

    open_port = opener if opener is not None else _serial_opener(baudrate, timeout_seconds)
    attempts = []
    for port in ports:
        try:
            replied = _port_answers_emm42(port, open_port=open_port, addresses=addresses)
        except OSError as error:
            # 打不开的口（被占用、刚拔掉）不是总线，继续看下一个。
            attempts.append(f"{port}（打不开：{error}）")
            continue
        if replied:
            _DETECTED_PORTS[key] = port
            return port
        attempts.append(f"{port}（无应答）")

    raise RuntimeError(
        f"port 写的是 {AUTO_PORT}，但没有任何串口回 Emm42 的读状态帧；"
        "请检查电机的电源与 USB-TTL 接线。已经试过："
        + ("、".join(attempts) if attempts else "本机没有枚举到 USB 串口")
    )


def _port_answers_emm42(
    port: str,
    *,
    open_port: Callable[[str], ProbedPort],
    addresses: Sequence[int],
) -> bool:
    """这条口上有没有驱动器回**读状态**帧。只发只读命令，不会让电机动作。"""
    serial_port = open_port(port)
    try:
        for address in addresses:
            # 每条命令前清空输入缓冲：上一条迟到的回包不能算到下一条头上。
            serial_port.reset_input_buffer()
            serial_port.write(bytes((address, PROBE_FUNCTION, FRAME_TAIL)))
            serial_port.flush()
            if _is_status_reply(serial_port.read(REPLY_LENGTH), address):
                return True
        return False
    finally:
        serial_port.close()


def _is_status_reply(reply: bytes, address: int) -> bool:
    """回包必须是**长度 4、地址与功能码都对、帧尾 0x6B** 的完整帧。"""
    return (
        len(reply) == REPLY_LENGTH
        and reply[0] == address
        and reply[1] == PROBE_FUNCTION
        and reply[3] == FRAME_TAIL
    )


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
    baudrate = _integer(bus, "baudrate", 1, 4_000_000, where=f"{hardware_path}: bus")
    timeout = _positive(bus, "timeout_seconds", where=f"{hardware_path}: bus")
    return MotorBusConfig(
        port=resolve_bus_port(
            port.strip(), baudrate=baudrate, timeout_seconds=timeout
        ),
        baudrate=baudrate,
        pulses_per_revolution=_integer(
            bus, "pulses_per_revolution", 1, 0xFFFFFFFF, where=f"{hardware_path}: bus"
        ),
        timeout_seconds=timeout,
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

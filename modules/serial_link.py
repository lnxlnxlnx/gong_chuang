"""打开一条 USB-TTL 串口：全仓只在这里 ``import serial``。

理由很朴素：pyserial 只在板子上装，桌面离线跑不了。**同一个"没装 pyserial"的报错**
以前在 8 处各写一遍（emm42 / mecanum / arm / robot / motor.config / bus_servo …），
改一句话要改八次。这里只做"导入 + 打开"，close 交给持有者——谁开的谁关，
借来的连接（arm 与 chassis 共用一条）不在这里管。

各模块的打开参数仍然自己决定：步进总线 115200、舵机总线 1000000，
timeout 由调用方给（步进的读回包超时、舵机的应答超时不是一回事）。
"""

from __future__ import annotations

from typing import Any

#: 板子上缺 pyserial 时的统一报错。桌面端离线单测注入假串口，不会走到这条。
MISSING_PYSERIAL = (
    "pyserial is required on Jetson. Install it in the target Python environment first."
)


def open_serial(port: str, *, baudrate: int, timeout: float) -> Any:
    """按给定参数打开串口并返回 pyserial 的 ``Serial``（或其替身）。"""
    try:
        import serial
    except ImportError as error:
        raise RuntimeError(MISSING_PYSERIAL) from error
    return serial.Serial(port, baudrate=baudrate, timeout=timeout)


def import_serial() -> Any:
    """需要直接摸 pyserial 本身时（例如枚举/探测）用这个，报错同样统一。"""
    try:
        import serial
    except ImportError as error:
        raise RuntimeError(MISSING_PYSERIAL) from error
    return serial

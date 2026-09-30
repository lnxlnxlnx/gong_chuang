"""Hiwonder half-duplex bus-servo protocol driver.

总线舵机通讯协议(串口)：

  帧格式:  55 55 | id(1B) | length(1B) | command(1B) | data... | checksum(1B)
  校验和:  checksum = (~sum(id, length, command, data...)) & 0xFF
  length  = len(data) + 3

说明:
  - 物理层为 5V 上拉的单线半双工 UART，必须用"总线舵机 USB 转接板"或带半双工
    收发切换 + 电平转换(3.3V/5V)的串口模块；普通 USB-TTL 的 TX/RX 不能直接
    并接在舵机信号线上。
  - 常用指令(见本类底部各常量):
      1   MOVE_TIME_WRITE     带时间定点运动 (位置 0..1000 + 时长 ms)
      12  MOVE_STOP           立即停止
      28  POS_READ            读取当前角度位置
      29  MOTOR_MODE_WRITE    切换为电机模式并设定速度 (-1000..1000)
      31  LOAD_OR_UNLOAD_WRITE 扭矩加载/卸载
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "servo.json"


@dataclass(frozen=True)
class BusServoConfig:
    """Port and servo identity read from ``config/servo.json``."""

    port: str
    baudrate: int
    servo_id: int


def _config_integer(section: dict, name: str, minimum: int, maximum: int) -> int:
    value = section.get(name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"bus.{name} 必须是 {minimum}..{maximum} 的整数")
    return value


def load_bus_config(
    platform: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> BusServoConfig:
    """Read the bus servo port, baud rate and ID from the shared servo file."""
    platform = platform or ("windows" if os.name == "nt" else "jetson")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取舵机配置：{config_path}") from error
    if not isinstance(document, dict):
        raise ValueError("舵机配置必须是 JSON 对象")
    profile = document.get(platform)
    if not isinstance(profile, dict) or not isinstance(profile.get("port"), str) or not profile["port"].strip():
        raise ValueError(f"请先填写 {config_path} 中的 {platform}.port")
    bus = document.get("bus")
    if not isinstance(bus, dict):
        raise ValueError("bus 必须是 JSON 对象")
    return BusServoConfig(
        port=profile["port"].strip(),
        # 协议固定 115200；允许配置覆盖，便于换用其它半双工转接板
        baudrate=_config_integer(bus, "baudrate", 1, 4000000),
        servo_id=_config_integer(bus, "servo_id", 1, 253),
    )


class SerialPort(Protocol):
    """Minimum serial interface used by the driver and its tests."""

    is_open: bool

    def write(self, data: bytes) -> int: ...

    def flush(self) -> None: ...

    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...


class BusServo:
    """Control one Hiwonder bus servo through a half-duplex serial adapter.

    The adapter must expose the servo's single-wire half-duplex bus. A normal
    full-duplex USB-TTL adapter cannot be connected directly to this bus.
    """

    # 串口参数：总线舵机固定使用 115200 波特率
    BAUDRATE = 115200
    # 帧头两字节，用于从串口数据流中定位一帧的起点
    HEADER = b"\x55\x55"

    # ---------- 协议指令码 ----------
    # 1: 带时间定点运动: [pos 2B 小端] + [duration_ms 2B 小端]
    MOVE_TIME_WRITE = 1
    # 12: 立即停止在当前角度
    MOVE_STOP = 12
    # 28: 读取当前位置, 舵机回复 55 55 id len 1C pos(2B 小端, 带符号) cksum
    POS_READ = 28
    # 29: 电机模式 + 设定速度: [0x01 0x00] + [speed 2B 小端, 带符号 -1000..1000]
    MOTOR_MODE_WRITE = 29
    # 31: 扭矩加载/卸载: [0/1]
    LOAD_OR_UNLOAD_WRITE = 31

    def __init__(
        self,
        port: str,
        servo_id: int,
        *,
        baudrate: int = BAUDRATE,
        timeout: float = 0.1,
        serial_port: SerialPort | None = None,
    ) -> None:
        # 地址为 0 时代表广播(所有舵机)，因此允许 0..253
        if not 0 <= servo_id <= 253:
            raise ValueError("servo_id must be in 0..253")
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")
        self.port = port
        self.servo_id = servo_id
        self.baudrate = baudrate
        self.timeout = timeout
        # 测试时可注入一个 FakeSerial，避免真正打开串口
        self._serial = serial_port

    def initialize(self) -> "BusServo":
        """Open the configured serial port at the protocol baud rate."""
        if self._serial is None:
            self._serial = self._open_serial()
        return self

    def close(self) -> None:
        """Close the serial port if it has been initialized."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def set_position(self, position: int, duration_ms: int) -> None:
        """Move to ``position`` (0..1000) within ``duration_ms`` (0..30000)."""
        self._require_range("position", position, 0, 1000)
        self._require_range("duration_ms", duration_ms, 0, 30000)
        # 位置和时长均按协议用小端序编码为 2 个字节
        self._write(self.MOVE_TIME_WRITE, position.to_bytes(2, "little") + duration_ms.to_bytes(2, "little"))

    def set_speed(self, speed: int) -> None:
        """Enter motor mode and set signed speed in the documented -1000..1000 range."""
        self._require_range("speed", speed, -1000, 1000)
        # 前两字节固定 01 00；速度带符号小端 2B
        self._write(self.MOTOR_MODE_WRITE, b"\x01\x00" + speed.to_bytes(2, "little", signed=True))

    def stop(self) -> None:
        """Immediately stop at the current position."""
        # MOVE_STOP 指令无数据段，帧为 55 55 id len=3 cmd cksum
        self._write(self.MOVE_STOP)

    def set_torque(self, enabled: bool) -> None:
        """Enable (load) or disable (unload) motor torque."""
        self._write(self.LOAD_OR_UNLOAD_WRITE, bytes((int(enabled),)))

    def read_position(self) -> int:
        """Read the signed current position value returned by the servo."""
        self._write(self.POS_READ)
        packet = self._read_packet()
        # packet[0:2]=55 55; packet[2]=id; packet[3]=长度; packet[4]=指令码
        if len(packet) != 8 or packet[2] != self.servo_id or packet[3] != 5 or packet[4] != self.POS_READ:
            raise RuntimeError("unexpected servo position response")
        # 数据段为 2 字节带符号小端位置值 (packet[5:7])
        return int.from_bytes(packet[5:7], "little", signed=True)

    def _open_serial(self) -> SerialPort:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required on Jetson. Install it in the target Python environment first."
            ) from error
        return serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)

    def _write(self, command: int, parameters: bytes = b"") -> None:
        serial_port = self._require_serial()
        # 长度字段统计: id + length + command + data (不含帧头和校验和)
        length = len(parameters) + 3
        body = bytes((self.servo_id, length, command)) + parameters
        # 校验和 = 除帧头与校验和自身外所有字节之和取反
        checksum = (~sum(body)) & 0xFF
        serial_port.write(self.HEADER + body + bytes((checksum,)))
        serial_port.flush()

    def _read_packet(self) -> bytes:
        serial_port = self._require_serial()
        # 一帧至少包含 55 55 + id + length
        header = serial_port.read(2)
        if header != self.HEADER:
            raise TimeoutError("servo response header was not received")
        prefix = serial_port.read(2)
        if len(prefix) != 2:
            raise TimeoutError("servo response prefix was incomplete")
        length = prefix[1]
        # Length 包含自身，因此除去它的剩余字节数为 length - 1
        remainder = serial_port.read(length - 1)
        packet = header + prefix + remainder
        if len(remainder) != length - 1:
            raise TimeoutError("servo response was incomplete")
        # 校验返回帧: 对除头2字节和末尾校验字节外的内容求和取反
        if (~sum(packet[2:-1])) & 0xFF != packet[-1]:
            raise RuntimeError("servo response checksum mismatch")
        return packet

    def _require_serial(self) -> SerialPort:
        if self._serial is None:
            raise RuntimeError("call initialize() before sending commands")
        return self._serial

    @staticmethod
    def _require_range(name: str, value: int, minimum: int, maximum: int) -> None:
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in {minimum}..{maximum}")

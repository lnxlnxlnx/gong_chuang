"""Emm42 V5.0 UART protocol driver.

Emm42 V5.0 闭环步进驱动器串口协议(ZDT Emm 固件)：

  帧格式:  [addr 1B] [function 1B] [data...] [0x6B 帧尾/固定校验]
  应答帧:  [addr] [function] [0x02 SUCCESS] [0x6B]   (每个命令驱动器都会回 ACK)

说明:
  - 波特率 115200, 8N1, TTL 电平；用 USB-TTL 模块连接，驱动器电源独立接 12-32V。
  - 接线: USB-TTL 的 TX -> 驱动器 R/A/H, RX -> T/B/L, GND -> GND。
  - direction: 0=CW(顺时针), 1=CCW(逆时针)。
  - sync 字节 0=立即执行, 1=缓存后由 ``00 FF 66 6B`` 同步帧触发。
  - 速度/加速度单位: RPM(0..3000) 与加速度档位(0..255)。
  - 位置模式下脉冲数 = |角度| / 360 * pulses_per_revolution；
    默认 pulses_per_revolution=3200(1.8°步距角 x 16细分 = 3200 脉冲/圈)。
"""

from __future__ import annotations

from enum import IntEnum
from typing import Protocol


class SerialPort(Protocol):
    """Minimum serial interface used by the driver and its tests."""

    is_open: bool

    def write(self, data: bytes) -> int: ...

    def flush(self) -> None: ...

    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...


class Direction(IntEnum):
    """运动方向字节。"""
    CW = 0
    CCW = 1


class HomeMode(IntEnum):
    """回零模式：就近/方向/无限位碰撞/限位开关。实际回零行为以驱动器面板配置为准。"""
    SINGLE_TURN_NEAREST = 0
    SINGLE_TURN_DIRECTION = 1
    MULTI_TURN_COLLISION = 2
    MULTI_TURN_LIMIT_SWITCH = 3


class Emm42Driver:
    """Control one Emm42 V5.0 driver through its TTL UART port."""

    # ---------- 协议常量 ----------
    BAUDRATE = 115200  # 驱动器固定波特率
    CHECKSUM = 0x6B    # 固定帧尾字节(该协议所有命令都以 0x6B 结尾)
    MAX_RPM = 3000     # 速度上限

    def __init__(
        self,
        port: str,
        motor_id: int,
        *,
        timeout: float = 0.1,
        pulses_per_revolution: int = 3200,
        serial_port: SerialPort | None = None,
    ) -> None:
        # 地址 1..255, 同一串口总线上可用不同地址挂多台驱动器
        if not 1 <= motor_id <= 255:
            raise ValueError("motor_id must be in 1..255")
        if pulses_per_revolution <= 0:
            raise ValueError("pulses_per_revolution must be positive")
        self.port = port
        self.motor_id = motor_id
        self.timeout = timeout
        # 角度<->脉冲换算系数; 若驱动器细分配置不同需在初始化时传入实际值
        self.pulses_per_revolution = pulses_per_revolution
        # 测试时可注入 FakeSerial, 避免真正打开串口
        self._serial = serial_port

    def initialize(self) -> "Emm42Driver":
        """Open the configured USB-TTL serial port at 115200 baud."""
        if self._serial is None:
            self._serial = self._open_serial()
        return self

    def close(self) -> None:
        """Close the serial port if it has been initialized."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def enable(self, enabled: bool) -> None:
        """Enable or disable motor torque output.

        帧: addr F3 AB 01 00 6B -> 使能锁轴; addr F3 AB 00 00 6B -> 失能松轴。
        """
        self._write(0xF3, b"\xAB" + bytes((int(enabled), 0)))

    def set_speed(self, direction: Direction, rpm: int, acceleration: int) -> None:
        """Run continuously in the requested direction at ``rpm``.

        速度模式：持续按 rpm 转动直到收到停止指令。
        帧: addr F6 [dir 1B] [rpm 2B 大端] [accel 1B] [sync 1B] 6B
        """
        self._require_range("rpm", rpm, 0, self.MAX_RPM)
        self._require_range("acceleration", acceleration, 0, 255)
        self._write(0xF6, bytes((Direction(direction),)) + rpm.to_bytes(2, "big") + bytes((acceleration, 0)))

    def move_relative(
        self,
        degrees: float,
        rpm: int,
        acceleration: int,
        *,
        synchronized: bool = False,
    ) -> None:
        """Move a signed relative angle; optionally buffer for synchronized start."""
        self._move(degrees, rpm, acceleration, absolute=False, synchronized=synchronized)

    def move_absolute(
        self,
        degrees: float,
        rpm: int,
        acceleration: int,
        *,
        synchronized: bool = False,
    ) -> None:
        """Move to a signed absolute angle; optionally buffer for synchronized start."""
        self._move(degrees, rpm, acceleration, absolute=True, synchronized=synchronized)

    def trigger_sync_motion(self) -> None:
        """Trigger all previously buffered synchronized commands on the bus."""
        self._write(0xFF, b"\x66", address=0)

    def stop(self) -> None:
        """Immediately brake the motor in either speed or position mode.

        帧: addr FE 98 [sync] 6B  (0xFE 0x98 为"立即停止")。
        若电机已经停止/失能，驱动器可能返回 E2，表示无需再次执行停止。
        """
        self._write(0xFE, b"\x98\x00", accepted_statuses=(0x02, 0xE2))

    def home(self, mode: HomeMode) -> None:
        """Trigger the requested homing mode configured in the motor driver.

        帧: addr 9A [mode 1B] [sync 1B] 6B
        """
        self._write(0x9A, bytes((HomeMode(mode), 0)))

    def _move(
        self,
        degrees: float,
        rpm: int,
        acceleration: int,
        *,
        absolute: bool,
        synchronized: bool,
    ) -> None:
        # 位置命令不允许 0 位移
        if degrees == 0:
            raise ValueError("degrees must be non-zero")
        self._require_range("rpm", rpm, 0, self.MAX_RPM)
        self._require_range("acceleration", acceleration, 0, 255)
        # 角度 -> 脉冲数, 再校验脉冲数能放进 4 字节大端无符号整数
        pulses = round(abs(degrees) * self.pulses_per_revolution / 360)
        if not 1 <= pulses <= 0xFFFFFFFF:
            raise ValueError("degrees is outside the representable pulse range")
        # 正角度 -> CW, 负角度 -> CCW
        direction = Direction.CW if degrees > 0 else Direction.CCW
        parameters = (
            bytes((direction,))
            + rpm.to_bytes(2, "big")
            + bytes((acceleration,))
            + pulses.to_bytes(4, "big")
            + bytes((int(absolute), int(synchronized)))
        )
        # 位置命令帧: addr FD dir rpm acc pulses absolute sync 6B
        self._write(0xFD, parameters)

    def _open_serial(self) -> SerialPort:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required on Jetson. Install it in the target Python environment first."
            ) from error
        return serial.Serial(self.port, baudrate=self.BAUDRATE, timeout=self.timeout)

    def _write(
        self,
        function: int,
        parameters: bytes,
        *,
        accepted_statuses: tuple[int, ...] = (0x02,),
        address: int | None = None,
    ) -> None:
        # 拼装完整帧并验证驱动器 ACK: [motor_id] [function] [status] [0x6B]
        serial_port = self._require_serial()
        target_address = self.motor_id if address is None else address
        frame = bytes((target_address, function)) + parameters + bytes((self.CHECKSUM,))
        serial_port.write(frame)
        serial_port.flush()
        reply = serial_port.read(4)
        expected_prefix = bytes((target_address, function))
        if not reply:
            raise RuntimeError(
                f"Emm42 motor {self.motor_id} command 0x{function:02X} received no ACK"
            )
        is_expected_reply = (
            len(reply) == 4
            and reply[:2] == expected_prefix
            and reply[2] in accepted_statuses
            and reply[3] == self.CHECKSUM
        )
        if not is_expected_reply:
            raise RuntimeError(
                f"Emm42 motor {self.motor_id} command 0x{function:02X} returned unexpected ACK: {reply.hex(' ')}"
            )

    def _require_serial(self) -> SerialPort:
        if self._serial is None:
            raise RuntimeError("call initialize() before sending commands")
        return self._serial

    @staticmethod
    def _require_range(name: str, value: int, minimum: int, maximum: int) -> None:
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in {minimum}..{maximum}")

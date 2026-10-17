"""Emm42 V5.0 UART protocol driver.

Emm42 V5.0 闭环步进驱动器串口协议(ZDT Emm 固件)：

  帧格式:  [addr 1B] [function 1B] [data...] [0x6B 帧尾/固定校验]

说明:
  - 波特率 115200, 8N1, TTL 电平；用 USB-TTL 模块连接，驱动器电源独立接 12-32V。
  - 接线: USB-TTL 的 TX -> 驱动器 R/A/H, RX -> T/B/L, GND -> GND。
  - direction: 0=CW(顺时针), 1=CCW(逆时针)。
  - sync 字节 0=立即执行, 1=缓存后由 ``00 FF 66 6B`` 同步帧触发。
  - 速度/加速度单位: RPM(0..3000) 与加速度档位(0..255)。
  - 位置模式下脉冲数 = |角度| / 360 * pulses_per_revolution；
    默认 pulses_per_revolution=3200(1.8°步距角 x 16细分 = 3200 脉冲/圈)。

## 写命令与读命令的应答规则不一样(说明书 4.2 / 8.1)

``Response`` 菜单只决定**控制动作命令列表**(说明书 6.3.1)里的命令回不回包：

| Response | 确认收到命令 | 位置模式的到位命令 |
|---|---|---|
| ``None`` | 不回 | 不回 |
| ``Receive`` | 回 | 不回 |
| ``Reached`` | 不回 | 回 ``地址 FD 9F 6B`` |
| ``Both`` | 回 | 回 |

**读取类命令**(说明书 6.3.2 / 6.3.4)不在那张表里，它们本来就是"发一条、回一条数据"，
不受 ``Response`` 影响。所以机械臂 5/6/7 号轴虽然是 ``Response=None``(构造时
``expect_ack=False``)，**照样能读位置和状态**——这也是 ``wait_until_in_position``
能在不回 ACK 的驱动器上工作的原因。

读取命令的返回帧长度见 :meth:`Emm42Driver._read` 的 ``REPLY_LENGTHS``。多字节数值一律
**大端**，带符号的量按说明书的通用套路编码为 ``符号字节(00 正 / 01 负) + 无符号数值``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Final, Optional, Protocol, Tuple


class SerialPort(Protocol):
    """Minimum serial interface used by the driver and its tests.

    ``write`` 的形状按 **pyserial 的真实实现**声明，而不是按"看起来更安全"的写法：

    - ``data`` 是位置参数（``/`` 之后不能再按关键字传；pyserial 与现有假串口都这样调）；
    - 返回值可能是 ``None``（部分后端/假件如此），所以是 ``Optional[int]``。

    本驱动只执行 ``serial_port.write(frame)`` 后 ``flush``，**不消费**这个返回值，
    所以这里只修正协议声明，不引入"短写要报错"之类的新行为——那是另一件事。
    """

    is_open: bool

    def write(self, data: bytes, /) -> Optional[int]: ...

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


#: 编码器(``0x31``)与实时位置(``0x36``)一圈的数值：内部 4 倍频，一圈 0-65535。
#: 注意 ``0x31`` 只有单圈绝对值，**区分不了圈数**；``0x36`` 的 4 字节字段可以。
POSITION_UNITS_PER_REVOLUTION: Final = 65536


@dataclass(frozen=True)
class MotorStatus:
    """``0x3A`` 电机状态标志字节(说明书 6.3.4)的逐位解释。"""

    #: bit0 ``0x01``：电机处于使能状态。
    enabled: bool
    #: bit1 ``0x02``：到位。判据是 |输入位置角度 - 电机实时位置角度| < 位置到达窗口。
    in_position: bool
    #: bit2 ``0x04``：堵转。撞到东西时软件能看见的就是这一位。
    stalled: bool
    #: bit3 ``0x08``：堵转保护已触发。置位后驱动器可能已经关闭输出，
    #: 需要发 ``0x0E 0x52``(或把 R/A/H 接 Gnd)才能解除。
    stall_protected: bool

    @classmethod
    def from_flags(cls, flags: int) -> "MotorStatus":
        return cls(
            enabled=bool(flags & 0x01),
            in_position=bool(flags & 0x02),
            stalled=bool(flags & 0x04),
            stall_protected=bool(flags & 0x08),
        )


@dataclass(frozen=True)
class HomingStatus:
    """``0x3B`` 回零状态标志字节(说明书 6.3.2)的逐位解释。"""

    #: bit0 ``0x01``：编码器就绪。没就绪说明没校准过，回零不可信。
    encoder_ready: bool
    #: bit1 ``0x02``：校准表就绪。
    calibration_ready: bool
    #: bit2 ``0x04``：正在回零。
    homing: bool
    #: bit3 ``0x08``：回零失败。说明书里这一位由 ``O_Tmo_Ms`` **回零超时**置位，
    #: 所以它只能证明"没转完"，**证明不了"回到了正确的那一圈"**。
    failed: bool

    @classmethod
    def from_flags(cls, flags: int) -> "HomingStatus":
        return cls(
            encoder_ready=bool(flags & 0x01),
            calibration_ready=bool(flags & 0x02),
            homing=bool(flags & 0x04),
            failed=bool(flags & 0x08),
        )


def _decode_signed(
    sign_byte: int, magnitude: bytes, *, name: str, motor_id: int
) -> int:
    """按说明书的通用套路解码 ``符号字节 + 大端无符号数值``。

    符号字节只允许 ``00``(正) 和 ``01``(负)。其它值说明帧已经错位，
    宁可报错也不要猜出一个看起来合理的数字。
    """
    if sign_byte not in (0x00, 0x01):
        raise RuntimeError(
            f"Emm42 motor {motor_id} returned an invalid {name} sign byte "
            f"0x{sign_byte:02X}"
        )
    value = int.from_bytes(magnitude, "big")
    return -value if sign_byte == 0x01 else value


class Emm42Driver:
    """Control one Emm42 V5.0 driver through its TTL UART port."""

    # ---------- 协议常量 ----------
    BAUDRATE = 115200  # 驱动器固定波特率
    CHECKSUM = 0x6B    # 固定帧尾字节(该协议所有命令都以 0x6B 结尾)
    MAX_RPM = 3000     # 速度上限

    #: 读取类命令的**整帧**长度(含地址、功能码与校验字节)，
    #: 逐条来自说明书 6.3.2 / 6.3.4。
    #: ``0x36`` 是 8 字节而不是 7——位置值前面还有一个符号字节，少算一个字节整帧就错位。
    REPLY_LENGTHS = {
        0x31: 5,  # 地址 + 31 + 编码器值(2B) + 校验
        0x36: 8,  # 地址 + 36 + 符号(1B) + 位置(4B) + 校验
        0x3A: 4,  # 地址 + 3A + 状态标志(1B) + 校验
        0x3B: 4,  # 地址 + 3B + 回零状态标志(1B) + 校验
    }

    def __init__(
        self,
        port: str,
        motor_id: int,
        *,
        timeout: float = 0.1,
        pulses_per_revolution: int = 3200,
        expect_ack: bool = True,
        serial_port: Optional[SerialPort] = None,
    ) -> None:
        # 地址 1..255, 同一串口总线上可用不同地址挂多台驱动器
        if not 1 <= motor_id <= 255:
            raise ValueError("motor_id must be in 1..255")
        if pulses_per_revolution <= 0:
            raise ValueError("pulses_per_revolution must be positive")
        self.port = port
        self.motor_id = motor_id
        self.timeout = timeout
        # Response=None 的驱动器会执行控制帧但不回 4 字节 ACK。
        self.expect_ack = expect_ack
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

    def move_relative_pulses(
        self,
        pulses: int,
        rpm: int,
        acceleration: int,
        *,
        synchronized: bool = False,
    ) -> None:
        """Move a signed pulse count; positive pulses mean CW.

        与 ``move_relative`` 的唯一区别是本方法直接给脉冲数，不再经过
        "角度 -> 脉冲" 的有损换算，机械臂和麦轮的位置模式都走这里。
        """
        self._move_pulses(pulses, rpm, acceleration, absolute=False, synchronized=synchronized)

    def move_absolute_pulses(
        self,
        pulses: int,
        rpm: int,
        acceleration: int,
        *,
        synchronized: bool = False,
    ) -> None:
        """Move to a signed absolute pulse position; optionally buffered."""
        self._move_pulses(pulses, rpm, acceleration, absolute=True, synchronized=synchronized)

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

    # ------------------------------------------------------- 读取类命令
    def read_encoder(self) -> int:
        """读线性化校准后的单圈绝对值编码器(``0x31``)，返回 ``0..65535``。

        这是**单圈**读数，一圈只有一个值，所以能回答"电机轴在本圈转到哪了"，
        不能回答"转了几圈"。要带圈数的位置用 :meth:`read_position`。

        帧: addr 31 6B  ->  addr 31 编码器值(2B 大端) 6B
        """
        return int.from_bytes(self._read(0x31), "big")

    def read_position(self) -> int:
        """读电机**实时位置**(``0x36``)，单位是 1/65536 电机圈，带符号。

        这是编码器测出来的位置，不是"发下去多少脉冲"——所以它是判断
        "命令发了、轴却没动"（丢步、堵转）的依据。手册的 4 字节字段可表达
        多圈，但**手册没有说明多圈计数在掉电后是否保持**，上电后的绝对值
        仍然需要回零或相机锚定其中一条来建立。

        帧: addr 36 6B  ->  addr 36 符号 + 位置(4B 大端) 6B
        """
        payload = self._read(0x36)
        return _decode_signed(
            payload[0], payload[1:], name="position", motor_id=self.motor_id
        )

    def read_status(self) -> MotorStatus:
        """读电机状态标志位(``0x3A``)：使能 / 到位 / 堵转 / 堵转保护。

        帧: addr 3A 6B  ->  addr 3A 状态标志(1B) 6B
        """
        return MotorStatus.from_flags(self._read(0x3A)[0])

    def read_homing_status(self) -> HomingStatus:
        """读回零状态标志位(``0x3B``)：编码器就绪 / 校准表就绪 / 正在回零 / 回零失败。

        帧: addr 3B 6B  ->  addr 3B 回零状态标志(1B) 6B
        """
        return HomingStatus.from_flags(self._read(0x3B)[0])

    def wait_until_in_position(
        self,
        timeout: float,
        *,
        poll_interval: float = 0.01,
        departure_timeout: Optional[float] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> MotorStatus:
        """轮询 ``0x3A`` 的到位位，直到到位、超时，或判定已经堵转。

        返回**最后一次读到的**状态对象，不抛异常：

        - ``status.in_position`` —— 到底等到了没有；
        - ``status.stalled`` / ``status.stall_protected`` —— 是不是撞上了东西。

        "没等到"在真实机械上是正常结果（目标被挡住、行程不够、负载太重），
        不是协议错误，所以由调用方决定怎么处理，这里不替它抛。

        刚发完位置命令时驱动器可能还没更新到位位，那一位还留着**上一次**动作的结果；
        此时立刻读会看到"已经到位"，等待就等于没等。所以先等它离开到位区、再等到位。
        ``departure_timeout`` 限定前一段等多久，默认取 ``timeout`` 的四分之一
        （最多 1 秒）：真的走了一段就会离开，没离开说明这次动作本来就极短，
        或者命令还没被处理。

        :param timeout: 总超时（秒）。超过就返回最后读到的状态。
        :param poll_interval: 两次读之间睡多久（秒）。
        :param departure_timeout: 等待"离开到位区"的上限（秒）；
            ``None`` 按上面的规则取。
        :param sleep: 睡函数，测试时注入假实现。
        :param clock: 单调时钟，测试时注入假实现（配合假 ``sleep`` 一起推进）。
        """
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if departure_timeout is None:
            departure_timeout = min(1.0, timeout / 4)
        elif departure_timeout < 0:
            raise ValueError("departure_timeout must be >= 0")

        deadline = clock() + timeout
        status = self.read_status()

        departure_deadline = min(deadline, clock() + departure_timeout)
        while status.in_position and clock() < departure_deadline:
            sleep(poll_interval)
            status = self.read_status()

        while not status.in_position:
            # 堵转不会再到位，继续等只是白等：直接把这一位交回调用方。
            if status.stalled or status.stall_protected:
                return status
            if clock() >= deadline:
                return status
            sleep(poll_interval)
            status = self.read_status()

        return status

    def _read(self, function: int) -> bytes:
        """发一条读取命令，返回**去掉地址/功能码/校验之后**的数据字节。

        读取类命令不受 ``Response`` 菜单影响（见模块文档字符串），所以
        ``expect_ack=False`` 的驱动器照样能读——这里**不看** ``expect_ack``。

        正常返回的长度由 :data:`REPLY_LENGTHS` 给出；命令不被接受时驱动器返回
        ``[地址] 00 EE 6B``，比正常帧短，先单独认出来再校验长度。
        """
        try:
            total = self.REPLY_LENGTHS[function]
        except KeyError:
            raise ValueError(
                f"0x{function:02X} is not a known Emm42 read command"
            ) from None

        serial_port = self._require_serial()
        serial_port.write(bytes((self.motor_id, function, self.CHECKSUM)))
        serial_port.flush()
        reply = serial_port.read(total)
        if not reply:
            raise RuntimeError(
                f"Emm42 motor {self.motor_id} read command "
                f"0x{function:02X} received no reply"
            )
        if (
            len(reply) >= 3
            and reply[0] == self.motor_id
            and reply[1] == 0x00
            and reply[2] == 0xEE
        ):
            raise RuntimeError(
                f"Emm42 motor {self.motor_id} rejected read command 0x{function:02X}: "
                f"{reply.hex(' ')}"
            )
        if (
            len(reply) != total
            or reply[0] != self.motor_id
            or reply[1] != function
            or reply[-1] != self.CHECKSUM
        ):
            raise RuntimeError(
                f"Emm42 motor {self.motor_id} read command 0x{function:02X} returned "
                f"unexpected reply: {reply.hex(' ')}"
            )
        return reply[2:-1]

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
        # 角度 -> 脉冲数（保留原来的校验顺序：先角度、再速度/加速度、最后脉冲范围）
        self._require_range("rpm", rpm, 0, self.MAX_RPM)
        self._require_range("acceleration", acceleration, 0, 255)
        pulses = round(abs(degrees) * self.pulses_per_revolution / 360)
        if not 1 <= pulses <= 0xFFFFFFFF:
            raise ValueError("degrees is outside the representable pulse range")
        self._write_position(
            Direction.CW if degrees > 0 else Direction.CCW,
            pulses,
            rpm,
            acceleration,
            absolute=absolute,
            synchronized=synchronized,
        )

    def _move_pulses(
        self,
        pulses: int,
        rpm: int,
        acceleration: int,
        *,
        absolute: bool,
        synchronized: bool,
    ) -> None:
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        # 相对模式下 0 位移没有意义；绝对模式下 0 是"回到原点"，必须允许——
        # 这是回零之后归位的唯一一条命令形式。
        if pulses == 0 and not absolute:
            raise ValueError("pulses must be non-zero in relative mode")
        if not -0xFFFFFFFF <= pulses <= 0xFFFFFFFF:
            raise ValueError("pulses is outside the representable range")
        self._require_range("rpm", rpm, 0, self.MAX_RPM)
        self._require_range("acceleration", acceleration, 0, 255)
        self._write_position(
            # 绝对模式下方向由驱动器自己算，所以 0 归到 CW 只是取一个确定值。
            Direction.CW if pulses >= 0 else Direction.CCW,
            abs(pulses),
            rpm,
            acceleration,
            absolute=absolute,
            synchronized=synchronized,
        )

    def _write_position(
        self,
        direction: Direction,
        pulses: int,
        rpm: int,
        acceleration: int,
        *,
        absolute: bool,
        synchronized: bool,
    ) -> None:
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
        accepted_statuses: Tuple[int, ...] = (0x02,),
        address: Optional[int] = None,
    ) -> None:
        # 拼装完整帧并验证驱动器 ACK: [motor_id] [function] [status] [0x6B]
        serial_port = self._require_serial()
        target_address = self.motor_id if address is None else address
        frame = bytes((target_address, function)) + parameters + bytes((self.CHECKSUM,))
        serial_port.write(frame)
        serial_port.flush()
        if not self.expect_ack:
            return
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

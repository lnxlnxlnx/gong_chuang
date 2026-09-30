"""Serial protocol for the 9600 baud multi-servo controller in the old project."""

from __future__ import annotations

from typing import Protocol


class SerialPort(Protocol):
    def write(self, data: bytes) -> int: ...
    def flush(self) -> None: ...
    def read(self, count: int) -> bytes: ...
    def close(self) -> None: ...


class ServoController:
    """Address one servo through the controller's ``55 55 length command`` frames."""

    BAUDRATE = 9600

    def __init__(self, port: str, *, serial_port: SerialPort | None = None,
                 timeout: float = 0.5) -> None:
        self.port = port
        self.timeout = timeout
        self._serial = serial_port

    def initialize(self) -> "ServoController":
        """Open the serial controller at its documented baud rate."""
        if self._serial is None:
            import serial
            self._serial = serial.Serial(self.port, self.BAUDRATE, timeout=self.timeout)
        return self

    def close(self) -> None:
        """Close the controller port after the manual test."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def read_position(self, servo_id: int) -> int:
        """Read a pulse position; zero returned servos means the target did not answer."""
        self._validate("servo_id", servo_id, 1, 253)
        self._write(bytes((0x55, 0x55, 0x04, 0x15, 0x01, servo_id)))
        header = self._require_serial().read(5)
        if header != bytes((0x55, 0x55, 0x03, 0x15, 0x00)) and header != bytes((0x55, 0x55, 0x06, 0x15, 0x01)):
            raise TimeoutError(f"舵机控制板位置回包异常：{header.hex(' ')}")
        if header[4] == 0:
            raise RuntimeError(f"舵机 ID {servo_id} 未返回位置；请检查舵机供电、数据线和 ID")
        payload = self._require_serial().read(3)
        if len(payload) != 3 or payload[0] != servo_id:
            raise RuntimeError(f"舵机 ID {servo_id} 位置回包不完整或 ID 不符：{payload.hex(' ')}")
        return int.from_bytes(payload[1:3], "little")

    def set_position(self, servo_id: int, *, position: int, duration_ms: int) -> None:
        """Send one confirmed position command in the controller's pulse units."""
        self._validate("servo_id", servo_id, 1, 253)
        self._validate("position", position, 0, 1000)
        self._validate("duration_ms", duration_ms, 1, 30000)
        frame = (bytes((0x55, 0x55, 0x08, 0x03, 0x01))
                 + duration_ms.to_bytes(2, "little") + bytes((servo_id,))
                 + position.to_bytes(2, "little"))
        self._write(frame)

    def _write(self, data: bytes) -> None:
        port = self._require_serial()
        if port.write(data) != len(data):
            raise IOError("舵机控制板指令发送不完整")
        port.flush()

    def _require_serial(self) -> SerialPort:
        if self._serial is None:
            raise RuntimeError("请先初始化舵机控制板")
        return self._serial

    @staticmethod
    def _validate(name: str, value: int, minimum: int, maximum: int) -> None:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} 必须在 {minimum}..{maximum} 之间")

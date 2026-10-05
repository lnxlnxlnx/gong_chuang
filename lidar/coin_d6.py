"""One COIN-D6 scanning-lidar session: start, collect a cloud, stop."""

from __future__ import annotations

import sys
import time
from typing import Callable, List, Optional

from .cloud import CloudPoint, PointCloud
from .config import ScanConfig
from .protocol import (
    ACK_STATUS_START,
    ACK_STATUS_STOP,
    START_COMMAND,
    STOP_COMMAND,
    Ack,
    DeviceInfo,
    LidarSerial,
    Packet,
    PacketReader,
)


class LidarError(RuntimeError):
    """Base class for every COIN-D6 failure."""


class LidarAckError(LidarError):
    """The sensor refused a start or stop command."""


class LidarTimeoutError(LidarError):
    """The sensor sent nothing usable before the deadline."""


class CoinD6:
    """Drive one COIN-D6 lidar over a serial port.

    The sensor powers up stopped and only streams after ``start()``; call
    ``scan()`` for the usual start/collect/stop cycle, or ``start()`` and
    ``collect_scan()`` separately when the rotation must stay running.
    """

    BAUDRATE = 230400

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = BAUDRATE,
        timeout: float = 0.2,
        ack_timeout: float = 1.0,
        strict_checksum: bool = True,
        require_ack: bool = True,
        serial_port: Optional[LidarSerial] = None,
    ) -> None:
        if not port:
            raise ValueError("port must not be empty")
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if ack_timeout <= 0:
            raise ValueError("ack_timeout must be positive")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ack_timeout = ack_timeout
        self.require_ack = require_ack
        self._strict_checksum = strict_checksum
        # Tests inject a fake serial port so no hardware is required.
        self._serial = serial_port
        self._reader: Optional[PacketReader] = None

    def initialize(self) -> "CoinD6":
        """Open the serial port and prepare the packet reader."""
        if self._serial is None:
            self._serial = self._open_serial()
        self._reader = PacketReader(self._serial, strict_checksum=self.strict_checksum)
        return self

    @property
    def strict_checksum(self) -> bool:
        return self._strict_checksum

    def close(self) -> None:
        """Close the serial port if it was opened."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None
            self._reader = None

    def __enter__(self) -> "CoinD6":
        return self.initialize()

    def __exit__(self, *exception) -> None:
        self.close()

    def start(self, timeout: Optional[float] = None) -> Optional[Ack]:
        """Send the start command and wait for the sensor's acknowledgement.

        Returns ``None`` when ``require_ack`` is false and the sensor stays
        silent; an acknowledgement that rejects the command always raises.
        """
        return self._command(START_COMMAND, ACK_STATUS_START, "启动", timeout)

    def stop(self, timeout: Optional[float] = None) -> Optional[Ack]:
        """Send the stop command and wait for the sensor's acknowledgement."""
        return self._command(STOP_COMMAND, ACK_STATUS_STOP, "停止", timeout)

    def read_packet(self, timeout: Optional[float] = None) -> Packet:
        """Return the next decoded packet, skipping noise and damaged frames."""
        try:
            return self._require_reader().read_packet(
                self.timeout if timeout is None else timeout
            )
        except TimeoutError as error:
            raise LidarTimeoutError("雷达在超时前没有上传任何数据包") from error

    def read_device_info(self, timeout: Optional[float] = None) -> DeviceInfo:
        """Read the device description the sensor uploads once after power-up.

        The packet is only sent when the sensor powers up, so a timeout here is
        normal and reported as a :class:`LidarTimeoutError` for the caller to
        ignore rather than a crash.
        """
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LidarTimeoutError(
                    "未读到设备信息包（该包只在雷达上电时发送一次，正常现象）"
                )
            try:
                packet = self.read_packet(remaining)
            except LidarTimeoutError:
                raise LidarTimeoutError(
                    "未读到设备信息包（该包只在雷达上电时发送一次，正常现象）"
                ) from None
            if isinstance(packet, DeviceInfo):
                return packet

    def collect_scan(
        self,
        scan: ScanConfig,
        *,
        on_revolution: Optional[Callable[[int], None]] = None,
    ) -> PointCloud:
        """Collect ``scan.revolutions`` complete revolutions of accepted points.

        The scan window is applied here: points outside ``scan.min_distance_mm``
        / ``scan.max_distance_mm`` or outside the configured angle window are
        dropped, and the collected angle is mirrored and offset so that ``0°``
        is forward and angles grow counter-clockwise.
        """
        reader = self._require_reader()
        deadline = time.monotonic() + scan.timeout_seconds
        points: List[CloudPoint] = []
        device: Optional[DeviceInfo] = None
        revolutions = 0
        started = False

        while len(points) < scan.max_points:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                packet = reader.read_packet(remaining)
            except TimeoutError:
                break
            if isinstance(packet, DeviceInfo):
                device = packet
                continue
            if isinstance(packet, Ack):
                continue
            if packet.is_start and started:
                revolutions += 1
                if on_revolution is not None:
                    on_revolution(revolutions)
                if revolutions >= scan.revolutions:
                    break
            started = True
            for sensor_point in packet.points:
                angle = correct_angle(sensor_point.angle_degrees, scan)
                if not scan.min_distance_mm <= sensor_point.distance_mm <= scan.max_distance_mm:
                    continue
                if not scan.angle_min_degrees <= angle <= scan.angle_max_degrees:
                    continue
                points.append(CloudPoint(
                    angle_degrees=angle,
                    distance_mm=sensor_point.distance_mm,
                    intensity=sensor_point.intensity,
                    high_reflection=sensor_point.high_reflection,
                ))
                if len(points) >= scan.max_points:
                    break

        if not points:
            raise LidarTimeoutError(
                f"未收到可用的点云数据（丢弃噪声 {reader.skipped_bytes} 字节、"
                f"损坏数据包 {reader.rejected_packets} 个），请检查端口、波特率与启动指令"
            )
        return PointCloud(points=tuple(points), revolutions=revolutions, device=device)

    def scan(
        self,
        scan: ScanConfig,
        *,
        on_revolution: Optional[Callable[[int], None]] = None,
    ) -> PointCloud:
        """Run the full start / collect / stop cycle and always stop afterwards."""
        self.initialize()
        self.start()
        try:
            return self.collect_scan(scan, on_revolution=on_revolution)
        finally:
            original_error = sys.exc_info()[0] is not None
            try:
                self.stop()
            except Exception as error:  # cleanup must not hide a scan failure
                print(f"停止激光雷达失败：{error}", file=sys.stderr)
                if not original_error:
                    raise

    def _command(
        self, command: bytes, expected_status: int, name: str, timeout: Optional[float]
    ) -> Optional[Ack]:
        """Write one four-byte command and require the matching acknowledgement."""
        serial_port = self._require_serial()
        reader = self._require_reader()
        serial_port.reset_input_buffer()
        reader.reset()
        serial_port.write(command)
        deadline = time.monotonic() + (self.ack_timeout if timeout is None else timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._missing_ack(name)
            try:
                packet = reader.read_packet(remaining)
            except TimeoutError:
                return self._missing_ack(name)
            if isinstance(packet, Ack):
                if packet.status != expected_status:
                    raise LidarAckError(
                        f"雷达拒绝{name}指令（应答状态 0x{packet.status:02X}）"
                    )
                return packet

    def _missing_ack(self, name: str) -> None:
        """Return ``None`` when acknowledgements are optional, otherwise fail."""
        if not self.require_ack:
            return None
        raise LidarAckError(
            f"雷达未回复{name}指令的应答；若该雷达不回应答，请在配置中把 device.require_ack 改为 false"
        )

    def _require_serial(self) -> LidarSerial:
        if self._serial is None:
            raise LidarError("串口尚未打开，请先调用 initialize()")
        return self._serial

    def _require_reader(self) -> PacketReader:
        if self._reader is None:
            raise LidarError("串口尚未打开，请先调用 initialize()")
        return self._reader

    def _open_serial(self) -> LidarSerial:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "COIN-D6 需要当前 Python 环境提供 pyserial"
            ) from error
        try:
            return serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
        except OSError as error:  # pyserial's SerialException is an OSError
            raise LidarError(f"无法打开串口 {self.port}：{error}") from error


def correct_angle(sensor_angle_degrees: float, scan: ScanConfig) -> float:
    """Mirror a clockwise sweep, apply the offset and normalise to [-180, 180)."""
    angle = -sensor_angle_degrees if scan.clockwise else sensor_angle_degrees
    angle += scan.angle_offset_degrees
    return (angle + 180.0) % 360.0 - 180.0

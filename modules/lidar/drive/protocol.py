"""Wire protocol of the COIN-D6 scanning lidar (国科光芯 D6).

The sensor speaks two documented layers:

* Host commands are four raw bytes: ``AA 55 F0 0F`` starts scanning and
  ``AA 55 F5 0A`` stops it.  The sensor answers with a fixed 12-byte
  acknowledgement that starts with ``A5 5A`` and ends with a status byte.
* Everything the sensor uploads is framed as ``A5 5A LEN(2) CS(2) TYPE DATA``
  with little-endian fields.  ``CS`` is the 16-bit sum of every byte of the
  upload packet except the two ``CS`` bytes themselves.  ``TYPE`` is ``0x01``
  for the one-shot device description and ``0x81`` for point-cloud data.
* Point-cloud data starts with its own ``AA 55`` sync:
  ``AA 55 M&T LSN FSA(2) LSA(2) CS(2)`` followed by ``LSN`` samples of three
  bytes.  The sample checksum is the documented two-byte XOR of the packet and
  is stored little-endian before the samples.

The constants and formulas below follow ``COIN-D6 激光雷达数据格式标准说明 V1.0``.
Where that manual and the vendor's other lidar driver disagree — the grouping
of a three-byte sample inside the sample checksum, and whether point-cloud
packets carry the outer ``A5 5A`` frame — this reader accepts both forms so the
first run on real hardware does not depend on resolving the difference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Union

START_COMMAND = bytes((0xAA, 0x55, 0xF0, 0x0F))
STOP_COMMAND = bytes((0xAA, 0x55, 0xF5, 0x0A))

ACK_HEADER = b"\xA5\x5A"
ACK_SIZE = 12
ACK_STATUS_START = 0x50
ACK_STATUS_STOP = 0x55
ACK_MARKER = 0x07

UPLOAD_HEADER = b"\xA5\x5A"
UPLOAD_HEADER_SIZE = 7  # A5 5A LEN(2) CS(2) TYPE(1)
DEVICE_INFO_TYPE = 0x01
POINT_CLOUD_TYPE = 0x81
DEVICE_INFO_SIZE = 20
MAX_UPLOAD_SIZE = 4096

POINT_SYNC = b"\xAA\x55"
POINT_HEADER_SIZE = 10  # AA 55 M&T LSN FSA(2) LSA(2) CS(2)
POINT_SIZE = 3
MAX_PACKET_POINTS = 128
#: Accepted sample-checksum groupings, most likely first; see :func:`point_checksum`.
CHECKSUM_VARIANTS = ("vendor", "manual")

# Angle_fsa = (FSA >> 1) / 64, so one raw step after the check bit is 1/64 degree.
ANGLE_DEGREES_PER_STEP = 1.0 / 64.0


class LidarSerial(Protocol):
    def read(self, size: int) -> bytes: ...

    def write(self, data: bytes) -> int: ...

    def reset_input_buffer(self) -> None: ...


@dataclass(frozen=True)
class Ack:
    """A 12-byte command acknowledgement from the sensor."""

    raw: bytes
    status: int
    checksum_valid: bool

    @property
    def is_start(self) -> bool:
        return self.status == ACK_STATUS_START


@dataclass(frozen=True)
class DeviceInfo:
    """The 20-byte description the sensor uploads once after power-up."""

    model: str
    zero_offset_raw: int
    clockwise: bool
    needs_angle_correction: bool
    software_revision: int


@dataclass(frozen=True)
class SensorPoint:
    """One sample in the sensor's own frame, before any angle correction."""

    angle_degrees: float
    distance_mm: int
    intensity: int
    high_reflection: bool


@dataclass(frozen=True)
class PointCloudPacket:
    """One ``0x81`` upload: a slice of a revolution."""

    speed_hz: int
    is_start: bool
    start_angle_degrees: float
    end_angle_degrees: float
    points: Tuple[SensorPoint, ...]
    checksum_valid: bool
    checksum_variant: Optional[str] = None


Packet = Union[Ack, DeviceInfo, PointCloudPacket]

_INCOMPLETE = object()
_SKIP = object()


def decode_angle(raw: int) -> float:
    """Decode a 16-bit angle field: the low bit is a fixed ``1`` check bit."""
    return (raw >> 1) * ANGLE_DEGREES_PER_STEP


def decode_sensor_point(raw: bytes, angle_degrees: float = 0.0) -> SensorPoint:
    """Decode one three-byte sample: 14-bit distance, 8-bit intensity, flags."""
    if len(raw) != POINT_SIZE:
        raise ValueError("a COIN-D6 sample is exactly three bytes")
    low, second, high = raw
    distance_mm = high * 64 + (second >> 2)
    intensity = (second & 0x03) * 64 + (low >> 2)
    return SensorPoint(
        angle_degrees=angle_degrees,
        distance_mm=distance_mm,
        intensity=intensity,
        high_reflection=bool(low & 0x01),
    )


def upload_checksum(packet: bytes) -> int:
    """Byte sum of an ``A5 5A`` upload frame, excluding its own checksum bytes."""
    if len(packet) < UPLOAD_HEADER_SIZE:
        raise ValueError("upload frame is too short to checksum")
    return (sum(packet[:4]) + sum(packet[6:])) & 0xFFFF


def point_checksum(header: bytes, samples: bytes, variant: str = "vendor") -> int:
    """Two-byte XOR checksum of a point-cloud packet (the ``CS`` field excluded).

    The 8-byte header is XORed as little-endian 16-bit words, which every source
    agrees on.  The three bytes of one sample are grouped differently by the
    manual and by the shipped firmware, so both groupings are supported:

    ``vendor``
        ``d0 ^ (d1 | d2 << 8)`` — what the shipped COIN lidar drivers do
        (``ROBOTIS-GIT/coin_d4_driver``, ``lukaskrnac/COIN-D6-lidar-ROS2``).
    ``manual``
        ``(d0 | d1 << 8) ^ d2`` — what the D6 manual's wording describes, where
        the sample's third byte is zero-extended into its own 16-bit word.

    ``header`` is ``AA 55 M&T LSN FSA(2) LSA(2)`` and ``samples`` holds whole
    three-byte samples.
    """
    if len(header) != POINT_HEADER_SIZE - 2 or len(samples) % POINT_SIZE:
        raise ValueError("point checksum needs an 8-byte header and whole samples")
    if variant not in CHECKSUM_VARIANTS:
        raise ValueError(f"unknown checksum variant: {variant}")
    checksum = 0
    for start in range(0, len(header), 2):
        checksum ^= int.from_bytes(header[start:start + 2], "little")
    for start in range(0, len(samples), POINT_SIZE):
        first, second, third = samples[start:start + POINT_SIZE]
        if variant == "vendor":
            checksum ^= first
            checksum ^= second | (third << 8)
        else:
            checksum ^= first | (second << 8)
            checksum ^= third
    return checksum & 0xFFFF


def matching_checksum_variant(header: bytes, samples: bytes, stored: int) -> Optional[str]:
    """Return the checksum grouping that reproduces ``stored``, or ``None``."""
    for variant in CHECKSUM_VARIANTS:
        if point_checksum(header, samples, variant) == stored:
            return variant
    return None


def parse_ack(data: bytes) -> Ack:
    """Decode a 12-byte command acknowledgement.

    The manual's own error-acknowledgement example does not agree with the XOR
    rule used by its two success examples, so a mismatching trailing byte is
    reported through ``checksum_valid`` instead of rejecting the frame.
    """
    if len(data) != ACK_SIZE or data[:2] != ACK_HEADER:
        raise ValueError("invalid COIN-D6 acknowledgement frame")
    expected = 0
    for byte in data[:ACK_SIZE - 1]:
        expected ^= byte
    return Ack(raw=bytes(data), status=data[2], checksum_valid=expected == data[-1])


def parse_device_info(payload: bytes) -> DeviceInfo:
    """Decode the 20-byte device description."""
    if len(payload) != DEVICE_INFO_SIZE:
        raise ValueError("the COIN-D6 device description is 20 bytes")
    model = payload[:12].split(b"\x00", 1)[0].decode("ascii", "replace")
    return DeviceInfo(
        model=model,
        zero_offset_raw=int.from_bytes(payload[12:14], "little"),
        clockwise=payload[14] == 0x00,
        needs_angle_correction=payload[15] != 0x00,
        software_revision=payload[19],
    )


def parse_point_packet(data: bytes, verify_checksum: bool = True) -> PointCloudPacket:
    """Decode one complete ``AA 55 ...`` point-cloud packet."""
    if len(data) < POINT_HEADER_SIZE or data[:2] != POINT_SYNC:
        raise ValueError("invalid COIN-D6 point-cloud packet header")
    count = data[3]
    expected_size = POINT_HEADER_SIZE + count * POINT_SIZE
    if len(data) != expected_size:
        raise ValueError(
            f"point-cloud packet declares {count} samples but carries {len(data)} bytes"
        )
    stored = int.from_bytes(data[8:10], "little")
    variant = matching_checksum_variant(data[:8], data[POINT_HEADER_SIZE:], stored)
    if verify_checksum and variant is None:
        raise ValueError("invalid COIN-D6 point-cloud checksum")

    start_angle = decode_angle(int.from_bytes(data[4:6], "little"))
    end_angle = decode_angle(int.from_bytes(data[6:8], "little"))
    points = []
    for index in range(count):
        sample = data[POINT_HEADER_SIZE + index * POINT_SIZE:][:POINT_SIZE]
        points.append(decode_sensor_point(
            sample, interpolate_angle(start_angle, end_angle, count, index)
        ))
    return PointCloudPacket(
        speed_hz=data[2] >> 1,
        is_start=bool(data[2] & 0x01),
        start_angle_degrees=start_angle,
        end_angle_degrees=end_angle,
        points=tuple(points),
        checksum_valid=variant is not None,
        checksum_variant=variant,
    )


def interpolate_angle(start_degrees: float, end_degrees: float, count: int, index: int) -> float:
    """Angle of sample ``index`` inside one packet, wrapping forward past 360."""
    if count <= 1:
        return start_degrees % 360.0
    span = (end_degrees - start_degrees) % 360.0
    return (start_degrees + span * index / (count - 1)) % 360.0


class PacketReader:
    """Turn a COIN-D6 byte stream into acknowledgements, info and point packets."""

    def __init__(self, serial_port: LidarSerial, strict_checksum: bool = True) -> None:
        self.serial_port = serial_port
        self.strict_checksum = strict_checksum
        self._buffer = bytearray()
        self.skipped_bytes = 0
        self.rejected_packets = 0

    def reset(self) -> None:
        """Forget buffered bytes, for example before sending a new command."""
        self._buffer.clear()

    def feed(self, chunk: bytes) -> None:
        """Append bytes received elsewhere to the decode buffer."""
        self._buffer.extend(chunk)

    def read_packet(self, timeout_seconds: float) -> Packet:
        """Read until one decodable packet arrives or the timeout expires."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            packet = self.next_packet()
            if packet is not None:
                return packet
            if time.monotonic() >= deadline:
                raise TimeoutError("no COIN-D6 packet received before the timeout")
            chunk = self.serial_port.read(256)
            if chunk:
                self.feed(chunk)

    def next_packet(self) -> Optional[Packet]:
        """Decode one buffered packet, dropping noise and damaged frames."""
        while len(self._buffer) >= 2:
            if self._buffer[:2] == POINT_SYNC:
                result = self._take_point_packet()
            elif self._buffer[:2] == UPLOAD_HEADER:
                result = self._take_upload_packet()
            else:
                self._resynchronise()
                continue
            if result is _INCOMPLETE:
                return None
            if result is _SKIP:
                continue
            return result
        return None

    def _take_point_packet(self):
        """Decode a bare ``AA 55`` packet, used when the sensor omits the envelope."""
        if len(self._buffer) < POINT_HEADER_SIZE:
            return _INCOMPLETE
        count = self._buffer[3]
        size = POINT_HEADER_SIZE + count * POINT_SIZE
        if count < 1 or count > MAX_PACKET_POINTS:
            self.rejected_packets += 1
            del self._buffer[0]
            return _SKIP
        if len(self._buffer) < size:
            return _INCOMPLETE
        data = bytes(self._buffer[:size])
        try:
            packet = parse_point_packet(data, verify_checksum=self.strict_checksum)
        except ValueError:
            self.rejected_packets += 1
            del self._buffer[0]
            return _SKIP
        del self._buffer[:size]
        return packet

    def _take_upload_packet(self):
        """Decode an ``A5 5A`` frame: acknowledgement, device info or point cloud."""
        if len(self._buffer) < 4:
            return _INCOMPLETE
        if self._buffer[3] == ACK_MARKER and self._buffer[2] in (ACK_STATUS_START, ACK_STATUS_STOP):
            if len(self._buffer) < ACK_SIZE:
                return _INCOMPLETE
            data = bytes(self._buffer[:ACK_SIZE])
            del self._buffer[:ACK_SIZE]
            try:
                return parse_ack(data)
            except ValueError:
                self.rejected_packets += 1
                return _SKIP
        if len(self._buffer) < UPLOAD_HEADER_SIZE:
            return _INCOMPLETE
        size = int.from_bytes(self._buffer[2:4], "little")
        if size < 1 or size > MAX_UPLOAD_SIZE:
            self.rejected_packets += 1
            del self._buffer[0]
            return _SKIP
        total = UPLOAD_HEADER_SIZE + size
        if len(self._buffer) < total:
            return _INCOMPLETE
        data = bytes(self._buffer[:total])
        if upload_checksum(data) != int.from_bytes(data[4:6], "little"):
            self.rejected_packets += 1
            del self._buffer[0]
            return _SKIP
        del self._buffer[:total]
        payload = data[UPLOAD_HEADER_SIZE:]
        packet_type = data[6]
        if packet_type == DEVICE_INFO_TYPE:
            try:
                return parse_device_info(payload)
            except ValueError:
                self.rejected_packets += 1
                return _SKIP
        if packet_type == POINT_CLOUD_TYPE:
            try:
                return parse_point_packet(payload, verify_checksum=self.strict_checksum)
            except ValueError:
                self.rejected_packets += 1
                return _SKIP
        self.rejected_packets += 1
        return _SKIP

    def _resynchronise(self) -> None:
        """Drop bytes up to the next plausible packet start."""
        candidates = [
            index for index in (
                self._buffer.find(ACK_HEADER, 1),
                self._buffer.find(POINT_SYNC, 1),
            ) if index > 0
        ]
        if candidates:
            index = min(candidates)
        else:
            index = len(self._buffer) - (1 if self._buffer[-1] in (0xA5, 0xAA) else 0)
        self.skipped_bytes += index
        del self._buffer[:index]

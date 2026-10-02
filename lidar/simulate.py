"""Synthetic COIN-D6 traffic: a fake sensor for tests and the no-hardware demo.

The encoder mirrors :mod:`lidar.protocol` exactly, so a simulated scan exercises
the real serial session, the real frame reader and the real renderer.  It is not
a substitute for testing against the sensor, but it proves the whole path works
before hardware is available.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .protocol import (
    ACK_HEADER,
    ACK_MARKER,
    ACK_STATUS_START,
    ACK_STATUS_STOP,
    DEVICE_INFO_TYPE,
    POINT_CLOUD_TYPE,
    POINT_SYNC,
    START_COMMAND,
    STOP_COMMAND,
    UPLOAD_HEADER,
    point_checksum,
    upload_checksum,
)

# A small rectangular room with one round pillar and a reflective strip.
ROOM_HALF_X_MM = 2200.0
ROOM_HALF_Y_MM = 1600.0
PILLAR_CENTER_MM = (1100.0, 500.0)
PILLAR_RADIUS_MM = 160.0
REFLECTOR_X_MM = -2200.0
REFLECTOR_HALF_Y_MM = 260.0
DEFAULT_POINTS_PER_PACKET = 12
DEFAULT_POINTS_PER_REVOLUTION = 144


def encode_sample(distance_mm: int, intensity: int = 120, high_reflection: bool = False) -> bytes:
    """Pack one sample into the documented three bytes."""
    distance = max(0, min(0x3FFF, int(distance_mm)))
    intensity = max(0, min(255, int(intensity)))
    low = ((intensity & 0x3F) << 2) | (0x01 if high_reflection else 0x00)
    second = ((distance & 0x3F) << 2) | ((intensity >> 6) & 0x03)
    high = (distance >> 6) & 0xFF
    return bytes((low, second, high))


def encode_angle(angle_degrees: float) -> bytes:
    """Encode one angle field, including the mandatory low check bit."""
    raw = (int(round((angle_degrees % 360.0) * 64.0)) << 1) | 0x01
    return (raw & 0xFFFF).to_bytes(2, "little")


def encode_point_packet(
    samples: Sequence[bytes],
    start_angle_degrees: float,
    end_angle_degrees: float,
    *,
    is_start: bool = False,
    speed_hz: int = 10,
    variant: str = "vendor",
    envelope: bool = False,
) -> bytes:
    """Build one ``0x81`` point-cloud packet, optionally inside an ``A5 5A`` frame."""
    control = ((speed_hz & 0x7F) << 1) | (0x01 if is_start else 0x00)
    header = (
        POINT_SYNC
        + bytes((control, len(samples)))
        + encode_angle(start_angle_degrees)
        + encode_angle(end_angle_degrees)
    )
    payload = b"".join(samples)
    checksum = point_checksum(header, payload, variant)
    packet = header + checksum.to_bytes(2, "little") + payload
    return encode_upload(POINT_CLOUD_TYPE, packet) if envelope else packet


def encode_device_info(model: str = "COIN-D6", *, clockwise: bool = True, revision: int = 1) -> bytes:
    """Build the 20-byte device description."""
    text = model.encode("ascii")[:12].ljust(12, b"\x00")
    body = (
        text
        + b"\x00\x00"
        + bytes((0x00 if clockwise else 0x01, 0x01))
        + b"\x00\x00\x00"
        + bytes((revision,))
    )
    return encode_upload(DEVICE_INFO_TYPE, body)


def encode_ack(status: int) -> bytes:
    """Build the 12-byte acknowledgement the sensor returns for a command."""
    body = ACK_HEADER + bytes((status, ACK_MARKER)) + b"\x00" * 7
    checksum = 0
    for byte in body:
        checksum ^= byte
    return body + bytes((checksum,))


def encode_upload(packet_type: int, payload: bytes) -> bytes:
    """Wrap a payload in the ``A5 5A LEN CS TYPE`` upload frame."""
    head = UPLOAD_HEADER + len(payload).to_bytes(2, "little") + b"\x00\x00" + bytes((packet_type,))
    checksum = upload_checksum(head + payload)
    return head[:4] + checksum.to_bytes(2, "little") + head[6:] + payload


def scene_distance(angle_degrees: float) -> Tuple[int, int, bool]:
    """Return ``(distance_mm, intensity, high_reflection)`` for one scene ray.

    ``angle_degrees`` is a normal counter-clockwise angle in the room frame.
    """
    radians = math.radians(angle_degrees)
    direction = (math.cos(radians), math.sin(radians))
    distance = _ray_to_box(direction)
    pillar = _ray_to_circle(direction, PILLAR_CENTER_MM, PILLAR_RADIUS_MM)
    if pillar is not None and pillar < distance:
        distance = pillar
    half_width = math.degrees(math.atan2(REFLECTOR_HALF_Y_MM, ROOM_HALF_X_MM))
    reflector = abs((angle_degrees - 180.0 + 180.0) % 360.0 - 180.0) < half_width
    distance += 6.0 * math.sin(math.radians(angle_degrees * 7.0))
    intensity = int(max(20, min(250, 250 - distance / 18.0)))
    return int(round(max(1.0, distance))), intensity, reflector


def _ray_to_box(direction: Tuple[float, float]) -> float:
    dx, dy = direction
    limits = []
    if abs(dx) > 1e-9:
        limits.append(ROOM_HALF_X_MM / abs(dx))
    if abs(dy) > 1e-9:
        limits.append(ROOM_HALF_Y_MM / abs(dy))
    return min(limits)


def _ray_to_circle(direction: Tuple[float, float], center: Tuple[float, float],
                   radius: float) -> Optional[float]:
    """Nearest positive intersection distance of a ray from the origin."""
    cx, cy = center
    projection = direction[0] * cx + direction[1] * cy
    perpendicular_squared = cx * cx + cy * cy - projection * projection
    if perpendicular_squared > radius * radius:
        return None
    offset = math.sqrt(radius * radius - perpendicular_squared)
    for distance in (projection - offset, projection + offset):
        if distance > 0.0:
            return distance
    return None


class SimulatedLidar:
    """A fake serial port that streams COIN-D6 packets until stopped.

    Implements the small serial surface :class:`lidar.coin_d6.CoinD6` needs, so
    the real driver can be driven without hardware.
    """

    def __init__(
        self,
        *,
        variant: str = "vendor",
        envelope: bool = False,
        points_per_packet: int = DEFAULT_POINTS_PER_PACKET,
        points_per_revolution: int = DEFAULT_POINTS_PER_REVOLUTION,
        revolutions: Optional[int] = None,
        device_info: bool = True,
        device_model: str = "COIN-D6",
    ) -> None:
        self.variant = variant
        self.envelope = envelope
        self.points_per_packet = points_per_packet
        self.points_per_revolution = points_per_revolution
        self.revolutions = revolutions
        self.is_open = True
        self.commands: List[bytes] = []
        self._pending = bytearray(encode_device_info(device_model) if device_info else b"")
        self._scanning = False
        self._index = 0
        self._revolution = 0
        self._start_sent = True

    # -- serial surface ---------------------------------------------------
    def read(self, size: int) -> bytes:
        while self.is_open and self._scanning and len(self._pending) < size:
            packet = self._next_packet()
            if packet is None:
                break
            self._pending.extend(packet)
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def write(self, data: bytes) -> int:
        self.commands.append(bytes(data))
        if data == START_COMMAND:
            self._scanning = True
            self._index = 0
            self._start_sent = False
            self._pending.extend(encode_ack(ACK_STATUS_START))
        elif data == STOP_COMMAND:
            self._scanning = False
            self._pending.extend(encode_ack(ACK_STATUS_STOP))
        return len(data)

    def reset_input_buffer(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        self.is_open = False

    # -- stream generation ------------------------------------------------
    @property
    def _step_degrees(self) -> float:
        return 360.0 / self.points_per_revolution

    def _next_packet(self) -> Optional[bytes]:
        """Emit the ring-start packet, then sample packets, revolution by revolution."""
        if self.revolutions is not None and self._revolution >= self.revolutions:
            return None
        if not self._start_sent:
            self._start_sent = True
            self._index = 0
            packet = self._packet(0, 1, is_start=True)
            self._index = 1
            return packet
        count = min(self.points_per_packet, self.points_per_revolution - self._index)
        packet = self._packet(self._index, count)
        self._index += count
        if self._index >= self.points_per_revolution:
            self._index = 0
            self._start_sent = False
            self._revolution += 1
        return packet

    def _packet(self, first_index: int, count: int, *, is_start: bool = False) -> bytes:
        """Build the packet covering samples ``first_index .. first_index+count``."""
        step = self._step_degrees
        samples = []
        for offset in range(count):
            sensor_angle = (first_index + offset) * step % 360.0
            # The sensor sweeps clockwise, so the room angle is its mirror.
            distance, intensity, reflection = scene_distance(-sensor_angle)
            samples.append(encode_sample(distance, intensity, reflection))
        return encode_point_packet(
            samples,
            first_index * step % 360.0,
            (first_index + max(0, count - 1)) * step % 360.0,
            is_start=is_start,
            variant=self.variant,
            envelope=self.envelope,
        )

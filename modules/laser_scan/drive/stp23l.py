"""Read checked STP-23L UART measurement frames."""

from __future__ import annotations

import struct
import time
from statistics import median
from typing import Optional, Protocol


HEADER = b"\xAA" * 4
POINT_COUNT = 12
POINT_SIZE = 15
PAYLOAD_SIZE = POINT_COUNT * POINT_SIZE + 4  # Twelve points and a timestamp.
FRAME_SIZE = 4 + 6 + PAYLOAD_SIZE + 1


class LidarSerial(Protocol):
    def read(self, size: int) -> bytes: ...

    def reset_input_buffer(self) -> None: ...


def parse_measurement_frame(frame: bytes) -> list[int]:
    """Decode one checked measurement packet into twelve distances in millimeters."""
    if len(frame) != FRAME_SIZE or frame[:4] != HEADER:
        raise ValueError("invalid STP-23L measurement frame length or header")
    if frame[4] != 0 or frame[5] != 0x02 or frame[6:8] != b"\x00\x00":
        raise ValueError("not an STP-23L measurement frame")
    if int.from_bytes(frame[8:10], "little") != PAYLOAD_SIZE:
        raise ValueError("invalid STP-23L measurement payload length")
    if sum(frame[4:-1]) & 0xFF != frame[-1]:
        raise ValueError("invalid STP-23L frame checksum")
    return [struct.unpack_from("<h", frame, 10 + index * POINT_SIZE)[0]
            for index in range(POINT_COUNT)]


class Stp23lReader:
    """Find fresh measurement packets in a continuous UART byte stream."""

    def __init__(self, serial_port: LidarSerial) -> None:
        self.serial_port = serial_port
        self._buffer = bytearray()

    def read_distances(self, timeout_seconds: float) -> list[int]:
        """Wait for one valid frame, skipping noise and damaged packets."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            header_index = self._buffer.find(HEADER)
            if header_index < 0:
                del self._buffer[:-3]
            elif header_index:
                del self._buffer[:header_index]
            if len(self._buffer) >= 10 and self._buffer[:4] == HEADER:
                payload_length = int.from_bytes(self._buffer[8:10], "little")
                if payload_length > 320:
                    del self._buffer[0]
                    continue
                frame_length = 11 + payload_length
                if len(self._buffer) >= frame_length:
                    frame = bytes(self._buffer[:frame_length])
                    try:
                        distances = parse_measurement_frame(frame)
                    except ValueError:
                        del self._buffer[0]
                        continue
                    del self._buffer[:frame_length]
                    return distances
            chunk = self.serial_port.read(64)
            if chunk:
                self._buffer.extend(chunk)
        raise TimeoutError("no valid STP-23L measurement frame received")

    def sample_distance(
        self, minimum_mm: int, maximum_mm: int, timeout_seconds: float
    ) -> Optional[int]:
        """Discard pre-motion data and take the median of valid fresh distances."""
        self.serial_port.reset_input_buffer()
        self._buffer.clear()
        distances = self.read_distances(timeout_seconds)
        valid = [distance for distance in distances if minimum_mm <= distance <= maximum_mm]
        return int(median(valid)) if valid else None

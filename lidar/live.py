"""Live COIN-D6 acquisition: a background reader with a rolling point cloud.

The reader thread only collects; filtering happens when a snapshot is taken, so
moving a slider in the GUI re-filters the buffered cloud immediately instead of
only affecting points that arrive later.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, List, Optional

from .cloud import CloudPoint, PointCloud
from .coin_d6 import CoinD6, LidarError, LidarTimeoutError, correct_angle
from .config import ScanConfig
from .protocol import Ack, DeviceInfo


@dataclass(frozen=True)
class LiveStats:
    """What the GUI shows about the acquisition thread."""

    running: bool
    points: int
    revolutions: int
    packets: int
    rejected: int
    skipped_bytes: int
    idle_seconds: Optional[float]
    error: Optional[str] = None

    @property
    def healthy(self) -> bool:
        return self.error is None


class LiveCloud:
    """Keep the newest revolutions of a continuously scanned point cloud.

    ``snapshot()`` returns the buffered points that pass the current distance
    and angle windows, so the view can be retuned without touching the sensor.
    """

    def __init__(
        self,
        lidar: CoinD6,
        scan: ScanConfig,
        *,
        revolutions_kept: int = 2,
        read_timeout: float = 0.2,
        on_note: Optional[Callable[[str], None]] = None,
    ) -> None:
        if revolutions_kept < 1:
            raise ValueError("revolutions_kept must be at least 1")
        self.lidar = lidar
        self.scan = scan
        self.read_timeout = read_timeout
        self._on_note = on_note
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._revolutions: List[List[CloudPoint]] = [[]]
        self._revolutions_kept = revolutions_kept
        self._packets = 0
        self._rejected = 0
        self._skipped_bytes = 0
        self._last_packet = 0.0
        self._error: Optional[str] = None
        self._device: Optional[DeviceInfo] = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "LiveCloud":
        """Start the background reader; returns immediately."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="coin-d6-live", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        """Ask the reader to finish and wait for the sensor to be stopped."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # -- data -------------------------------------------------------------
    def snapshot(self, *, max_points: int = 20000) -> PointCloud:
        """Filter the buffered revolutions with the current windows."""
        with self._lock:
            revolutions = [list(points) for points in self._revolutions]
            device = self._device
        points = []
        for revolution in revolutions:
            for point in revolution:
                if not self.scan.min_distance_mm <= point.distance_mm <= self.scan.max_distance_mm:
                    continue
                if not self.scan.angle_min_degrees <= point.angle_degrees <= self.scan.angle_max_degrees:
                    continue
                points.append(point)
                if len(points) >= max_points:
                    return PointCloud(points=tuple(points), revolutions=len(revolutions), device=device)
        return PointCloud(points=tuple(points), revolutions=len(revolutions), device=device)

    def set_filters(
        self,
        *,
        min_distance_mm: Optional[int] = None,
        max_distance_mm: Optional[int] = None,
        angle_min_degrees: Optional[float] = None,
        angle_max_degrees: Optional[float] = None,
    ) -> None:
        """Replace the display windows; the sensor itself is not reconfigured.

        Swapped limits are accepted and put back in order, so dragging two
        sliders past each other never produces an empty window.
        """
        changes = {}
        if min_distance_mm is not None:
            changes["min_distance_mm"] = int(min_distance_mm)
        if max_distance_mm is not None:
            changes["max_distance_mm"] = int(max_distance_mm)
        if angle_min_degrees is not None:
            changes["angle_min_degrees"] = float(angle_min_degrees)
        if angle_max_degrees is not None:
            changes["angle_max_degrees"] = float(angle_max_degrees)
        scan = replace(self.scan, **changes)
        if scan.min_distance_mm > scan.max_distance_mm:
            scan = replace(
                scan, min_distance_mm=scan.max_distance_mm, max_distance_mm=scan.min_distance_mm
            )
        if scan.angle_min_degrees > scan.angle_max_degrees:
            scan = replace(
                scan,
                angle_min_degrees=scan.angle_max_degrees,
                angle_max_degrees=scan.angle_min_degrees,
            )
        self.scan = scan

    def set_revolutions_kept(self, revolutions: int) -> None:
        """Change how many revolutions are retained for the view."""
        if revolutions < 1:
            raise ValueError("revolutions must be at least 1")
        with self._lock:
            self._revolutions_kept = revolutions
            self._trim()

    @property
    def revolutions_kept(self) -> int:
        return self._revolutions_kept

    def clear(self) -> None:
        """Drop every buffered point."""
        with self._lock:
            self._revolutions = [[]]

    @property
    def stats(self) -> LiveStats:
        with self._lock:
            points = sum(len(revolution) for revolution in self._revolutions)
            revolutions = sum(1 for revolution in self._revolutions if revolution)
            idle = time.monotonic() - self._last_packet if self._last_packet else None
            return LiveStats(
                running=self.running,
                points=points,
                revolutions=revolutions,
                packets=self._packets,
                rejected=self._rejected,
                skipped_bytes=self._skipped_bytes,
                idle_seconds=idle,
                error=self._error,
            )

    # -- reader thread ----------------------------------------------------
    def _run(self) -> None:
        try:
            self.lidar.initialize()
            self.lidar.start()
        except (LidarError, OSError) as error:
            self._fail(error)
            return
        try:
            while not self._stop.is_set():
                try:
                    packet = self.lidar.read_packet(self.read_timeout)
                except LidarTimeoutError:
                    continue  # the sensor is simply quiet for a moment
                self._absorb(packet)
        except (LidarError, OSError) as error:
            self._fail(error)
        finally:
            self._shutdown_sensor()

    def _absorb(self, packet) -> None:
        with self._lock:
            self._packets += 1
            self._last_packet = time.monotonic()
            if isinstance(packet, DeviceInfo):
                self._device = packet
                return
            if isinstance(packet, Ack):
                return
            if packet.is_start and self._revolutions[-1]:
                self._revolutions.append([])
                self._trim()
            for sensor_point in packet.points:
                self._revolutions[-1].append(CloudPoint(
                    angle_degrees=correct_angle(sensor_point.angle_degrees, self.scan),
                    distance_mm=sensor_point.distance_mm,
                    intensity=sensor_point.intensity,
                    high_reflection=sensor_point.high_reflection,
                ))

    def _trim(self) -> None:
        while len(self._revolutions) > self._revolutions_kept:
            self._revolutions.pop(0)

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            self._error = str(error)
        if self._on_note is not None:
            self._on_note(str(error))

    def _shutdown_sensor(self) -> None:
        """Stop the rotation and close the port; never raise from the thread."""
        try:
            self.lidar.stop()
        except Exception as error:  # noqa: BLE001 - shutdown must not propagate
            if self._on_note is not None:
                self._on_note(f"停止激光雷达失败：{error}")
        finally:
            try:
                self.lidar.close()
            except Exception:  # noqa: BLE001 - nothing left to do
                pass

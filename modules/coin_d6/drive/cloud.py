"""Point-cloud value objects shared by the COIN-D6 driver and its renderer."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

from .protocol import DeviceInfo


@dataclass(frozen=True)
class CloudPoint:
    """One measured point, expressed in the sensor plane.

    ``angle_degrees`` is the corrected sensor angle: counter-clockwise from the
    sensor's zero mark, so ``x`` points forward and ``y`` points to the left.
    """

    angle_degrees: float
    distance_mm: int
    intensity: int
    high_reflection: bool

    @property
    def x_mm(self) -> float:
        return self.distance_mm * math.cos(math.radians(self.angle_degrees))

    @property
    def y_mm(self) -> float:
        return self.distance_mm * math.sin(math.radians(self.angle_degrees))


@dataclass(frozen=True)
class PointCloud:
    """Every accepted point of one acquisition, plus the sensor that produced it."""

    points: Tuple[CloudPoint, ...] = ()
    revolutions: int = 0
    device: Optional[DeviceInfo] = None

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[CloudPoint]:
        return iter(self.points)

    @property
    def empty(self) -> bool:
        return not self.points

    @property
    def nearest_mm(self) -> Optional[int]:
        return min((point.distance_mm for point in self.points), default=None)

    @property
    def farthest_mm(self) -> Optional[int]:
        return max((point.distance_mm for point in self.points), default=None)

    def save_csv(self, path: Path) -> Path:
        """Write one row per point; the parent directory is created on demand."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"point cloud output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("angle_degrees", "distance_mm", "intensity", "high_reflection"))
            for point in self.points:
                writer.writerow((
                    f"{point.angle_degrees:.3f}",
                    point.distance_mm,
                    point.intensity,
                    int(point.high_reflection),
                ))
        return path

    @classmethod
    def load_csv(cls, path: Path) -> "PointCloud":
        """Read back a cloud saved by :meth:`save_csv` (used by offline demos)."""
        path = Path(path)
        points = []
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                points.append(CloudPoint(
                    angle_degrees=float(row["angle_degrees"]),
                    distance_mm=int(row["distance_mm"]),
                    intensity=int(row["intensity"]),
                    high_reflection=bool(int(row["high_reflection"])),
                ))
        return cls(points=tuple(points))

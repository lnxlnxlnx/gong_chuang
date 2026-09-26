"""Bounded 90-degree step-and-sample sweep for a motor-mounted STP-23L."""

from __future__ import annotations

import json
import csv
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

from motor_drivers.emm42 import Emm42Driver


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "laser_scan.json"


@dataclass(frozen=True)
class ScanConfig:
    """Hardware connection, sweep geometry, and obstacle threshold."""

    motor_id: int
    motor_port: str
    motor_expect_ack: bool
    lidar_port: str
    pulses_per_revolution: int
    rpm: int
    acceleration: int
    max_ccw_pulses: int
    step_pulses: int
    sweep_degrees: float
    settle_seconds: float
    sample_timeout_seconds: float
    min_distance_mm: int
    max_distance_mm: int
    obstacle_distance_mm: int

    def __post_init__(self) -> None:
        if isinstance(self.motor_id, bool) or not isinstance(self.motor_id, int) or not 1 <= self.motor_id <= 255:
            raise ValueError("motor.id must be an integer in 1..255")
        if not self.motor_port or not self.lidar_port:
            raise ValueError("motor.port and lidar.port must be configured")
        if self.motor_port == self.lidar_port:
            raise ValueError("motor.port and lidar.port must be different serial ports")
        if not isinstance(self.motor_expect_ack, bool):
            raise ValueError("motor.expect_ack must be a boolean")
        for name, value, minimum, maximum in (
            ("pulses_per_revolution", self.pulses_per_revolution, 1, 0xFFFFFFFF),
            ("rpm", self.rpm, 1, Emm42Driver.MAX_RPM),
            ("acceleration", self.acceleration, 0, 255),
            ("max_ccw_pulses", self.max_ccw_pulses, 1, 0xFFFFFFFF),
            ("step_pulses", self.step_pulses, 1, 0xFFFFFFFF),
            ("min_distance_mm", self.min_distance_mm, 1, 32767),
            ("max_distance_mm", self.max_distance_mm, 1, 32767),
            ("obstacle_distance_mm", self.obstacle_distance_mm, 1, 32767),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
        if self.step_pulses > self.max_ccw_pulses:
            raise ValueError("scan.step_pulses cannot exceed scan.max_ccw_pulses")
        if not 0 < self.sweep_degrees <= 360:
            raise ValueError("scan.sweep_degrees must be in (0, 360]")
        if self.settle_seconds < 0 or self.sample_timeout_seconds <= 0:
            raise ValueError("scan timing values must be non-negative, with a positive timeout")
        if not self.min_distance_mm <= self.obstacle_distance_mm <= self.max_distance_mm:
            raise ValueError("obstacle_distance_mm must be within the measurement range")


@dataclass(frozen=True)
class ScanPoint:
    pulses: int
    angle_degrees: float
    distance_mm: Optional[int]
    obstacle: bool


def load_scan_config(path: Path = DEFAULT_CONFIG_PATH) -> ScanConfig:
    """Read JSON once before any serial port is opened."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read laser scan configuration: {path}") from error
    try:
        motor, lidar, scan = (document[name] for name in ("motor", "lidar", "scan"))
        return ScanConfig(
            motor_id=motor["id"],
            motor_port=motor["port"],
            motor_expect_ack=motor["expect_ack"],
            lidar_port=lidar["port"],
            pulses_per_revolution=motor["pulses_per_revolution"],
            rpm=motor["rpm"],
            acceleration=motor["acceleration"],
            max_ccw_pulses=scan["max_ccw_pulses"],
            step_pulses=scan["step_pulses"],
            sweep_degrees=scan["sweep_degrees"],
            settle_seconds=scan["settle_seconds"],
            sample_timeout_seconds=scan["sample_timeout_seconds"],
            min_distance_mm=scan["min_distance_mm"],
            max_distance_mm=scan["max_distance_mm"],
            obstacle_distance_mm=scan["obstacle_distance_mm"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"incomplete laser scan configuration: {error}") from error


class Motor(Protocol):
    def enable(self, enabled: bool) -> None: ...

    def move_relative(self, degrees: float, rpm: int, acceleration: int) -> None: ...

    def stop(self) -> None: ...


class DistanceReader(Protocol):
    def sample_distance(
        self, minimum_mm: int, maximum_mm: int, timeout_seconds: float
    ) -> Optional[int]: ...


def scan_sector(
    motor: Motor,
    lidar: DistanceReader,
    config: ScanConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
    confirm_return: Optional[Callable[[], None]] = None,
) -> list[ScanPoint]:
    """Sample from the operator-set CW endpoint to 90 degrees CCW and return."""
    samples = []
    commanded_pulses = 0

    def sample() -> None:
        distance = lidar.sample_distance(
            config.min_distance_mm, config.max_distance_mm, config.sample_timeout_seconds
        )
        samples.append(ScanPoint(
            pulses=commanded_pulses,
            angle_degrees=config.sweep_degrees * commanded_pulses / config.max_ccw_pulses,
            distance_mm=distance,
            obstacle=distance is not None and distance <= config.obstacle_distance_mm,
        ))

    try:
        motor.enable(True)
        sample()
        while commanded_pulses < config.max_ccw_pulses:
            step = min(config.step_pulses, config.max_ccw_pulses - commanded_pulses)
            motor.move_relative(-step * 360 / config.pulses_per_revolution,
                                config.rpm, config.acceleration)
            commanded_pulses += step
            sleep(step * 60 / (config.pulses_per_revolution * config.rpm)
                  + config.settle_seconds)
            sample()
        motor.move_relative(commanded_pulses * 360 / config.pulses_per_revolution,
                            config.rpm, config.acceleration)
        sleep(commanded_pulses * 60 / (config.pulses_per_revolution * config.rpm)
              + config.settle_seconds)
        if confirm_return is not None:
            confirm_return()
        return samples
    finally:
        original_error = sys.exc_info()[0] is not None
        cleanup_error = None
        for action in (motor.stop, lambda: motor.enable(False)):
            try:
                action()
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
                print(f"扫描清理失败：{error}", file=sys.stderr)
        if cleanup_error is not None and not original_error:
            raise cleanup_error


def save_scan(
    samples: list[ScanPoint], config: ScanConfig, prefix: Path
) -> tuple[Path, Path]:
    """Save numerical scan data and a full radar with third-quadrant samples."""
    csv_path = prefix.with_suffix(".csv")
    svg_path = prefix.with_suffix(".svg")
    if csv_path.exists() or svg_path.exists():
        raise FileExistsError(f"scan output already exists: {prefix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("pulses", "angle_degrees", "distance_mm", "obstacle"))
        for point in samples:
            writer.writerow((point.pulses, point.angle_degrees,
                             "" if point.distance_mm is None else point.distance_mm,
                             point.obstacle))

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": "0 0 600 650",
        "width": "600", "height": "650", "role": "img",
        "aria-label": "STP-23L 90 degree counterclockwise obstacle scan",
    })
    ET.SubElement(svg, "rect", {"width": "600", "height": "650", "fill": "#ffffff"})
    ET.SubElement(svg, "text", {
        "x": "32", "y": "38", "fill": "#172528", "font-size": "18",
        "font-family": "Arial, sans-serif",
    }).text = "STP-23L | 90° CCW scan"
    origin_x, origin_y, radius = 300, 315, 250
    ET.SubElement(svg, "path", {
        "d": (f"M {origin_x} {origin_y} L {origin_x - radius} {origin_y} "
              f"A {radius} {radius} 0 0 0 {origin_x} {origin_y + radius} Z"),
        "fill": "#e9f5f2",
    })
    for fraction in (0.25, 0.5, 0.75, 1.0):
        ring = radius * fraction
        distance = config.max_distance_mm * fraction
        ET.SubElement(svg, "circle", {
            "class": "distance-ring", "cx": str(origin_x), "cy": str(origin_y),
            "r": f"{ring:.1f}", "fill": "none", "stroke": "#d4dedf",
            "stroke-width": "1",
        })
        ET.SubElement(svg, "text", {
            "x": str(origin_x + 8), "y": f"{origin_y - ring + 14:.1f}",
            "fill": "#56676a", "font-size": "12", "font-family": "Arial, sans-serif",
        }).text = f"{distance:g} mm"
    threshold_radius = radius * config.obstacle_distance_mm / config.max_distance_mm
    ET.SubElement(svg, "circle", {
        "class": "threshold-ring", "cx": str(origin_x), "cy": str(origin_y),
        "r": f"{threshold_radius:.1f}", "fill": "none", "stroke": "#d58b43",
        "stroke-width": "2", "stroke-dasharray": "5 4",
    })
    ET.SubElement(svg, "text", {
        "x": f"{origin_x + threshold_radius + 8:.1f}", "y": str(origin_y + 24),
        "fill": "#9a5926", "font-size": "12", "font-family": "Arial, sans-serif",
    }).text = f"Obstacle <= {config.obstacle_distance_mm} mm"
    for x1, y1, x2, y2 in (
        (origin_x - radius, origin_y, origin_x + radius, origin_y),
        (origin_x, origin_y - radius, origin_x, origin_y + radius),
    ):
        ET.SubElement(svg, "line", {
            "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
            "stroke": "#aabbbc", "stroke-width": "1",
        })
    for angle in (0, 30, 60, 90):
        radians = math.radians(180 + angle)
        x = origin_x + radius * math.cos(radians)
        y = origin_y - radius * math.sin(radians)
        if angle in (30, 60):
            ET.SubElement(svg, "line", {
                "x1": str(origin_x), "y1": str(origin_y),
                "x2": f"{x:.1f}", "y2": f"{y:.1f}", "stroke": "#d4dedf",
            })
        label = "0° start" if angle == 0 else "90° end" if angle == 90 else f"{angle}°"
        ET.SubElement(svg, "text", {
            "x": f"{x:.1f}", "y": f"{y + (20 if angle else -8):.1f}", "fill": "#56676a",
            "font-size": "12", "font-family": "Arial, sans-serif",
            "text-anchor": "middle",
        }).text = label
    for point in samples:
        if point.distance_mm is None:
            continue
        scaled = radius * min(point.distance_mm, config.max_distance_mm) / config.max_distance_mm
        radians = math.radians(180 + point.angle_degrees)
        x = origin_x + scaled * math.cos(radians)
        y = origin_y - scaled * math.sin(radians)
        ET.SubElement(svg, "circle", {
            "class": "scan-point", "cx": f"{x:.1f}", "cy": f"{y:.1f}", "r": "5",
            "fill": "#c64538" if point.obstacle else "#147d85",
        })
    ET.SubElement(svg, "text", {
        "x": "32", "y": "625", "fill": "#56676a", "font-size": "12",
        "font-family": "Arial, sans-serif",
    }).text = "Red: obstacle    Teal: farther    Empty quadrants: not scanned"
    ET.ElementTree(svg).write(svg_path, encoding="utf-8", xml_declaration=True)
    return csv_path, svg_path

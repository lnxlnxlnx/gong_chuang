"""Load the editable COIN-D6 lidar settings from one JSON file.

Everything an operator normally changes between machines lives in
``config/coin_d6.json``: the serial port per platform, the scan window
(angle and distance limits) and the point-cloud rendering options.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

COLOR_MODES = ("distance", "intensity", "angle")
# The COIN-D6 encodes distance in 14 bits, so it can never report more than this.
MAX_DISTANCE_MM = 16383


@dataclass(frozen=True)
class DeviceConfig:
    """Serial link settings for one COIN-D6 sensor."""

    port: str
    baudrate: int
    ack_timeout_seconds: float
    read_timeout_seconds: float
    strict_checksum: bool
    require_ack: bool


@dataclass(frozen=True)
class ScanConfig:
    """Which part of each revolution is kept as point-cloud data."""

    revolutions: int
    angle_min_degrees: float
    angle_max_degrees: float
    min_distance_mm: int
    max_distance_mm: int
    clockwise: bool
    angle_offset_degrees: float
    max_points: int
    timeout_seconds: float


@dataclass(frozen=True)
class CloudView:
    """Rendering options for the saved point-cloud figure."""

    range_mm: float
    color_by: str
    point_size: float
    colormap: str
    show_window: bool


@dataclass(frozen=True)
class LidarConfig:
    """One validated device + scan + view configuration."""

    profile: str
    device: DeviceConfig
    scan: ScanConfig
    view: CloudView


def default_profile_name() -> str:
    """Return the profile that matches the machine this code runs on."""
    return "windows" if os.name == "nt" else "jetson"


def load_lidar_config(
    profile_name: Optional[str] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> LidarConfig:
    """Read one platform profile and reject settings the sensor cannot honour."""
    profile = profile_name or default_profile_name()
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read laser lidar configuration: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON laser lidar configuration: {config_path}") from error

    root = _mapping(document, "root")
    profiles = _mapping(root.get("profiles"), "profiles")
    if profile not in profiles:
        raise ValueError(f"unknown profile: {profile}")
    profile_document = _mapping(profiles[profile], f"profiles.{profile}")
    device = _mapping(root.get("device", {}), "device")
    scan = _mapping(root.get("scan", {}), "scan")
    view = _mapping(root.get("visualization", {}), "visualization")

    scan_config = ScanConfig(
        revolutions=_integer(scan.get("revolutions", 3), "scan.revolutions", 1, 1000),
        angle_min_degrees=_number(
            scan.get("angle_min_degrees", -180.0), "scan.angle_min_degrees", -360.0, 360.0
        ),
        angle_max_degrees=_number(
            scan.get("angle_max_degrees", 180.0), "scan.angle_max_degrees", -360.0, 360.0
        ),
        min_distance_mm=_integer(
            scan.get("min_distance_mm", 30), "scan.min_distance_mm", 1, MAX_DISTANCE_MM
        ),
        max_distance_mm=_integer(
            scan.get("max_distance_mm", 12000), "scan.max_distance_mm", 1, MAX_DISTANCE_MM
        ),
        clockwise=_boolean(scan.get("clockwise", True), "scan.clockwise"),
        angle_offset_degrees=_number(
            scan.get("angle_offset_degrees", 0.0), "scan.angle_offset_degrees", -360.0, 360.0
        ),
        max_points=_integer(scan.get("max_points", 40000), "scan.max_points", 1, 1_000_000),
        timeout_seconds=_number(
            scan.get("timeout_seconds", 15.0), "scan.timeout_seconds", 0.0, exclusive=True
        ),
    )
    if scan_config.angle_min_degrees >= scan_config.angle_max_degrees:
        raise ValueError("scan.angle_min_degrees must be smaller than scan.angle_max_degrees")
    if scan_config.min_distance_mm > scan_config.max_distance_mm:
        raise ValueError("scan.min_distance_mm must not exceed scan.max_distance_mm")

    return LidarConfig(
        profile=profile,
        device=DeviceConfig(
            port=_string(profile_document.get("port"), f"profiles.{profile}.port"),
            baudrate=_integer(device.get("baudrate", 230400), "device.baudrate", 9600, 3_000_000),
            ack_timeout_seconds=_number(
                device.get("ack_timeout_seconds", 1.0), "device.ack_timeout_seconds", 0.0, exclusive=True
            ),
            read_timeout_seconds=_number(
                device.get("read_timeout_seconds", 0.2), "device.read_timeout_seconds", 0.0, exclusive=True
            ),
            strict_checksum=_boolean(device.get("strict_checksum", True), "device.strict_checksum"),
            require_ack=_boolean(device.get("require_ack", True), "device.require_ack"),
        ),
        scan=scan_config,
        view=CloudView(
            range_mm=_number(view.get("range_mm", 4000.0), "visualization.range_mm", 1.0, exclusive=True),
            color_by=_choice(
                view.get("color_by", "distance"), "visualization.color_by", COLOR_MODES
            ),
            point_size=_number(
                view.get("point_size", 6.0), "visualization.point_size", 0.0, exclusive=True
            ),
            colormap=_string(view.get("colormap", "turbo"), "visualization.colormap"),
            show_window=_boolean(view.get("show_window", True), "visualization.show_window"),
        ),
    )


# The COIN-D6 encodes distance in 14 bits, so it can never report more than this.
MAX_DISTANCE_MM = 16383


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _number(value: object, name: str, minimum: float, exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if number < minimum or (exclusive and number == minimum):
        operator = ">" if exclusive else ">="
        raise ValueError(f"{name} must be {operator} {minimum}")
    return number


def _choice(value: object, name: str, allowed: tuple) -> str:
    text = _string(value, name)
    if text not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    return text


def describe_config(config: LidarConfig) -> str:
    """Summarise the active settings for the manual test menu."""
    return (
        f"配置档 {config.profile} | 端口 {config.device.port} | {config.device.baudrate} bps | "
        f"采集 {config.scan.revolutions} 圈 | 距离 {config.scan.min_distance_mm}~"
        f"{config.scan.max_distance_mm} mm | 角度 {config.scan.angle_min_degrees:g}~"
        f"{config.scan.angle_max_degrees:g}°"
    )

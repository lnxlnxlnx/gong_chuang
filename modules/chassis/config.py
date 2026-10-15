"""Load editable platform profiles for mecanum hardware tests.

**只读并校验配置**：不建驱动、不开串口。装配在
:meth:`modules.chassis.runtime.MecanumChassis.open`（它会建 :class:`MecanumDrive` 并
打开或借用那条共享步进总线），所以这里不导入任何上层类型、也不导入 MecanumDrive——
那样会让"配置"反过来依赖"装配"，成环。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from modules.chassis.drive.mecanum import WheelPosition
from modules.motor.drive.emm42 import Emm42Driver

#: 轮速上限直接取自驱动器，不再绕经 ``MecanumDrive``——那会让配置反过来依赖装配。
MAX_RPM = Emm42Driver.MAX_RPM


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_WHEEL_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class HardwareProfile:
    """All calibration and test settings selected from one platform profile.

    The serial port is deliberately absent: it belongs to the shared stepper bus
    and is read from :mod:`modules.motor.config`.
    """

    motor_ids: Mapping[WheelPosition, int]
    wheel_signs: Mapping[WheelPosition, int]
    acceleration: int
    pattern: str
    rotation_gain: float
    motion_transform: str
    individual_rpm: int
    individual_duration: float
    chassis_rpm: int
    chassis_duration: float
    chassis_pause: float


def load_profile(profile_name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> HardwareProfile:
    """Load one named platform profile and reject incomplete or unsafe configuration."""
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read configuration: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration: {config_path}") from error

    root = _mapping(document, "root")
    vehicle = _mapping(root.get("vehicle"), "vehicle")
    profiles = _mapping(root.get("profiles"), "profiles")
    if profile_name not in profiles:
        raise ValueError(f"unknown profile: {profile_name}")
    profile = _mapping(profiles[profile_name], f"profiles.{profile_name}")

    motor_ids = _wheel_mapping(vehicle.get("motor_ids"), "vehicle.motor_ids", allow_negative=False)
    if len(set(motor_ids.values())) != len(motor_ids):
        raise ValueError("vehicle.motor_ids must be unique")
    wheel_signs = _wheel_mapping(vehicle.get("wheel_signs"), "vehicle.wheel_signs", allow_negative=True)
    if any(sign not in (-1, 1) for sign in wheel_signs.values()):
        raise ValueError("vehicle.wheel_signs values must be -1 or 1")

    acceleration = _integer(vehicle.get("acceleration"), "vehicle.acceleration", minimum=0, maximum=255)
    pattern = _string(vehicle.get("pattern"), "vehicle.pattern")
    if pattern not in ("X", "O"):
        raise ValueError("vehicle.pattern must be X or O")
    rotation_gain = _number(vehicle.get("rotation_gain"), "vehicle.rotation_gain", minimum=0, exclusive=True)
    motion_transform = _string(vehicle.get("motion_transform", "standard"), "vehicle.motion_transform")
    if motion_transform not in ("standard", "calibrated"):
        raise ValueError("vehicle.motion_transform must be standard or calibrated")

    individual = _mapping(profile.get("individual_wheels"), f"profiles.{profile_name}.individual_wheels")
    chassis = _mapping(profile.get("chassis_motion"), f"profiles.{profile_name}.chassis_motion")
    return HardwareProfile(
        motor_ids=motor_ids,
        wheel_signs=wheel_signs,
        acceleration=acceleration,
        pattern=pattern,
        rotation_gain=rotation_gain,
        motion_transform=motion_transform,
        individual_rpm=_integer(individual.get("rpm"), "individual_wheels.rpm", minimum=1, maximum=MAX_RPM),
        individual_duration=_number(individual.get("duration"), "individual_wheels.duration", minimum=0, exclusive=True),
        chassis_rpm=_integer(chassis.get("rpm"), "chassis_motion.rpm", minimum=1, maximum=MAX_RPM),
        chassis_duration=_number(chassis.get("duration"), "chassis_motion.duration", minimum=0, exclusive=True),
        chassis_pause=_number(chassis.get("pause"), "chassis_motion.pause", minimum=0),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _wheel_mapping(value: object, name: str, *, allow_negative: bool) -> dict[WheelPosition, int]:
    mapping = _mapping(value, name)
    if set(mapping) != set(_WHEEL_NAMES):
        raise ValueError(f"{name} must define exactly: {', '.join(_WHEEL_NAMES)}")
    minimum = -255 if allow_negative else 1
    return {
        WheelPosition[wheel_name]: _integer(mapping[wheel_name], f"{name}.{wheel_name}", minimum=minimum, maximum=255)
        for wheel_name in _WHEEL_NAMES
    }


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _number(value: object, name: str, *, minimum: float, exclusive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if number < minimum or (exclusive and number == minimum):
        operator = ">" if exclusive else ">="
        raise ValueError(f"{name} must be {operator} {minimum}")
    return number

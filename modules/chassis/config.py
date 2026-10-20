"""Load the mecanum chassis settings from **two** files.

- ``hardware.json`` —— 本机硬件事实：四个轮子的驱动器地址、单轮转向、驱动器要不要回 ACK、
  辊子排布、实车标定矩阵的选择器。换接线、换装法时改这里。
- ``config.json`` —— 运动调参：加速度档、旋转增益，以及各平台的真机演示 profile。

脉冲/圈是驱动器细分，只有一份，读 ``modules/motor/hardware.json``。

**只读并校验配置**：不建驱动、不开串口。装配在
:meth:`modules.chassis.runtime.MecanumChassis.open`，所以这里不导入任何上层类型、
也不导入 MecanumDrive——那样会让"配置"反过来依赖"装配"，成环。

两个文件都读完之后仍然返回**合并后的** :class:`HardwareProfile` 快照，下游不需要知道
参数来自哪一份文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from modules.chassis.drive.mecanum import WheelPosition
from modules.motor.config import DEFAULT_HARDWARE_PATH as DEFAULT_BUS_HARDWARE_PATH
from modules.motor.config import load_pulses_per_revolution
from modules.motor.drive.emm42 import Emm42Driver

#: 轮速上限直接取自驱动器，不再绕经 ``MecanumDrive``——那会让配置反过来依赖装配。
MAX_RPM = Emm42Driver.MAX_RPM


#: 运动调参：手改。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
#: 硬件事实：换机器/换接线时改。
DEFAULT_HARDWARE_PATH = Path(__file__).resolve().parent / "hardware.json"

_WHEEL_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class HardwareProfile:
    """硬件事实 + 本次平台的运动调参，合并成一个装配用快照。

    串口**故意不在这里**：它属于共享的步进总线，由 :mod:`modules.motor.config` 读。
    """

    motor_ids: Mapping[WheelPosition, int]
    wheel_signs: Mapping[WheelPosition, int]
    expect_ack: bool
    pulses_per_revolution: int
    acceleration: int
    pattern: str
    rotation_gain: float
    motion_transform: str
    individual_rpm: int
    individual_duration: float
    chassis_rpm: int
    chassis_duration: float
    chassis_pause: float


def load_profile(
    profile_name: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    hardware_path: Path = DEFAULT_HARDWARE_PATH,
    bus_hardware_path: Optional[Path] = None,
) -> HardwareProfile:
    """Load one named platform profile and reject incomplete or unsafe configuration.

    硬件键缺失时报出**具体文件与键名**，不退回代码默认值。
    """
    tuning = _read_json(config_path)
    hardware = _read_json(hardware_path)
    pulses_per_revolution = load_pulses_per_revolution(
        DEFAULT_BUS_HARDWARE_PATH if bus_hardware_path is None else bus_hardware_path
    )

    vehicle = _mapping(hardware.get("vehicle"), f"{hardware_path}: vehicle")
    tuning_vehicle = _mapping(tuning.get("vehicle"), f"{config_path}: vehicle")
    profiles = _mapping(tuning.get("profiles"), f"{config_path}: profiles")
    if profile_name not in profiles:
        raise ValueError(f"unknown profile：{config_path} 里没有 {profile_name!r}")
    profile = _mapping(profiles[profile_name], f"profiles.{profile_name}")

    motor_ids = _wheel_mapping(vehicle.get("motor_ids"), "vehicle.motor_ids", allow_negative=False)
    if len(set(motor_ids.values())) != len(motor_ids):
        raise ValueError("vehicle.motor_ids must be unique（不能有重复地址）")
    wheel_signs = _wheel_mapping(vehicle.get("wheel_signs"), "vehicle.wheel_signs", allow_negative=True)
    if any(sign not in (-1, 1) for sign in wheel_signs.values()):
        raise ValueError("vehicle.wheel_signs 只能是 -1 或 1")

    acceleration = _integer(
        tuning_vehicle.get("acceleration"), "vehicle.acceleration", minimum=0, maximum=255
    )
    pattern = _string(vehicle.get("pattern"), "vehicle.pattern")
    if pattern not in ("X", "O"):
        raise ValueError("vehicle.pattern must be X or O")
    rotation_gain = _number(
        tuning_vehicle.get("rotation_gain"), "vehicle.rotation_gain", minimum=0, exclusive=True
    )
    # ``motion_transform`` 没有默认值：漏写必须报错，不能静默退回未标定的标准矩阵。
    motion_transform = _string(vehicle.get("motion_transform"), "vehicle.motion_transform")
    if motion_transform not in ("standard", "calibrated"):
        raise ValueError("vehicle.motion_transform must be standard or calibrated")

    individual = _mapping(
        profile.get("individual_wheels"), f"profiles.{profile_name}.individual_wheels"
    )
    chassis = _mapping(profile.get("chassis_motion"), f"profiles.{profile_name}.chassis_motion")
    return HardwareProfile(
        motor_ids=motor_ids,
        wheel_signs=wheel_signs,
        expect_ack=_boolean(vehicle.get("expect_ack"), "vehicle.expect_ack"),
        pulses_per_revolution=pulses_per_revolution,
        acceleration=acceleration,
        pattern=pattern,
        rotation_gain=rotation_gain,
        motion_transform=motion_transform,
        individual_rpm=_integer(
            individual.get("rpm"), "individual_wheels.rpm", minimum=1, maximum=MAX_RPM
        ),
        individual_duration=_number(
            individual.get("duration"), "individual_wheels.duration", minimum=0, exclusive=True
        ),
        chassis_rpm=_integer(chassis.get("rpm"), "chassis_motion.rpm", minimum=1, maximum=MAX_RPM),
        chassis_duration=_number(
            chassis.get("duration"), "chassis_motion.duration", minimum=0, exclusive=True
        ),
        chassis_pause=_number(chassis.get("pause"), "chassis_motion.pause", minimum=0),
    )


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取配置：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"配置不是合法 JSON：{path}") from error
    return _mapping(document, str(path))


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
        WheelPosition[wheel_name]: _integer(
            mapping[wheel_name], f"{name}.{wheel_name}", minimum=minimum, maximum=255
        )
        for wheel_name in _WHEEL_NAMES
    }


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是 true 或 false，实际 {value!r}")
    return value


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

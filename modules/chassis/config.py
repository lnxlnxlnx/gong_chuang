"""Load the mecanum chassis settings from **two** files.

- ``hardware.json`` —— 本机硬件事实：四个轮子的驱动器地址、单轮转向、驱动器要不要回 ACK、
  辊子排布、实车标定矩阵的选择器，以及**轮子转一圈车走多少毫米**
  （``vehicle.mm_per_wheel_rev``）。换接线、换装法、换轮子时改这里。
- ``config.json`` —— 运动调参：默认轮速、加速度档、旋转增益，以及各平台的真机演示 profile。

脉冲/圈是驱动器细分，只有一份，读 ``modules/motor/hardware.json``。

``vehicle.mm_per_wheel_rev`` 是**填了才能用毫米指令**的标定值：它不改脉冲/圈的换算，
只给"车心要走多少毫米"提供一把尺。没填（``null`` 或整键缺失）时脉冲与计时动作照旧可用，
毫米动作会被明确拒绝，而不是拿一个默认轮径算出一个没人验证过的距离。

**只读并校验配置**：不建驱动、不开串口。装配在
:meth:`modules.chassis.runtime.MecanumChassis.open`，所以这里不导入任何上层类型、
也不导入 MecanumDrive——那样会让"配置"反过来依赖"装配"，成环。

两个文件都读完之后仍然返回**合并后的** :class:`HardwareProfile` 快照，下游不需要知道
参数来自哪一份文件。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
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

    ``rpm`` 是"调用方没给轮速时用它"的默认轮速（``config.json`` 的 ``vehicle.rpm``）；
    ``mm_per_wheel_rev`` 是轮子转一圈车走多少毫米的实测标定（一份值管四个轮子），
    ``None`` 表示还没填——那时毫米动作会被拒绝，脉冲与计时动作不受影响。
    ``mm_per_wheel_rev`` 带默认值所以排在最后，其余字段仍然是必填的。
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
    rpm: int
    mm_per_wheel_rev: Optional[float] = None
    #: 每轮有效行程增益；空 = 不修正（等价于四轮都是 1.0），见 :func:`_wheel_gains`。
    wheel_gains: Mapping[WheelPosition, float] = field(default_factory=dict)


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

    # 默认轮速是运动调参，毫米标定是硬件事实；两者都不退回代码里的默认值。
    rpm = _integer(tuning_vehicle.get("rpm"), "vehicle.rpm", minimum=1, maximum=MAX_RPM)
    mm_per_wheel_rev = _optional_positive_number(
        vehicle.get("mm_per_wheel_rev"), "vehicle.mm_per_wheel_rev"
    )
    wheel_gains = _wheel_gains(vehicle.get("wheel_gains"), "vehicle.wheel_gains")

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
        rpm=rpm,
        mm_per_wheel_rev=mm_per_wheel_rev,
        wheel_gains=wheel_gains,
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


def _wheel_gains(value: object, name: str) -> dict[WheelPosition, float]:
    """可选的每轮有效行程增益：``null``/缺失 = 不修正（四轮默认 1.0）。

    实测"四轮同样脉冲、车却稳定画了一段弧"时填这里：偏快那侧（车头转向的反侧）
    给 <1，两侧对称地各分一半。整键缺失不报错——没标定就是没修正，和 ``mm_per_wheel_rev`` 同理；
    但一旦写了就要求四个轮子齐全，避免漏一轮时静默按 1.0 跑。
    """
    if value is None:
        return {}
    mapping = _mapping(value, name)
    if set(mapping) != set(_WHEEL_NAMES):
        raise ValueError(f"{name} must define exactly: {', '.join(_WHEEL_NAMES)}")
    return {
        WheelPosition[wheel_name]: _number(
            mapping[wheel_name], f"{name}.{wheel_name}", minimum=0, exclusive=True
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


def _optional_positive_number(value: object, name: str) -> Optional[float]:
    """可选的正数标定值：``null`` / 缺失都表示**还没填**，不退回代码默认值。"""
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} 必须是正数或 null，实际 {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必须是有限正数或 null，实际 {value!r}")
    return number

"""The two bus-servo devices' settings, split into **hardware** and **tuning**.

- ``hardware.json`` —— 本机硬件事实：总线端口、波特率、串口超时；夹爪与转盘的舵机 ID；
  夹爪两个机械端位；转盘的**输出行程**（0..1000 对应多少度）、到位容差与各工位位置。
- ``config.json`` —— 运动调参：两个动作的 ``duration_ms`` 与转盘的停稳余量。

端口是**总线**的属性、不是某个设备的属性，所以由
:func:`modules.servo.drive.bus_servo.load_servo_bus_config` 从同一份 ``hardware.json`` 读。

两个舵机只靠 ID 区分，所以 :func:`load_turntable_config` 会拒绝和夹爪用同一个
``servo_id``：那个错误发出去之后，转盘指令会去开合夹爪（或反过来），现场看到的只是
"动作不对"，很难倒查到配置。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


#: 运动调参：手改。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
#: 硬件事实：换机器/换接线/重新装机时改。
DEFAULT_HARDWARE_PATH = Path(__file__).resolve().parent / "hardware.json"


@dataclass(frozen=True)
class GripperConfig:
    """夹爪舵机的地址、两个机械端位与动作耗时。"""

    servo_id: int
    open_position: int
    closed_position: int
    duration_ms: int


#: 转盘"回零"落在哪个工位上：``turntable.stations`` 里名为 ``home`` 的那一项。
HOME_STATION = "home"

#: 转盘至少要有"回零位"和另一个工位，否则分度没有意义。
MIN_TURNTABLE_STATIONS = 2

#: 舵机位置量程：0..1000。对应多少度由 ``turntable.travel_degrees`` 给出（本机 240°）。
POSITION_UNITS = 1000


@dataclass(frozen=True)
class TurntableConfig:
    """转盘舵机的地址、各工位位置、转动耗时与到位判据。

    ``stations`` 是**工位名 → 舵机位置**（0..1000）。名字由配置决定，代码不写死它，
    所以重新标定后只改 ``hardware.json``，调用方继续用 ``go_to_station("<名字>")``。

    ``travel_degrees`` 是输出轴总行程（0..1000 对应多少度）。它决定读回位置换算成
    角度时报出多少度，是**装机事实**，不是调参。
    """

    servo_id: int
    duration_ms: int
    stations: Mapping[str, int]
    travel_degrees: float = 240.0
    position_tolerance: int = 30
    settle_seconds: float = 0.2

    def station(self, name: str) -> int:
        """Return one station's position, or explain which names exist."""
        try:
            return self.stations[name]
        except KeyError:
            known = "、".join(sorted(self.stations))
            raise ValueError(f"未知工位 {name!r}；已配置的工位：{known}") from None

    @property
    def home_position(self) -> int:
        """Return the position that "回零" goes to."""
        return self.station(HOME_STATION)

    @property
    def degrees_per_unit(self) -> float:
        """一个舵机位置单位对应多少度。"""
        return self.travel_degrees / POSITION_UNITS


def load_gripper_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    hardware_path: Path = DEFAULT_HARDWARE_PATH,
) -> GripperConfig:
    """Read the gripper's hardware block plus its tuning block."""
    hardware = _read_json(hardware_path)
    tuning = _read_json(config_path)
    gripper = _mapping(hardware.get("gripper"), f"{hardware_path}: gripper")
    gripper_tuning = _mapping(tuning.get("gripper"), f"{config_path}: gripper")

    config = GripperConfig(
        servo_id=_integer(gripper.get("servo_id"), "gripper.servo_id", 1, 253),
        open_position=_integer(gripper.get("open_position"), "gripper.open_position", 0, 1000),
        closed_position=_integer(gripper.get("closed_position"), "gripper.closed_position", 0, 1000),
        duration_ms=_integer(gripper_tuning.get("duration_ms"), "gripper.duration_ms", 1, 30000),
    )
    if config.open_position == config.closed_position:
        raise ValueError("open_position 和 closed_position 不能相同")
    return config


def load_turntable_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    hardware_path: Path = DEFAULT_HARDWARE_PATH,
) -> TurntableConfig:
    """Read the turntable's hardware block plus its tuning block.

    读完之后还会拿 ``gripper.servo_id`` 对一遍：同一条总线上的两个舵机共用一个 ID 是
    **静默的**错配，不会报通信错误。
    """
    hardware = _read_json(hardware_path)
    tuning = _read_json(config_path)
    turntable = _mapping(hardware.get("turntable"), f"{hardware_path}: turntable")
    turntable_tuning = _mapping(tuning.get("turntable"), f"{config_path}: turntable")

    config = TurntableConfig(
        servo_id=_integer(turntable.get("servo_id"), "turntable.servo_id", 1, 253),
        duration_ms=_integer(turntable_tuning.get("duration_ms"), "turntable.duration_ms", 1, 30000),
        stations=_stations(turntable.get("stations")),
        travel_degrees=_positive_number(
            turntable.get("travel_degrees"), "turntable.travel_degrees"
        ),
        position_tolerance=_integer(
            turntable.get("position_tolerance_units"),
            "turntable.position_tolerance_units",
            0,
            POSITION_UNITS,
        ),
        settle_seconds=_non_negative_number(
            turntable_tuning.get("settle_seconds", 0.2), "turntable.settle_seconds"
        ),
    )
    _require_servo_id_not_shared_with_gripper(hardware, config.servo_id)
    return config


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取舵机配置：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"舵机配置不是合法 JSON：{path}") from error
    return _mapping(document, str(path))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 JSON 对象")
    return value


def _stations(value: object) -> Mapping[str, int]:
    """Parse and validate the ``turntable.stations`` mapping."""
    if not isinstance(value, dict) or len(value) < MIN_TURNTABLE_STATIONS:
        raise ValueError(
            f"turntable.stations 至少要有 {MIN_TURNTABLE_STATIONS} 个工位"
            "（工位名 → 0..1000 的位置）"
        )
    stations = {}
    for name, position in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("turntable.stations 的工位名必须是非空字符串")
        if type(position) is not int or not 0 <= position <= 1000:
            raise ValueError(f"turntable.stations.{name} 必须是 0..1000 的整数")
        stations[name] = position
    if HOME_STATION not in stations:
        raise ValueError(
            f"turntable.stations 必须有一个名为 {HOME_STATION!r} 的工位（回零位）"
        )
    if len(set(stations.values())) != len(stations):
        raise ValueError("turntable.stations 里两个工位不能停在同一个位置")
    return stations


def _require_servo_id_not_shared_with_gripper(document: Mapping[str, object], servo_id: int) -> None:
    """Reject an id that the gripper section already claims on this bus."""
    gripper = document.get("gripper")
    if isinstance(gripper, dict) and gripper.get("servo_id") == servo_id:
        raise ValueError(
            f"turntable.servo_id 和 gripper.servo_id 都是 {servo_id}；"
            "两个舵机挂在同一条总线上，ID 必须不同"
        )


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须是 {minimum}..{maximum} 的整数")
    return value


def _positive_number(value: object, name: str) -> float:
    number = _non_negative_number(value, name)
    if not number > 0:
        raise ValueError(f"{name} 必须大于 0")
    return number


def _non_negative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    if float(value) < 0:
        raise ValueError(f"{name} 不能为负")
    return float(value)

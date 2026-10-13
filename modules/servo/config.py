"""The devices' own settings on the shared bus servo.

The bus port, baud rate and the bus-wide servo id live in
``modules/servo/config.json`` too, but they are read through
:func:`modules.servo.drive.bus_servo.load_bus_config`: the port is a property of
the physical servo bus, not of any one device.  Only a device's own servo id,
travel positions and duration are configured here.

Two devices currently share that one bus, so the file has two sections:

``gripper``
    夹爪：张开/闭合两个位置。
``turntable``
    转盘：若干**物料放取工位**（``stations``），其中一个必须叫 ``home``——那就是"回零"
    落在的位置。

两个舵机只靠 ID 区分，所以 :func:`load_turntable_config` 会拒绝和夹爪用同一个
``servo_id``：那个错误发出去之后，转盘指令会去开合夹爪（或反过来），现场看到的只是
"动作不对"，很难倒查到配置。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


#: 模块自己的配置和加载器放在一起，因此与仓库深度、当前目录都无关。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass(frozen=True)
class GripperConfig:
    """The gripper servo's id, open/closed pulse positions and travel time."""

    servo_id: int
    open_position: int
    closed_position: int
    duration_ms: int


#: 转盘"回零"落在哪个工位上：``turntable.stations`` 里名为 ``home`` 的那一项。
#: ``go_home()`` 就是走到它；没有它，"回零"就没有确定的目标。
HOME_STATION = "home"

#: 转盘至少要有"回零位"和另一个工位，否则分度没有意义。
MIN_TURNTABLE_STATIONS = 2


@dataclass(frozen=True)
class TurntableConfig:
    """转盘舵机的地址、各工位位置与转动耗时。

    ``stations`` 是**工位名 → 舵机位置**（0..1000）。名字由配置决定，代码不写死它，
    所以重新标定后只改 ``modules/servo/config.json``，调用方继续用
    ``go_to_station("<名字>")``，不用跟着改代码。
    """

    servo_id: int
    duration_ms: int
    stations: Mapping[str, int]

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


def load_gripper_config(config_path: Path = DEFAULT_CONFIG_PATH) -> GripperConfig:
    """Read the gripper section of the servo configuration file."""
    document = _load_document(config_path)
    gripper = document.get("gripper")
    if not isinstance(gripper, dict):
        raise ValueError("gripper 必须是 JSON 对象")

    def integer(name: str, minimum: int, maximum: int) -> int:
        value = gripper.get(name)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"gripper.{name} 必须是 {minimum}..{maximum} 的整数")
        return value

    config = GripperConfig(
        servo_id=integer("servo_id", 1, 253),
        open_position=integer("open_position", 0, 1000),
        closed_position=integer("closed_position", 0, 1000),
        duration_ms=integer("duration_ms", 1, 30000),
    )
    if config.open_position == config.closed_position:
        raise ValueError("open_position 和 closed_position 不能相同")
    return config


def load_turntable_config(config_path: Path = DEFAULT_CONFIG_PATH) -> TurntableConfig:
    """Read the turntable section of the servo configuration file.

    读完之后还会拿 ``gripper.servo_id`` 对一遍（如果那份配置里写了夹爪段）：
    同一条总线上的两个舵机共用一个 ID 是**静默的**错配，不会报通信错误。
    """
    document = _load_document(config_path)
    turntable = document.get("turntable")
    if not isinstance(turntable, dict):
        raise ValueError("turntable 必须是 JSON 对象")

    def integer(name: str, minimum: int, maximum: int) -> int:
        value = turntable.get(name)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"turntable.{name} 必须是 {minimum}..{maximum} 的整数")
        return value

    config = TurntableConfig(
        servo_id=integer("servo_id", 1, 253),
        duration_ms=integer("duration_ms", 1, 30000),
        stations=_stations(turntable.get("stations")),
    )
    _require_servo_id_not_shared_with_gripper(document, config.servo_id)
    return config


def _load_document(config_path: Path) -> dict:
    """Read the one JSON file every device on this bus shares."""
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取舵机配置：{config_path}") from error
    if not isinstance(document, dict):
        raise ValueError("舵机配置必须是 JSON 对象")
    return document


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


def _require_servo_id_not_shared_with_gripper(document: dict, servo_id: int) -> None:
    """Reject an id that the gripper section already claims on this bus."""
    gripper = document.get("gripper")
    if isinstance(gripper, dict) and gripper.get("servo_id") == servo_id:
        raise ValueError(
            f"turntable.servo_id 和 gripper.servo_id 都是 {servo_id}；"
            "两个舵机挂在同一条总线上，ID 必须不同"
        )

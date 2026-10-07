"""The gripper's own settings on the shared bus servo.

The bus port, baud rate and the bus-wide servo id live in
``modules/servo/config.json`` too, but they are read through
:func:`modules.servo.drive.bus_servo.load_bus_config`: the port is a property of
the physical servo bus, not of the gripper.  Only the gripper's own servo id,
travel positions and duration are configured here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


#: 模块自己的配置和加载器放在一起，因此与仓库深度、当前目录都无关。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass(frozen=True)
class GripperConfig:
    """The gripper servo's id, open/closed pulse positions and travel time."""

    servo_id: int
    open_position: int
    closed_position: int
    duration_ms: int


def load_gripper_config(config_path: Path = DEFAULT_CONFIG_PATH) -> GripperConfig:
    """Read the gripper section of the servo configuration file."""
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取舵机配置：{config_path}") from error
    if not isinstance(document, dict):
        raise ValueError("舵机配置必须是 JSON 对象")
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

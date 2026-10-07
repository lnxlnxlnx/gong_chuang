"""Open and close the gripper through the 9600 baud multi-servo controller."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from motor_drivers.servo_controller import ServoController
from project_paths import config_path


DEFAULT_CONFIG_PATH = config_path("servo.json")


@dataclass(frozen=True)
class GripperConfig:
    port: str
    servo_id: int
    open_position: int
    closed_position: int
    duration_ms: int


def load_gripper_config(
    platform: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> GripperConfig:
    """Read the controller port and gripper settings from one servo JSON file."""
    platform = platform or ("windows" if os.name == "nt" else "jetson")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取舵机配置：{config_path}") from error
    if not isinstance(document, dict):
        raise ValueError("舵机配置必须是 JSON 对象")
    profile = document.get(platform)
    if not isinstance(profile, dict) or not isinstance(profile.get("port"), str) or not profile["port"].strip():
        raise ValueError(f"请先填写 {config_path} 中的 {platform}.port")
    gripper = document.get("gripper")
    if not isinstance(gripper, dict):
        raise ValueError("gripper 必须是 JSON 对象")

    def integer(name: str, minimum: int, maximum: int) -> int:
        value = gripper.get(name)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"gripper.{name} 必须是 {minimum}..{maximum} 的整数")
        return value

    config = GripperConfig(
        port=profile["port"].strip(),
        servo_id=integer("servo_id", 1, 253),
        open_position=integer("open_position", 0, 1000),
        closed_position=integer("closed_position", 0, 1000),
        duration_ms=integer("duration_ms", 1, 30000),
    )
    if config.open_position == config.closed_position:
        raise ValueError("open_position 和 closed_position 不能相同")
    return config


class Gripper:
    """Named gripper actions sent to one servo on the multi-servo controller."""

    def __init__(self, config: GripperConfig | None = None, *, controller: ServoController | None = None) -> None:
        self.config = config if config is not None else load_gripper_config()
        self._controller = controller if controller is not None else ServoController(self.config.port)

    def initialize(self) -> "Gripper":
        self._controller.initialize()
        return self

    def open_gripper(self) -> None:
        self._controller.set_position(
            self.config.servo_id, position=self.config.open_position, duration_ms=self.config.duration_ms
        )

    def close_gripper(self) -> None:
        self._controller.set_position(
            self.config.servo_id, position=self.config.closed_position, duration_ms=self.config.duration_ms
        )

    def read_position(self) -> int:
        return self._controller.read_position(self.config.servo_id)

    def close(self) -> None:
        """Close the serial port; this does not move the gripper."""
        self._controller.close()

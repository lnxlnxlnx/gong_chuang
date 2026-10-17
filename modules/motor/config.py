"""Load the shared Emm42 stepper-bus settings.

The chassis and the arm hang off the same stepper bus, so its port belongs here
rather than in either module's own configuration; ``modules.chassis`` and
``modules.arm`` both read it.  The bus baud rate is a fixed protocol constant
(:data:`modules.motor.drive.emm42.Emm42Driver.BAUDRATE`), so only the
per-platform port is configurable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


#: 模块自己的配置和加载器放在一起，因此与仓库深度、当前目录都无关。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass(frozen=True)
class MotorBusConfig:
    """The serial port every Emm42 device on the stepper bus shares."""

    port: str


def load_bus_config(
    platform: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> MotorBusConfig:
    """Read the stepper-bus port for one platform; reject an unfilled profile."""
    platform = platform or ("windows" if os.name == "nt" else "jetson")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取步进总线配置：{config_path}") from error
    if not isinstance(document, dict):
        raise ValueError("步进总线配置必须是 JSON 对象")
    profile = document.get(platform)
    if (
        not isinstance(profile, dict)
        or not isinstance(profile.get("port"), str)
        or not profile["port"].strip()
    ):
        raise ValueError(f"请先填写 {config_path} 中的 {platform}.port")
    return MotorBusConfig(port=profile["port"].strip())

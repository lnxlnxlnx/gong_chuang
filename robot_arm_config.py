"""Load the editable hardware settings used by the robot-arm controller."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from motor_drivers.emm42 import Emm42Driver
from project_paths import config_path


DEFAULT_CONFIG_PATH = config_path("robot_arm.json")
AXIS_NAMES = ("base", "lift", "extension")


@dataclass(frozen=True)
class AxisConfig:
    """One physical axis address and, when applicable, its output reduction."""

    motor_id: int
    reduction_ratio: Optional[int] = None


@dataclass(frozen=True)
class ArmConfig:
    """Validated arm settings shared by the controller and hardware-test CLI."""

    pulses_per_revolution: int
    rpm: int
    acceleration: int
    default_pulses: int
    max_pulses: int
    axes: Mapping[str, AxisConfig]

    def motor_id(self, axis_name: str) -> int:
        """Return the configured Emm42 address for a named axis."""
        return self.axes[axis_name].motor_id

    @property
    def base_reduction(self) -> int:
        """Return the confirmed motor-turns/output-turn ratio of the base axis."""
        reduction_ratio = self.axes["base"].reduction_ratio
        if reduction_ratio is None:
            raise ValueError("axes.base.reduction_ratio is required")
        return reduction_ratio


def load_arm_config(config_path: Path = DEFAULT_CONFIG_PATH) -> ArmConfig:
    """Load one JSON file and reject settings outside the supported protocol range."""
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read configuration: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration: {config_path}") from error

    root = _mapping(document, "root")
    motion = _mapping(root.get("motion"), "motion")
    axes_document = _mapping(root.get("axes"), "axes")
    if set(axes_document) != set(AXIS_NAMES):
        raise ValueError(f"axes must define exactly: {', '.join(AXIS_NAMES)}")

    config = ArmConfig(
        pulses_per_revolution=_integer(
            motion.get("pulses_per_revolution"),
            "motion.pulses_per_revolution",
            minimum=1,
            maximum=0xFFFFFFFF,
        ),
        rpm=_integer(motion.get("rpm"), "motion.rpm", minimum=0, maximum=Emm42Driver.MAX_RPM),
        acceleration=_integer(
            motion.get("acceleration"), "motion.acceleration", minimum=0, maximum=255),
        default_pulses=_integer(
            motion.get("default_pulses"),
            "motion.default_pulses",
            minimum=1,
            maximum=0xFFFFFFFF,
        ),
        max_pulses=_integer(
            motion.get("max_pulses"),
            "motion.max_pulses",
            minimum=1,
            maximum=0xFFFFFFFF,
        ),
        axes={
            axis_name: _axis_config(
                _mapping(axes_document[axis_name], f"axes.{axis_name}"),
                axis_name,
            )
            for axis_name in AXIS_NAMES
        },
    )
    if config.default_pulses > config.max_pulses:
        raise ValueError("motion.default_pulses must not exceed motion.max_pulses")
    if len({axis.motor_id for axis in config.axes.values()}) != len(AXIS_NAMES):
        raise ValueError("axes motor_id values must be unique")
    return config


def _axis_config(document: Mapping[str, object], axis_name: str) -> AxisConfig:
    reduction_ratio = document.get("reduction_ratio")
    if axis_name == "base":
        reduction_ratio = _integer(
            reduction_ratio,
            "axes.base.reduction_ratio",
            minimum=1,
            maximum=0xFFFFFFFF,
        )
    elif reduction_ratio is not None:
        raise ValueError(f"axes.{axis_name}.reduction_ratio is not supported")
    return AxisConfig(
        motor_id=_integer(document.get("motor_id"), f"axes.{axis_name}.motor_id", minimum=1, maximum=255),
        reduction_ratio=reduction_ratio,
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value

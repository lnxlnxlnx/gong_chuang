"""Load the editable hardware settings used by the robot-arm controller.

这里既读 ``motion``（速度、加速度、脉冲上限）和 ``axes``（驱动器地址、传动常数），
也读**标定结果**：每轴的软行程范围和抓取前的安全位。

``limits`` 与 ``safe_pose`` 都是可选的。缺失等于"还没标定"，此时
:func:`modules.arm.drive.kinematics.check_reachable` 不做检查——**没标定就等于没有软件
行程保护**，调用方必须清楚这一点，所以这里用 ``None`` 如实表达，而不是塞一个假的默认范围。

``safe_pose`` 用的是绝对圆柱坐标，看起来和"只有相对位置模式"矛盾，其实不矛盾：
抓取流程每一轮都用相机重新测量靶标，得到的就是绝对位姿，与上一轮无关。所以只要标定
文件不变，同一个物理姿态永远对应同一组 ``(θ, r, h)``，绝对安全位是可复现的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from modules.arm.drive.kinematics import CylindricalPose, TravelLimit
from modules.motor.drive.emm42 import Emm42Driver

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
AXIS_NAMES = ("base", "lift", "extension")

#: ``limits`` / ``safe_pose`` 里三个量的名字，对应 ``CylindricalPose`` 的字段。
POSE_FIELDS = ("base_degrees", "radial_mm", "height_mm")


@dataclass(frozen=True)
class AxisConfig:
    """One physical axis: its driver address and its transmission constant.

    - ``reduction_ratio``（仅底座）：电机转几圈输出轴转一圈；
    - ``mm_per_motor_rev``（仅升降/伸缩）：电机转一圈走多少毫米，即丝杠导程与减速比之商。
    """

    motor_id: int
    reduction_ratio: Optional[int] = None
    mm_per_motor_rev: Optional[float] = None


@dataclass(frozen=True)
class ArmConfig:
    """Validated arm settings shared by the controller and hardware-test CLI."""

    pulses_per_revolution: int
    rpm: int
    acceleration: int
    default_pulses: int
    max_pulses: int
    axes: Mapping[str, AxisConfig]
    base_limit: Optional[TravelLimit] = None
    radial_limit: Optional[TravelLimit] = None
    height_limit: Optional[TravelLimit] = None
    safe_pose: Optional[CylindricalPose] = None

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

    def mm_per_motor_rev(self, axis_name: str) -> Optional[float]:
        """Return the linear transmission constant, or ``None`` when uncalibrated."""
        return self.axes[axis_name].mm_per_motor_rev

    def require_mm_per_motor_rev(self, axis_name: str) -> float:
        """Return the linear constant or explain which experiment calibrates it."""
        value = self.axes[axis_name].mm_per_motor_rev
        if value is None:
            raise ValueError(
                f"axes.{axis_name}.mm_per_motor_rev 还没标定；"
                "先用实验 B/C 测出电机一圈走多少毫米（见 docs/抓取坐标链路.md）"
            )
        return value

    def require_safe_pose(self) -> CylindricalPose:
        """Return the configured safe pose or explain what is missing."""
        if self.safe_pose is None:
            raise ValueError(
                "safe_pose 还没配置；抓取流程必须先有一个已标定的安全位"
                "（见 docs/抓取坐标链路.md）"
            )
        return self.safe_pose


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

    limits = root.get("limits")
    limits = {} if limits is None else _mapping(limits, "limits")

    config = ArmConfig(
        pulses_per_revolution=_integer(
            motion.get("pulses_per_revolution"),
            "modules.motor.drive.motion.pulses_per_revolution",
            minimum=1,
            maximum=0xFFFFFFFF,
        ),
        rpm=_integer(motion.get("rpm"), "modules.motor.drive.motion.rpm", minimum=0, maximum=Emm42Driver.MAX_RPM),
        acceleration=_integer(
            motion.get("acceleration"), "modules.motor.drive.motion.acceleration", minimum=0, maximum=255),
        default_pulses=_integer(
            motion.get("default_pulses"),
            "modules.motor.drive.motion.default_pulses",
            minimum=1,
            maximum=0xFFFFFFFF,
        ),
        max_pulses=_integer(
            motion.get("max_pulses"),
            "modules.motor.drive.motion.max_pulses",
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
        base_limit=_optional_limit(limits, "base_degrees"),
        radial_limit=_optional_limit(limits, "radial_mm"),
        height_limit=_optional_limit(limits, "height_mm"),
        safe_pose=_optional_safe_pose(root.get("safe_pose")),
    )
    if config.default_pulses > config.max_pulses:
        raise ValueError("modules.motor.drive.motion.default_pulses must not exceed modules.motor.drive.motion.max_pulses")
    if len({axis.motor_id for axis in config.axes.values()}) != len(AXIS_NAMES):
        raise ValueError("axes motor_id values must be unique")
    return config


def _axis_config(document: Mapping[str, object], axis_name: str) -> AxisConfig:
    reduction_ratio = document.get("reduction_ratio")
    mm_per_motor_rev = document.get("mm_per_motor_rev")
    if axis_name == "base":
        reduction_ratio = _integer(
            reduction_ratio,
            "axes.base.reduction_ratio",
            minimum=1,
            maximum=0xFFFFFFFF,
        )
        if mm_per_motor_rev is not None:
            raise ValueError("axes.base.mm_per_motor_rev is not supported")
    else:
        if reduction_ratio is not None:
            raise ValueError(f"axes.{axis_name}.reduction_ratio is not supported")
        mm_per_motor_rev = _optional_positive_number(
            mm_per_motor_rev, f"axes.{axis_name}.mm_per_motor_rev"
        )
    return AxisConfig(
        motor_id=_integer(document.get("motor_id"), f"axes.{axis_name}.motor_id", minimum=1, maximum=255),
        reduction_ratio=reduction_ratio,
        mm_per_motor_rev=mm_per_motor_rev,
    )


def _optional_limit(document: Mapping[str, object], field: str) -> Optional[TravelLimit]:
    value = document.get(field)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"limits.{field} must be a two-element [min, max] list or null")
    return TravelLimit(
        minimum=_number(value[0], f"limits.{field}[0]"),
        maximum=_number(value[1], f"limits.{field}[1]"),
    )


def _optional_safe_pose(value: object) -> Optional[CylindricalPose]:
    if value is None:
        return None
    document = _mapping(value, "safe_pose")
    fields = {field: document.get(field) for field in POSE_FIELDS}
    missing = [field for field, item in fields.items() if item is None]
    if len(missing) == len(POSE_FIELDS):
        return None
    if missing:
        raise ValueError(
            "safe_pose 的 " + " / ".join(POSE_FIELDS) + " 必须同时给出或同时为 null"
        )
    return CylindricalPose(
        base_degrees=_number(fields["base_degrees"], "safe_pose.base_degrees"),
        radial_mm=_number(fields["radial_mm"], "safe_pose.radial_mm"),
        height_mm=_number(fields["height_mm"], "safe_pose.height_mm"),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_positive_number(value: object, name: str) -> Optional[float]:
    if value is None:
        return None
    number = _number(value, name)
    if not number > 0:
        raise ValueError(f"{name} must be positive when given")
    return number

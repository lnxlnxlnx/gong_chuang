"""抓取循环里可调的余量与容差。

这里**只放策略参数**，不放机器常数。机器常数各有归属：传动比和行程在
``modules/arm/config.json``，相机标定在 ``modules/vision/calibration.json``，
靶标与物料的尺寸颜色在 ``modules/vision/config.json``。这样换一台机器只要重标，
不用改这个文件。

所有长度单位都是毫米，角度是度。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from modules.motor.drive.emm42 import Emm42Driver

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass(frozen=True)
class CorrectionSettings:
    """视觉闭环微调：把爪上靶标对到物料正上方。"""

    #: 最多迭代几次。每次都要重新取一帧并测量，所以次数直接决定耗时。
    #: 必须 >= 1：这一层存在的意义就是闭环，想开环直抓就把 ``tolerance_mm`` 放宽即可。
    iterations: int = 3
    #: 平面误差小于这个值就认为对上了（毫米）。
    tolerance_mm: float = 3.0
    #: 每次只走误差的这个比例，留出过冲余量；(0, 1]。
    gain: float = 0.8


@dataclass(frozen=True)
class GraspConfig:
    """一次抓取的余量、夹持高度与闭环参数。"""

    #: 接近与离开时相对夹持高度额外抬起的量（毫米）。
    clearance_mm: float = 25.0
    #: 侧向夹取夹在物料的哪个高度：0 贴台面、0.5 半高、1 顶面。
    grip_ratio: float = 0.5
    #: 靶标相对实际夹持点的竖直偏移（毫米），靶标在夹持点上方时为正。
    marker_to_grasp_mm: float = 0.0
    #: 回零后核对 ``safe_pose`` 的容差（毫米）。单圈回零落到错误的圈上时驱动器不会报错，
    #: 只有这一次相机校验能发现，所以容差不能放得太松。
    home_tolerance_mm: float = 5.0
    #: 抓取流程统一使用的转速；``None`` 表示沿用 ``modules/arm/config.json`` 的值。
    motion_rpm: Optional[int] = None
    correction: CorrectionSettings = field(default_factory=CorrectionSettings)


def load_grasp_config(config_path: Path = DEFAULT_CONFIG_PATH) -> GraspConfig:
    """Read and validate the grasp strategy file."""
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read grasp configuration: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in grasp configuration: {config_path}") from error

    root = _mapping(document, "root")
    correction_document = root.get("correction")
    correction = (
        CorrectionSettings()
        if correction_document is None
        else _correction(_mapping(correction_document, "correction"))
    )
    return GraspConfig(
        clearance_mm=_non_negative(root.get("clearance_mm", 25.0), "clearance_mm"),
        grip_ratio=_ratio(root.get("grip_ratio", 0.5), "grip_ratio"),
        marker_to_grasp_mm=_number(root.get("marker_to_grasp_mm", 0.0), "marker_to_grasp_mm"),
        home_tolerance_mm=_positive(root.get("home_tolerance_mm", 5.0), "home_tolerance_mm"),
        motion_rpm=_optional_rpm(root.get("motion_rpm")),
        correction=correction,
    )


def _correction(document: Mapping[str, object]) -> CorrectionSettings:
    gain = _number(document.get("gain", 0.8), "correction.gain")
    if not 0.0 < gain <= 1.0:
        raise ValueError("correction.gain must be in (0, 1]")
    iterations = document.get("iterations", 3)
    if type(iterations) is not int or iterations < 1:
        raise ValueError("correction.iterations must be a positive integer")
    return CorrectionSettings(
        iterations=iterations,
        tolerance_mm=_positive(document.get("tolerance_mm", 3.0), "correction.tolerance_mm"),
        gain=gain,
    )


def _optional_rpm(value: object) -> Optional[int]:
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= Emm42Driver.MAX_RPM:
        raise ValueError(f"motion_rpm must be an integer in 1..{Emm42Driver.MAX_RPM} or null")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _non_negative(value: object, name: str) -> float:
    number = _number(value, name)
    if number < 0:
        raise ValueError(f"{name} must not be negative")
    return number


def _positive(value: object, name: str) -> float:
    number = _number(value, name)
    if not number > 0:
        raise ValueError(f"{name} must be positive")
    return number


def _ratio(value: object, name: str) -> float:
    number = _number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in 0..1")
    return number

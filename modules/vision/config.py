"""Vision settings: hand-edited targets plus machine-calibrated numbers.

两个文件、两种性质，刻意分开：

- ``config.json`` —— **手改**。摄像头编号，以及抓取用的目标尺寸与颜色。
- ``calibration.json`` —— **标定产物**。像素焦距、镜头到台面的深度、平面标定结果，
  由标定脚本写入（见 ``modules/grasp/docs/抓取坐标链路.md``）。

分开的理由是它们的生命周期不同：换摄像头要改 ``config.json``，重新标定只动
``calibration.json``，两者不会互相覆盖，git 里也一眼能看出"这台机器标定过了"。

``config.json`` 里的 ``targets`` 块整体可选。缺失时物料按默认 30 mm 圆面处理，
靶标视为未配置——此时抓取流程必须在打开摄像头之前拒绝启动，而不是拿一个假直径去算距离。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

from modules.vision.drive.calibration import PlaneCalibration

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"

#: 物料顶部内侧圆面的实际直径；与 ``modules/vision/drive/material_locate`` 的默认值一致。
DEFAULT_MATERIAL_DIAMETER_MM = 30.0

#: 默认颜色匹配容差（BGR 欧氏距离）。同一盏灯下同一颜色的实测散布通常远小于这个值。
DEFAULT_COLOR_TOLERANCE = 60.0

#: 标定脚本支持的平台名，与 ``config.json`` 顶层键一致。
PLATFORMS = ("windows", "jetson")


@dataclass(frozen=True)
class TargetSettings:
    """抓取流程要区分的两类圆面目标。"""

    #: 物料顶部内圆的真实直径（毫米）。
    material_diameter_mm: float = DEFAULT_MATERIAL_DIAMETER_MM
    #: 夹爪上靶标的真实直径（毫米）；``None`` 表示还没装/没量。
    marker_diameter_mm: Optional[float] = None
    #: 物料内圆的参考颜色（BGR）；``None`` 表示不按颜色区分，全部圆面都当物料。
    material_bgr: Optional[Tuple[float, float, float]] = None
    #: 靶标的参考颜色（BGR）；``None`` 表示不按颜色挑靶标。
    marker_bgr: Optional[Tuple[float, float, float]] = None
    #: 颜色匹配容差。
    color_tolerance: float = DEFAULT_COLOR_TOLERANCE

    @property
    def can_split_by_color(self) -> bool:
        """两类目标都有参考颜色时才可能在同一帧里把它们分开。"""
        return self.material_bgr is not None and self.marker_bgr is not None


@dataclass(frozen=True)
class VisionConfig:
    """Which camera to open, what the two targets look like, and how it was calibrated."""

    camera_ids: Mapping[str, int] = field(default_factory=dict)
    targets: TargetSettings = field(default_factory=TargetSettings)
    #: 像素焦距；``None`` 表示沿用粗标定或调用方自备值。
    focal_px: Optional[float] = None
    #: 镜头到台面的深度（毫米）；换算离台面高度必需。
    table_depth_mm: Optional[float] = None
    #: 平面标定结果；``None`` 表示还没做实验 A。
    plane: Optional[PlaneCalibration] = None

    def camera_id(self, platform: str) -> int:
        """Return the camera index for one platform or explain what is missing."""
        camera_id = self.camera_ids.get(platform)
        if camera_id is None:
            raise ValueError(f"config.json 缺少 {platform}.camera_id")
        return camera_id

    def require_plane(self) -> PlaneCalibration:
        """Return the plane calibration or point at the experiment that produces it."""
        if self.plane is None:
            raise ValueError(
                "还没有平面标定结果；先跑实验 A 写入 calibration.json"
                "（见 modules/grasp/docs/抓取坐标链路.md）"
            )
        return self.plane

    def require_table_depth(self) -> float:
        """Return the lens-to-table depth or explain how to measure it."""
        if self.table_depth_mm is None:
            raise ValueError(
                "calibration.json 里还没有 table_depth_mm；"
                "把已知直径的圆片平放在台面上量一次即可（见 modules/grasp/docs/抓取坐标链路.md）"
            )
        return self.table_depth_mm

    def require_marker_diameter(self) -> float:
        """Return the gripper-marker diameter or explain what is missing."""
        if self.targets.marker_diameter_mm is None:
            raise ValueError(
                "config.json 的 targets.marker_diameter_mm 还没填；"
                "相机就是靠这个已知直径反推爪上靶标的三维位置的"
            )
        return self.targets.marker_diameter_mm


def load_vision_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
) -> VisionConfig:
    """Load both vision files; a missing calibration file means "uncalibrated"."""
    document = _read_json(config_path, "vision configuration")
    root = _mapping(document, "root")

    calibration_document = (
        _read_json(calibration_path, "vision calibration") if calibration_path.exists() else {}
    )
    calibration_root = _mapping(calibration_document, "calibration root")

    return VisionConfig(
        camera_ids=_camera_ids(root),
        targets=_targets(root.get("targets")),
        focal_px=_optional_positive_number(
            calibration_root.get("focal_px"), "calibration.focal_px"
        ),
        table_depth_mm=_optional_positive_number(
            calibration_root.get("table_depth_mm"), "calibration.table_depth_mm"
        ),
        plane=_plane_calibration(calibration_root.get("plane")),
    )


def save_vision_calibration(
    *,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    plane: Optional[PlaneCalibration] = None,
    focal_px: Optional[float] = None,
    table_depth_mm: Optional[float] = None,
) -> None:
    """Write ``calibration.json``, keeping any field the caller does not pass.

    只改传入的字段：标完焦距不会把已有的平面标定抹掉，反之亦然。
    """
    existing = (
        _mapping(_read_json(calibration_path, "vision calibration"), "calibration root")
        if calibration_path.exists()
        else {}
    )
    document = dict(existing)
    if focal_px is not None:
        document["focal_px"] = float(focal_px)
    if table_depth_mm is not None:
        document["table_depth_mm"] = float(table_depth_mm)
    if plane is not None:
        document["plane"] = {
            "center_x_mm": plane.center_x_mm,
            "center_y_mm": plane.center_y_mm,
            "rotation_degrees": plane.rotation_degrees,
            "rms_mm": plane.rms_mm,
            "samples": plane.samples,
        }
    calibration_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {name}: {path}") from error


def _camera_ids(root: Mapping[str, object]) -> Mapping[str, int]:
    found = {}
    for platform in PLATFORMS:
        profile = root.get(platform)
        if profile is None:
            continue
        camera_id = _mapping(profile, platform).get("camera_id")
        if camera_id is None:
            continue
        if type(camera_id) is not int or camera_id < 0:
            raise ValueError(f"{platform}.camera_id must be a non-negative integer")
        found[platform] = camera_id
    return found


def _targets(value: object) -> TargetSettings:
    if value is None:
        return TargetSettings()
    document = _mapping(value, "targets")
    material_diameter = document.get("material_diameter_mm")
    return TargetSettings(
        material_diameter_mm=(
            DEFAULT_MATERIAL_DIAMETER_MM
            if material_diameter is None
            else _positive_number(material_diameter, "targets.material_diameter_mm")
        ),
        marker_diameter_mm=_optional_positive_number(
            document.get("marker_diameter_mm"), "targets.marker_diameter_mm"
        ),
        material_bgr=_optional_bgr(document.get("material_bgr"), "targets.material_bgr"),
        marker_bgr=_optional_bgr(document.get("marker_bgr"), "targets.marker_bgr"),
        color_tolerance=_positive_number(
            document.get("color_tolerance", DEFAULT_COLOR_TOLERANCE),
            "targets.color_tolerance",
        ),
    )


def _plane_calibration(value: object) -> Optional[PlaneCalibration]:
    if value is None:
        return None
    document = _mapping(value, "calibration.plane")
    fields = ("center_x_mm", "center_y_mm", "rotation_degrees")
    if all(document.get(field) is None for field in fields):
        return None
    return PlaneCalibration(
        center_x_mm=_number(document.get("center_x_mm"), "calibration.plane.center_x_mm"),
        center_y_mm=_number(document.get("center_y_mm"), "calibration.plane.center_y_mm"),
        rotation_degrees=_number(
            document.get("rotation_degrees"), "calibration.plane.rotation_degrees"
        ),
        rms_mm=_number(document.get("rms_mm", 0.0), "calibration.plane.rms_mm"),
        samples=int(document.get("samples", 0)),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _positive_number(value: object, name: str) -> float:
    number = _number(value, name)
    if not number > 0:
        raise ValueError(f"{name} must be positive")
    return number


def _optional_positive_number(value: object, name: str) -> Optional[float]:
    if value is None:
        return None
    return _positive_number(value, name)


def _optional_bgr(value: object, name: str) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element [B, G, R] list or null")
    return tuple(_number(channel, f"{name}[{index}]") for index, channel in enumerate(value))  # type: ignore[return-value]

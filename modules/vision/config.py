"""Vision settings: hand-edited targets plus machine-calibrated numbers.

两个文件、两种性质，刻意分开：

- ``config.json`` —— **手改**。摄像头编号、**用哪个模型**，以及抓取用的目标尺寸与颜色。
- ``calibration.json`` —— **标定产物**。像素焦距、镜头到台面的深度、平面标定结果，
  由标定脚本写入（见 ``modules/grasp/docs/抓取坐标链路.md``）。

分开的理由是它们的生命周期不同：换摄像头要改 ``config.json``，重新标定只动
``calibration.json``，两者不会互相覆盖，git 里也一眼能看出"这台机器标定过了"。

``config.json`` 里的 ``targets`` 块整体可选。缺失时物料按默认 30 mm 圆面处理，
靶标视为未配置——此时抓取流程必须在打开摄像头之前拒绝启动，而不是拿一个假直径去算距离。

``config.json`` 里的 ``model`` 块决定"加载哪个模型文件"（见 :class:`ModelSettings`）。
它是**唯一**换模型的地方：文件路径、类别表、输入边长、阈值都在这里，代码里不再写死
``models/best.onnx`` / ``best.engine``。相对路径按项目根解析（不是当前工作目录），
绝对路径原样使用。整块缺失时全部走内置默认值，行为与加这个块之前一致。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

from project_paths import PROJECT_ROOT
from modules.vision.drive.calibration import PlaneCalibration
from modules.vision.drive.pinhole import normalize_depth_range
from modules.vision.drive.yolov8 import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_CONF_THRESH,
    DEFAULT_INPUT_SIZE,
    DEFAULT_IOU_THRESH,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"

#: 物料顶部内侧圆面的实际直径；与 ``modules/vision/drive/material_locate`` 的默认值一致。
DEFAULT_MATERIAL_DIAMETER_MM = 30.0

#: 默认颜色匹配容差（BGR 欧氏距离）。同一盏灯下同一颜色的实测散布通常远小于这个值。
DEFAULT_COLOR_TOLERANCE = 60.0

#: 标定脚本支持的平台名，与 ``config.json`` 顶层键一致。
PLATFORMS = ("windows", "jetson")

#: 没配 ``model.onnx`` 时的 ONNX 路径（相对项目根）。
DEFAULT_ONNX_MODEL = "models/best.onnx"

#: 没配 ``model.engine`` 时的 TensorRT 引擎路径（相对项目根）。
DEFAULT_ENGINE_MODEL = "models/best.engine"

#: 推理后端名。``auto`` 按平台选：Windows 走 OpenCV ONNX，Jetson 走 TensorRT。
BACKEND_AUTO = "auto"
BACKEND_TENSORRT = "tensorrt"
BACKEND_ONNX = "onnx"
BACKENDS = (BACKEND_AUTO, BACKEND_TENSORRT, BACKEND_ONNX)


def current_platform() -> str:
    """本机对应的配置平台名：Windows 是 ``"windows"``，Jetson/Linux 是 ``"jetson"``。"""
    return "windows" if os.name == "nt" else "jetson"


def _project_file(value: str) -> Path:
    """配置里的模型路径：相对路径按项目根解析（不按当前工作目录）。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class ModelSettings:
    """用哪个模型文件、它有哪些类别、怎么解码——全部来自 ``config.json``。

    换模型的完整流程就是改这一个块，然后（Jetson 上）用
    :mod:`modules.vision.training.build_engine` 重新生成引擎。
    """

    #: ``"auto"`` / ``"tensorrt"`` / ``"onnx"``。
    backend: str = BACKEND_AUTO
    #: OpenCV ONNX 通道要加载的文件。
    onnx_path: Path = field(default_factory=lambda: _project_file(DEFAULT_ONNX_MODEL))
    #: TensorRT 通道要加载的文件；引擎与板子/GPU/TensorRT 版本绑定，只能在 Jetson 上生成。
    engine_path: Path = field(default_factory=lambda: _project_file(DEFAULT_ENGINE_MODEL))
    #: 类别名，顺序必须与训练时的类别索引一致（ONNX 的 ``names`` 元数据）。
    class_names: Tuple[str, ...] = tuple(DEFAULT_CLASS_NAMES)
    #: :attr:`class_names` 里算作物料的子集；没配时等于全部类别。
    materials: Tuple[str, ...] = tuple(DEFAULT_CLASS_NAMES)
    #: 模型输入边长（正方形）；``None`` 表示让代码去问模型自己。
    input_size: Optional[int] = None
    #: 解码阈值。
    conf_thresh: float = DEFAULT_CONF_THRESH
    iou_thresh: float = DEFAULT_IOU_THRESH
    #: 读取它的那份 config.json，只用于打印提示。
    source: Optional[Path] = None

    def resolve_backend(self, platform: Optional[str] = None) -> str:
        """本平台实际用哪个后端：显式配了就听配置的，``auto`` 才按平台猜。"""
        if self.backend != BACKEND_AUTO:
            return self.backend
        return BACKEND_ONNX if (platform or current_platform()) == "windows" else BACKEND_TENSORRT

    def path_for(self, backend: str) -> Path:
        """某个后端要加载的文件路径。"""
        return self.engine_path if backend == BACKEND_TENSORRT else self.onnx_path

    def model_path(self, platform: Optional[str] = None) -> Path:
        """本平台默认加载的模型文件路径。"""
        return self.path_for(self.resolve_backend(platform))


def load_model_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> ModelSettings:
    """读 ``config.json`` 的 ``model`` 块；整块缺失时返回内置默认值。"""
    config_path = Path(config_path)
    document = _read_json(config_path, "vision configuration") if config_path.exists() else {}
    root = _mapping(document, "root")
    value = root.get("model")
    if value is None:
        return ModelSettings(source=config_path)
    return _model_settings(_mapping(value, "model"), config_path)


def resolve_target_settings(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Tuple[TargetSettings, Optional[str]]:
    """读 ``config.json`` 的 ``targets`` 块，**读不出来也不抛异常**。

    返回 ``(参数, 原因)``：成功时原因是 ``None``；文件坏了、字段写错时返回内置
    默认值 + 一句中文原因，由调用方决定怎么提示。

    为什么不做成"读失败就报错退出"：``targets`` 是**把识别调得更准**的那部分
    （物料颜色、工作距离先验），不是"没有就跑不动"的那部分。预览入口在用户面前
    开着，为了一个写错的字段直接退出，比按默认值跑一遍、把原因打在终端上更糟。
    真正需要这些参数的抓取流程会在自己的入口显式要求它们（``VisionConfig``
    的 ``require_*``）。
    """
    try:
        return load_vision_config(Path(config_path)).targets, None
    except ValueError as error:
        return TargetSettings(), str(error)


def _model_settings(document: Mapping[str, object], source: Path) -> ModelSettings:
    backend = document.get("backend", BACKEND_AUTO)
    if backend not in BACKENDS:
        raise ValueError(f"model.backend 必须是 {'/'.join(BACKENDS)} 之一，实际是 {backend!r}")
    class_names = _class_names(document.get("class_names"))
    materials = _materials(document.get("materials"), class_names)
    return ModelSettings(
        backend=str(backend),
        onnx_path=_model_file(document.get("onnx"), DEFAULT_ONNX_MODEL, "model.onnx"),
        engine_path=_model_file(document.get("engine"), DEFAULT_ENGINE_MODEL, "model.engine"),
        class_names=class_names,
        materials=materials,
        input_size=_optional_input_size(document.get("input_size")),
        conf_thresh=_probability(
            document.get("conf_thresh"), DEFAULT_CONF_THRESH, "model.conf_thresh"
        ),
        iou_thresh=_probability(document.get("iou_thresh"), DEFAULT_IOU_THRESH, "model.iou_thresh"),
        source=source,
    )


def _model_file(value: object, fallback: str, name: str) -> Path:
    if value is None:
        return _project_file(fallback)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串路径，或 null 表示用默认值 {fallback}")
    return _project_file(value.strip())


def _class_names(value: object) -> Tuple[str, ...]:
    """读类别表；``null`` 表示沿用内置表（见 ``yolov8.DEFAULT_CLASS_NAMES``）。"""
    if value is None:
        return tuple(DEFAULT_CLASS_NAMES)
    return _name_list(value, "model.class_names")


def _materials(value: object, class_names: Tuple[str, ...]) -> Tuple[str, ...]:
    """读"哪些类别算物料"；``null`` 表示全部类别都算。"""
    if value is None:
        return class_names
    names = _name_list(value, "model.materials")
    unknown = [name for name in names if name not in set(class_names)]
    if unknown:
        # 写错一个名字就会让那一类永远被判成"非物料"，静默漏检；这里直接拒绝启动。
        raise ValueError(
            f"model.materials 里的 {', '.join(unknown)} 不在 model.class_names "
            f"（{', '.join(class_names)}）里"
        )
    return names


def _name_list(value: object, name: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} 必须是非空的类别名列表，或 null")
    names = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}[{index}] 必须是非空字符串")
        names.append(item.strip())
    if len(set(names)) != len(names):
        raise ValueError(f"{name} 里有重复的类别名")
    return tuple(names)


def _optional_input_size(value: object) -> Optional[int]:
    """读输入边长；``null`` 表示让代码去问模型/引擎自己（问不出来才用兜底值）。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("model.input_size 必须是正整数（方形输入的边长），或 null")
    return int(value)


def _probability(value: object, fallback: float, name: str) -> float:
    if value is None:
        return float(fallback)
    number = _number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} 要在 0–1 之间")
    return number


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
    #: 相机到物料顶面的深度范围 ``(近, 远)`` 毫米；``None`` 表示不设先验。
    #: 给了它，物料定位的霍夫半径窗口就按针孔模型算，夹爪弧线一类的假圆会被排除。
    working_distance_mm: Optional[Tuple[float, float]] = None
    #: 物料调色板 ``{类别名: (B, G, R)}``。换一块料只在这里加一行，不用改代码；
    #: 给了它，物料定位就走颜色块（夹爪、台面、线材进不了掩膜）。空表示不按颜色区分。
    material_palette: Mapping[str, Tuple[float, float, float]] = field(default_factory=dict)

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
        working_distance_mm=_optional_depth_range(
            document.get("working_distance_mm"), "targets.working_distance_mm"
        ),
        material_palette=_material_palette(
            document.get("material_palette"), "targets.material_palette"
        ),
    )


def _material_palette(value: object, name: str) -> Mapping[str, Tuple[float, float, float]]:
    """读 ``{"green": [165, 252, 114], ...}``；空对象表示不按颜色区分。"""
    if value is None:
        return {}
    document = _mapping(value, name)
    palette = {}
    for key, colour in document.items():
        palette[str(key)] = _optional_bgr(colour, f"{name}.{key}")  # type: ignore[assignment]
    return palette


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


def _optional_depth_range(value: object, name: str) -> Optional[Tuple[float, float]]:
    """读 ``[近, 远]`` 毫米；校验规则与针孔模型那边共用一份。"""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element [near_mm, far_mm] list or null")
    try:
        return normalize_depth_range((_number(value[0], name), _number(value[1], name)))
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error

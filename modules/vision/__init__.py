"""Camera and YOLOv8 detection for Windows and Jetson.

This package provides a self-contained USB camera + YOLOv8
detection stack meant to be reused by robot applications:

  - :class:`modules.vision.drive.yolov8.YOLOv8Detector`: TensorRT detection on Jetson.
  - :class:`modules.vision.drive.yolov8.YOLOv8ONNXDetector`: OpenCV ONNX detection on Windows.
  - :class:`modules.vision.drive.camera.USBCamera`: USB camera capture wrapper.
  - :func:`modules.vision.drive.preview.open_preview`: HighGUI preview window (ASCII name,
    created explicitly; see the module docstring for the Windows traps).
  - :class:`modules.vision.drive.distance.DistanceEstimator`: monocular distance estimate.

物料定位流水线（每个功能一个模块，每个模块都能单独 ``python -m`` 自测）:

  - :mod:`modules.vision.drive.hough_lock`: 霍夫圆变换定位物料顶部圆面（圆心 + 半径）。
  - :mod:`modules.vision.drive.material_yolo`: YOLO 推理出物料候选框（可只跑锁定区域）。
  - :mod:`modules.vision.drive.material_presence`: 基于 YOLO 框的独立存在性辅助工具。
  - :mod:`modules.vision.drive.material_class`: "是哪一种物料" + 中文标签。
  - :mod:`modules.vision.drive.material_distance`: 基于 YOLO 框的独立测距辅助工具。
  - :mod:`modules.vision.drive.material_locate`: 圆面定位和测距，再用 YOLO 补充类别。

Example::

    import cv2
    from modules.vision import USBCamera, close_previews, default_model_path, open_preview, run_preview

    window = open_preview()
    try:
        with USBCamera(0).initialize() as cam, \
                create_detector(default_model_path()) as detector:
            # 取帧与推理各跑一条线程：窗口跑在摄像头帧率上，
            # 推理跟不上的帧被跳过而不是拖慢显示（见 drive/stream.py）。
            run_preview(cam, detector.detect, detector.draw, window=window)
    finally:
        close_previews()

需要"结果与画面严格同帧"的场合（抓取控制）不要用 :func:`run_preview`，
直接同步调 ``detector.detect(frame)`` / ``locator.locate(frame)``。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .drive.camera import USBCamera, load_camera_id, looks_blank
from .drive.distance import DistanceEstimator, focal_length_from_fov
from .drive.preview import (
    BLANK_FRAME_WARNING,
    PREVIEW_WINDOW_NAME,
    close_previews,
    open_preview,
    warn_blank_preview,
)
from .drive.yolov8 import DEFAULT_CLASS_NAMES, Detection, Detector, YOLOv8Detector, YOLOv8ONNXDetector

#: 物料定位流水线的符号 -> 定义它的模块。这些模块按需导入（见 __getattr__）：
#: 一是 ``python -m modules.vision.drive.hough_lock`` 这类自测入口不会因为在包初始化时就导入过
#: 而触发 runpy 的 "found in sys.modules" 警告，二是 ``import vision`` 不必为
#: 用不到的功能买单。
_LAZY_ATTRIBUTES: dict[str, str] = {
    # stream（预览的两条线程；自己没有 __main__ 自测入口的话也可以直接导入，
    # 但 python -m modules.vision.drive.stream 需要它别在包初始化时就被导入）
    "DetectionResult": "drive.stream",
    "DetectionWorker": "drive.stream",
    "FrameGrabber": "drive.stream",
    "GrabbedFrame": "drive.stream",
    "PreviewStats": "drive.stream",
    "RateMeter": "drive.stream",
    "run_preview": "drive.stream",
    # pinhole
    "CameraFix": "drive.pinhole",
    "DEFAULT_FOCAL_PX_AT_1280": "drive.pinhole",
    "PinholeCamera": "drive.pinhole",
    "focal_px_from_known_target": "drive.pinhole",
    "resolve_focal_px": "drive.pinhole",
    # calibration
    "CalibrationSample": "drive.calibration",
    "PlaneCalibration": "drive.calibration",
    "fit_plane_calibration": "drive.calibration",
    # cell_targets
    "CellTarget": "drive.cell_targets",
    "CellTargetDetector": "drive.cell_targets",
    "CellTargets": "drive.cell_targets",
    "MARKER_ROLE": "drive.cell_targets",
    "MATERIAL_ROLE": "drive.cell_targets",
    # config
    "ModelSettings": "config",
    "TargetSettings": "config",
    "VisionConfig": "config",
    "current_platform": "config",
    "load_model_settings": "config",
    "load_vision_config": "config",
    "save_vision_calibration": "config",
    # hough_lock
    "HoughRingLocator": "drive.hough_lock",
    "LockedRegion": "drive.hough_lock",
    "RingLock": "drive.hough_lock",
    # material_yolo
    "MATERIAL_CLASS_NAMES": "drive.material_yolo",
    "MaterialYOLO": "drive.material_yolo",
    "is_material": "drive.material_yolo",
    # material_presence
    "MaterialPresence": "drive.material_presence",
    "PresenceResult": "drive.material_presence",
    # material_class
    "DEFAULT_MATERIAL_LABELS": "drive.material_class",
    "ClassResult": "drive.material_class",
    "MaterialCandidate": "drive.material_class",
    "MaterialClassifier": "drive.material_class",
    # material_distance
    "MaterialDistance": "drive.material_distance",
    "MaterialDistanceEstimator": "drive.material_distance",
    # material_locate
    "MaterialLocator": "drive.material_locate",
    "MaterialReport": "drive.material_locate",
    "RegionReport": "drive.material_locate",
}

if TYPE_CHECKING:
    # 与 _LAZY_ATTRIBUTES 一一对应的显式再导出：``X as X`` 才算再导出。
    # 运行时仍由 __getattr__ 按需导入（保住 ``python -m modules.vision.drive.hough_lock``
    # 这类自测入口不触发 runpy 的 "found in sys.modules" 警告），静态分析则
    # 从这里拿到真实类型，``from vision import MaterialLocator`` 不再是 Any。
    from .config import (
        ModelSettings as ModelSettings,
        TargetSettings as TargetSettings,
        VisionConfig as VisionConfig,
        current_platform as current_platform,
        load_model_settings as load_model_settings,
        load_vision_config as load_vision_config,
        save_vision_calibration as save_vision_calibration,
    )
    from .drive.calibration import (
        CalibrationSample as CalibrationSample,
        PlaneCalibration as PlaneCalibration,
        fit_plane_calibration as fit_plane_calibration,
    )
    from .drive.cell_targets import (
        MARKER_ROLE as MARKER_ROLE,
        MATERIAL_ROLE as MATERIAL_ROLE,
        CellTarget as CellTarget,
        CellTargetDetector as CellTargetDetector,
        CellTargets as CellTargets,
    )
    from .drive.hough_lock import (
        HoughRingLocator as HoughRingLocator,
        LockedRegion as LockedRegion,
        RingLock as RingLock,
    )
    from .drive.pinhole import (
        DEFAULT_FOCAL_PX_AT_1280 as DEFAULT_FOCAL_PX_AT_1280,
        CameraFix as CameraFix,
        PinholeCamera as PinholeCamera,
        focal_px_from_known_target as focal_px_from_known_target,
        resolve_focal_px as resolve_focal_px,
    )
    from .drive.stream import (
        DetectionResult as DetectionResult,
        DetectionWorker as DetectionWorker,
        FrameGrabber as FrameGrabber,
        GrabbedFrame as GrabbedFrame,
        PreviewStats as PreviewStats,
        RateMeter as RateMeter,
        run_preview as run_preview,
    )
    from .drive.material_class import (
        DEFAULT_MATERIAL_LABELS as DEFAULT_MATERIAL_LABELS,
        ClassResult as ClassResult,
        MaterialCandidate as MaterialCandidate,
        MaterialClassifier as MaterialClassifier,
    )
    from .drive.material_distance import (
        MaterialDistance as MaterialDistance,
        MaterialDistanceEstimator as MaterialDistanceEstimator,
    )
    from .drive.material_locate import (
        MaterialLocator as MaterialLocator,
        MaterialReport as MaterialReport,
        RegionReport as RegionReport,
    )
    from .drive.material_presence import (
        MaterialPresence as MaterialPresence,
        PresenceResult as PresenceResult,
    )
    from .drive.material_yolo import (
        MATERIAL_CLASS_NAMES as MATERIAL_CLASS_NAMES,
        MaterialYOLO as MaterialYOLO,
        is_material as is_material,
    )

__all__ = [
    "BLANK_FRAME_WARNING",
    "CalibrationSample",
    "CameraFix",
    "CellTarget",
    "CellTargetDetector",
    "CellTargets",
    "DEFAULT_CLASS_NAMES",
    "DEFAULT_FOCAL_PX_AT_1280",
    "DEFAULT_MATERIAL_LABELS",
    "Detection",
    "DetectionResult",
    "DetectionWorker",
    "Detector",
    "DistanceEstimator",
    "FrameGrabber",
    "GrabbedFrame",
    "PreviewStats",
    "RateMeter",
    "run_preview",
    "ClassResult",
    "HoughRingLocator",
    "LockedRegion",
    "MARKER_ROLE",
    "MATERIAL_CLASS_NAMES",
    "MATERIAL_ROLE",
    "MaterialCandidate",
    "MaterialClassifier",
    "MaterialDistance",
    "MaterialDistanceEstimator",
    "MaterialLocator",
    "MaterialPresence",
    "MaterialReport",
    "MaterialYOLO",
    "ModelSettings",
    "PREVIEW_WINDOW_NAME",
    "PinholeCamera",
    "PlaneCalibration",
    "PresenceResult",
    "RegionReport",
    "RingLock",
    "TargetSettings",
    "USBCamera",
    "VisionConfig",
    "close_previews",
    "create_configured_detector",
    "current_platform",
    "fit_plane_calibration",
    "focal_px_from_known_target",
    "is_material",
    "load_camera_id",
    "load_model_settings",
    "load_vision_config",
    "looks_blank",
    "open_preview",
    "require_model_path",
    "resolve_focal_px",
    "save_vision_calibration",
    "warn_blank_preview",
    "YOLOv8Detector",
    "YOLOv8ONNXDetector",
    "focal_length_from_fov",
    "default_model_path",
    "create_detector",
]


def __getattr__(name: str):
    """Resolve the material-locator symbols on first use (PEP 562)."""
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # 缓存，之后不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))


def default_model_path() -> Path:
    """本平台默认加载的模型文件，取自 ``modules/vision/config.json`` 的 ``model`` 块。

    ``model.backend`` 是 ``auto``（默认）时：Windows 取 ``model.onnx``（OpenCV 推理），
    Jetson 取 ``model.engine``（TensorRT 推理）。想在某平台上强制另一条通道，
    把 ``backend`` 写成 ``"onnx"`` 或 ``"tensorrt"``。
    """
    from .config import load_model_settings

    return load_model_settings().model_path()


def require_model_path(settings: "ModelSettings | None" = None) -> Path:
    """同 :func:`default_model_path`，但文件不存在时直接说清楚该做什么。"""
    from .config import BACKEND_TENSORRT, load_model_settings

    settings = settings if settings is not None else load_model_settings()
    path = settings.model_path()
    if path.is_file():
        return path
    if settings.resolve_backend() == BACKEND_TENSORRT:
        raise FileNotFoundError(
            f"未找到 TensorRT 引擎：{path}\n"
            "引擎与板子/GPU/TensorRT 版本绑定，只能在 Jetson 上生成：\n"
            "    python -m modules.vision.training.build_engine\n"
            "（换成 OpenCV 的 ONNX 通道：把 modules/vision/config.json 的 "
            'model.backend 改成 "onnx"）'
        )
    raise FileNotFoundError(
        f"未找到模型文件：{path}\n"
        "把导出的模型放到这个路径，或改 modules/vision/config.json 的 model.onnx"
    )


def create_configured_detector(
    *,
    model_path: str | Path | None = None,
    class_names: "Sequence[str] | None" = None,
    conf_thresh: float | None = None,
    iou_thresh: float | None = None,
    input_size: int | None = None,
    settings: "ModelSettings | None" = None,
) -> Detector:
    """按 ``config.json`` 建检测器；显式传进来的参数才覆盖配置。

    这是"换模型只改配置"的落点：路径、类别表、输入边长、阈值都从这里进。
    显式参数是给临时试验和测试用的。
    """
    from .config import load_model_settings

    settings = settings if settings is not None else load_model_settings()
    path = Path(model_path) if model_path is not None else require_model_path(settings)

    kwargs = {
        "class_names": list(class_names) if class_names else list(settings.class_names),
        "conf_thresh": settings.conf_thresh if conf_thresh is None else float(conf_thresh),
        "iou_thresh": settings.iou_thresh if iou_thresh is None else float(iou_thresh),
    }
    size = settings.input_size if input_size is None else input_size
    if size is not None:
        # 引擎的尺寸写在引擎里，探针会把配置和它对一遍；ONNX 没有这条退路，
        # 只能靠配置（或 onnx 元数据）告诉它输入多大。
        kwargs["input_size"] = int(size)
    return create_detector(path, **kwargs)


def create_detector(model_path: str | Path | None = None, **kwargs) -> Detector:
    """Open a detector matching the explicitly selected model file format.

    ``model_path`` 为 ``None`` 时等价于 :func:`create_configured_detector`
    （即完全按 ``config.json`` 来）。
    """
    if model_path is None:
        return create_configured_detector(**kwargs)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{path}")
    if path.suffix == ".onnx":
        return YOLOv8ONNXDetector(str(path), **kwargs)
    if path.suffix == ".engine":
        return YOLOv8Detector(str(path), **kwargs)
    raise ValueError(f"unsupported vision model format: {path.suffix}")

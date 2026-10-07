"""Camera and YOLOv8 detection for Windows and Jetson.

This package provides a self-contained USB camera + YOLOv8
detection stack meant to be reused by robot applications:

  - :class:`vision.yolov8.YOLOv8Detector`: TensorRT detection on Jetson.
  - :class:`vision.yolov8.YOLOv8ONNXDetector`: OpenCV ONNX detection on Windows.
  - :class:`vision.camera.USBCamera`: USB camera capture wrapper.
  - :func:`vision.preview.open_preview`: HighGUI preview window (ASCII name,
    created explicitly; see the module docstring for the Windows traps).
  - :class:`vision.distance.DistanceEstimator`: monocular distance estimate.

物料定位流水线（每个功能一个模块，每个模块都能单独 ``python -m`` 自测）:

  - :mod:`vision.hough_lock`: 霍夫圆变换定位物料顶部圆面（圆心 + 半径）。
  - :mod:`vision.material_yolo`: YOLO 推理出物料候选框（可只跑锁定区域）。
  - :mod:`vision.material_presence`: 基于 YOLO 框的独立存在性辅助工具。
  - :mod:`vision.material_class`: "是哪一种物料" + 中文标签。
  - :mod:`vision.material_distance`: 基于 YOLO 框的独立测距辅助工具。
  - :mod:`vision.material_locate`: 圆面定位和测距，再用 YOLO 补充类别。

Example::

    import cv2
    from vision import USBCamera, close_previews, create_detector, default_model_path, open_preview

    window = open_preview()
    try:
        with USBCamera(0).initialize() as cam, \
                create_detector(default_model_path()) as detector:
            while True:
                frame = cam.read()
                if frame is None:
                    break
                cv2.imshow(window, detector.draw(frame, detector.detect(frame)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        close_previews()
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .camera import USBCamera, load_camera_id, looks_blank
from .distance import DistanceEstimator, focal_length_from_fov
from .preview import (
    BLANK_FRAME_WARNING,
    PREVIEW_WINDOW_NAME,
    close_previews,
    open_preview,
    warn_blank_preview,
)
from .yolov8 import DEFAULT_CLASS_NAMES, Detection, Detector, YOLOv8Detector, YOLOv8ONNXDetector

#: 物料定位流水线的符号 -> 定义它的模块。这些模块按需导入（见 __getattr__）：
#: 一是 ``python -m vision.hough_lock`` 这类自测入口不会因为在包初始化时就导入过
#: 而触发 runpy 的 "found in sys.modules" 警告，二是 ``import vision`` 不必为
#: 用不到的功能买单。
_LAZY_ATTRIBUTES: dict[str, str] = {
    # hough_lock
    "HoughRingLocator": "hough_lock",
    "LockedRegion": "hough_lock",
    "RingLock": "hough_lock",
    # material_yolo
    "MATERIAL_CLASS_NAMES": "material_yolo",
    "MaterialYOLO": "material_yolo",
    "is_material": "material_yolo",
    # material_presence
    "MaterialPresence": "material_presence",
    "PresenceResult": "material_presence",
    # material_class
    "DEFAULT_MATERIAL_LABELS": "material_class",
    "ClassResult": "material_class",
    "MaterialCandidate": "material_class",
    "MaterialClassifier": "material_class",
    # material_distance
    "MaterialDistance": "material_distance",
    "MaterialDistanceEstimator": "material_distance",
    # material_locate
    "MaterialLocator": "material_locate",
    "MaterialReport": "material_locate",
    "RegionReport": "material_locate",
}

if TYPE_CHECKING:
    # 与 _LAZY_ATTRIBUTES 一一对应的显式再导出：``X as X`` 才算再导出。
    # 运行时仍由 __getattr__ 按需导入（保住 ``python -m vision.hough_lock``
    # 这类自测入口不触发 runpy 的 "found in sys.modules" 警告），静态分析则
    # 从这里拿到真实类型，``from vision import MaterialLocator`` 不再是 Any。
    from .hough_lock import (
        HoughRingLocator as HoughRingLocator,
        LockedRegion as LockedRegion,
        RingLock as RingLock,
    )
    from .material_class import (
        DEFAULT_MATERIAL_LABELS as DEFAULT_MATERIAL_LABELS,
        ClassResult as ClassResult,
        MaterialCandidate as MaterialCandidate,
        MaterialClassifier as MaterialClassifier,
    )
    from .material_distance import (
        MaterialDistance as MaterialDistance,
        MaterialDistanceEstimator as MaterialDistanceEstimator,
    )
    from .material_locate import (
        MaterialLocator as MaterialLocator,
        MaterialReport as MaterialReport,
        RegionReport as RegionReport,
    )
    from .material_presence import (
        MaterialPresence as MaterialPresence,
        PresenceResult as PresenceResult,
    )
    from .material_yolo import (
        MATERIAL_CLASS_NAMES as MATERIAL_CLASS_NAMES,
        MaterialYOLO as MaterialYOLO,
        is_material as is_material,
    )

__all__ = [
    "BLANK_FRAME_WARNING",
    "DEFAULT_CLASS_NAMES",
    "DEFAULT_MATERIAL_LABELS",
    "Detection",
    "Detector",
    "DistanceEstimator",
    "ClassResult",
    "HoughRingLocator",
    "LockedRegion",
    "MATERIAL_CLASS_NAMES",
    "MaterialCandidate",
    "MaterialClassifier",
    "MaterialDistance",
    "MaterialDistanceEstimator",
    "MaterialLocator",
    "MaterialPresence",
    "MaterialReport",
    "MaterialYOLO",
    "PREVIEW_WINDOW_NAME",
    "PresenceResult",
    "RegionReport",
    "RingLock",
    "USBCamera",
    "close_previews",
    "is_material",
    "load_camera_id",
    "looks_blank",
    "open_preview",
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
    """Select the copied ONNX model on Windows or TensorRT engine on Jetson."""
    extension = ".onnx" if os.name == "nt" else ".engine"
    return Path(__file__).resolve().parent.parent / "models" / f"best{extension}"


def create_detector(model_path: str | Path, **kwargs) -> YOLOv8Detector:
    """Open a detector matching the explicitly selected model file format."""
    path = Path(model_path)
    if path.suffix == ".onnx":
        return YOLOv8ONNXDetector(str(path), **kwargs)
    if path.suffix == ".engine":
        return YOLOv8Detector(str(path), **kwargs)
    raise ValueError(f"unsupported vision model format: {path.suffix}")

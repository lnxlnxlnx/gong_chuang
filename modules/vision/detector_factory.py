"""按配置（或显式参数）建检测器：``.onnx`` 走 OpenCV，``.engine`` 走 TensorRT。

## 为什么从包出口搬到这里

工厂原先定义在 ``modules/vision/__init__.py``。包出口放实现代码有两个后果：
一是工厂要回调包出口里的 :func:`~modules.vision.config.require_model_path`，
二是不是"只要结果类型"的调用方也被迫经过包初始化。搬到这个模块之后，
依赖方向是单向的：``detector_factory`` → ``config`` / ``drive.yolov8``，
而包出口只剩显式再导出。

``PROJECT_ROOT`` 的路径规则在 :mod:`modules.vision.config` 里（配置里的相对路径
按项目根解析，不按当前工作目录），本模块不重复实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from modules.vision.config import ModelSettings, load_model_settings, require_model_path
from modules.vision.drive.yolov8 import Detector, YOLOv8Detector, YOLOv8ONNXDetector


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

    ``settings`` 是本会话已经读好的模型快照；给 ``None`` 时才现读默认配置文件
    （只有"完全没配置上下文"的调用方会走到那一支）。
    """
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

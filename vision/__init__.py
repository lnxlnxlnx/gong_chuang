"""Camera and YOLOv8 detection for Windows and Jetson.

This package provides a self-contained USB camera + YOLOv8
detection stack meant to be reused by robot applications:

  - :class:`vision.yolov8.YOLOv8Detector`: TensorRT detection on Jetson.
  - :class:`vision.yolov8.YOLOv8ONNXDetector`: OpenCV ONNX detection on Windows.
  - :class:`vision.camera.USBCamera`: USB camera capture wrapper.
  - :class:`vision.distance.DistanceEstimator`: monocular distance estimate.

Example::

    from vision import USBCamera, create_detector, default_model_path

    with USBCamera(0).initialize() as cam, \
            create_detector(default_model_path()) as detector:
        while True:
            frame = cam.read()
            if frame is None:
                break
            for hit in detector.detect(frame):
                print(hit)
"""

from __future__ import annotations

import os
from pathlib import Path

from .camera import USBCamera, load_camera_id
from .distance import DistanceEstimator, focal_length_from_fov
from .yolov8 import DEFAULT_CLASS_NAMES, Detection, YOLOv8Detector, YOLOv8ONNXDetector

__all__ = [
    "DEFAULT_CLASS_NAMES",
    "Detection",
    "DistanceEstimator",
    "USBCamera",
    "load_camera_id",
    "YOLOv8Detector",
    "YOLOv8ONNXDetector",
    "focal_length_from_fov",
    "default_model_path",
    "create_detector",
]


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

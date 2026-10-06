"""Camera and YOLOv8 detection for Windows and Jetson.

This package provides a self-contained USB camera + YOLOv8
detection stack meant to be reused by robot applications:

  - :class:`vision.yolov8.YOLOv8Detector`: TensorRT detection on Jetson.
  - :class:`vision.yolov8.YOLOv8ONNXDetector`: OpenCV ONNX detection on Windows.
  - :class:`vision.camera.USBCamera`: USB camera capture wrapper.
  - :func:`vision.preview.open_preview`: HighGUI preview window (ASCII name,
    created explicitly; see the module docstring for the Windows traps).
  - :class:`vision.distance.DistanceEstimator`: monocular distance estimate.

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

import os
from pathlib import Path

from .camera import USBCamera, load_camera_id, looks_blank
from .distance import DistanceEstimator, focal_length_from_fov
from .preview import (
    BLANK_FRAME_WARNING,
    PREVIEW_WINDOW_NAME,
    close_previews,
    open_preview,
    warn_blank_preview,
)
from .yolov8 import DEFAULT_CLASS_NAMES, Detection, YOLOv8Detector, YOLOv8ONNXDetector

__all__ = [
    "BLANK_FRAME_WARNING",
    "DEFAULT_CLASS_NAMES",
    "Detection",
    "DistanceEstimator",
    "PREVIEW_WINDOW_NAME",
    "USBCamera",
    "close_previews",
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

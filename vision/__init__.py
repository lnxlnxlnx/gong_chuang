"""Vision peripherals for the Jetson dev board.

This package provides a self-contained USB camera + YOLOv8 (TensorRT)
detection stack meant to be reused by robot applications:

  - :class:`vision.yolov8.YOLOv8Detector`: end-to-end detection on BGR frames.
  - :class:`vision.camera.USBCamera`: USB camera capture wrapper.
  - :class:`vision.distance.DistanceEstimator`: monocular distance estimate.

Example::

    from vision import DistanceEstimator, USBCamera, YOLOv8Detector

    with USBCamera(0).initialize() as cam, \
            YOLOv8Detector("models/best.engine") as detector:
        while True:
            frame = cam.read()
            if frame is None:
                break
            for hit in detector.detect(frame):
                print(hit)
"""

from .camera import USBCamera
from .distance import DistanceEstimator, focal_length_from_fov
from .yolov8 import DEFAULT_CLASS_NAMES, Detection, YOLOv8Detector

__all__ = [
    "DEFAULT_CLASS_NAMES",
    "Detection",
    "DistanceEstimator",
    "USBCamera",
    "YOLOv8Detector",
    "focal_length_from_fov",
]
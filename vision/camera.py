"""Minimal USB camera helper shared by the vision demo and applications.

Wraps :class:`cv2.VideoCapture` behind a small ``initialize()`` / ``read()`` /
``release()`` interface consistent with the other gong_chuang peripherals.
OpenCV is imported lazily so this module can be loaded on machines without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from project_paths import config_path


DEFAULT_CONFIG_PATH = config_path("vision.json")


def load_camera_id(platform: str, config_path: Path = DEFAULT_CONFIG_PATH) -> int:
    """Read the selected camera index for one platform; reject invalid settings."""
    try:
        settings = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read vision configuration: {config_path}") from error
    profile = settings.get(platform) if isinstance(settings, dict) else None
    camera_id = profile.get("camera_id") if isinstance(profile, dict) else None
    if type(camera_id) is not int or camera_id < 0:
        raise ValueError(f"{platform}.camera_id must be a non-negative integer")
    return camera_id


def _cv2():
    """Import and return ``cv2`` lazily."""
    import cv2
    return cv2


def looks_blank(frame, *, std_threshold: float = 2.0) -> bool:
    """Return ``True`` when a frame is one flat colour (all black / all white).

    A real camera picture always has some variation, so a near-zero standard
    deviation means the capture produced nothing usable even though ``read()``
    reported success. Used by the preview entry points to print a diagnosis
    instead of silently showing a black window.
    """
    if frame is None:
        return True
    import numpy as np

    return bool(float(np.asarray(frame).std()) < std_threshold)


class USBCamera:
    """A V4L2/USB camera wrapper producing BGR frames.

    Usage::

        with USBCamera(device_id=0).initialize() as cam:
            frame = cam.read()   # None when no frame is available

    Attributes:
        device_id: V4L2 device index (``0`` matches ``/dev/video0``).
        width, height: requested resolution (best effort, driver dependent).
        fps: requested frame rate.
        capture: the underlying ``cv2.VideoCapture``, open after initialize().
    """

    def __init__(self, device_id: int = 0, *, width: int = 640,
                 height: int = 480, fps: int = 30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None

    def initialize(self) -> "USBCamera":
        """Open the camera and apply resolution / frame-rate settings."""
        cv2 = _cv2()
        capture = cv2.VideoCapture(self.device_id)
        if not capture.isOpened():
            raise RuntimeError(
                f"USB camera {self.device_id} failed to open; "
                "check the configured camera index and connection"
            )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture = capture
        return self

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or ``None`` if capture failed."""
        if self.capture is None:
            raise RuntimeError("USBCamera not initialized; call initialize() first")
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self) -> None:
        """Release the underlying capture (safe to call multiple times)."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "USBCamera":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

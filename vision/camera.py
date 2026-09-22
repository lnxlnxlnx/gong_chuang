"""Minimal USB camera helper shared by the vision demo and applications.

Wraps :class:`cv2.VideoCapture` behind a small ``initialize()`` / ``read()`` /
``release()`` interface consistent with the other gong_chuang peripherals.
OpenCV is imported lazily so this module can be loaded on machines without it.
"""

from __future__ import annotations


def _cv2():
    """Import and return ``cv2`` lazily."""
    import cv2
    return cv2


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
                "check `ls /dev/video*` for available devices"
            )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture = capture
        return self

    def read(self):
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
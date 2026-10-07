"""Monocular distance estimation for known-width objects.

Uses the pinhole camera model:  ``distance = focal_px * real_width / box_width_px``.
No depth camera needed; it is only accurate when the real width of the object
class is known and roughly uniform (see the README calibration notes).
"""

from __future__ import annotations

import math
from typing import Mapping


def focal_length_from_fov(fov_deg: float, image_width_px: int) -> float:
    """Derive the horizontal focal length (px) from a horizontal FOV (degrees)."""
    return (image_width_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


class DistanceEstimator:
    """Estimate distance to detections using class real widths.

    Parameters:
        focal_px: horizontal focal length in pixels
            (see :func:`focal_length_from_fov`, e.g. ~500-600 px for a typical
            90-degree USB webcam at 640 px width).
        real_widths: mapping of ``class name -> real width in millimetres``.
            Only classes present here are measurable; everything else reports
            ``None``.  Using millimetres yields distances in millimetres.
    """

    def __init__(self, focal_px: float, real_widths: Mapping[str, float]):
        self.focal_px = float(focal_px)
        self.real_widths = dict(real_widths)

    def estimate(self, detection) -> float | None:
        """Return the estimated distance (same unit as ``real_widths``).

        ``None`` when the detection's class width is unknown or the box has no
        usable width.  The input may be any object exposing ``name`` and
        ``width`` attributes (e.g. :class:`vision.yolov8.Detection`).
        """
        width_mm = self.real_widths.get(detection.name)
        if width_mm is None or detection.width <= 0.0:
            return None
        return self.focal_px * width_mm / detection.width
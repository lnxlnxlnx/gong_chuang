"""相对物料距离：由检测框算"物料离摄像头多远 / 相对锁定位置偏多少"。

两套量一起给，因为"相对距离"在不同场合说法不同：

- **真实距离** ``distance_mm``：单目针孔模型 ``焦距 × 真实宽度 / 框宽``，需要先标定
  焦距（``docs``/README 的焦距标定）和该类别物料的真实宽度；没标定时是 ``None``，
  不会瞎猜一个数。
- **画面内相对量**：``offset_x_px`` / ``offset_y_px`` 是相对**锁定圆心**（没给锁定
  圆环时相对画面中心）的像素偏移，``relative_x`` / ``relative_y`` 是归一化到
  ``-1..1`` 的量，``region_distance_px`` 是到锁定圆心的直线像素距离 —— 这几个不需要
  标定，适合直接喂给控制环做对准。

命令行自测::

    python -m modules.vision.drive.material_distance     # 合成检测框自测，不需要摄像头/模型
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from modules.vision.drive.distance import DistanceEstimator


@dataclass(frozen=True)
class MaterialDistance:
    """一个物料相对摄像头 / 锁定位置的量（长度单位与标定一致，默认毫米）。"""

    name: str
    box_width_px: float
    box_height_px: float
    #: 针孔模型估的距离；没标定真实宽度时为 ``None``。
    distance_mm: float | None
    offset_x_px: float
    offset_y_px: float
    relative_x: float
    relative_y: float
    region_distance_px: float | None
    #: 偏移的参考点：``"ring"`` = 锁定圆环圆心，``"image-center"`` = 画面中心。
    reference: str

    @property
    def distance_cm(self) -> float | None:
        """Distance in centimetres (``None`` when uncalibrated)."""
        return None if self.distance_mm is None else self.distance_mm / 10.0

    @property
    def calibrated(self) -> bool:
        """Whether a real-world distance could be computed."""
        return self.distance_mm is not None


class MaterialDistanceEstimator:
    """Estimate distance / offsets for material detections.

    Parameters:
        focal_px: horizontal focal length in pixels (see
            :func:`modules.vision.drive.distance.focal_length_from_fov`).
        real_widths_mm: ``class name -> real width in mm``; only those classes
            get a real distance. Without it, only pixel offsets are returned.
        image_size: ``(width, height)`` used to normalise the offsets when the
            caller does not pass ``frame_shape``.
    """

    def __init__(
        self,
        *,
        focal_px: float = 560.0,
        real_widths_mm: Mapping[str, float] | None = None,
        image_size: tuple[int, int] = (640, 480),
    ) -> None:
        if focal_px <= 0:
            raise ValueError("focal_px must be positive")
        self.focal_px = float(focal_px)
        self.real_widths_mm = dict(real_widths_mm or {})
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._estimator = (
            DistanceEstimator(self.focal_px, self.real_widths_mm) if self.real_widths_mm else None
        )

    def estimate(self, hit, *, region=None, frame_shape=None) -> MaterialDistance:
        """Return the distance / offsets of one detection.

        ``region`` is a :class:`modules.vision.drive.hough_lock.LockedRegion`; when omitted the
        offsets are relative to the image centre.
        """
        width, height = self._frame_size(frame_shape)
        center_x, center_y = hit.center
        frame_center_x, frame_center_y = width / 2.0, height / 2.0

        if region is not None:
            reference_x, reference_y = region.center
            reference = "ring"
            region_distance = math.hypot(center_x - reference_x, center_y - reference_y)
        else:
            reference_x, reference_y = frame_center_x, frame_center_y
            reference = "image-center"
            region_distance = None

        distance_mm = None
        if self._estimator is not None:
            distance_mm = self._estimator.estimate(hit)

        return MaterialDistance(
            name=hit.name,
            box_width_px=float(hit.width),
            box_height_px=float(hit.height),
            distance_mm=None if distance_mm is None else float(distance_mm),
            offset_x_px=float(center_x - reference_x),
            offset_y_px=float(center_y - reference_y),
            relative_x=float((center_x - frame_center_x) / (width / 2.0)),
            relative_y=float((center_y - frame_center_y) / (height / 2.0)),
            region_distance_px=None if region_distance is None else float(region_distance),
            reference=reference,
        )

    def estimate_all(self, hits, *, region=None, frame_shape=None) -> list[MaterialDistance]:
        """Estimate every hit, closest to the reference first."""
        results = [self.estimate(hit, region=region, frame_shape=frame_shape) for hit in hits]
        results.sort(key=lambda item: self._sort_key(item))
        return results

    @staticmethod
    def _sort_key(item: MaterialDistance) -> tuple:
        distance = item.region_distance_px
        if distance is None:
            return (1, abs(item.offset_x_px) + abs(item.offset_y_px))
        return (0, distance)

    def _frame_size(self, frame_shape) -> tuple[int, int]:
        if frame_shape is None:
            return self.image_size
        if len(frame_shape) < 2:
            raise ValueError("frame_shape must be (height, width) or (height, width, channels)")
        return (int(frame_shape[1]), int(frame_shape[0]))


# ---------------------------------------------------------------------- 自测
def _synthetic_hits():
    from modules.vision.drive.yolov8 import Detection

    return {
        # 框宽 120 px；焦距 560 px、真实宽度 60 mm -> 560*60/120 = 280 mm
        "red": Detection(x1=260.0, y1=180.0, x2=380.0, y2=300.0, score=0.9, class_id=0, name="red"),
        # 离锁定圆心更近的一个
        "blue": Detection(x1=330.0, y1=240.0, x2=390.0, y2=300.0, score=0.8, class_id=2, name="blue"),
    }


def _self_test() -> int:
    from modules.vision.drive.hough_lock import LockedRegion

    hits = _synthetic_hits()
    estimator = MaterialDistanceEstimator(
        focal_px=560.0,
        real_widths_mm={"red": 60.0, "blue": 50.0},
        image_size=(640, 480),
    )
    region = LockedRegion(center_x=320.0, center_y=240.0, radius=80.0)
    problems: list[str] = []

    red = estimator.estimate(hits["red"], region=region, frame_shape=(480, 640, 3))
    if red.distance_mm is None or abs(red.distance_mm - 280.0) > 1e-6:
        problems.append(f"针孔模型应为 280 mm，实际 {red.distance_mm}")
    if red.distance_cm is None or abs(red.distance_cm - 28.0) > 1e-6:
        problems.append(f"厘米换算应为 28 cm，实际 {red.distance_cm}")
    if red.reference != "ring":
        problems.append(f"给了锁定圆环时 reference 应为 ring，实际 {red.reference}")
    if abs(red.offset_x_px - 0.0) > 1e-9 or abs(red.offset_y_px - 0.0) > 1e-9:
        problems.append(f"红框中心正好在圆心，偏移应为 0，实际 ({red.offset_x_px}, {red.offset_y_px})")
    if red.region_distance_px is None or abs(red.region_distance_px) > 1e-9:
        problems.append(f"到圆心距离应为 0，实际 {red.region_distance_px}")
    if abs(red.relative_x) > 1e-9 or abs(red.relative_y) > 1e-9:
        problems.append("画面归一化坐标应为 (0, 0)")

    blue = estimator.estimate(hits["blue"], region=region, frame_shape=(480, 640, 3))
    if abs(blue.offset_x_px - 40.0) > 1e-9 or abs(blue.offset_y_px - 30.0) > 1e-9:
        problems.append(f"蓝框中心 (360,270) 相对圆心 (320,240) 应为 (40,30)，实际 ({blue.offset_x_px}, {blue.offset_y_px})")
    if blue.region_distance_px is None or abs(blue.region_distance_px - 50.0) > 1e-9:
        problems.append(f"到圆心距离应为 50 px，实际 {blue.region_distance_px}")
    if blue.distance_mm is None or abs(blue.distance_mm - 560.0 * 50.0 / 60.0) > 1e-6:
        problems.append(f"蓝框针孔距离应为 {560.0 * 50.0 / 60.0:.1f} mm，实际 {blue.distance_mm}")

    ordered = estimator.estimate_all([hits["red"], hits["blue"]], region=region, frame_shape=(480, 640))
    if [item.name for item in ordered] != ["red", "blue"]:
        problems.append(f"estimate_all 应按离圆心由近到远排序，实际 {[item.name for item in ordered]}")

    # 没标定真实宽度：只给像素量，绝不编造一个距离
    uncalibrated = MaterialDistanceEstimator(focal_px=560.0).estimate(hits["red"], frame_shape=(480, 640))
    if uncalibrated.distance_mm is not None or uncalibrated.calibrated:
        problems.append("没标定真实宽度时不应给出 distance_mm")
    if uncalibrated.reference != "image-center":
        problems.append(f"没给锁定圆环时 reference 应为 image-center，实际 {uncalibrated.reference}")
    if abs(uncalibrated.offset_x_px) > 1e-9 or abs(uncalibrated.offset_y_px) > 1e-9:
        problems.append("画面中心参考下，红框中心 (320,240) 的偏移应为 0")

    print(f"标定 {estimator.real_widths_mm}、焦距 {estimator.focal_px:.0f} px")
    print(f"红框：{red.distance_mm:.1f} mm，偏移 ({red.offset_x_px:.0f}, {red.offset_y_px:.0f}) px")
    print(f"蓝框：{blue.distance_mm:.1f} mm，到圆心 {blue.region_distance_px:.0f} px，归一化 ({blue.relative_x:.3f}, {blue.relative_y:.3f})")
    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：针孔距离、像素偏移、归一化坐标、未标定回退都正确。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="相对物料距离：检测框 -> 距离 / 像素偏移")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

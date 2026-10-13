"""用霍夫圆变换定位物料顶部圆面，返回圆心和像素半径。

输入是相机原始 BGR 帧；灰度化和中值滤波保留圆边缘，再做霍夫检测。
默认半径窗口随实际帧尺寸变化，不假设画面中恰好有三个目标。

命令行自测::

    python -m modules.vision.drive.hough_lock                  # 合成图自测（不需要摄像头/模型）
    python -m modules.vision.drive.hough_lock --image a.jpg    # 在一张图片上跑
    python -m modules.vision.drive.hough_lock --camera 1       # 摄像头实时预览，按 q 退出
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _cv2():
    """Import and return ``cv2`` lazily."""
    import cv2
    return cv2


#: 严格参数一无所获时退回来的 Canny 高阈值。
#:
#: 默认 100 只适合锐利画面。近距（约 7 cm）离焦时物料顶面那圈边的梯度被摊平，
#: 100 会只剩夹爪那种高对比弧线，于是锁定到半径偏大的假圆（实测 248 px / 61 mm
#: 对真值 218 px / 70 mm）；50 才留得住真实圆边。放松只在严格参数没找到"像顶面
#: 的圆"时发生，见 ``modules.vision.drive.material_locate``。
RELAXED_PARAM1 = 50.0


@dataclass(frozen=True)
class LockedRegion:
    """一个被锁定的圆形区域（像素坐标，原点在画面左上角）。"""

    center_x: float
    center_y: float
    radius: float

    @property
    def center(self) -> tuple[float, float]:
        """Return ``(x, y)`` of the circle centre."""
        return (self.center_x, self.center_y)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return the circle's bounding box as ``(x1, y1, x2, y2)``."""
        return (
            int(round(self.center_x - self.radius)),
            int(round(self.center_y - self.radius)),
            int(round(self.center_x + self.radius)),
            int(round(self.center_y + self.radius)),
        )

    def contains(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        """Return ``True`` when ``(x, y)`` is inside the circle (+``margin`` px)."""
        dx = float(x) - self.center_x
        dy = float(y) - self.center_y
        limit = self.radius + float(margin)
        return dx * dx + dy * dy <= limit * limit

    def offset_from_center(self, x: float, y: float) -> tuple[float, float]:
        """Return ``(dx, dy)`` of a point relative to the locked centre, in pixels."""
        return (float(x) - self.center_x, float(y) - self.center_y)

    def scaled(self, factor: float) -> "LockedRegion":
        """Return a copy with the radius multiplied by ``factor`` (same centre)."""
        if factor <= 0:
            raise ValueError("factor must be positive")
        return LockedRegion(self.center_x, self.center_y, self.radius * float(factor))

    def expanded(self, margin_px: float) -> "LockedRegion":
        """Return a copy grown by ``margin_px`` (negative values shrink it)."""
        return LockedRegion(self.center_x, self.center_y, max(1.0, self.radius + float(margin_px)))


@dataclass(frozen=True)
class RingLock:
    """按 x 从左到右排序的物料顶部圆面。"""

    regions: tuple[LockedRegion, ...]
    #: 可选的期望数量；默认不限制物料个数。
    expected: int | None = None

    @property
    def complete(self) -> bool:
        """Return ``True`` when the detected count matches ``expected``."""
        return self.expected is None or len(self.regions) == self.expected

    @property
    def centers(self) -> tuple[tuple[float, float], ...]:
        """Return every locked centre in left-to-right order."""
        return tuple(region.center for region in self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)


class HoughRingLocator:
    """定位物料顶部圆面；半径窗口默认按帧尺寸计算。

    ``param1`` 是 Canny 的高阈值。画面锐利时 100 足够；画面发虚（近距、离焦）时
    真实圆边的梯度被摊平，阈值太高会只剩夹爪那种高对比弧线，于是锁定到错误的圆，
    半径偏大、距离跟着偏小——所以要按实际画面放宽，见
    :data:`modules.vision.drive.material_locate.DEFAULT_PARAM1`。
    """

    def __init__(
        self,
        *,
        expected: int | None = None,
        min_radius: int | None = None,
        max_radius: int | None = None,
        min_distance: int | None = None,
        dp: float = 1.2,
        param1: float = 100.0,
        param2: float = 45.0,
    ) -> None:
        if expected is not None and expected < 1:
            raise ValueError("expected must be a positive integer or None")
        if min_radius is not None and min_radius < 1:
            raise ValueError("min_radius must be positive")
        if max_radius is not None and max_radius < 1:
            raise ValueError("max_radius must be positive")
        if min_radius is not None and max_radius is not None and max_radius <= min_radius:
            raise ValueError("max_radius must exceed min_radius")
        if min_distance is not None and min_distance < 1:
            raise ValueError("min_distance must be positive")
        if dp <= 0 or param1 <= 0 or param2 <= 0:
            raise ValueError("Hough parameters must be positive")
        self.expected = expected
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_distance = min_distance
        self.dp = float(dp)
        self.param1 = float(param1)
        self.param2 = float(param2)

    def relaxed(self) -> "HoughRingLocator":
        """Return a copy with a lower Canny threshold, for soft (defocused) frames.

        只在严格参数一无所获时用：放松会带来更多候选圆，必须配合
        ``modules.vision.drive.material_face`` 的顶面判据挑出真正的那一个。
        """
        return HoughRingLocator(
            expected=self.expected,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            min_distance=self.min_distance,
            dp=self.dp,
            param1=min(self.param1, RELAXED_PARAM1),
            param2=self.param2,
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Return the smoothed grayscale image used by HoughCircles."""
        cv2 = _cv2()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame is empty")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.medianBlur(gray, 5)

    def locate(
        self,
        frame: np.ndarray,
        *,
        min_radius: int | None = None,
        max_radius: int | None = None,
    ) -> RingLock:
        """Locate top faces in image coordinates, left to right.

        ``min_radius`` / ``max_radius`` override the instance window for this
        call only.  调用方知道物料顶面大概在什么深度时，应当把针孔模型算出来的
        半径窗口传进来：霍夫的尺度先验越准，越不会把背景上凑出来的圆当成物料。
        """
        min_radius = self.min_radius if min_radius is None else round(min_radius)
        max_radius = self.max_radius if max_radius is None else round(max_radius)
        if min_radius is not None and min_radius < 1:
            raise ValueError("min_radius must be positive")
        if max_radius is not None and max_radius < 1:
            raise ValueError("max_radius must be positive")
        if min_radius is not None and max_radius is not None and max_radius <= min_radius:
            raise ValueError("max_radius must exceed min_radius")

        gray = self.preprocess(frame)
        height, width = gray.shape
        min_radius = min_radius if min_radius is not None else max(8, round(width * 0.02))
        max_radius = max_radius if max_radius is not None else round(min(width, height) * 0.36)
        min_distance = self.min_distance if self.min_distance is not None else max(1, round(min(width, height) * 0.08))
        circles = _cv2().HoughCircles(
            gray,
            _cv2().HOUGH_GRADIENT,
            self.dp,
            min_distance,
            param1=self.param1,
            param2=self.param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        regions = [] if circles is None else [LockedRegion(*map(float, circle)) for circle in circles[0]]
        regions.sort(key=lambda region: region.center_x)
        return RingLock(tuple(regions), self.expected)

    def locate_regions(
        self,
        frame: np.ndarray,
        *,
        min_radius: int | None = None,
        max_radius: int | None = None,
    ) -> list[LockedRegion]:
        """Return material top circles as a plain list (left to right)."""
        return list(self.locate(frame, min_radius=min_radius, max_radius=max_radius).regions)

    # ------------------------------------------------------------------ 画图
    def draw(
        self,
        frame: np.ndarray,
        lock: RingLock,
        *,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with every top circle and centre marked."""
        cv2 = _cv2()
        annotated = frame.copy()
        for index, region in enumerate(lock.regions, start=1):
            centre = (int(round(region.center_x)), int(round(region.center_y)))
            cv2.circle(annotated, centre, int(round(region.radius)), color, thickness)
            cv2.circle(annotated, centre, 2, (255, 0, 0), 2)
            cv2.putText(
                annotated,
                f"material{index}",
                (centre[0] - 20, centre[1] - int(region.radius) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return annotated

    def debug_image(self, frame: np.ndarray):
        """Return the smoothed grayscale image as BGR for previews."""
        return _cv2().cvtColor(self.preprocess(frame), _cv2().COLOR_GRAY2BGR)


# ---------------------------------------------------------------------- 自测
def synthetic_rings(
    width: int = 640,
    height: int = 480,
    *,
    radius: int = 40,
    thickness: int = 8,
    centers: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build a test frame with three rings under uneven lighting.

    The background has a dark-to-bright ramp so the algorithm is exercised on
    the uneven illumination it is designed for, and no hardware is needed.
    """
    cv2 = _cv2()
    if centers is None:
        centers = ((width // 4, height // 2), (width // 2, height // 2), (width * 3 // 4, height // 2))

    base = np.full((height, width, 3), 60, dtype=np.float32)
    ramp = np.linspace(-30.0, 30.0, width, dtype=np.float32)
    frame = np.clip(base + ramp[None, :, None], 0, 255).astype(np.uint8)
    for center_x, center_y in centers:
        cv2.circle(frame, (int(center_x), int(center_y)), int(radius), (235, 235, 235), int(thickness))
    return frame


def _self_test() -> int:
    """Run the hardware-free synthetic check used by ``python -m modules.vision.drive.hough_lock``."""
    locator = HoughRingLocator()
    frame = synthetic_rings()
    lock = locator.locate(frame)
    expected_centres = [(160, 240), (320, 240), (480, 240)]

    print(f"合成图 {frame.shape[1]}x{frame.shape[0]}：检出 {len(lock)} 个圆")
    for index, region in enumerate(lock.regions):
        print(f"  第 {index + 1} 个：圆心 ({region.center_x:.1f}, {region.center_y:.1f})，半径 {region.radius:.1f}")

    problems: list[str] = []
    if len(lock) != 3:
        problems.append(f"数量不对：检出 {len(lock)}，期望 3")
    for index, (region, (expected_x, expected_y)) in enumerate(zip(lock.regions, expected_centres)):
        if abs(region.center_x - expected_x) > 16 or abs(region.center_y - expected_y) > 16:
            problems.append(
                f"第 {index + 1} 个圆心偏离合成位置：({region.center_x:.0f},{region.center_y:.0f}) "
                f"!= ({expected_x},{expected_y})"
            )
    radius_error = [abs(region.radius - 40.0) for region in lock.regions]
    if radius_error and max(radius_error) > 8:
        problems.append(f"半径偏差过大：{['%.1f' % error for error in radius_error]}")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：三个合成圆都被锁定，且按 x 从左到右排序。")
    return 0


def _run_image(locator: HoughRingLocator, image_path: Path) -> int:
    frame = _cv2().imread(str(image_path))
    if frame is None:
        print(f"读不出图片：{image_path}")
        return 1
    lock = locator.locate(frame)
    print(f"{image_path}：检出 {len(lock)} 个圆面")
    for index, region in enumerate(lock.regions, start=1):
        print(f"  第 {index} 个：圆心 ({region.center_x:.1f}, {region.center_y:.1f})，半径 {region.radius:.1f}")
    return 0 if lock.regions else 1


def _run_camera(locator: HoughRingLocator, camera_id: int) -> int:
    import cv2

    from modules.vision import (
        BLANK_FRAME_WARNING,
        USBCamera,
        close_previews,
        looks_blank,
        open_preview,
        warn_blank_preview,
    )

    window = open_preview()
    blank_frames = 0
    warned = False
    try:
        with USBCamera(camera_id, width=1280, height=720).initialize() as camera:
            print(f"摄像头 {camera_id}；窗口 {window!r}，按 q 退出（按 d 切换灰度图）")
            show_debug = False
            while True:
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("摄像头未返回图像")
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned:
                    warned = True
                    warn_blank_preview(camera_id)

                lock = locator.locate(frame)
                shown = locator.debug_image(frame) if show_debug else locator.draw(frame, lock)
                cv2.putText(
                    shown,
                    f"materials={len(lock)}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    show_debug = not show_debug
    finally:
        close_previews()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="霍夫圆变换定位物料顶部圆面")
    parser.add_argument("--image", type=Path, help="在一张图片上跑（不给就跑合成图自测）")
    parser.add_argument("--camera", type=int, help="打开摄像头实时预览")
    parser.add_argument("--min-radius", type=int, help="最小半径（像素，默认随画幅计算）")
    parser.add_argument("--max-radius", type=int, help="最大半径（像素，默认随画幅计算）")
    parser.add_argument("--param2", type=float, default=45.0, help="霍夫累加器阈值（默认 45）")
    args = parser.parse_args(argv)

    locator = HoughRingLocator(min_radius=args.min_radius, max_radius=args.max_radius, param2=args.param2)
    if args.camera is not None:
        return _run_camera(locator, args.camera)
    if args.image is not None:
        return _run_image(locator, args.image)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

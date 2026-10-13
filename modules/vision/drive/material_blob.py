"""按颜色找物料顶面：颜色分割 → 连通域 → 等效圆半径。

霍夫圆在两种场景下都会失效：夹爪的圆弧与物料顶面尺度相近（误判），以及背景
纹理多、候选几十个的时候（漏判、挑错圆）。物料的顶面是**已知颜色的单色圆面**，
把它当颜色块找比当圆找稳得多：夹爪、背景、线材、结构件根本进不了掩膜。

同一台机器上实测（1280×720，焦距按 1012 px 粗标定，顶面按 30 mm）：

===========  ==========  ==========  ==========
画面          霍夫        颜色块      卷尺
===========  ==========  ==========  ==========
7 cm 机位     70.1 mm     71.0 mm     ≈70 mm
12 cm 机位     98–113 mm   123.8 mm    ≈120 mm
===========  ==========  ==========  ==========

而且颜色块是**每帧都稳**的：同一距离下连续帧的等效半径只差 0.3 px（约 0.2%），
霍夫则在几十个候选之间跳。半径用**面积等效半径** ``sqrt(area/π)``：顶面被夹爪
挡住一角时它仍然稳，外接圆半径会被挡住的那一角带偏。

命令行自测::

    python -m modules.vision.drive.material_blob     # 合成画面自测，不需要摄像头
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modules.vision.drive.hough_lock import LockedRegion, RingLock


def _cv2():
    """Import and return ``cv2`` lazily."""
    import cv2
    return cv2


#: 掩膜形态学去噪/补洞的核大小（像素）：先开运算去散点，再闭运算补回顶面高光打出的洞。
DEFAULT_OPEN_PX = 5
DEFAULT_CLOSE_PX = 9

#: 小于这么多像素的色块不当物料（噪点、反光、背景上一小块同色）。
DEFAULT_MIN_AREA_PX = 500


class ColourBlobLocator:
    """按参考颜色分割出物料顶面，输出圆心与**面积等效**像素半径。

    与 :class:`~modules.vision.drive.hough_lock.HoughRingLocator` 接口一致
    （``locate(frame, min_radius=..., max_radius=...)`` 返回 :class:`RingLock`），
    所以 ``MaterialLocator`` 换定位器不用改流程。

    Parameters:
        reference_bgr: 物料顶面的参考颜色（BGR），例如 ``(165, 252, 114)``。
        tolerance: 颜色容差（BGR 欧氏距离），默认 60——同一盏灯下同一块料的实测散布
            远小于它，而深色台面、白色夹爪离它有 150 以上。
        min_area_px: 小于这个面积的色块丢掉。
        min_radius / max_radius: 像素半径窗口；给定时按**等效半径**过滤，
            用来挡住"远处的同色物体"（例如更远处另一块料）。
    """

    def __init__(
        self,
        reference_bgr: Sequence[float],
        *,
        tolerance: float = 60.0,
        min_area_px: float = DEFAULT_MIN_AREA_PX,
        min_radius: float | None = None,
        max_radius: float | None = None,
        open_px: int = DEFAULT_OPEN_PX,
        close_px: int = DEFAULT_CLOSE_PX,
    ) -> None:
        if len(tuple(reference_bgr)) != 3:
            raise ValueError("reference_bgr must be a three-element [B, G, R] colour")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if min_area_px < 1:
            raise ValueError("min_area_px must be positive")
        if min_radius is not None and min_radius < 1:
            raise ValueError("min_radius must be positive")
        if max_radius is not None and max_radius < 1:
            raise ValueError("max_radius must be positive")
        if min_radius is not None and max_radius is not None and max_radius <= min_radius:
            raise ValueError("max_radius must exceed min_radius")
        self.reference_bgr = tuple(float(channel) for channel in reference_bgr)
        self.tolerance = float(tolerance)
        self.min_area_px = float(min_area_px)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.open_px = int(open_px)
        self.close_px = int(close_px)

    # ------------------------------------------------------------------ 颜色
    def mask(self, frame: np.ndarray) -> np.ndarray:
        """返回顶面的二值掩膜（``uint8``，255 表示在容差内）。"""
        cv2 = _cv2()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame is empty")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be a BGR image")
        difference = frame[:, :, :3].astype(np.float32) - np.asarray(
            self.reference_bgr, dtype=np.float32
        )[None, None, :]
        distance = np.sqrt((difference**2).sum(axis=2))
        mask = (distance <= self.tolerance).astype(np.uint8) * 255
        if self.open_px >= 3:
            kernel = np.ones((self.open_px, self.open_px), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if self.close_px >= 3:
            kernel = np.ones((self.close_px, self.close_px), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ------------------------------------------------------------------ 定位
    def locate(
        self,
        frame: np.ndarray,
        *,
        min_radius: int | None = None,
        max_radius: int | None = None,
    ) -> RingLock:
        """找出所有够大的同色块，按 x 从左到右返回。"""
        cv2 = _cv2()
        low = self.min_radius if min_radius is None else float(min_radius)
        high = self.max_radius if max_radius is None else float(max_radius)
        if low is not None and high is not None and high <= low:
            raise ValueError("max_radius must exceed min_radius")

        mask = self.mask(frame)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        regions = []
        for index in range(1, count):
            area = float(stats[index, cv2.CC_STAT_AREA])
            if area < self.min_area_px:
                continue
            radius = float(np.sqrt(area / np.pi))
            if low is not None and radius < low:
                continue
            if high is not None and radius > high:
                continue
            regions.append(LockedRegion(float(centroids[index][0]), float(centroids[index][1]), radius))
        regions.sort(key=lambda region: region.center_x)
        return RingLock(tuple(regions))

    def locate_regions(
        self,
        frame: np.ndarray,
        *,
        min_radius: int | None = None,
        max_radius: int | None = None,
    ) -> list[LockedRegion]:
        """返回色块列表（接口与霍夫定位器一致）。"""
        return list(self.locate(frame, min_radius=min_radius, max_radius=max_radius).regions)

    def debug_image(self, frame: np.ndarray):
        """把掩膜画成三通道图，方便看"到底分割到了什么"。"""
        return _cv2().cvtColor(self.mask(frame), _cv2().COLOR_GRAY2BGR)


# ---------------------------------------------------------------------- 自测
def synthetic_scene(width: int = 640, height: int = 480):
    """一块绿色顶面 + 深色台面 + 白色夹爪圆弧 + 同色小噪点。"""
    cv2 = _cv2()
    frame = np.full((height, width, 3), (40, 70, 55), dtype=np.uint8)
    cv2.circle(frame, (300, 240), 90, (114, 252, 165), -1)          # 物料顶面
    cv2.ellipse(frame, (300, 240), (120, 100), 0, 100, 260, (250, 250, 250), 18)  # 夹爪
    cv2.circle(frame, (560, 120), 6, (114, 252, 165), -1)           # 同色噪点
    return frame


def _self_test() -> int:
    frame = synthetic_scene()
    locator = ColourBlobLocator((114.0, 252.0, 165.0))
    lock = locator.locate(frame)
    print(f"合成画面 {frame.shape[1]}x{frame.shape[0]}：检出 {len(lock)} 个同色块")
    for region in lock.regions:
        print(f"  圆心 ({region.center_x:.1f}, {region.center_y:.1f})，等效半径 {region.radius:.1f}")

    problems: list[str] = []
    if len(lock) != 1:
        problems.append(f"应当只有物料那一块（噪点太小），实际 {len(lock)} 块")
    elif abs(lock.regions[0].center_x - 300) > 4 or abs(lock.regions[0].radius - 90) > 4:
        problems.append(f"圆心/半径不对：{lock.regions[0]}")

    # 半径窗口按等效半径过滤：给一个不含 90 px 的窗口就应当什么都不留
    if len(ColourBlobLocator((114.0, 252.0, 165.0)).locate(frame, min_radius=150, max_radius=300)):
        problems.append("半径窗口外应当被过滤掉")

    # 夹爪与台面不是这个颜色，掩膜里不该出现
    mask = ColourBlobLocator((114.0, 252.0, 165.0)).mask(frame)
    if int((mask > 0).sum()) > 3 * np.pi * 90**2:
        problems.append("掩膜面积远超顶面，说明夹爪或台面被算进去了")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：颜色块提取、噪点过滤与半径窗口都正确。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按颜色定位物料顶面")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

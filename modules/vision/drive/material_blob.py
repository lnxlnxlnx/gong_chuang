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
from typing import Mapping, Sequence

import numpy as np

from modules.vision.drive.hough_lock import LockedRegion, RingLock


def _cv2():
    """Import and return ``cv2`` lazily."""
    import cv2
    return cv2


def normalize_references(value) -> tuple[tuple[float, float, float], ...]:
    """把「一个颜色」「一串颜色」「``{类别名: 颜色}``」统一成颜色列表。

    物料有六种颜色，换一块料就要换参考色，所以参考色天生是一个**调色板**：
    传单个三元组仍然照旧（向后兼容），传字典就是按类别名登记的那种。
    """
    if isinstance(value, Mapping):
        items = list(value.values())
    elif isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(channel, (int, float)) and not isinstance(channel, bool)
        for channel in value
    ):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError("reference colours must be a [B, G, R] colour or a list of them")
    if not items:
        raise ValueError("at least one reference colour is required")
    references = []
    for item in items:
        try:
            channels = tuple(float(channel) for channel in item)
        except (TypeError, ValueError) as error:
            raise ValueError("each reference colour must be a three-element [B, G, R] colour") from error
        if len(channels) != 3:
            raise ValueError("each reference colour must be a three-element [B, G, R] colour")
        references.append(channels)
    return tuple(references)


#: 掩膜形态学去噪/补洞的核大小（像素）：先开运算去散点，再闭运算补回顶面高光打出的洞。
DEFAULT_OPEN_PX = 5
DEFAULT_CLOSE_PX = 9

#: 小于这么多像素的色块不当物料（噪点、反光、背景上一小块同色）。
DEFAULT_MIN_AREA_PX = 500

#: 边界起伏上限：轮廓点到形心距离的变异系数（标准差 / 均值）。
#:
#: 调色板里有蓝色，画面里的蓝色 3D 打印件也会命中同一颜色——光靠颜色分不开。
#: 但物料顶面是个**圆**，结构件不是。实测边界起伏：物料 0.02（7 cm）/ 0.11（蓝色）/
#: 0.17（12 cm，顶面外圈有台阶），蓝色打印件 0.38–0.51。取 0.25 两边都留 1.5 倍余量。
DEFAULT_MAX_BOUNDARY_VARIATION = 0.25

#: 填充率下限：等效半径 / 外接圆半径。
#:
#: 边界起伏只能用在**完整入画**的色块上；贴着画面边缘的打印件量不出起伏，
#: 就得靠这一项：物料顶面的填充率是 0.87–0.96，被画面切掉一半的打印件只有 0.46。
DEFAULT_MIN_FILL_RATIO = 0.75


class ColourBlobLocator:
    """按参考颜色分割出物料顶面，输出圆心与**面积等效**像素半径。

    与 :class:`~modules.vision.drive.hough_lock.HoughRingLocator` 接口一致
    （``locate(frame, min_radius=..., max_radius=...)`` 返回 :class:`RingLock`），
    所以 ``MaterialLocator`` 换定位器不用改流程。

    Parameters:
        reference_bgr: 参考颜色。可以是单个 ``(B, G, R)``、一组颜色，或
            ``{"green": (165, 252, 114), "blue": (248, 122, 41)}`` 这样的调色板——
            换物料颜色时只改配置，不用改代码。
        tolerance: 颜色容差（BGR 欧氏距离），默认 60——同一盏灯下同一块料的实测散布
            远小于它，而深色台面、白色夹爪、异色结构件离它有 150 以上。
        min_area_px: 小于这个面积的色块丢掉。
        min_radius / max_radius: 像素半径窗口；给定时按**等效半径**过滤，
            用来挡住"远处的同色物体"（例如更远处另一块料）。
        max_boundary_variation: 边界起伏上限（见 :data:`DEFAULT_MAX_BOUNDARY_VARIATION`）；
            用来把"同色的结构件"和"同色的物料顶面"分开。色块贴着画面边缘时跳过这一项，
            因为被画面切掉的一角本来就会让起伏变大。
        min_fill_ratio: 填充率下限（见 :data:`DEFAULT_MIN_FILL_RATIO`）；贴边的色块
            量不出起伏，靠这一项兜住。
    """

    def __init__(
        self,
        reference_bgr,
        *,
        tolerance: float = 60.0,
        min_area_px: float = DEFAULT_MIN_AREA_PX,
        min_radius: float | None = None,
        max_radius: float | None = None,
        max_boundary_variation: float | None = DEFAULT_MAX_BOUNDARY_VARIATION,
        min_fill_ratio: float | None = DEFAULT_MIN_FILL_RATIO,
        open_px: int = DEFAULT_OPEN_PX,
        close_px: int = DEFAULT_CLOSE_PX,
    ) -> None:
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
        if max_boundary_variation is not None and max_boundary_variation <= 0:
            raise ValueError("max_boundary_variation must be positive or None")
        if min_fill_ratio is not None and not 0 < min_fill_ratio <= 1:
            raise ValueError("min_fill_ratio must be within (0, 1] or None")
        self.references = normalize_references(reference_bgr)
        self.reference_bgr = self.references[0]
        self.tolerance = float(tolerance)
        self.min_area_px = float(min_area_px)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_boundary_variation = max_boundary_variation
        self.min_fill_ratio = min_fill_ratio
        self.open_px = int(open_px)
        self.close_px = int(close_px)

    # ------------------------------------------------------------------ 颜色
    def mask_for(self, frame: np.ndarray, reference) -> np.ndarray:
        """某一个参考色的二值掩膜（``uint8``，255 表示在容差内）。"""
        cv2 = _cv2()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame is empty")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be a BGR image")
        difference = frame[:, :, :3].astype(np.float32) - np.asarray(
            reference, dtype=np.float32
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

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """整个调色板的掩膜并集（调试图用）。"""
        cv2 = _cv2()
        combined = None
        for reference in self.references:
            current = self.mask_for(frame, reference)
            combined = current if combined is None else cv2.bitwise_or(combined, current)
        return combined

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

        found: list[LockedRegion] = []
        for reference in self.references:
            mask = self.mask_for(frame, reference)
            count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            for index in range(1, count):
                area = float(stats[index, cv2.CC_STAT_AREA])
                if area < self.min_area_px:
                    continue
                radius = float(np.sqrt(area / np.pi))
                if low is not None and radius < low:
                    continue
                if high is not None and radius > high:
                    continue
                if not self._is_round(mask, stats[index], centroids[index]):
                    continue
                found.append(
                    LockedRegion(float(centroids[index][0]), float(centroids[index][1]), radius)
                )

        # 调色板里两个颜色相近时同一个物体会被找到两次：按半径从大到小去重，
        # 圆心落在已保留区域内的（即同一个物体的小圈）丢掉，保留物体真正的边界。
        kept: list[LockedRegion] = []
        for region in sorted(found, key=lambda item: item.radius, reverse=True):
            if any(other.contains(region.center_x, region.center_y) for other in kept):
                continue
            kept.append(region)
        kept.sort(key=lambda region: region.center_x)
        return RingLock(tuple(kept))

    def locate_regions(
        self,
        frame: np.ndarray,
        *,
        min_radius: int | None = None,
        max_radius: int | None = None,
    ) -> list[LockedRegion]:
        """返回色块列表（接口与霍夫定位器一致）。"""
        return list(self.locate(frame, min_radius=min_radius, max_radius=max_radius).regions)

    def blob_shape(self, mask: np.ndarray, stats, centroid) -> tuple[float, float | None]:
        """色块的 ``(填充率, 边界起伏)``；贴边的色块测不准起伏，第二项返回 ``None``。"""
        cv2 = _cv2()
        x = int(stats[cv2.CC_STAT_LEFT]); y = int(stats[cv2.CC_STAT_TOP])
        width = int(stats[cv2.CC_STAT_WIDTH]); height = int(stats[cv2.CC_STAT_HEIGHT])
        area = float(stats[cv2.CC_STAT_AREA])
        clipped = (
            x <= 1 or y <= 1
            or x + width >= mask.shape[1] - 1 or y + height >= mask.shape[0] - 1
        )
        crop = mask[y:y + height, x:x + width]
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0, None
        contour = max(contours, key=cv2.contourArea)
        _centre, enclosing = cv2.minEnclosingCircle(contour)
        fill = float(np.sqrt(area / np.pi) / enclosing) if enclosing > 0 else 0.0
        if clipped:
            return fill, None
        points = contour.reshape(-1, 2).astype(np.float64)
        points[:, 0] += x
        points[:, 1] += y
        distances = np.hypot(points[:, 0] - centroid[0], points[:, 1] - centroid[1])
        mean = float(distances.mean())
        variation = float(distances.std() / mean) if mean > 0 else None
        return fill, variation

    def boundary_variation(self, mask: np.ndarray, stats, centroid) -> float | None:
        """只要边界起伏那一项（自测与排查用）。"""
        return self.blob_shape(mask, stats, centroid)[1]

    def _is_round(self, mask: np.ndarray, stats, centroid) -> bool:
        """色块够不够圆（同色的结构件会被这一步挡掉）。"""
        if self.max_boundary_variation is None and self.min_fill_ratio is None:
            return True
        fill, variation = self.blob_shape(mask, stats, centroid)
        if self.min_fill_ratio is not None and fill < self.min_fill_ratio:
            return False
        if (
            self.max_boundary_variation is not None
            and variation is not None
            and variation > self.max_boundary_variation
        ):
            return False
        return True

    def debug_image(self, frame: np.ndarray):
        """把掩膜画成三通道图，方便看"到底分割到了什么"。"""
        return _cv2().cvtColor(self.mask(frame), _cv2().COLOR_GRAY2BGR)


# ---------------------------------------------------------------------- 自测
def synthetic_scene(width: int = 640, height: int = 480):
    """一块绿色顶面 + 深色台面 + 白色夹爪圆弧 + 同色小噪点 + 同色异形结构件。

    结构件画成"掏了一块的打印件"而不是正方形：正方形在"边界起伏"这个指标上其实
    很接近圆（实测 0.12），真实打印件（实测 0.42）才分得开。
    """
    cv2 = _cv2()
    frame = np.full((height, width, 3), (40, 70, 55), dtype=np.uint8)
    cv2.circle(frame, (300, 240), 90, (114, 252, 165), -1)          # 物料顶面
    cv2.ellipse(frame, (300, 240), (120, 100), 0, 100, 260, (250, 250, 250), 18)  # 夹爪
    cv2.circle(frame, (560, 120), 6, (114, 252, 165), -1)           # 同色噪点
    cv2.rectangle(frame, (60, 40), (210, 120), (114, 252, 165), -1)  # 同色结构件
    cv2.rectangle(frame, (100, 70), (170, 120), (40, 70, 55), -1)    # 掏空一块
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

    # 调色板：多给几个颜色不该把物料找成好几块（同物体按半径去重）
    palette = ColourBlobLocator({"green": (114.0, 252.0, 165.0), "blue": (248.0, 122.0, 41.0)})
    palette_lock = palette.locate(frame)
    if len(palette_lock) != 1:
        problems.append(f"调色板模式应当还是 1 块，实际 {len(palette_lock)} 块")
    try:
        normalize_references({})
    except ValueError:
        pass
    else:
        problems.append("空调色板必须报错")

    # 同色的异形结构件不是物料：边界起伏/填充率把它挡掉（两项都关掉时它就会被找出来）
    loose = ColourBlobLocator(
        (114.0, 252.0, 165.0), max_boundary_variation=None, min_fill_ratio=None
    ).locate(frame)
    if len(loose) != 2:
        problems.append(f"关掉圆度检查时结构件应当也被找到（2 块），实际 {len(loose)} 块")

    # 贴着画面边缘的色块测不准起伏，只能放过（不然物料一贴边就丢）
    edge_frame = np.full((200, 200, 3), (40, 70, 55), dtype=np.uint8)
    _cv2().rectangle(edge_frame, (0, 60), (60, 140), (114, 252, 165), -1)
    if not ColourBlobLocator((114.0, 252.0, 165.0)).locate(edge_frame).regions:
        problems.append("贴边的色块不该因为测不准起伏而被丢掉")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：颜色块提取、噪点过滤、半径窗口与调色板去重都正确。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按颜色定位物料顶面")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

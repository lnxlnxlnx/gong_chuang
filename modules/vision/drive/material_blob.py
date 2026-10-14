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
#:
#: 开运算取 15 是有讲究的：画面里常有**同色的细长结构**——红料旁边就是红色线材，
#: 掩膜会把料和线连成一块，填充率掉到 0.53、边界起伏涨到 0.38，于是真料被判掉。
#: 实测 5 px 连在一起、11 px 分开、15 px 起就是干净的圆（fill 0.96 / var 0.04）。
#: 代价是半径小于 15 px 的同色块会消失，而这个尺度本来也测不准距离。
DEFAULT_OPEN_PX = 15
DEFAULT_CLOSE_PX = 9

#: 送进 ``cv2.minEnclosingCircle`` 的轮廓点上限（超出就等间隔抽样）。
#:
#: OpenCV 的 ``minEnclosingCircle`` 点数一多就急剧变慢（本机实测，同一条轮廓：
#: 3862 点 826 ms、1931 点 110 ms、966 点 14.5 ms、483 点 2.1 ms，大致是三次方），
#: 而它在这里只用来算"等效半径 / 外接圆半径"这一个比值。抽到几百个点以后，
#: 外接圆半径与全量结果的差只有 0.15%（实测 431.61 对 432.27 px），而填充率判据
#: 两边留的余量是**阈值 0.65 对实测的 0.74–0.96（物料）/ 0.36–0.53（同色结构件）**：
#: 0.15% 挪不动任何一次判定。这一步是颜色块路径上最贵的一处（实测一帧 826 ms
#: 全花在它上面），所以必须治。
ENCLOSING_MAX_POINTS = 256

#: 小于这么多像素的色块不当物料（噪点、反光、背景上一小块同色）。
DEFAULT_MIN_AREA_PX = 500

#: 边界起伏上限：轮廓点到形心距离的变异系数（标准差 / 均值）。
#:
#: 调色板里有蓝色，画面里的蓝色 3D 打印件也会命中同一颜色——光靠颜色分不开。
#: 但物料顶面是个**圆**，结构件不是。实测边界起伏：物料 0.02（7 cm）/ 0.11（蓝色）/
#: 0.17–0.31（顶面被光照分成亮暗两块时，掩膜边界会毛糙），蓝色打印件 0.38–0.51。
#: 取 0.35：真顶面最差的一帧 0.31 仍有富余，打印件最"圆"的一帧 0.38 也还被挡住。
DEFAULT_MAX_BOUNDARY_VARIATION = 0.35

#: 填充率下限：等效半径 / 外接圆半径。
#:
#: 边界起伏只能用在**完整入画**的色块上；贴着画面边缘、或者顶面有一半落在阴影里
#: （掩膜只剩亮的那半）时，就得靠这一项。实测：物料顶面 0.74–0.96，
#: 同色的打印件/结构件 0.36–0.53。取 0.65 两边都留 15% 以上余量。
DEFAULT_MIN_FILL_RATIO = 0.65

#: 色相容差（OpenCV 的 0–179 标度）。
#:
#: 参考色是**在某一盏灯下**取的，换了光照、搬了机位，同一个物料的 BGR 会整片平移：
#: 实测同一块绿料在两种光照下是 (165,252,114) 与 (130,204,48)，BGR 欧氏距离 88——
#: 早就超出容差 60，于是"换了个位置就识别不到"。但这两者的**色相**只差 5
#: （71 与 76），而台面那种暗绿差 14 以上。所以按色相匹配、按亮度设下限。
DEFAULT_HUE_TOLERANCE = 10.0

#: 亮度下限：候选像素的 V 至少是参考色 V 的这个倍数。
#: 台面与物料色相接近，差的就是明暗（实测 V 85 对 204–252），取 0.65 分开两边。
DEFAULT_VALUE_RATIO = 0.65


def _decimate_contour(contour, limit: int = ENCLOSING_MAX_POINTS):
    """等间隔抽样轮廓点，专供 ``cv2.minEnclosingCircle`` 用。

    见 :data:`ENCLOSING_MAX_POINTS`：那一步的耗时随点数急剧增长（三次方量级），
    而外接圆半径只用来算填充率，抽样几百个点已经足够。
    """
    count = len(contour)
    if count <= limit:
        return contour
    stride = count // limit + 1
    return np.ascontiguousarray(contour[::stride])


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
        hue_tolerance: float = DEFAULT_HUE_TOLERANCE,
        value_ratio: float = DEFAULT_VALUE_RATIO,
        min_area_px: float = DEFAULT_MIN_AREA_PX,
        min_radius: float | None = None,
        max_radius: float | None = None,
        max_boundary_variation: float | None = DEFAULT_MAX_BOUNDARY_VARIATION,
        min_fill_ratio: float | None = DEFAULT_MIN_FILL_RATIO,
        open_px: int = DEFAULT_OPEN_PX,
        close_px: int = DEFAULT_CLOSE_PX,
    ) -> None:
        if tolerance <= 0 or hue_tolerance <= 0:
            raise ValueError("tolerance and hue_tolerance must be positive")
        if not 0 < value_ratio <= 1:
            raise ValueError("value_ratio must be within (0, 1]")
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
        self.hue_tolerance = float(hue_tolerance)
        self.value_ratio = float(value_ratio)
        self.min_area_px = float(min_area_px)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_boundary_variation = max_boundary_variation
        self.min_fill_ratio = min_fill_ratio
        self.open_px = int(open_px)
        self.close_px = int(close_px)

    # ------------------------------------------------------------------ 颜色
    def mask_for(self, frame: np.ndarray, reference) -> np.ndarray:
        """某一个参考色的二值掩膜（``uint8``，255 表示算这块颜色）。

        两条**或**关系：BGR 距离在 ``tolerance`` 内（同一盏灯下的原样命中），或者
        色相在 ``hue_tolerance`` 内且亮度不低于参考色 V 的 ``value_ratio`` 倍
        （换了光照、搬了机位以后仍然认得出）。
        """
        cv2 = _cv2()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame is empty")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be a BGR image")
        reference = np.asarray(reference, dtype=np.float32)
        difference = frame[:, :, :3].astype(np.float32) - reference[None, None, :]
        distance = np.sqrt((difference**2).sum(axis=2))
        mask = (distance <= self.tolerance).astype(np.uint8) * 255

        hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HSV)
        reference_hsv = cv2.cvtColor(
            np.uint8([[np.clip(reference, 0, 255)]]), cv2.COLOR_BGR2HSV
        )[0, 0]
        hue_delta = np.abs(hsv[:, :, 0].astype(np.int16) - int(reference_hsv[0]))
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        value_floor = self.value_ratio * float(reference_hsv[2])
        similar = (hue_delta <= self.hue_tolerance) & (hsv[:, :, 2] >= value_floor)
        mask[similar] = 255

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
        _centre, enclosing = cv2.minEnclosingCircle(_decimate_contour(contour))
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


def sample_reference(frame: np.ndarray, box) -> tuple[float, float, float] | None:
    """从检测框里估出这块料的参考色（``box`` 是 ``(x1, y1, x2, y2)``）。

    取框**中心那半块**像素的中位色：模型的框一般贴着一块料，中心基本就是料本身，
    边角上才有台面/背景。这样"换一块料"就不必先去量颜色，模型认得出就够。
    参考色的亮度不影响匹配（匹配只看色相 + 亮度下限），所以阴影里的暗红也照样能用。
    """
    cv2 = _cv2()
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("frame is empty")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    centre = frame[
        y1 + (y2 - y1) // 4: y2 - (y2 - y1) // 4,
        x1 + (x2 - x1) // 4: x2 - (x2 - x1) // 4,
        :3,
    ]
    if centre.size < 3 * 16:
        centre = frame[y1:y2, x1:x2, :3]
    if centre.size < 3 * 16:
        return None
    return tuple(float(value) for value in np.median(centre.reshape(-1, 3), axis=0))


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

    # 从检测框里估参考色：换一块料不必先量颜色
    sampled = sample_reference(synthetic_scene(), (210, 150, 390, 330))
    if sampled is None or max(abs(a - b) for a, b in zip(sampled, (114.0, 252.0, 165.0))) > 2:
        problems.append(f"从框中心估出来的参考色不对：{sampled}")
    if sample_reference(frame, (0, 0, 3, 3)) is not None:
        problems.append("太小的框应当估不出参考色")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：颜色块提取、噪点过滤、半径窗口、调色板去重与参考色取样都正确。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按颜色定位物料顶面")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

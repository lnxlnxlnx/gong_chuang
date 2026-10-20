"""环内数字与环心坐标：把"环"接成机械臂能直接对准的读数。

本模块有两个入口，对应两种取景方式，**别混用**：

- :class:`NumberedRingLocator` —— 宽视野，三个环同时在画面里（固定相机/整排工位）。
  数字靠左右顺序（``ring_digits.order``），适合"工位摆正了没有"那种场合；
- :class:`RingTargetLocator` —— **窄视野，一次只看到一个环**（相机装在臂上时通常如此）。
  不做"选出三个"，只挑出画面里最像环的那一个，给它的环心相机坐标。

## 它补的是哪一段

``ring_station`` 只回答"环在哪"（像素）与"相对参考偏了多少"（差量），两者都不是
机械臂要的东西。机械臂要的是**位置**：这个环的圆心现在在相机系的哪一点。

换算与 :class:`~modules.vision.reports.RegionReport` 同一套坐标——原点在光轴、
x 向右、y 向下、单位毫米——所以可以直接喂给 ``align_xy(reading=...)``::

    target = vision.ring_center()          # 视野里那个环的 (x_mm, y_mm)
    if target is not None:
        robot.grasp.align_xy(reading=target)

相机装在臂上（eye-in-hand），``align_xy`` 本来就把任意平面读数伺服到爪子正下方，
机械臂那边一行新代码都不用写。

## 为什么还要核验"环里有没有数字"

视野里那个环**可能是空环**（换纸、还没摆好）。空环的圆心照样能算出来，但按它去对准
等于朝一个"不是工位"的位置动，而动作本身看起来完全正常。所以每个环都做一次**墨迹
核验**：环心取内圆，二值化后看墨迹占比与字形高度，两者都过门槛才算"这个环里确实有数字"。

核验不过时环心读数**仍然给出来**（机械臂可能只是要跟住那个圆圈），但
``digit_present=False`` 会如实标在结果里，调用方可以据此拒绝——不替你决定，也不假装
它没发生。

## 单位与坐标系（别和工位偏差混用）

- 本模块的 ``x_mm / y_mm`` 是**相机系位置**：原点在光轴、x 向右、y 向下、单位毫米；
- :class:`~modules.vision.drive.ring_station.RingStationDetection` 的 ``error_x_mm``
  是**相对参考圆心的位移**，是差量，不是位置，不要拿它当环心坐标。

## 比例尺只有一种来源

只用 ``station.mm_per_px``（台面各向同性比例尺）。没配就**不报毫米**（报 ``None``），
不拿 ``calibration.json`` 的焦距凑：那两者描述的是同一个台面，凑出来只会多一套尺度，
而错一个倍数看起来完全正常。``station.image_to_station_affine`` 在这里**不用**——
它映射到工位坐标系，机械臂要的是相机坐标系。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

from modules.vision.config import (
    RingDigitSettings,
    StationDetectionSettings,
    StationGeometrySettings,
)
from modules.vision.drive.ring_station import (
    RingStationDetection,
    RingStationDetector,
    cluster_observations,
    extract_observations,
    median,
)


def normalize_digit(digit: object) -> int:
    """校验"要几号环"：只接受从 1 开始的整数。

    写错就**当场报错**，而不是当成"画面里没有它"：``want=0`` 那种写法如果被静默吞掉，
    现场只会看到"一直识别不到"，查半天才发现是参数写错了（与 ``normalize_want`` 同理）。
    """
    if isinstance(digit, bool) or not isinstance(digit, int):
        raise ValueError(f"环号要是整数（1/2/3），收到 {digit!r}")
    if digit < 1:
        raise ValueError("环号从 1 开始（0 和负数指不到任何环）")
    return digit


@dataclass(frozen=True)
class NumberedRing:
    """一个环：它是几号、环心在哪，以及这个环里到底有没有数字。"""

    #: 环上标着的数字（``ring_digits.order`` 给的，不是识别出来的）。
    digit: int
    #: 在工位检测器的 ``centers_px`` 里的序号（1..3），与 ``ring1_side`` 一致。
    ring_index: int
    #: 环心（像素）。
    center_px: Tuple[float, float]
    #: 环的等效半径（像素）。
    radius_px: float
    #: 环心在**相机系**的毫米坐标；没有比例尺时是 ``None``。
    x_mm: Optional[float]
    y_mm: Optional[float]
    #: 环心内圆里是否检测到墨迹（数字）。
    digit_present: bool
    #: 内圆里的墨迹占比与字形高度占比（诊断"为什么没认成有数字"）。
    ink_ratio: float
    glyph_ratio: float
    #: 三圆环检测的置信度。
    confidence: float

    @property
    def usable(self) -> bool:
        """能不能拿去对准：环里有数字、毫米坐标也算出来了。"""
        return self.digit_present and self.x_mm is not None and self.y_mm is not None

    def reading(self) -> Optional[Tuple[float, float]]:
        """相机系 ``(x_mm, y_mm)``；不可用时 ``None``。"""
        if not self.usable:
            return None
        return (float(self.x_mm), float(self.y_mm))

    def summary(self) -> str:
        where = (
            f"xyz=({self.x_mm:+.1f},{self.y_mm:+.1f})mm"
            if self.x_mm is not None and self.y_mm is not None
            else "无毫米（缺比例尺）"
        )
        state = "有数字" if self.digit_present else "**没有数字**"
        return (
            f"环 {self.digit}（第 {self.ring_index} 个）："
            f"@({self.center_px[0]:.0f},{self.center_px[1]:.0f}) r={self.radius_px:.0f}px "
            f"{where}，{state}（墨迹 {self.ink_ratio:.1%}、字形 {self.glyph_ratio:.1%}）"
        )


@dataclass(frozen=True)
class NumberedRings:
    """一帧的"带数字的环"：三个环、它们各自的数字与相机系环心。"""

    detected: bool
    #: 三个环，按**左→右**排列（纸上的顺序）。
    rings: Tuple[NumberedRing, ...] = ()
    #: 这一帧的工位检测结果（像素、偏角、偏差都在里面）。
    station: Optional[RingStationDetection] = None
    #: 左→右的数字顺序（来自 ``ring_digits.order``）。
    order: Tuple[int, ...] = ()
    #: 环心像素 → 毫米用的比例尺；``None`` 表示没配，毫米字段不可用。
    scale_mm_per_px: Optional[float] = None
    #: 人话说明：为什么没检出、为什么没有毫米。
    detail: str = ""

    @property
    def scale_known(self) -> bool:
        return self.scale_mm_per_px is not None

    def ring(self, digit: int) -> Optional[NumberedRing]:
        """按数字取环；取不到或那个环不可用时返回 ``None``（**不回退**到别的环）。"""
        wanted = normalize_digit(digit)
        for item in self.rings:
            if item.digit == wanted and item.usable:
                return item
        return None

    def reading(self, digit: int) -> Optional[Tuple[float, float]]:
        """按数字取相机系 ``(x_mm, y_mm)``，可直接喂给 ``align_xy(reading=...)``。"""
        item = self.ring(digit)
        return None if item is None else item.reading()

    def summary(self) -> str:
        if not self.detected:
            return f"带数字的环：未识别到（{self.detail}）"
        lines = [f"带数字的环：{len(self.rings)} 个（左→右 {self.order}）"]
        if not self.scale_known:
            lines.append(f"  注意：{self.detail}")
        for item in self.rings:
            lines.append("  " + item.summary())
        return "\n".join(lines)


class NumberedRingLocator:
    """一帧 BGR → 三个环各自是几号、环心在相机系的毫米坐标。

    本类**不拥有相机**，与 ``RingStationDetector`` 一样是"给一帧就还一个读数"，
    所以能在没有硬件的合成帧上完整测一遍。

    Parameters:
        station: 三圆环检测器；``None`` 用内置默认配置。
        geometry: 真实尺寸与比例尺（``station`` 块）；``None`` 用默认值。
        digits: 数字顺序与墨迹判据（``ring_digits`` 块）；``None`` 用默认值。
    """

    def __init__(
        self,
        *,
        station: Optional[RingStationDetector] = None,
        geometry: Optional[StationGeometrySettings] = None,
        digits: Optional[RingDigitSettings] = None,
    ) -> None:
        self.station = station if station is not None else RingStationDetector()
        self.geometry = geometry if geometry is not None else StationGeometrySettings()
        self.digits = digits if digits is not None else RingDigitSettings()

    # ------------------------------------------------------------- 主流程
    def detect(self, frame: np.ndarray) -> NumberedRings:
        """识别一帧：三个环的顺序、数字与环心毫米。"""
        if frame is None:
            raise ValueError("frame 不能是 None")
        detection = self.station.detect(frame)
        if not detection.detected:
            return NumberedRings(
                detected=False,
                station=detection,
                detail=f"三圆环未识别：{detection.detail}",
            )

        order = self._left_to_right(detection.centers_px)
        digits = self._digit_order(len(order))
        scale = self.geometry.mm_per_px
        gray = _grayscale(frame)
        height, width = gray.shape[:2]

        rings = []
        for position, index in enumerate(order):
            center = detection.centers_px[index]
            diameter = (
                detection.diameters_px[index]
                if index < len(detection.diameters_px)
                else 0.0
            )
            radius = max(0.0, float(diameter) * 0.5)
            ink_ratio, glyph_ratio = self._digit_metrics(gray, center, radius)
            present = self._digit_present(ink_ratio, glyph_ratio)
            x_mm, y_mm = self._to_millimetres(center, width, height, scale)
            rings.append(
                NumberedRing(
                    digit=digits[position],
                    ring_index=index + 1,
                    center_px=(float(center[0]), float(center[1])),
                    radius_px=radius,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    digit_present=present,
                    ink_ratio=ink_ratio,
                    glyph_ratio=glyph_ratio,
                    confidence=detection.confidence,
                )
            )

        detail = (
            f"三个环心已按 {scale:.4f} mm/px 换算到相机系"
            if scale is not None
            else "没有比例尺（station.mm_per_px）：只给像素，给不出毫米"
        )
        return NumberedRings(
            detected=True,
            rings=tuple(rings),
            station=detection,
            order=tuple(digits),
            scale_mm_per_px=None if scale is None else float(scale),
            detail=detail,
        )

    # ------------------------------------------------------------- 内部
    def _left_to_right(self, centers) -> "list[int]":
        """按 x 排序的下标（左→右）；x 相同再按 y，保证顺序稳定。"""
        return sorted(
            range(len(centers)), key=lambda index: (centers[index][0], centers[index][1])
        )

    def _digit_order(self, count: int) -> Tuple[int, ...]:
        """左→右的数字；配置里的数字比环少时**报错**，不自己编一个。

        编一个出来会变成"配置漏了一个数字，代码补了一个看起来正常的号"——那种错误在
        现场没有任何表现，只有对准到空位时才发现。
        """
        order = tuple(self.digits.order)
        if len(order) < count:
            raise ValueError(
                f"ring_digits.order 只有 {len(order)} 个数字，但这一帧检出了 {count} 个环；"
                "请把三个数字都配上"
            )
        return order[:count]

    def _digit_present(self, ink_ratio: float, glyph_ratio: float) -> bool:
        return digit_present(ink_ratio, glyph_ratio, self.digits)

    def _digit_metrics(
        self, gray: np.ndarray, center, radius: float
    ) -> Tuple[float, float]:
        return digit_metrics(gray, center, radius, self.digits)

    def _to_millimetres(
        self, center, width: int, height: int, scale: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """像素 → 相机系毫米：以画面中心为光轴，与 ``RegionReport`` 同一套坐标。"""
        if scale is None:
            return (None, None)
        x_mm = (float(center[0]) - width / 2.0) * float(scale)
        y_mm = (float(center[1]) - height / 2.0) * float(scale)
        return (x_mm, y_mm)


def _grayscale(frame: np.ndarray) -> np.ndarray:
    """BGR → 灰度；本来就是单通道就原样返回。"""
    if frame.ndim == 2:
        return frame
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def digit_metrics(
    gray: np.ndarray, center, radius: float, digits: RingDigitSettings
) -> Tuple[float, float]:
    """环心内圆的墨迹占比与字形高度占比（两个入口共用同一份实现）。

    内圆半径取环半径的 ``inner_ratio``（默认 0.55）：环本身的虚线在半径处，
    取太大就会把环的笔画算成"数字"。墨迹按灰度阈值判（默认 < 100）。
    """
    import cv2

    inner = int(round(radius * digits.inner_ratio))
    if inner < 4:
        return (0.0, 0.0)
    x = int(round(center[0]))
    y = int(round(center[1]))
    x0, x1 = max(0, x - inner), min(gray.shape[1], x + inner)
    y0, y1 = max(0, y - inner), min(gray.shape[0], y + inner)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return (0.0, 0.0)

    crop = gray[y0:y1, x0:x1]
    circle = np.zeros(crop.shape, dtype=np.uint8)
    cv2.circle(circle, (x - x0, y - y0), inner, 255, -1)
    area = float(np.count_nonzero(circle))
    if area <= 0:
        return (0.0, 0.0)
    ink = np.logical_and(crop < digits.threshold, circle > 0)
    ink_ratio = float(np.count_nonzero(ink)) / area
    rows = int(np.count_nonzero(ink.any(axis=1)))
    glyph_ratio = rows / float(max(1, 2 * inner))
    return (ink_ratio, glyph_ratio)


def digit_present(ink_ratio: float, glyph_ratio: float, digits: RingDigitSettings) -> bool:
    """墨迹占比与字形高度是否都落在"印了一个数字"的区间里。"""
    low, high = digits.ink_ratio_range
    return low <= ink_ratio <= high and glyph_ratio >= digits.min_glyph_ratio


def _distance_to_centre(center, width: int, height: int) -> float:
    """环心到画面中心（光轴）的距离，像素。"""
    return float(np.hypot(center[0] - width / 2.0, center[1] - height / 2.0))


def innermost_circle_diameter(
    observations, gap: float
) -> Tuple[float, Tuple[float, ...]]:
    """候选环里**最内那一圈**的像素直径，以及所有直径带（供现场核对）。

    一条真实圆圈会给出**一组同直径**的观测（内外两道边 + 断续的虚线段）；
    数字的轮廓也会被 ``fitEllipse`` 拟成一个"圆"，但它是**单独一条**，凑不出第二个
    同直径的观测。所以"至少有两条观测的直径带"才是"一圈圆"——按这个把字迹剔掉，
    再取最内的一圈当比例尺基准。

    返回 ``(最内圈的直径, 各直径带的中位直径)``；一个候选都没有时返回 ``(0.0, ())``。
    """
    diameters = sorted(float(item.diameter_px) for item in observations)
    if not diameters:
        return (0.0, ())
    bands: "list[list[float]]" = []
    for value in diameters:
        if bands and abs(value - bands[-1][-1]) <= max(0.0, float(gap)):
            bands[-1].append(value)
        else:
            bands.append([value])
    medians = tuple(float(median(band)) for band in bands)
    for band in bands:
        if len(band) >= 2:
            return (float(median(band)), medians)
    # 所有带都只有一条观测：没有可信的"圈"，退回最内那条（调用方可能在诊断里看出来）。
    return (diameters[0], medians)


@dataclass(frozen=True)
class RingTarget:
    """**视野里的那一个环**：圆心在哪、能不能直接拿去对准。

    与 :class:`NumberedRing` 的区别是适用场景：那个要三个环同时在画面里（宽视野、
    固定相机）；这个只要求画面里有**一个**环——相机视野装不下整排时就用它。
    """

    #: 环心（像素）。
    center_px: Tuple[float, float]
    #: 环的等效半径（像素）：取候选环的**中位直径**，用于"环心内圆"的墨迹核验。
    radius_px: float
    #: 用来定比例尺的那一圈的像素直径（默认是**最内圈**）。
    reference_diameter_px: float
    #: 环心在**相机系**的毫米坐标；没有比例尺时是 ``None``。
    x_mm: Optional[float]
    y_mm: Optional[float]
    #: 环心内圆里是否检测到墨迹（数字）。
    digit_present: bool
    ink_ratio: float
    glyph_ratio: float
    #: 这个候选环由几条椭圆观测支撑、跨几个直径带（证据强度）。
    support: int
    diameter_bands: int
    #: 圆心是否落在画面外（只看到一段弧）：那时圆心是**外推**出来的。
    partial: bool
    #: 所有直径带的中位直径（诊断用：现场可以核对哪一圈是 53 mm）。
    band_diameters_px: Tuple[float, ...] = ()
    #: 本帧实际用的比例尺（毫米/像素）与它的来源（``ring diameter`` / ``depth/focal`` /
    #: ``station.mm_per_px`` / ``none``）。
    scale_mm_per_px: Optional[float] = None
    scale_source: str = ""
    #: 人话说明：选的是哪一个、为什么。
    detail: str = ""

    @property
    def usable(self) -> bool:
        """能不能拿去对准：毫米坐标算出来了。"""
        return self.x_mm is not None and self.y_mm is not None

    def reading(self) -> Optional[Tuple[float, float]]:
        """相机系 ``(x_mm, y_mm)``，可直接喂给 ``align_xy(reading=...)``。"""
        if not self.usable:
            return None
        return (float(self.x_mm), float(self.y_mm))

    def summary(self) -> str:
        where = (
            f"xyz=({self.x_mm:+.1f},{self.y_mm:+.1f})mm"
            if self.usable
            else "无毫米（缺比例尺）"
        )
        state = "有数字" if self.digit_present else "**没有数字**"
        shape = "**圆心在画面外（外推）**" if self.partial else "圆心在画面内"
        scale = (
            f"比例尺 {self.scale_mm_per_px:.4f} mm/px（{self.scale_source}）"
            if self.scale_mm_per_px is not None
            else "无比例尺"
        )
        return (
            f"环心 @({self.center_px[0]:.0f},{self.center_px[1]:.0f}) "
            f"r={self.radius_px:.0f}px 基准圈 {self.reference_diameter_px:.0f}px {where}，"
            f"{state}，{shape}（{scale}）"
            f"（支撑 {self.support} 条 / {self.diameter_bands} 个直径带）"
        )


class RingTargetLocator:
    """一帧 BGR → **视野里那一个环**的环心（像素 + 相机系毫米）。

    用三圆环检测的同一套候选（``extract_observations`` + ``cluster_observations``），
    但**不做"选出三个"那一步**：视野里只有一个环时，那一步必然失败——而"只看到一个环"
    本来就是正常取景，不是错误。

    选哪一个：支撑观测最多的那个（它最完整地落在画面里）。条数相同再看直径带数量，
    还相同就取离画面中心最近的。这三个键都是"证据"，不是"猜测"。

    本类不拥有相机，能在合成帧上完整测一遍。

    Parameters:
        diameter_mm: 环**外圈**的真实直径（毫米）。配了就优先用它当比例尺：
            ``毫米/像素 = diameter_mm / 外圈直径像素``——与镜头距离无关，也不需要标定。
        focal_px / table_depth_mm: 台面的针孔标定。没配直径时用 ``table_depth_mm / focal_px``，
            它与 ``RegionReport`` 的毫米是同一个来源，环心读数与物料读数同一套单位。
        geometry: 以上都没有时的兜底（``station.mm_per_px``）。
    """

    def __init__(
        self,
        *,
        geometry: Optional[StationGeometrySettings] = None,
        digits: Optional[RingDigitSettings] = None,
        detection: Optional[StationDetectionSettings] = None,
        focal_px: Optional[float] = None,
        table_depth_mm: Optional[float] = None,
        diameter_mm: Optional[float] = None,
    ) -> None:
        self.geometry = geometry if geometry is not None else StationGeometrySettings()
        self.digits = digits if digits is not None else RingDigitSettings()
        self.detection = detection if detection is not None else StationDetectionSettings()
        self.focal_px = focal_px
        self.table_depth_mm = table_depth_mm
        self.diameter_mm = diameter_mm
        #: 最近一次 ``detect`` 走到了哪一步（给真机入口打印"为什么没找到"）。
        self.last_diagnosis = ""

    def _millimetres_per_pixel(
        self, reference_diameter_px: float
    ) -> Tuple[Optional[float], str]:
        """本帧的比例尺与来源，优先级：已知直径 > 针孔标定 > 配置固定值。"""
        if self.diameter_mm is not None and reference_diameter_px > 0:
            return (
                float(self.diameter_mm) / float(reference_diameter_px),
                "ring diameter",
            )
        focal = self.focal_px
        depth = self.table_depth_mm
        if focal is not None and depth is not None and focal > 0 and depth > 0:
            return (float(depth) / float(focal), "depth/focal")
        if self.geometry.mm_per_px is not None:
            return (float(self.geometry.mm_per_px), "station.mm_per_px")
        return (None, "none")

    def detect(self, frame: np.ndarray) -> Optional[RingTarget]:
        """找出最像"视野里那个环"的候选；一个都没有时返回 ``None``。"""
        if frame is None:
            raise ValueError("frame 不能是 None")
        height, width = frame.shape[:2]
        # 单环取景时环可能占满画面，而 ``station`` 块的 ``max_major_axis_px``（默认 320）
        # 是按"整排三个 95 mm 环"定的，会把大环当成杂项丢掉。上限抬到画面短边：
        # 完整可见的环，直径不可能超过它；再大就只可能是别的边缘拼出来的。
        detection = replace(
            self.detection,
            max_major_axis_px=max(
                self.detection.max_major_axis_px, float(min(width, height))
            ),
        )
        observations = extract_observations(frame, detection)
        self.last_diagnosis = f"{len(observations)} 条椭圆观测"
        candidates = cluster_observations(observations, detection)
        if not candidates:
            self.last_diagnosis += "，0 个候选环（条数或直径带不够）"
            return None

        gray = _grayscale(frame)

        def rank(item) -> tuple:
            """先看圆心在不在画面里，再比证据，最后比离光轴多近。"""
            inside = 0.0 <= item.center[0] <= width and 0.0 <= item.center[1] <= height
            return (
                1 if inside else 0,
                item.support,
                item.diameter_bands,
                -_distance_to_centre(item.center, width, height),
            )

        chosen = max(candidates, key=rank)
        # 墨迹核验用**中位直径**那一圈（与三圆环的选法一致）；比例尺用**最内圈**：
        # 53 mm 说的是最里面那一圈，拿别的圈去比会整体差一个倍数。
        radius = max(0.0, float(chosen.diameter_px) * 0.5)
        reference_diameter, band_diameters = innermost_circle_diameter(
            chosen.observations, detection.diameter_band_gap_px
        )
        if reference_diameter <= 0:
            reference_diameter = float(chosen.diameter_px)
        ink_ratio, glyph_ratio = digit_metrics(gray, chosen.center, radius, self.digits)
        scale, scale_source = self._millimetres_per_pixel(reference_diameter)
        x_mm = y_mm = None
        if scale is not None:
            x_mm = (float(chosen.center[0]) - width / 2.0) * float(scale)
            y_mm = (float(chosen.center[1]) - height / 2.0) * float(scale)
        inside = (
            0.0 <= chosen.center[0] <= width and 0.0 <= chosen.center[1] <= height
        )
        detail = (
            f"{len(candidates)} 个候选环，取支撑最多的那个"
            if len(candidates) > 1
            else "1 个候选环"
        )
        if not inside:
            detail += "；圆心在画面外，毫米是外推的"
        if scale is None:
            detail += "；没有比例尺（ring_target.diameter_mm / depth+focal / station.mm_per_px），给不出毫米"
        else:
            detail += f"；比例尺 {scale:.4f} mm/px 来自 {scale_source}"
        self.last_diagnosis = (
            f"{len(observations)} 条椭圆观测、{len(candidates)} 个候选环，"
            f"选中支撑 {chosen.support} 条 / {chosen.diameter_bands} 个直径带；"
            f"各带直径 {[round(value) for value in band_diameters]} px"
        )
        return RingTarget(
            center_px=(float(chosen.center[0]), float(chosen.center[1])),
            radius_px=radius,
            reference_diameter_px=float(reference_diameter),
            x_mm=x_mm,
            y_mm=y_mm,
            digit_present=digit_present(ink_ratio, glyph_ratio, self.digits),
            ink_ratio=ink_ratio,
            glyph_ratio=glyph_ratio,
            support=int(chosen.support),
            diameter_bands=int(chosen.diameter_bands),
            partial=not inside,
            band_diameters_px=tuple(float(value) for value in band_diameters),
            scale_mm_per_px=None if scale is None else float(scale),
            scale_source=scale_source,
            detail=detail,
        )


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """合成三环 + 三个数字：验证顺序、毫米换算与"没有数字"的判定。

    不需要摄像头、模型，也不开窗口：``synthetic_station`` 画环，``putText`` 画数字，
    配置给一份与合成帧对得上的参考圆心与比例尺。
    """
    import cv2

    from modules.vision.config import StationReferenceSettings
    from modules.vision.drive.ring_station import synthetic_station

    problems = []
    size = (1280, 720)
    centers = ((300.0, 360.0), (640.0, 360.0), (980.0, 360.0))
    # 白纸黑环（现场就是这样）：合成帧的背景/前景反过来。
    frame = synthetic_station(
        size=size, centers=centers, diameter_px=120.0, thickness=6,
        background=255, foreground=0,
    )
    for (center_x, center_y), digit in zip(centers, "123"):
        cv2.putText(
            frame, digit, (int(center_x) - 18, int(center_y) + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4,
        )

    reference = StationReferenceSettings(ring_centers_px=centers)
    geometry = StationGeometrySettings(mm_per_px=0.5)
    result = NumberedRingLocator(
        station=RingStationDetector(geometry=geometry, reference=reference),
        geometry=geometry,
    ).detect(frame)
    if not result.detected:
        problems.append(f"合成三环应当检出：{result.detail}")
    else:
        digits = [item.digit for item in result.rings]
        if digits != [1, 2, 3]:
            problems.append(f"左→右应当是 1/2/3，实际 {digits}")
        if not all(item.digit_present for item in result.rings):
            problems.append(
                "三个环里都画了数字，应当都判成有数字："
                + "；".join(item.summary() for item in result.rings)
            )
        middle = result.reading(2)
        # 检测器给出的圆心会差零点几像素（中位直径那条观测），毫米换算后就差零点几毫米。
        if middle is None or abs(middle[0]) > 0.5 or abs(middle[1]) > 0.5:
            problems.append(f"中间环应当在光轴上 (0,0)，实际 {middle}")
        left = result.reading(1)
        if left is None or abs(left[0] + 170.0) > 0.5:
            problems.append(f"左边环应当在 x=-170mm，实际 {left}")
        print(result.summary())

    # 空环：没有数字就**不能**报成有数字，也不能给出可对准的读数
    blank = synthetic_station(
        size=size, centers=centers, diameter_px=120.0, thickness=6,
        background=255, foreground=0,
    )
    blank_result = NumberedRingLocator(
        station=RingStationDetector(geometry=geometry, reference=reference),
        geometry=geometry,
    ).detect(blank)
    if blank_result.detected and any(item.digit_present for item in blank_result.rings):
        problems.append("空环里没有数字，不该判成有数字")
    if blank_result.reading(2) is not None:
        problems.append("空环不该给出可对准的读数")

    # 没有比例尺时：环仍然列出来，但不给毫米
    no_scale = NumberedRingLocator(
        station=RingStationDetector(reference=reference),
        # 默认配置里 mm_per_px 是有值的（0.495），所以这里必须显式清掉才叫"没配"。
        geometry=StationGeometrySettings(mm_per_px=None),
    ).detect(frame)
    if no_scale.detected and no_scale.rings and no_scale.rings[0].x_mm is not None:
        problems.append("没配比例尺时不该给出毫米坐标")
    if no_scale.detected and no_scale.reading(2) is not None:
        problems.append("没配比例尺时不该给出可对准的读数")

    # 窄视野：画面里只有一个环（相机装臂上时的常态）
    single = synthetic_station(
        size=size, centers=((640.0, 360.0),), diameter_px=200.0, thickness=8,
        background=255, foreground=0,
    )
    cv2.putText(single, "2", (622, 382), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
    target = RingTargetLocator(geometry=geometry).detect(single)
    if target is None:
        problems.append("画面里只有一个环时也应当给出环心")
    else:
        centre = target.reading()
        if centre is None or abs(centre[0]) > 1.0 or abs(centre[1]) > 1.0:
            problems.append(f"单环环心应当在光轴上，实际 {centre}")
        if not target.digit_present:
            problems.append(f"单环里画了数字，应当判成有数字：{target.summary()}")
        print(target.summary())

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：顺序 1/2/3、相机系毫米、单环环心、空环与缺比例尺都按约定处理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

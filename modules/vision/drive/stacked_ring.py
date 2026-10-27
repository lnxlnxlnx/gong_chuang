"""叠放位识别：找出中环与上下两条基准线，给出工位偏角。

## 与三圆环的区别

三圆环找的是**三个并排的环**，靠"三点共线"定方向；叠放位找的是**一个环 + 两条平行线**，
靠"线的倾角"定方向。两者的判据没有一处共用，所以是两个检测器而不是一个带开关的检测器。

两条线分别在环心**上方与下方**各 ``line_offset_mm``（默认 75 mm）处，是工位的两条基准边。
它们本该平行，所以：

- 两条都找到时，偏角取**两条线倾角的平均**——单条线的角度噪声被摊掉一半。
- 两条线的夹角超过 ``max_line_disagreement_deg``（默认 2°）就**判为不可信**：
  它们本该平行，不平行说明至少有一条认错了（台面纹理、物料边缘都可能拼出一条像线的边）。
  宁可报"没识别到"，也不要拿一条错线算出来的偏角去转底盘。
- 只找到一条时仍然出结果，但 ``line_quality`` 会低一档，调用方据此决定要不要重拍。

## 距离怎么变成像素

``line_offset_mm`` 是**毫米**，要落到画面上得先有比例尺：优先用
``station.image_to_station_affine`` 反解出的竖直尺度，否则用 ``station.mm_per_px``。
两个都没配时，用中环半径按"环外径 95 mm"反推一个临时尺度——这样至少能在没标定的
机器上跑起来看画面对不对，但结果的 ``scale_source`` 会写成 ``"ring"`` 提醒你它不可靠。

## 依赖边界

本模块用 cv2 做霍夫变换与最小二乘直线拟合（**函数内 import**，见 :func:`_cv2`），
用 numpy 做数值。不碰相机、不碰串口、不开文件，所以能完全离线测试。
**不控制底盘**：偏角算出来交给谁、要不要纠，是调用方的事。

命令行自测::

    python -m modules.vision.drive.stacked_ring                  # 合成帧自测（不需要摄像头）
    python -m modules.vision.drive.stacked_ring --image a.jpg    # 在一张图片上跑
    python -m modules.vision.drive.stacked_ring --camera 1       # 摄像头实时预览，按 q 退出
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from modules.vision.config import StationGeometrySettings, StackedRingSettings

#: 中环参考半径（像素）：中环外径 95 mm 在 1280×720 画幅下约 70 px 半径。
#: 只在"没配任何比例尺"时用来临时反推尺度，见模块文档的 ``scale_source == "ring"``。
DEFAULT_MIDDLE_RADIUS_PX = 70.0

#: 尺度从哪来。前两个是标定结果，第三个是临时反推的，**不可信**。
SCALE_FROM_AFFINE = "affine"
SCALE_FROM_MM_PER_PX = "mm_per_px"
SCALE_FROM_RING = "ring"
SCALE_UNKNOWN = "none"


def _cv2():
    """Import and return ``cv2`` lazily（理由见 ``ring_station._cv2``）。"""
    import cv2

    return cv2


@dataclass(frozen=True)
class ReferenceLine:
    """一条基准线：拟合出来的直线参数、它的倾角、以及有多少点支撑它。"""

    #: 直线上的一个点（像素）。
    point: Tuple[float, float]
    #: 直线方向（单位向量）。
    direction: Tuple[float, float]
    #: 倾角（度）：相对画面水平，右端抬高为正。画面 y 轴向下，所以这里已经取过反号。
    angle_deg: float
    #: 支撑这条线的边缘点数。
    points: int
    #: 线在 x 方向的跨度（像素）。跨度太短的线多半是某段纹理。
    x_span_px: float

    @property
    def slope(self) -> float:
        """``dy/dx``（画面坐标下，正数表示向右下走）。"""
        if self.direction[0] == 0:
            return math.inf
        return self.direction[1] / self.direction[0]

    def y_at(self, x: float) -> float:
        """这条线在给定 x 处的 y。"""
        if self.direction[0] == 0:
            return self.point[1]
        return self.point[1] + self.slope * (x - self.point[0])

    def summary(self) -> str:
        return (
            f"过 ({self.point[0]:.0f},{self.point[1]:.0f})、倾角 {self.angle_deg:+.2f}°，"
            f"{self.points} 点、跨 {self.x_span_px:.0f} px"
        )


@dataclass(frozen=True)
class StackedRingDetection:
    """一帧的叠放位识别结果。

    ``detected`` 为 ``False`` 时其余几何量都不可用，但 :attr:`detail` 一定有话说。
    """

    detected: bool
    #: 中环圆心（像素）；未检出时是 ``None``。
    middle_center_px: Optional[Tuple[float, float]] = None
    #: 中环像素半径。
    middle_radius_px: Optional[float] = None
    #: 中环圆周上的边缘支撑率（0–1）。
    circle_support: Optional[float] = None
    #: 上、下基准线；没找到的那条是 ``None``。
    top_line: Optional[ReferenceLine] = None
    bottom_line: Optional[ReferenceLine] = None
    #: 工位偏角（度）：两条基准线倾角的平均，右端抬高为正。
    station_yaw_deg: Optional[float] = None
    #: 两条线倾角之差（度）；只找到一条时是 ``None``。
    line_disagreement_deg: Optional[float] = None
    #: 线质量 0–1：两条都比一条高。
    line_quality: float = 0.0
    #: 综合置信度 0–1。
    confidence: float = 0.0
    #: 中环圆心相对参考圆心的位移（毫米）；没配尺度时是 ``None``。
    error_x_mm: Optional[float] = None
    error_y_mm: Optional[float] = None
    #: 位移的模（毫米）；没配尺度时是 ``None``。
    position_error_mm: Optional[float] = None
    #: 毫米换算用的比例尺（毫米/像素）。
    scale_mm_per_px: Optional[float] = None
    #: 尺度从哪来：``"affine"`` / ``"mm_per_px"`` / ``"ring"``（不可信）/ ``"none"``。
    scale_source: str = SCALE_UNKNOWN
    #: 中环圆心在最近若干帧里的最大抖动（像素）；样本不足时是 ``None``。
    center_jitter_px: Optional[float] = None
    #: 是否已经连续若干帧稳定。
    stable: bool = False
    #: 这一帧花了多少毫秒。
    processing_ms: float = 0.0
    #: 人话说明。
    detail: str = ""

    @property
    def line_count(self) -> int:
        """找到几条基准线（0、1 或 2）。"""
        return int(self.top_line is not None) + int(self.bottom_line is not None)

    @property
    def scale_known(self) -> bool:
        """有没有可用的尺度（``"ring"`` 也算有，但那是临时反推的）。"""
        return self.scale_source != SCALE_UNKNOWN

    @property
    def scale_trustworthy(self) -> bool:
        """尺度是不是标定出来的。``False`` 时毫米读数只能当参考。"""
        return self.scale_source in (SCALE_FROM_AFFINE, SCALE_FROM_MM_PER_PX)

    def summary(self) -> str:
        if not self.detected:
            return f"叠放位：未识别到（{self.detail}）"
        center = self.middle_center_px or (0.0, 0.0)
        parts = [
            f"叠放位：中环 ({center[0]:.1f}, {center[1]:.1f}) px、半径 "
            f"{self.middle_radius_px:.1f} px",
            f"偏角 {self.station_yaw_deg:+.2f}°" if self.station_yaw_deg is not None else "偏角未知",
            f"基准线 {self.line_count} 条",
        ]
        if self.line_disagreement_deg is not None:
            parts.append(f"两线夹角 {self.line_disagreement_deg:.2f}°")
        if self.scale_known:
            parts.append(
                f"偏差 ({self.error_x_mm:+.2f}, {self.error_y_mm:+.2f}) mm"
                f"（尺度来自 {self.scale_source}）"
            )
        else:
            parts.append("未配尺度（只有像素，没有毫米）")
        parts.append(f"置信度 {self.confidence:.2f}")
        if self.stable:
            parts.append("已稳定")
        return "，".join(parts)


class StackedRingDetector:
    """一帧 BGR 画面 → 中环圆心、上下基准线与工位偏角。

    本类**不拥有相机、不控制底盘**：给一帧就还一个读数。

    Parameters:
        stacked: 叠放位的检测参数与参考（``config.json`` 的 ``stacked_ring`` 块）。
        geometry: 三圆环那一块的**真实尺寸与比例尺**（``config.json`` 的 ``station`` 块）。
            基准线的毫米偏移要落到像素上就得有比例尺，所以这里复用 ``station`` 的标定，
            而不是在 ``stacked_ring`` 里再配一份——同一台机器只该有一个比例尺。
        middle_reference_px: 中环参考圆心（像素）；``None`` 表示用 ``stacked`` 里的值。
    """

    def __init__(
        self,
        *,
        stacked: Optional[StackedRingSettings] = None,
        geometry: Optional[StationGeometrySettings] = None,
        middle_reference_px: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.stacked = stacked if stacked is not None else StackedRingSettings()
        self.geometry = geometry if geometry is not None else StationGeometrySettings()
        self.middle_reference_px = (
            middle_reference_px if middle_reference_px is not None else self.stacked.middle_reference_px
        )
        if self.stacked.middle_min_radius_px >= self.stacked.middle_max_radius_px:
            raise ValueError("middle_min_radius_px must be smaller than middle_max_radius_px")
        if self.stacked.line_offset_mm <= 0:
            raise ValueError("line_offset_mm must be positive")
        if self.stacked.stable_frame_count < 1:
            raise ValueError("stable_frame_count must be a positive integer")
        if self.stacked.min_line_points < 2:
            raise ValueError("min_line_points must be at least 2")
        self._history: list = []

    # ------------------------------------------------------------- 生命周期
    def reset(self) -> None:
        """清掉稳定窗口。换机位、重新摆工位之后必须调用。"""
        self._history = []

    def close(self) -> None:
        """Release anything held; safe to call twice and safe when nothing was opened."""
        self._history = []

    # --------------------------------------------------------------- 主算
    def detect(self, frame: np.ndarray) -> StackedRingDetection:
        """在一帧 BGR 图上找叠放位。

        :raises ValueError: ``frame`` 为空、或不是三通道 BGR。
        """
        started = time.perf_counter()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame is empty")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR image")

        cv2 = _cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        middle = self._detect_middle_ring(gray)
        if middle is None:
            self._history = []
            return StackedRingDetection(
                detected=False,
                processing_ms=(time.perf_counter() - started) * 1000.0,
                detail="没有找到中环（半径窗口内没有边缘支撑够高的圆）",
            )

        center, radius, support = middle
        scale, scale_source = self._resolve_scale(radius)
        lines = self._detect_reference_lines(gray, center, scale)

        top = lines.get("top")
        bottom = lines.get("bottom")
        yaw, disagreement = _combine_lines(top, bottom)
        quality = _line_quality(top, bottom, self.stacked)

        if yaw is None:
            # 一条基准线都没有：中环找到了也没法定方向，如实报未检出。
            self._history = []
            return StackedRingDetection(
                detected=False,
                middle_center_px=center,
                middle_radius_px=radius,
                circle_support=support,
                line_quality=0.0,
                scale_mm_per_px=scale,
                scale_source=scale_source,
                processing_ms=(time.perf_counter() - started) * 1000.0,
                detail="找到中环，但上下基准线一条都没找到（定不了方向）",
            )

        confidence = _confidence(support, quality, disagreement, self.stacked)
        jitter, stable = self._update_stability(center)
        error_x, error_y, position_error = self._position_error(center, scale)
        detail = (
            f"中环支撑 {support:.2f}，基准线 {int(top is not None) + int(bottom is not None)} 条，"
            f"线质量 {quality:.2f}"
        )
        if disagreement is not None and disagreement > self.stacked.max_line_disagreement_deg:
            detail += f"（两线夹角 {disagreement:.2f}° 超过上限，偏角不可信）"

        return StackedRingDetection(
            detected=True,
            middle_center_px=center,
            middle_radius_px=radius,
            circle_support=support,
            top_line=top,
            bottom_line=bottom,
            station_yaw_deg=yaw,
            line_disagreement_deg=disagreement,
            line_quality=quality,
            confidence=confidence,
            error_x_mm=error_x,
            error_y_mm=error_y,
            position_error_mm=position_error,
            scale_mm_per_px=scale,
            scale_source=scale_source,
            center_jitter_px=jitter,
            stable=stable,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            detail=detail,
        )

    #: 别名：``stream.run_preview`` 约定用 ``analyze``。
    analyze = detect

    def draw(
        self,
        frame: np.ndarray,
        detection: StackedRingDetection,
        *,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with the middle ring and the two lines marked.

        画面上只写 ASCII（``cv2.putText`` 画不出中文），中文结论走终端。**不改输入帧**。
        """
        cv2 = _cv2()
        annotated = frame.copy()
        if detection.middle_center_px is not None and detection.middle_radius_px is not None:
            point = (
                int(round(detection.middle_center_px[0])),
                int(round(detection.middle_center_px[1])),
            )
            cv2.circle(annotated, point, int(round(detection.middle_radius_px)), color, thickness)
            cv2.circle(annotated, point, 3, (0, 255, 255), -1)

        for line, tint in ((detection.top_line, (255, 128, 0)), (detection.bottom_line, (255, 128, 0))):
            if line is None:
                continue
            half = int(round(0.5 * line.x_span_px)) + 40
            start = (int(round(line.point[0])) - half, int(round(line.y_at(line.point[0] - half))))
            end = (int(round(line.point[0])) + half, int(round(line.y_at(line.point[0] + half))))
            cv2.line(annotated, start, end, tint, thickness)

        yaw = "?" if detection.station_yaw_deg is None else f"{detection.station_yaw_deg:+.2f}deg"
        cv2.putText(annotated, f"yaw={yaw}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if detection.detected else (0, 0, 255), thickness)
        cv2.putText(annotated, f"lines={detection.line_count}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), thickness)
        return annotated

    # ------------------------------------------------------------- 内部
    def _detect_middle_ring(self, gray):
        """在参考圆心附近的窗口里找中环，返回 ``(圆心, 半径, 边缘支撑率)`` 或 ``None``。

        先 ``HoughCircles`` 出候选，再对每个候选量"圆周上有多少比例的角度真的落在边缘上"
        （:func:`edge_support`），取支撑率最高且过门槛的那个。

        **为什么不直接信霍夫的票数**：霍夫对同一个环的内外两条边会各投一次，
        票数高不代表这个圆真的贴合边缘；而支撑率是拿检测到的圆去**回头验**，
        这是"像不像一个环"的直接读数。
        """
        cv2 = _cv2()
        height, width = gray.shape[:2]
        window = self.stacked.middle_search_radius_px
        if window is None or self.middle_reference_px is None:
            offset_x, offset_y = 0, 0
            roi = gray
        else:
            center_x, center_y = self.middle_reference_px
            x0 = max(0, int(round(center_x - window)))
            y0 = max(0, int(round(center_y - window)))
            x1 = min(width, int(round(center_x + window)))
            y1 = min(height, int(round(center_y + window)))
            if x1 - x0 < 10 or y1 - y0 < 10:
                return None
            offset_x, offset_y = x0, y0
            roi = gray[y0:y1, x0:x1]

        blurred = cv2.medianBlur(roi, 5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(20.0, 1.5 * self.stacked.middle_min_radius_px),
            param1=100.0,
            param2=20.0,
            minRadius=int(round(self.stacked.middle_min_radius_px)),
            maxRadius=int(round(self.stacked.middle_max_radius_px)),
        )
        if circles is None:
            return None

        edges = cv2.Canny(blurred, 50.0, 150.0)
        best = None
        best_support = 0.0
        for circle_x, circle_y, radius in circles[0]:
            center = (float(circle_x) + offset_x, float(circle_y) + offset_y)
            # 离参考圆心太远的候选直接丢：中环不会跑到工位外面去。
            if self.middle_reference_px is not None:
                if math.hypot(
                    center[0] - self.middle_reference_px[0], center[1] - self.middle_reference_px[1]
                ) > self.stacked.middle_max_reference_distance_px:
                    continue
            support = edge_support(edges, (float(circle_x), float(circle_y)), float(radius))
            if support > best_support:
                best_support = support
                best = (center, float(radius), support)
        if best is None or best_support < self.stacked.middle_edge_support:
            return None
        return best

    def _resolve_scale(self, radius_px: float) -> Tuple[Optional[float], str]:
        """定出"一个像素算几毫米"，并如实报告它是从哪来的。

        优先仿射矩阵的竖直尺度（工位被斜着看时横竖不一样），其次各向同性 ``mm_per_px``，
        最后才用中环半径临时反推——那一条**不可信**，只为了在没标定的机器上也能跑起来。
        """
        affine = self.geometry.image_to_station_affine
        if affine is not None:
            (_, _), (c, d) = affine
            vertical = math.hypot(c, d)
            if vertical > 0:
                return vertical, SCALE_FROM_AFFINE
        if self.geometry.mm_per_px is not None:
            return self.geometry.mm_per_px, SCALE_FROM_MM_PER_PX
        if radius_px > 0:
            # 中环外径 95 mm -> 半径 47.5 mm；用它除以实测像素半径就是临时尺度。
            return (0.5 * self.geometry.outer_diameter_mm) / radius_px, SCALE_FROM_RING
        return None, SCALE_UNKNOWN

    def _detect_reference_lines(self, gray, center, scale) -> dict:
        """在中环上下各 ``line_offset_mm`` 处找基准线，返回 ``{"top": .., "bottom": ..}``。

        比例尺未知时**不猜偏移**：那条线该在哪都不知道，搜出来的东西没有意义，
        所以直接返回空字典，让上层如实报"定不了方向"。
        """
        if scale is None:
            return {}
        offset_px = self.stacked.line_offset_mm / scale
        found = {}
        for name, sign in (("top", -1.0), ("bottom", +1.0)):
            target_y = center[1] + sign * offset_px
            line = _find_line_near(gray, target_y, self.stacked)
            if line is not None:
                found[name] = line
        return found

    def _position_error(self, center, scale):
        """中环圆心相对参考圆心的毫米位移；没配参考或尺度时全是 ``None``。"""
        if scale is None or self.middle_reference_px is None:
            return None, None, None
        dx = (center[0] - self.middle_reference_px[0]) * scale
        dy = (center[1] - self.middle_reference_px[1]) * scale
        return dx, dy, math.hypot(dx, dy)

    def _update_stability(self, center) -> Tuple[Optional[float], bool]:
        """滑动窗口里中环圆心的最大抖动；样本不足时 ``(None, False)``。"""
        window = self.stacked.stable_frame_count
        self._history.append(center)
        if len(self._history) > window:
            self._history = self._history[-window:]
        if len(self._history) < window:
            return None, False
        xs = [entry[0] for entry in self._history]
        ys = [entry[1] for entry in self._history]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        worst = max(math.hypot(x - mean_x, y - mean_y) for x, y in zip(xs, ys))
        return worst, worst <= self.stacked.maximum_center_jitter_px


# ------------------------------------------------------------------ 纯算法
def edge_support(
    edges: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    *,
    rays: int = 360,
    tolerance_px: float = 4.0,
) -> float:
    """圆周上有多少比例的角度真的落在边缘上（0–1）。

    这是"这个圆像不像一个真环"的直接读数：拿检测到的圆回头验边缘。
    半径为 0 时返回 0（不是一个圆），而不是抛错——调用方正在筛候选，
    一个退化候选只是"不合格"，不是程序错误。
    """
    if radius <= 0 or edges is None or getattr(edges, "size", 0) == 0:
        return 0.0
    height, width = edges.shape[:2]
    hits = 0
    for index in range(rays):
        angle = 2.0 * math.pi * index / rays
        x = int(round(center[0] + radius * math.cos(angle)))
        y = int(round(center[1] + radius * math.sin(angle)))
        if not (0 <= x < width and 0 <= y < height):
            continue
        half = int(round(tolerance_px))
        x0, x1 = max(0, x - half), min(width, x + half + 1)
        y0, y1 = max(0, y - half), min(height, y + half + 1)
        if edges[y0:y1, x0:x1].any():
            hits += 1
    return hits / float(rays)


def _find_line_near(gray: np.ndarray, target_y: float, settings: StackedRingSettings):
    """在 ``target_y`` 附近找一条近似水平的直线，找不到返回 ``None``。

    做法：``Canny`` 出边缘 → 取目标 y 上下一个带 → 对带内的边缘点用
    ``cv2.fitLine(DIST_HUBER)`` 做**抗差**最小二乘拟合。用 ``DIST_HUBER`` 而不是默认的
    ``DIST_L2``：带内总会混进别的边缘（物料轮廓、台面纹理），L2 会被它们拽偏，
    Huber 只让它们有一点发言权。
    """
    cv2 = _cv2()
    height, width = gray.shape[:2]
    half_band = max(6.0, settings.line_offset_mm * 0.35)
    y0 = max(0, int(round(target_y - half_band)))
    y1 = min(height, int(round(target_y + half_band)))
    if y1 - y0 < 4:
        return None

    edges = cv2.Canny(gray, 50.0, 150.0)
    band = edges[y0:y1, :]
    ys, xs = np.nonzero(band)
    if xs.size < settings.min_line_points:
        return None
    x_span = float(xs.max() - xs.min())
    if x_span < settings.min_line_x_span_px:
        return None

    points = np.column_stack([xs.astype(np.float32), (ys + y0).astype(np.float32)])
    direction = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    vx, vy, x0_line, y0_line = (float(direction[0]), float(direction[1]),
                                float(direction[2]), float(direction[3]))
    length = math.hypot(vx, vy)
    if length <= 0:
        return None
    vx, vy = vx / length, vy / length
    if vx < 0:
        # 统一朝向：方向向量的符号是拟合出来的副产品，统一到 +x 才不会让倾角差 180°。
        vx, vy = -vx, -vy
    # 画面 y 轴向下，所以取反号才是"右端抬高为正"。
    angle_deg = math.degrees(math.atan2(-vy, vx))
    if abs(angle_deg) > settings.max_line_angle_deg:
        return None
    return ReferenceLine(
        point=(x0_line, y0_line),
        direction=(vx, vy),
        angle_deg=angle_deg,
        points=int(xs.size),
        x_span_px=x_span,
    )


def _combine_lines(top, bottom) -> Tuple[Optional[float], Optional[float]]:
    """把两条线收成一个偏角 ``(偏角, 两线夹角)``。

    两条都有 → 取平均（角度噪声摊掉一半），夹角如实报出来；
    只有一条 → 就用它，夹角是 ``None``（不是 0——"只有一条线"和"两条完全平行"是两回事）。
    一条都没有 → ``(None, None)``。
    """
    if top is not None and bottom is not None:
        disagreement = abs(top.angle_deg - bottom.angle_deg)
        return 0.5 * (top.angle_deg + bottom.angle_deg), disagreement
    only = top if top is not None else bottom
    if only is None:
        return None, None
    return only.angle_deg, None


def _line_quality(top, bottom, settings: StackedRingSettings) -> float:
    """线质量 0–1：**两条线永远不比一条线差**。

    单条线固定 0.5；两条线从 1.0 随夹角线性降到 0.5，夹角到上限时正好等于"只有一条线"。

    刻意**不**让两条线掉到 0.5 以下：两个线不平行时"至少找到了一条线"这个事实并没有
    变坏，不该被罚得比"根本没找到第二条"还惨。真正要表达"夹角太大所以结论不可信"的
    地方是 :func:`_confidence`（那里对超限直接打五折）和结果里的 ``detail``，
    不是这里——一个质量指标同时承担"有多少证据"和"证据有多可信"两件事，
    调用方就没法从它读出"是不是只看到一条线"了。
    """
    if top is None and bottom is None:
        return 0.0
    if top is None or bottom is None:
        return 0.5
    disagreement = abs(top.angle_deg - bottom.angle_deg)
    limit = settings.max_line_disagreement_deg
    return 1.0 - 0.5 * min(1.0, disagreement / limit)


def _confidence(
    support: float, quality: float, disagreement: Optional[float], settings: StackedRingSettings
) -> float:
    """综合置信度 0–1：中环支撑占 0.4，线质量占 0.6。

    线占得比环多，是因为**方向完全由线决定**：环只给位置。两条线都不平行的时候，
    中环找得再准，偏角也是错的。
    """
    score = 0.4 * max(0.0, min(1.0, support)) + 0.6 * max(0.0, min(1.0, quality))
    if disagreement is not None and disagreement > settings.max_line_disagreement_deg:
        score *= 0.5
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------- 自测
def synthetic_stacked(
    *,
    size: Tuple[int, int] = (1280, 720),
    center: Tuple[float, float] = (640.0, 360.0),
    radius_px: float = 70.0,
    offset_px: float = 150.0,
    yaw_deg: float = 0.0,
    line_half_span_px: float = 400.0,
    line_thickness: int = 3,
    ring_thickness: int = 5,
    draw_lines: bool = True,
) -> np.ndarray:
    """画一张合成叠放位图：黑底上一个中环 + 上下两条基准线。

    ``yaw_deg`` 是线的倾角，**右端抬高为正**——和 :attr:`ReferenceLine.angle_deg`
    以及检测结果的 ``station_yaw_deg`` 是同一套符号。画面 y 轴向下，所以画的时候
    要取反号（``+yaw`` 意味着右端的 y 更小）。
    """
    cv2 = _cv2()
    width, height = size
    frame = np.full((height, width, 3), 0, dtype=np.uint8)
    # 取反号：画面里 y 越小越靠上，所以"右端抬高"= 右端 y 更小。
    slope = -math.tan(math.radians(yaw_deg))
    if draw_lines:
        for sign in (-1.0, 1.0):
            mid_y = center[1] + sign * offset_px
            start_x = int(round(center[0] - line_half_span_px))
            end_x = int(round(center[0] + line_half_span_px))
            start = (start_x, int(round(mid_y + slope * (start_x - center[0]))))
            end = (end_x, int(round(mid_y + slope * (end_x - center[0]))))
            cv2.line(frame, start, end, (255, 255, 255), line_thickness)
    cv2.circle(
        frame,
        (int(round(center[0])), int(round(center[1]))),
        int(round(radius_px)),
        (255, 255, 255),
        ring_thickness,
    )
    return frame


def _self_test() -> int:
    problems: list[str] = []

    # 合成图按 mm_per_px = 0.5 设计：偏移 75 mm 就是 150 px。
    geometry = StationGeometrySettings(mm_per_px=0.5)
    stacked = StackedRingSettings(
        middle_reference_px=(640.0, 360.0),
        middle_search_radius_px=260.0,
    )
    detector = StackedRingDetector(stacked=stacked, geometry=geometry)

    # ---- 纯函数先测 ----
    if abs(_combine_lines(None, None)[0] is not None):
        problems.append("一条线都没有时偏角必须是 None")
    fake_top = ReferenceLine((0.0, 0.0), (1.0, 0.0), 2.0, 100, 500.0)
    fake_bottom = ReferenceLine((0.0, 0.0), (1.0, 0.0), 4.0, 100, 500.0)
    yaw, disagreement = _combine_lines(fake_top, fake_bottom)
    if abs(yaw - 3.0) > 1e-9 or abs(disagreement - 2.0) > 1e-9:
        problems.append(f"两条线应平均成 3.0°、夹角 2.0°，实际 {yaw} / {disagreement}")
    single_yaw, single_disagreement = _combine_lines(fake_top, None)
    if abs(single_yaw - 2.0) > 1e-9 or single_disagreement is not None:
        problems.append("只有一条线时应直接用它的角度，夹角必须是 None（不是 0）")
    # 两条**几乎平行**线的质量必须高于只有一条；两条不平行时反过来（那是有问题的读数）。
    # 这里刻意用 0.5° 而不是刚好等于上限的 2°：夹角正好在上限时两条线的质量就等于
    # 一条线（见 _line_quality 的说明），那是设计如此，不是用来测"更高"的样本。
    near_top = ReferenceLine((0.0, 0.0), (1.0, 0.0), 2.0, 100, 500.0)
    near_bottom = ReferenceLine((0.0, 0.0), (1.0, 0.0), 2.5, 100, 500.0)
    parallel_quality = _line_quality(near_top, near_bottom, stacked)
    single_quality = _line_quality(fake_top, None, stacked)
    if parallel_quality <= single_quality:
        problems.append(
            f"两条几乎平行线的质量（{parallel_quality}）必须高于只有一条（{single_quality}）"
        )
    skew_top = ReferenceLine((0.0, 0.0), (1.0, 0.0), 0.0, 100, 500.0)
    skew_bottom = ReferenceLine((0.0, 0.0), (1.0, 0.0), 9.0, 100, 500.0)
    if _line_quality(skew_top, skew_bottom, stacked) < single_quality - 1e-9:
        problems.append("两条线再不平行也不该比只有一条线更差（质量只表达有多少证据）")
    # "夹角太大所以不可信"由置信度表达，不是由质量表达。
    if _confidence(0.8, 1.0, 9.0, stacked) >= _confidence(0.8, 1.0, 0.0, stacked):
        problems.append("两线夹角超限时置信度必须被打折")

    # ---- 退化输入必须当场报错 ----
    for bad in (None, np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8)):
        try:
            detector.detect(bad)
        except ValueError:
            pass
        else:
            problems.append("空帧 / 非三通道帧必须报 ValueError")

    # ---- 正例：中环 + 两条水平基准线 ----
    frame = synthetic_stacked()
    detection = detector.detect(frame)
    if not detection.detected:
        problems.append(f"合成叠放位应当被识别，实际未识别（{detection.detail}）")
    else:
        if detection.middle_center_px is None:
            problems.append("识别到之后应当给出中环圆心")
        else:
            got = detection.middle_center_px
            if math.hypot(got[0] - 640.0, got[1] - 360.0) > 4.0:
                problems.append(f"中环圆心应为 (640,360)，实际 {got}")
        if detection.middle_radius_px is None or abs(detection.middle_radius_px - 70.0) > 6.0:
            # 霍夫在"画出来的环"上会偏向环的外沿（实测 74 px 对真值 70 px），
            # 所以容差按几个像素给，不按亚像素给。
            problems.append(f"中环半径应约 70 px（±6），实际 {detection.middle_radius_px}")
        if detection.line_count != 2:
            problems.append(f"应当找到两条基准线，实际 {detection.line_count} 条")
        if detection.station_yaw_deg is None or abs(detection.station_yaw_deg) > 0.6:
            problems.append(f"水平摆放的偏角应接近 0°，实际 {detection.station_yaw_deg}")
        if detection.scale_source != SCALE_FROM_MM_PER_PX:
            problems.append(f"配了 mm_per_px 时尺度来源应为 mm_per_px，实际 {detection.scale_source}")
        if detection.position_error_mm is None or detection.position_error_mm > 3.0:
            problems.append(f"与参考同心时位移应接近 0 mm，实际 {detection.position_error_mm}")

    # ---- 稳定窗口 ----
    for _ in range(stacked.stable_frame_count):
        stable_detection = detector.detect(frame)
    if not stable_detection.stable:
        problems.append(f"同一位置连续 {stacked.stable_frame_count} 帧后应当判为稳定")

    # ---- 倾斜：偏角应当跟着线走 ----
    detector.reset()
    tilted = synthetic_stacked(yaw_deg=5.0)
    tilted_detection = detector.detect(tilted)
    if not tilted_detection.detected:
        problems.append("倾斜摆放的叠放位应当仍被识别")
    elif tilted_detection.station_yaw_deg is None or not (
        4.0 < tilted_detection.station_yaw_deg < 6.0
    ):
        problems.append(
            f"线倾角 5° 时偏角应约 +5°，实际 {tilted_detection.station_yaw_deg}"
        )

    # ---- 两线不平行：偏角不可信，但如实报出来 ----
    detector.reset()
    crossed = synthetic_stacked(yaw_deg=0.0)
    cv2 = _cv2()
    cv2.line(crossed, (240, 205), (1040, 245), (255, 255, 255), 3)  # 下方那条改成斜的
    crossed_detection = detector.detect(crossed)
    if crossed_detection.detected and crossed_detection.line_disagreement_deg is not None:
        if crossed_detection.line_disagreement_deg > stacked.max_line_disagreement_deg:
            if "不可信" not in crossed_detection.detail:
                problems.append("两线夹角超限时 detail 里应当说明偏角不可信")

    # ---- 平移：位移应当跟着走 ----
    detector.reset()
    shifted = synthetic_stacked(center=(660.0, 375.0))
    shifted_detection = detector.detect(shifted)
    if not shifted_detection.detected:
        problems.append("平移后的叠放位应当仍被识别")
    else:
        if shifted_detection.error_x_mm is None or shifted_detection.error_x_mm <= 0:
            problems.append("整体右移后 error_x_mm 应为正")
        if shifted_detection.error_y_mm is None or shifted_detection.error_y_mm <= 0:
            problems.append("整体下移后 error_y_mm 应为正（y 轴向下）")
        # (20, 15) px @ 0.5 mm/px = (10, 7.5) mm，模约 12.5 mm。
        if shifted_detection.position_error_mm is None or not (
            11.0 < shifted_detection.position_error_mm < 14.0
        ):
            problems.append(
                f"平移 (20,15) px、0.5 mm/px 的位移应约 12.5 mm，"
                f"实际 {shifted_detection.position_error_mm}"
            )

    # ---- 空画面：如实说没识别到，不抛异常 ----
    detector.reset()
    empty = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    if empty.detected:
        problems.append("纯黑画面不该识别出叠放位")
    if not empty.detail:
        problems.append("未识别时应当给出原因（detail）")

    # ---- 只有中环、没有基准线：定不了方向，必须报未检出 ----
    detector.reset()
    ring_only = synthetic_stacked(draw_lines=False)
    ring_only_detection = detector.detect(ring_only)
    if ring_only_detection.detected:
        problems.append("没有基准线时定不了方向，不该报检出")
    if ring_only_detection.middle_center_px is None:
        problems.append("没有基准线时仍然应当报出找到的中环（便于排查）")

    # ---- 没配尺度：用中环反推临时尺度，并如实标注来源 ----
    detector.reset()
    uncalibrated = StackedRingDetector(
        stacked=stacked,
        geometry=StationGeometrySettings(mm_per_px=None, image_to_station_affine=None),
    )
    uncalibrated_detection = uncalibrated.detect(frame)
    if uncalibrated_detection.scale_source != SCALE_FROM_RING:
        problems.append(
            f"没配尺度时应当用中环反推（来源 ring），实际 {uncalibrated_detection.scale_source}"
        )
    if uncalibrated_detection.scale_trustworthy:
        problems.append("从环反推的尺度不可信，scale_trustworthy 必须是 False")

    # ---- 非法参数 ----
    for kwargs, name in (
        ({"stacked": StackedRingSettings(middle_min_radius_px=90.0, middle_max_radius_px=60.0)},
         "中环半径窗口颠倒"),
        ({"stacked": StackedRingSettings(line_offset_mm=0.0)}, "基准线偏移为 0"),
        ({"stacked": StackedRingSettings(stable_frame_count=0)}, "稳定帧数为 0"),
        ({"stacked": StackedRingSettings(min_line_points=1)}, "直线点数下限低于 2"),
    ):
        try:
            StackedRingDetector(**kwargs)
        except ValueError:
            continue
        problems.append(f"{name} 应被拒绝，实际被接受了")

    print(f"合成叠放位：{detection.summary()}")
    print(f"倾斜 5°：  {tilted_detection.summary()}")
    print(f"平移后：   {shifted_detection.summary()}")
    print(f"只有中环： {ring_only_detection.summary()}")
    print(f"空画面：   {empty.summary()}")
    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：中环定位、双线偏角、两线平行性、毫米位移与尺度来源都正确。")
    return 0


def _run_image(detector: StackedRingDetector, image_path: Path) -> int:
    """在一张图片上跑一次；**没检出也算失败**（返回码 1）。"""
    cv2 = _cv2()
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"读不出图片：{image_path}")
        return 1
    detection = detector.detect(frame)
    print(f"{image_path}：{detection.summary()}")
    return 0 if detection.detected else 1


def _run_camera(detector: StackedRingDetector, camera_id: int) -> int:
    """摄像头实时预览；按 ``q`` 退出。"""
    import cv2

    from modules.vision.drive.camera import USBCamera, looks_blank
    from modules.vision.drive.preview import (
        BLANK_FRAME_WARNING,
        close_previews,
        open_preview,
        warn_blank_preview,
    )

    window = open_preview()
    blank_frames = 0
    warned = False
    last = None
    try:
        with USBCamera(camera_id, width=1280, height=720).initialize() as camera:
            print(f"摄像头 {camera_id}；窗口 {window!r}，按 q 退出")
            while True:
                frame = camera.read()
                if frame is None:
                    break
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned:
                    warned = True
                    warn_blank_preview(camera_id)
                last = detector.detect(frame)
                cv2.imshow(window, detector.draw(frame, last))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if last is not None:
                print(last.summary())
    finally:
        close_previews()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="叠放位识别：中环 + 上下基准线 + 工位偏角")
    parser.add_argument("--image", type=Path, help="在一张图片上跑（不给就跑合成帧自测）")
    parser.add_argument("--camera", type=int, help="打开摄像头实时预览")
    args = parser.parse_args(argv)

    if args.camera is not None or args.image is not None:
        from modules.vision.config import load_vision_config

        settings = load_vision_config()
        detector = StackedRingDetector(
            stacked=settings.stacked_ring,
            geometry=settings.station_geometry,
            middle_reference_px=settings.station_reference.ring_centers_px[1],
        )
    else:
        detector = StackedRingDetector()

    if args.camera is not None:
        return _run_camera(detector, args.camera)
    if args.image is not None:
        return _run_image(detector, args.image)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

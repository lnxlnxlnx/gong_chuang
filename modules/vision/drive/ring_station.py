"""三圆环工位识别：一帧里找出三个圆环工位，给出圆心、工位偏角与偏差。

## 这一步在整条链路上的位置

物料定位回答"料在哪"，本模块回答"**工位摆正了没有**"。它只测几何、只出数，
**不控制底盘也不控制机械臂**——偏差算出来交给谁、要不要纠、纠多少，是调用方的事。
这条边界和 ``modules.vision.drive.material_locate`` 一致：``drive`` 里全是纯算法。

## 三个圆环为什么要"选"而不是"找"

画面里的椭圆轮廓远不止三个：工位环本身有内外两圈边缘、台面上还有别的圆形结构件。
所以本模块不假设"最大的三个就是工位"，而是**先攒候选、再穷举组合、最后按几何打分**：

1. :func:`extract_observations` 把每个像环的椭圆轮廓收成一条观测（``fitEllipse``）。
2. :func:`cluster_observations` 用并查集把圆心挨得近的观测并成一个**候选环**，
   并要求它至少由若干条不同直径的观测支撑——一个真环的内外边缘会给出好几条。
3. :func:`select_triple` 穷举三环组合，要求**近似共线、跨度够大、中间环落在两端之间**，
   按共线度 / 间距均衡 / 直径一致 / 支撑数打分，取最高分。

打分而不是"取前三个"，是因为现场总会有第 4、5 个像环的东西（法兰、垫片、料盘边缘）。
判据全部写在 ``config.json`` 的 ``station`` 块里，换工位不用改代码。

## 坐标与符号（沿用仓库既有约定）

- 像素坐标：左上角原点，x 向右、y 向下。
- **工位偏角**：从环 1 圆心指向环 3 圆心的方向角，单位度，画面右方为 0、**下方为正**。
  和 :mod:`modules.vision.drive.pixel_measure` 的方向角同一套符号，转置到
  "底盘该往哪边转"时不会多一次反号。
- **偏差**：环的实测圆心相对参考圆心的位移。给 ``image_to_station_affine`` 就按它换算成
  工位系毫米（各向异性），否则按 ``mm_per_px`` 各向同性缩放。两个都没配时**只报像素**，
  毫米字段是 ``None``——不猜一个尺度出来。

## 依赖边界

本模块用 cv2 做轮廓与椭圆拟合（**函数内 import**，见 :func:`_cv2`），用 numpy 做数值。
不碰相机、不碰串口、不开文件，所以能完全离线测试。

命令行自测::

    python -m modules.vision.drive.ring_station                  # 合成帧自测（不需要摄像头）
    python -m modules.vision.drive.ring_station --image a.jpg    # 在一张图片上跑
    python -m modules.vision.drive.ring_station --camera 1       # 摄像头实时预览，按 q 退出
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from modules.vision.config import (
    StationDetectionSettings,
    StationGeometrySettings,
    StationReferenceSettings,
)

#: 一个工位由几个环组成。写成常量是因为"三个"这件事贯穿整条链路（配置、报告、验收），
#: 散在各处的字面量 3 迟早会有一处漏改。
RING_COUNT = 3

#: 自测用的合成画面尺寸。
DEFAULT_SELF_TEST_SIZE = (1280, 720)


def _cv2():
    """Import and return ``cv2`` lazily.

    模块级 import cv2 会让 ``import modules.vision.drive.ring_station`` 连带拖起整个
    OpenCV，而本模块的纯几何部分（共线度、角度折算）本该能在没装 cv2 的机器上被导入。
    这是仓库的统一做法（见 ``hough_lock`` / ``preview``）。
    """
    import cv2

    return cv2


@dataclass(frozen=True)
class RingObservation:
    """一条椭圆轮廓观测：某个像环的圆心、长短轴与支撑点数。"""

    center: Tuple[float, float]
    #: 长短轴（像素直径，不是半径）——``cv2.fitEllipse`` 给的就是直径。
    major_axis_px: float
    minor_axis_px: float
    angle_degrees: float
    points: int

    @property
    def diameter_px(self) -> float:
        """等效直径：长短轴的算术平均。"""
        return 0.5 * (self.major_axis_px + self.minor_axis_px)

    def summary(self) -> str:
        return (
            f"@({self.center[0]:.1f},{self.center[1]:.1f}) "
            f"d={self.diameter_px:.1f} px，{self.points} 点"
        )


@dataclass(frozen=True)
class RingCandidate:
    """一个候选环：由若干条观测并簇而来。"""

    center: Tuple[float, float]
    diameter_px: float
    #: 支撑它的观测条数。
    support: int
    #: 这些观测落在几个不同的直径带上（真环的内外边缘会给出多个）。
    diameter_bands: int
    observations: Tuple[RingObservation, ...] = ()

    @property
    def radius_px(self) -> float:
        """Return the ring radius in pixels."""
        return 0.5 * self.diameter_px

    def summary(self) -> str:
        return (
            f"@({self.center[0]:.1f},{self.center[1]:.1f}) d={self.diameter_px:.1f} px，"
            f"支撑 {self.support}，直径带 {self.diameter_bands}"
        )


@dataclass(frozen=True)
class RingStationDetection:
    """一帧的三圆环工位识别结果。

    字段分三组，**不要混用**：

    - **像素**：:attr:`centers_px`、:attr:`diameters_px`、:attr:`station_yaw_deg`
    - **偏差**：:attr:`error_x_mm` / :attr:`error_y_mm` / :attr:`position_error_mm`
      （``None`` 表示没配尺度，只测了像素）
    - **判定**：:attr:`within_tolerance`、:attr:`stable`、:attr:`confidence`

    ``detected`` 为 ``False`` 时其余几何量都不可用，但 :attr:`detail` 一定有话说
    （为什么没检出），所以调用方永远能打印一句人话而不是"识别失败"。
    """

    detected: bool
    #: 三个环的实测圆心（像素），顺序固定为 环1、环2（中间）、环3。
    centers_px: Tuple[Tuple[float, float], ...] = ()
    #: 三个环的等效直径（像素）。
    diameters_px: Tuple[float, ...] = ()
    #: 从环 1 指向环 3 的方向角（度），画面右方为 0、下方为正。
    station_yaw_deg: Optional[float] = None
    #: 实测偏角相对参考偏角的差（度）。
    angle_error_deg: Optional[float] = None
    #: 中间环相对参考圆心的位移（毫米）；没配尺度时是 ``None``。
    error_x_mm: Optional[float] = None
    error_y_mm: Optional[float] = None
    #: 三个环里最大的位移（毫米）；没配尺度时是 ``None``。
    position_error_mm: Optional[float] = None
    #: 各环的逐环位移（毫米），顺序同 :attr:`centers_px`。
    ring_errors_mm: Tuple[float, ...] = ()
    #: 三环共线度残差（中间环到两端连线的垂距 / 跨度）；越小越像一条直线。
    affine_residual: Optional[float] = None
    #: 间距比 ``d12 / d23``；三个环等距时接近 1。
    perspective_ratio: Optional[float] = None
    #: 综合置信度 0–1。
    confidence: float = 0.0
    #: 三环圆心在最近若干帧里的最大抖动（像素）；样本不足时是 ``None``。
    center_jitter_px: Optional[float] = None
    #: 是否已经连续若干帧稳定。
    stable: bool = False
    #: 偏差是否落在验收门槛内。
    within_tolerance: bool = False
    #: 参与打分的候选环个数（诊断"为什么没选中"用）。
    candidates: int = 0
    #: 每个环的支撑观测数。
    support: Tuple[int, ...] = ()
    #: 这一帧花了多少毫秒。
    processing_ms: float = 0.0
    #: 人话说明：没检测到时是原因，检测到时是结论。
    detail: str = ""

    @property
    def station_center_px(self) -> Optional[Tuple[float, float]]:
        """工位中心取中间那个环（环 2）的圆心——三个环等距时它就是工位几何中心。"""
        if len(self.centers_px) < RING_COUNT:
            return None
        return self.centers_px[1]

    @property
    def middle_center_px(self) -> Optional[Tuple[float, float]]:
        """中间环 R2 的像素圆心；没检出时是 ``None``。"""
        return self.station_center_px

    @property
    def scale_known(self) -> bool:
        """有没有配尺度（决定毫米字段是数还是 ``None``）。"""
        return self.position_error_mm is not None

    def summary(self) -> str:
        if not self.detected:
            return f"三圆环工位：未识别到（{self.detail}）"
        center = self.station_center_px or (0.0, 0.0)
        parts = [
            f"三圆环工位：中心 ({center[0]:.1f}, {center[1]:.1f}) px",
            f"偏角 {self.station_yaw_deg:+.2f}°" if self.station_yaw_deg is not None else "偏角未知",
        ]
        if self.scale_known:
            parts.append(
                f"偏差 ({self.error_x_mm:+.2f}, {self.error_y_mm:+.2f}) mm、"
                f"最大 {self.position_error_mm:.2f} mm"
            )
            parts.append("在容差内" if self.within_tolerance else "**超出容差**")
        else:
            parts.append("未配尺度（只有像素，没有毫米）")
        parts.append(f"置信度 {self.confidence:.2f}")
        if self.stable:
            parts.append("已稳定")
        return "，".join(parts)


class RingStationDetector:
    """一帧 BGR 画面 → 三个圆环的圆心、工位偏角与毫米偏差。

    本类**不拥有相机、不控制底盘**：给一帧就还一个读数，什么时候拍、拍完动不动由调用方
    决定。这样它能在没有硬件的情况下被合成帧完整测一遍。

    Parameters:
        geometry / detection / reference: 三个设置块，``None`` 表示用内置默认值
            （与 ``config.json`` 缺失这两个块时的行为一致）。

    检测器**有状态**（只有稳定窗口），所以同一个实例可以跨帧复用；
    :meth:`reset` 清掉稳定窗口，换机位或重新对准时调用。
    """

    def __init__(
        self,
        *,
        geometry: Optional[StationGeometrySettings] = None,
        detection: Optional[StationDetectionSettings] = None,
        reference: Optional[StationReferenceSettings] = None,
    ) -> None:
        self.geometry = geometry if geometry is not None else StationGeometrySettings()
        self.detection = detection if detection is not None else StationDetectionSettings()
        self.reference = reference if reference is not None else StationReferenceSettings()
        if self.detection.center_cluster_radius_px <= 0:
            raise ValueError("center_cluster_radius_px must be positive")
        if self.detection.min_contour_points < 5:
            # fitEllipse 至少要 5 个点；低于 5 的阈值会让每个轮廓都走到 fitEllipse 再炸。
            raise ValueError("min_contour_points must be at least 5")
        if self.detection.min_cluster_support < 1:
            raise ValueError("min_cluster_support must be a positive integer")
        if self.detection.min_diameter_bands < 1:
            raise ValueError("min_diameter_bands must be a positive integer")
        if self.reference.stable_frame_count < 1:
            raise ValueError("stable_frame_count must be a positive integer")
        if len(self.reference.ring_centers_px) != RING_COUNT:
            raise ValueError(f"ring_centers_px must hold exactly {RING_COUNT} points")
        self._history: list = []

    # ------------------------------------------------------------- 生命周期
    def reset(self) -> None:
        """清掉稳定窗口。换机位、重新摆工位之后必须调用，否则会拿旧帧的抖动做判断。"""
        self._history = []

    def close(self) -> None:
        """Release anything held; safe to call twice and safe when nothing was opened."""
        self._history = []

    # --------------------------------------------------------------- 主算
    def detect(self, frame: np.ndarray) -> RingStationDetection:
        """在一帧 BGR 图上找三圆环工位。

        :raises ValueError: ``frame`` 为空、或不是三通道 BGR。
        """
        started = time.perf_counter()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame is empty")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR image")

        observations = extract_observations(frame, self.detection)
        candidates = cluster_observations(observations, self.detection)
        triple = select_triple(candidates, self.detection, frame_width=frame.shape[1])
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if triple is None:
            # 未检出就清窗口：下一次检出时若拿旧帧比抖动，会把"刚进来"读成"很稳"。
            self._history = []
            if not observations:
                reason = "画面里没有像环的椭圆轮廓"
            elif len(candidates) < RING_COUNT:
                reason = f"{len(observations)} 条观测只并出 {len(candidates)} 个候选环"
            else:
                reason = f"{len(candidates)} 个候选环里没有共线且等距的三个"
            return RingStationDetection(
                detected=False,
                candidates=len(candidates),
                processing_ms=elapsed_ms,
                detail=reason,
            )

        return self._report(triple, candidates=len(candidates), processing_ms=elapsed_ms)

    #: 别名：``stream.run_preview`` 约定用 ``analyze``。两者是同一个方法，不是两份实现。
    analyze = detect

    def draw(
        self,
        frame: np.ndarray,
        detection: RingStationDetection,
        *,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with the three centres and the station line marked.

        画面上只写 ASCII：``cv2.putText`` 画不出中文（见 ``modules/vision/drive/preview.py``），
        中文结论一律走终端。**不改输入帧**。
        """
        cv2 = _cv2()
        annotated = frame.copy()
        if not detection.detected:
            cv2.putText(annotated, "no station", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), thickness)
            return annotated

        for index, center in enumerate(detection.centers_px, start=1):
            point = (int(round(center[0])), int(round(center[1])))
            cv2.circle(annotated, point, 6, color, thickness)
            cv2.putText(annotated, f"R{index}", (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        first, last = detection.centers_px[0], detection.centers_px[-1]
        cv2.line(
            annotated,
            (int(round(first[0])), int(round(first[1]))),
            (int(round(last[0])), int(round(last[1]))),
            (255, 128, 0),
            thickness,
        )
        yaw = "?" if detection.station_yaw_deg is None else f"{detection.station_yaw_deg:+.2f}deg"
        cv2.putText(annotated, f"yaw={yaw}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), thickness)
        if detection.scale_known:
            cv2.putText(
                annotated,
                f"err={detection.position_error_mm:.2f}mm",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if detection.within_tolerance else (0, 0, 255),
                thickness,
            )
        return annotated

    # ------------------------------------------------------------- 内部
    def _report(
        self, triple: Sequence[RingCandidate], *, candidates: int, processing_ms: float
    ) -> RingStationDetection:
        centers = tuple(candidate.center for candidate in triple)
        diameters = tuple(candidate.diameter_px for candidate in triple)

        # 环 1 -> 环 3 的方向角。取 1->3 而不是 1->2，是因为两端离得最远、
        # 同样的像素噪声摊到角度上最小。
        yaw = math.degrees(
            math.atan2(centers[2][1] - centers[0][1], centers[2][0] - centers[0][0])
        )
        angle_error = wrap_degrees(yaw - self.reference.station_yaw_deg)

        errors_px = tuple(
            (center[0] - reference[0], center[1] - reference[1])
            for center, reference in zip(centers, self.reference.ring_centers_px)
        )
        signed_mm, uniform_mm = self._to_millimetres(errors_px)

        residual, perspective = line_quality(centers)
        confidence = confidence_of(residual, perspective, triple)
        jitter, stable = self._update_stability(centers)
        within = self._within_tolerance(
            uniform_mm, angle_error, residual, perspective, confidence
        )

        detail = (
            f"三环支撑 {[candidate.support for candidate in triple]}，"
            f"共线残差 {residual:.3f}，间距比 {perspective:.3f}"
        )
        return RingStationDetection(
            detected=True,
            centers_px=centers,
            diameters_px=diameters,
            station_yaw_deg=yaw,
            angle_error_deg=angle_error,
            error_x_mm=None if signed_mm is None else signed_mm[1][0],
            error_y_mm=None if signed_mm is None else signed_mm[1][1],
            position_error_mm=None if uniform_mm is None else max(uniform_mm),
            ring_errors_mm=() if uniform_mm is None else uniform_mm,
            affine_residual=residual,
            perspective_ratio=perspective,
            confidence=confidence,
            center_jitter_px=jitter,
            stable=stable,
            within_tolerance=within,
            candidates=candidates,
            support=tuple(candidate.support for candidate in triple),
            processing_ms=processing_ms,
            detail=detail,
        )

    def _to_millimetres(self, errors_px):
        """像素位移 → 毫米。两种尺度二选一；都没配就返回 ``(None, None)``。

        返回 ``(带符号的二维位移, 各环的位移模)``。带符号的那份只取中间环（它代表
        工位整体平移），模的那份三个环都要——"有一个环偏得多"和"整体偏得多"是两回事，
        验收门槛卡的是后者。
        """
        affine = self.geometry.image_to_station_affine
        if affine is not None:
            (a, b), (c, d) = affine
            signed = tuple((a * dx + b * dy, c * dx + d * dy) for dx, dy in errors_px)
            return signed, tuple(math.hypot(*item) for item in signed)
        scale = self.geometry.mm_per_px
        if scale is not None:
            signed = tuple((dx * scale, dy * scale) for dx, dy in errors_px)
            return signed, tuple(math.hypot(*item) for item in signed)
        return None, None

    def _within_tolerance(
        self,
        uniform_mm,
        angle_error: float,
        residual: float,
        perspective: float,
        confidence: float,
    ) -> bool:
        """验收：置信度、共线度、间距比、位移、偏角**全部**要过。

        尺度未知时**不返回 True**：那时连毫米都没算出来，"在容差内"这个结论无从谈起，
        报 True 会让调用方以为可以收工了。
        """
        if uniform_mm is None:
            return False
        if confidence < self.reference.minimum_confidence:
            return False
        if residual > self.reference.maximum_affine_residual:
            return False
        # 间距比跑出区间说明选中的三个环不是工位（或透视太强）：此时位置算得再准，
        # 也是在拿一组错的目标做判断，不能收工。
        low, high = self.reference.perspective_ratio_range
        if not math.isfinite(perspective) or not low <= perspective <= high:
            return False
        if max(uniform_mm) > self.reference.maximum_position_error_mm:
            return False
        if abs(angle_error) > self.reference.maximum_angle_error_deg:
            return False
        return True

    def _update_stability(self, centers) -> Tuple[Optional[float], bool]:
        """把最近若干帧的圆心收进滑动窗口，返回 ``(最大抖动, 是否稳定)``。

        抖动定义成"窗口内每个环的圆心相对窗口均值的最大位移"，而不是"首末帧之差"——
        后者在匀速漂移时恰好是 0，会把"一直在走"读成"很稳"。
        """
        window = self.reference.stable_frame_count
        self._history.append(centers)
        if len(self._history) > window:
            self._history = self._history[-window:]
        if len(self._history) < window:
            return None, False

        worst = 0.0
        for index in range(RING_COUNT):
            xs = [entry[index][0] for entry in self._history]
            ys = [entry[index][1] for entry in self._history]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            worst = max(
                worst,
                max(math.hypot(x - mean_x, y - mean_y) for x, y in zip(xs, ys)),
            )
        return worst, worst <= self.reference.maximum_center_jitter_px


# ------------------------------------------------------------------ 纯算法
def extract_observations(
    frame: np.ndarray,
    detection: Optional[StationDetectionSettings] = None,
) -> Tuple[RingObservation, ...]:
    """把一帧里的椭圆轮廓收成观测。

    ``Canny → findContours(RETR_LIST) → fitEllipse``，然后按长短轴与扁平度过滤。
    用 ``RETR_LIST`` 而不是 ``RETR_EXTERNAL``：环的内圈是一条**独立的**轮廓，
    取外部轮廓会把内外两圈丢掉一圈，而"多圈支撑"正是把真环和杂项分开的主要依据。
    """
    cv2 = _cv2()
    settings = detection if detection is not None else StationDetectionSettings()
    if frame is None or getattr(frame, "size", 0) == 0:
        return ()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, settings.canny_low, settings.canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    observations = []
    for contour in contours:
        if len(contour) < settings.min_contour_points:
            continue
        # fitEllipse 至少要 5 个点，上面那道门槛已经保证了，这里不再重复判。
        (center_x, center_y), (major, minor), angle = cv2.fitEllipse(contour)
        if minor < settings.min_minor_axis_px or major > settings.max_major_axis_px:
            continue
        if major <= 0 or minor / major < settings.min_ellipse_aspect:
            continue
        observations.append(
            RingObservation(
                center=(float(center_x), float(center_y)),
                major_axis_px=float(major),
                minor_axis_px=float(minor),
                angle_degrees=float(angle),
                points=len(contour),
            )
        )
    return tuple(observations)


def cluster_observations(
    observations: Sequence[RingObservation],
    detection: Optional[StationDetectionSettings] = None,
) -> Tuple[RingCandidate, ...]:
    """把圆心挨得近的观测并成候选环（并查集），再按支撑条件筛。

    支撑条件有两条，缺一不可：

    - **条数** ``>= min_cluster_support``：一个真环的内外边缘会给出好几条椭圆。
    - **不同直径带** ``>= min_diameter_bands``：这一条才是真正管用的——同一条边缘被
      Canny 断成几截会产生一堆**直径相同**的碎片，条数够了但直径带只有 1 个。
      没有这一条，画面里随便一段弧都会变成"候选环"。
    """
    settings = detection if detection is not None else StationDetectionSettings()
    if not observations:
        return ()

    parent = list(range(len(observations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    radius = settings.center_cluster_radius_px
    for first in range(len(observations)):
        for second in range(first + 1, len(observations)):
            if (
                math.hypot(
                    observations[first].center[0] - observations[second].center[0],
                    observations[first].center[1] - observations[second].center[1],
                )
                <= radius
            ):
                union(first, second)

    groups: dict = {}
    for index, observation in enumerate(observations):
        groups.setdefault(find(index), []).append(observation)

    candidates = []
    for members in groups.values():
        if len(members) < settings.min_cluster_support:
            continue
        diameters = sorted(item.diameter_px for item in members)
        bands = count_bands(diameters, settings.diameter_band_gap_px)
        if bands < settings.min_diameter_bands:
            continue
        # 圆心取**中位直径那条观测**的圆心，不取所有观测的平均：
        # 内外两圈边缘的圆心本来就差一点点，平均会把圆心拽向"谁都不是"的位置
        # （实测合成三环里被拽偏 0.9 px，倾斜摆放时偏差报成 10 mm）。
        # 中位直径那一条最有代表性，取它的圆心与直径，两者来自同一次拟合、自洽。
        representative = min(members, key=lambda item: abs(item.diameter_px - median(diameters)))
        candidates.append(
            RingCandidate(
                center=representative.center,
                diameter_px=representative.diameter_px,
                support=len(members),
                diameter_bands=bands,
                observations=tuple(members),
            )
        )
    return tuple(candidates)


def select_triple(
    candidates: Sequence[RingCandidate],
    detection: Optional[StationDetectionSettings] = None,
    *,
    frame_width: Optional[int] = None,
    ring1_side: str = "left",
) -> Optional[Tuple[RingCandidate, ...]]:
    """从候选环里穷举出最像工位的三个，返回按 环1/环2/环3 排好序的三元组。

    判据（全部可配）：

    - **跨度**：两端环的距离至少占画面宽的 ``min_station_span_ratio``。三个环挤在一角
      不是工位，是料盘上的一堆垫片。
    - **共线**：中间环到两端连线的垂距 / 跨度 ``<= max_collinearity_error_ratio``。
    - **居中**：中间环在连线上的投影比例落在 ``(0.2, 0.8)``——太靠边说明选中的
      "中间那个"其实是端点，三个环的顺序就错了。
    - **排序**：按 ``ring1_side`` 决定哪一端是环 1（``"left"`` 取 x 小的一端，
      ``"right"`` 取 x 大的一端）。工位翻面装时改这一个字段，不用改代码。
    """
    settings = detection if detection is not None else StationDetectionSettings()
    if len(candidates) < RING_COUNT:
        return None

    span_floor = 0.0 if frame_width is None else settings.min_station_span_ratio * frame_width
    best = None
    best_score = -math.inf
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            for third in range(second + 1, len(candidates)):
                triple = order_triple(
                    (candidates[first], candidates[second], candidates[third]), ring1_side
                )
                # 排不出"中间那个"时这一组直接作废：把 None 交给打分函数会在那里炸掉，
                # 而"画面上只有一两个环"本来是**正常结果**（detected=False），不是异常。
                if triple is None:
                    continue
                score = score_triple(triple, settings, span_floor)
                if score is not None and score > best_score:
                    best_score = score
                    best = triple
    return best


def order_triple(triple, ring1_side: str = "left"):
    """把三个候选排成 环1 / 环2（中间）/ 环3，排不出来返回 ``None``。

    **不能只按共线残差挑中间那个**：三个环本来就近似共线，此时三个残差都是 0，
    谁先被遍历到谁就成了"中间"，于是顺序随机——实测就是这样把一个端点当成了中间环，
    间距比算出来是 2.0。真正的判据是**投影位置**：中间环在两端连线上的投影必须落在
    中段（0.2–0.8），端点则会在 0 或 1 附近。先按这一条筛，再在合格的里面取残差最小的。
    """
    best_middle = None
    best_key = (math.inf, math.inf)
    for index in range(RING_COUNT):
        others = [triple[position] for position in range(RING_COUNT) if position != index]
        first, last = others[0].center, others[1].center
        middle = triple[index].center
        span = math.hypot(last[0] - first[0], last[1] - first[1])
        if span <= 0:
            continue
        projection = (
            (middle[0] - first[0]) * (last[0] - first[0])
            + (middle[1] - first[1]) * (last[1] - first[1])
        ) / (span * span)
        # 端点会被投影到 0 或 1 附近；中间那个落在中段。
        if not 0.2 < projection < 0.8:
            continue
        residual = collinearity_residual(first, middle, last)
        # 先比残差（谁更共线），再比"谁更居中"（越接近 0.5 越像中间）。
        key = (residual, abs(projection - 0.5))
        if key < best_key:
            best_key = key
            best_middle = index
    if best_middle is None:
        return None

    ends = [triple[position] for position in range(RING_COUNT) if position != best_middle]
    # 按 ring1_side 决定哪一端是环 1。
    ends.sort(key=lambda candidate: candidate.center[0], reverse=(ring1_side == "right"))
    return (ends[0], triple[best_middle], ends[1])


def score_triple(triple, settings: StationDetectionSettings, span_floor: float):
    """给一个排好序的三元组打分；不合判据返回 ``None``。"""
    first, middle, last = (candidate.center for candidate in triple)

    span = math.hypot(last[0] - first[0], last[1] - first[1])
    if span <= 0 or span < span_floor:
        return None

    residual = collinearity_residual(first, middle, last)
    if residual > settings.max_collinearity_error_ratio:
        return None

    # 中间环在两端连线上的投影比例；太靠边说明"中间那个"其实是端点。
    projection = (
        (middle[0] - first[0]) * (last[0] - first[0])
        + (middle[1] - first[1]) * (last[1] - first[1])
    ) / (span * span)
    if not 0.2 < projection < 0.8:
        return None

    balance = 1.0 - abs(projection - 0.5) * 2.0
    diameters = [candidate.diameter_px for candidate in triple]
    mean_diameter = sum(diameters) / RING_COUNT
    if mean_diameter <= 0:
        return None
    diameter_penalty = max(diameters) / mean_diameter - 1.0
    support = sum(candidate.support for candidate in triple)

    # 权重沿用仓库既有打分函数的形式：几何项占大头，支撑数只做小修正，
    # 直径不一致直接扣分（三个环本该一样大）。
    return 8.0 * (1.0 - residual) + 0.35 * balance - 2.0 * diameter_penalty - 0.015 * support


def collinearity_residual(first, middle, last) -> float:
    """中间点到两端连线的垂距 / 跨度。共线时为 0，与画面尺度无关。

    ``2×三角形面积 / 底边 = 高``，再除以跨度就得到无量纲的比值：
    垂距 1 px、跨度 10 px 时是 0.1。
    """
    span = math.hypot(last[0] - first[0], last[1] - first[1])
    if span <= 0:
        return math.inf
    area2 = abs(
        (last[0] - first[0]) * (middle[1] - first[1])
        - (last[1] - first[1]) * (middle[0] - first[0])
    )
    return (area2 / span) / span


def line_quality(centers) -> Tuple[float, float]:
    """返回 ``(共线残差, 间距比 d12/d23)``。"""
    first, middle, last = centers
    residual = collinearity_residual(first, middle, last)
    gap_12 = math.hypot(middle[0] - first[0], middle[1] - first[1])
    gap_23 = math.hypot(last[0] - middle[0], last[1] - middle[1])
    if gap_23 <= 0:
        return residual, math.inf
    return residual, gap_12 / gap_23


#: 置信度公式里"支撑数够了"和"直径带够了"的分母。取值理由：一个边缘完整的真环
#: 会给出 6–10 条椭圆观测（内外两圈、每圈被 Canny 断成两三截），直径带 4 个左右。
#: 这两个数决定"什么样的环算证据充分"，调大只会让验收更难通过。
CONFIDENCE_FULL_SUPPORT = 8.0
CONFIDENCE_FULL_BANDS = 4.0


def confidence_of(residual: float, perspective: float, triple) -> float:
    """综合置信度 0–1：几何 + 支撑 + 直径带，三项加权。

    几何项占一半：三个环共线、间距均匀，这件事本身比"边缘检出多少条"更能说明
    "选中的确实是工位"。支撑与直径带各占 0.3 / 0.2，用来把"边缘完整的真环"和
    "勉强凑够门槛的杂项"分开。

    标定方式：几何满分、支撑与直径带都满分的检测恰好得 1.0。所以
    ``config.json`` 里 ``minimum_confidence`` 默认 0.9 的含义是"证据要基本齐全"，
    而不是一个够不着的数——早先的权重（0.55/0.25/0.20，分母 15 与 12）即使
    完美检出也只有 0.85，那条验收门槛永远过不去。
    """
    if not math.isfinite(perspective) or perspective <= 0:
        return 0.0
    geometry = math.exp(-10.0 * residual - 0.5 * abs(math.log(perspective)))
    support = min(1.0, sum(candidate.support for candidate in triple) / CONFIDENCE_FULL_SUPPORT)
    bands = min(1.0, sum(candidate.diameter_bands for candidate in triple) / CONFIDENCE_FULL_BANDS)
    return max(0.0, min(1.0, 0.5 * geometry + 0.3 * support + 0.2 * bands))


def count_bands(diameters: Sequence[float], gap: float) -> int:
    """数一串直径里有几个"带"：相邻差超过 ``gap`` 就算新的一带。"""
    if not diameters:
        return 0
    bands = 1
    for previous, current in zip(diameters, diameters[1:]):
        if current - previous > gap:
            bands += 1
    return bands


def median(values: Sequence[float]) -> float:
    """中位数；空序列抛错（调用方保证非空）。"""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median of an empty sequence")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def wrap_degrees(value: float) -> float:
    """把角度折进 ``(-180, 180]``。"""
    wrapped = math.fmod(value + 180.0, 360.0)
    if wrapped <= 0:
        wrapped += 360.0
    return wrapped - 180.0


# ---------------------------------------------------------------------- 自测
def synthetic_station(
    *,
    size: Tuple[int, int] = DEFAULT_SELF_TEST_SIZE,
    centers: Sequence[Tuple[float, float]] = (),
    diameter_px: float = 96.0,
    thickness: int = 6,
    background: int = 0,
    foreground: int = 255,
) -> np.ndarray:
    """画一张合成工位图：黑底上三个白色**空心圆环**，供离线自测与单测复用。

    画成有厚度的环而不是一条细线，是为了让 Canny 在环的内外两侧各给出一条边——
    真实的工位环就是这样，而"多个直径带"这条判据正是靠它才成立。
    """
    cv2 = _cv2()
    width, height = size
    frame = np.full((height, width, 3), background, dtype=np.uint8)
    for center_x, center_y in centers:
        cv2.circle(
            frame,
            (int(round(center_x)), int(round(center_y))),
            int(round(diameter_px * 0.5)),
            (foreground, foreground, foreground),
            thickness,
        )
    return frame


def _self_test() -> int:
    problems: list[str] = []

    geometry = StationGeometrySettings(mm_per_px=0.5)
    reference = StationReferenceSettings(
        ring_centers_px=((380.0, 320.0), (633.0, 320.0), (886.0, 320.0)),
        stable_frame_count=3,
        minimum_confidence=0.0,
    )
    detector = RingStationDetector(geometry=geometry, reference=reference)

    # ---- 纯函数先测：它们没有 cv2 依赖，坏了最容易定位 ----
    if abs(collinearity_residual((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))) > 1e-12:
        problems.append("严格共线时残差应为 0")
    if abs(collinearity_residual((0.0, 0.0), (5.0, 1.0), (10.0, 0.0)) - 0.1) > 1e-9:
        # 垂距 1 px、跨度 10 px：残差 = 垂距/跨度 = 0.1（无量纲）。
        problems.append("垂距 1 / 跨度 10 的残差应为 0.1")
    if abs(collinearity_residual((0.0, 0.0), (5.0, 2.0), (10.0, 0.0)) - 0.2) > 1e-9:
        problems.append("垂距 2 / 跨度 10 的残差应为 0.2（残差随垂距线性增长）")
    if abs(wrap_degrees(190.0) + 170.0) > 1e-9:
        problems.append("190° 应折算成 -170°")
    if abs(wrap_degrees(-190.0) - 170.0) > 1e-9:
        problems.append("-190° 应折算成 170°")
    if count_bands([10.0, 11.0, 20.0], 5.0) != 2:
        problems.append("直径 [10, 11, 20]、间隔 5 应当数出 2 个带")
    if median([3.0, 1.0, 2.0]) != 2.0 or median([4.0, 1.0, 3.0, 2.0]) != 2.5:
        problems.append("中位数（奇偶两种长度）算错了")

    # ---- 退化输入必须当场报错 ----
    for bad in (None, np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8)):
        try:
            detector.detect(bad)
        except ValueError:
            pass
        else:
            problems.append("空帧 / 非三通道帧必须报 ValueError")

    # ---- 正例：三个等距圆环 ----
    frame = synthetic_station(centers=((380.0, 320.0), (633.0, 320.0), (886.0, 320.0)))
    detection = detector.detect(frame)
    if not detection.detected:
        problems.append(f"合成三环应当被识别，实际未识别（{detection.detail}）")
    else:
        if len(detection.centers_px) != RING_COUNT:
            problems.append(f"应当给出 {RING_COUNT} 个圆心，实际 {len(detection.centers_px)}")
        for index, (got, want) in enumerate(
            zip(detection.centers_px, ((380.0, 320.0), (633.0, 320.0), (886.0, 320.0)))
        ):
            if math.hypot(got[0] - want[0], got[1] - want[1]) > 3.0:
                problems.append(f"环 {index + 1} 圆心应为 {want}，实际 {got}")
        if detection.station_yaw_deg is None or abs(detection.station_yaw_deg) > 1.0:
            problems.append(f"水平摆放的工位偏角应接近 0°，实际 {detection.station_yaw_deg}")
        if detection.perspective_ratio is None or abs(detection.perspective_ratio - 1.0) > 0.05:
            problems.append(f"等距三环的间距比应接近 1，实际 {detection.perspective_ratio}")
        if not detection.scale_known:
            problems.append("配了 mm_per_px 就应当能给出毫米偏差")
        elif detection.position_error_mm is None or detection.position_error_mm > 3.0:
            problems.append(f"与参考同心时应接近 0 mm，实际 {detection.position_error_mm}")
        if detection.processing_ms <= 0.0:
            problems.append("应当记录处理耗时")

    # ---- 稳定窗口：同一位置连喂 3 帧才算稳定 ----
    for _ in range(reference.stable_frame_count):
        stable_detection = detector.detect(frame)
    if not stable_detection.stable:
        problems.append(
            f"同一位置连续 {reference.stable_frame_count} 帧后应当判为稳定，"
            f"抖动 {stable_detection.center_jitter_px}"
        )

    # ---- 平移：偏差应当跟着走，并且方向正确 ----
    shifted = synthetic_station(centers=((390.0, 330.0), (643.0, 330.0), (896.0, 330.0)))
    shifted_detection = detector.detect(shifted)
    if not shifted_detection.detected:
        problems.append("平移后的三环应当仍被识别")
    else:
        if shifted_detection.error_x_mm is None or shifted_detection.error_x_mm <= 0:
            problems.append("整体右移后 error_x_mm 应为正")
        if shifted_detection.error_y_mm is None or shifted_detection.error_y_mm <= 0:
            problems.append("整体下移后 error_y_mm 应为正（y 轴向下）")
        # 10 px @ 0.5 mm/px = 5 mm，两轴合起来 sqrt(50) ≈ 7.07 mm。
        if shifted_detection.position_error_mm is None or not (
            6.0 < shifted_detection.position_error_mm < 8.5
        ):
            problems.append(
                f"平移 (10,10) px、0.5 mm/px 时最大偏差应约 7.07 mm，"
                f"实际 {shifted_detection.position_error_mm}"
            )

    # ---- 旋转：偏角应当报出来 ----
    detector.reset()
    tilted = synthetic_station(centers=((380.0, 300.0), (633.0, 320.0), (886.0, 340.0)))
    tilted_detection = detector.detect(tilted)
    if not tilted_detection.detected:
        problems.append("倾斜摆放的三环应当仍被识别")
    elif tilted_detection.station_yaw_deg is None or not (
        3.0 < tilted_detection.station_yaw_deg < 6.0
    ):
        problems.append(
            f"两端高差 40 px / 跨度 506 px 的偏角应约 4.5°，"
            f"实际 {tilted_detection.station_yaw_deg}"
        )

    # ---- 空画面与两个环：如实说没识别到，不抛异常 ----
    detector.reset()
    empty = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    if empty.detected or empty.centers_px or empty.within_tolerance:
        problems.append("纯黑画面不该识别出工位，也不该判为在容差内")
    if not empty.detail:
        problems.append("未识别时应当给出原因（detail）")
    detector.reset()
    two = synthetic_station(centers=((380.0, 320.0), (633.0, 320.0)))
    if detector.detect(two).detected:
        problems.append("只有两个环时不该识别出三圆环工位")

    # ---- 没配尺度时：只报像素，毫米字段必须是 None 而不是 0 ----
    # ``mm_per_px`` 的**默认值不是 None**（默认 0.495），所以这里必须显式配一个
    # "没有尺度"的 geometry，否则测的是默认比例尺而不是"没标定"那条路。
    detector.reset()
    blind = RingStationDetector(
        geometry=StationGeometrySettings(mm_per_px=None, image_to_station_affine=None),
        reference=StationReferenceSettings(
            ring_centers_px=reference.ring_centers_px,
            stable_frame_count=3,
            minimum_confidence=0.0,
        ),
    )
    if blind.geometry.has_scale:
        problems.append("显式配成没有尺度时 has_scale 应为 False")
    blind_detection = blind.detect(frame)
    if not blind_detection.detected:
        problems.append("没配尺度也该能识别（只是没有毫米）")
    elif blind_detection.scale_known or blind_detection.position_error_mm is not None:
        problems.append("没配尺度时 position_error_mm 必须是 None（不是 0）")
    elif blind_detection.within_tolerance:
        problems.append("没配尺度时不该判为在容差内")

    # ---- 仿射换算：各向异性时两个方向的比例应当不同 ----
    detector.reset()
    affine = RingStationDetector(
        geometry=StationGeometrySettings(
            mm_per_px=None, image_to_station_affine=((0.5, 0.0), (0.0, 1.0))
        ),
        reference=reference,
    )
    anisotropic = affine.detect(shifted)
    if not anisotropic.detected:
        problems.append("配了仿射矩阵时应当能识别")
    elif anisotropic.error_x_mm is None or anisotropic.error_y_mm is None:
        problems.append("配了仿射矩阵就应当给出毫米偏差")
    elif not (4.0 < anisotropic.error_x_mm < 6.0 and 9.0 < anisotropic.error_y_mm < 11.0):
        problems.append(
            f"仿射 [[0.5,0],[0,1]] 下 (10,10) px 应为 (5,10) mm，"
            f"实际 ({anisotropic.error_x_mm}, {anisotropic.error_y_mm})"
        )

    # ---- 非法参数 ----
    for kwargs, name in (
        ({"detection": StationDetectionSettings(center_cluster_radius_px=0.0)}, "聚类半径为 0"),
        ({"detection": StationDetectionSettings(min_contour_points=2)}, "轮廓点数下限低于 5"),
        ({"detection": StationDetectionSettings(min_cluster_support=0)}, "支撑数为 0"),
        ({"detection": StationDetectionSettings(min_diameter_bands=0)}, "直径带数为 0"),
        ({"reference": StationReferenceSettings(stable_frame_count=0)}, "稳定帧数为 0"),
    ):
        try:
            RingStationDetector(**kwargs)
        except ValueError:
            continue
        problems.append(f"{name} 应被拒绝，实际被接受了")

    print(f"合成工位：{detection.summary()}")
    print(f"平移后：  {shifted_detection.summary()}")
    print(f"倾斜后：  {tilted_detection.summary()}")
    print(f"空画面：  {empty.summary()}")
    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：三环选取、圆心、偏角、间距比、毫米偏差、仿射换算与容差判定都正确。")
    return 0


def _run_image(detector: RingStationDetector, image_path: Path) -> int:
    """在一张图片上跑一次；**没检出也算失败**（返回码 1），与 ``hough_lock`` 一致。"""
    cv2 = _cv2()
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"读不出图片：{image_path}")
        return 1
    detection = detector.detect(frame)
    print(f"{image_path}：{detection.summary()}")
    return 0 if detection.detected else 1


def _run_camera(detector: RingStationDetector, camera_id: int) -> int:
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
                detection = detector.detect(frame)
                cv2.imshow(window, detector.draw(frame, detection))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            print(detector.detect(frame).summary() if frame is not None else "没有画面")
    finally:
        close_previews()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="三圆环工位识别：圆心、偏角与偏差")
    parser.add_argument("--image", type=Path, help="在一张图片上跑（不给就跑合成帧自测）")
    parser.add_argument("--camera", type=int, help="打开摄像头实时预览")
    args = parser.parse_args(argv)

    # 真机入口读 config.json 的 station 块；自测用内置默认值，不读文件。
    if args.camera is not None or args.image is not None:
        from modules.vision.config import load_vision_config

        settings = load_vision_config()
        detector = RingStationDetector(
            geometry=settings.station_geometry,
            detection=settings.station_detection,
            reference=settings.station_reference,
        )
    else:
        detector = RingStationDetector()

    if args.camera is not None:
        return _run_camera(detector, args.camera)
    if args.image is not None:
        return _run_image(detector, args.image)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

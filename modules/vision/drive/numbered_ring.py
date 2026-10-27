"""视野里那个环：环心在哪、它是几号（模型给的，不是代码猜的）。

    python test.py   # 选 26

相机装在臂上、镜头离纸很近时，画面里通常**只有一个环**——"同时看到三个"那种检测
在这个取景下必然失败，而"只看到一个环"本来就不是错误。本模块只为这种取景服务。

## 两件事都由模型给，但环心要再修一步

随包的 ``models/best.onnx`` 有 7 类：四色物料 + ``tg1`` / ``tg2`` / ``tg3``。后三个就是
带数字的靶标，所以一帧前向同时给出：

- **是哪个环** = 命中的类别，查 ``config.json`` 的 ``ring_target.classes``；
- **环大概在哪** = 框中心。

框是回归出来的，框心只在框正好套住图案时才等于环心（实测中位偏 3.8 px，最坏一帧偏
81 px）。所以还有一步**框内精修**：框内那一圈圈同心圆本来共圆心，用
``Canny + fitEllipse`` 收它们的圆心投票，取中位数当环心（实测中位偏 2.2 px、最坏
7.7 px）。三条验收不过就退回框心，见 :meth:`RingTargetLocator._refine_center`。

因此这里**没有**整帧 Canny → 椭圆观测 → 聚类 → 按支撑数排序那一套，也**没有**
"环里墨渍占比够不够"的墨迹判据：那些是已经训练进模型的能力，代码里再算一遍只会
多出一套会和模型结论打架的规则。精修只在**框内**做，不是全画面搜索。

## 框贴边就**不是**环心

模型给的是框，框心只有在"整个靶标都落在画面里"时才等于环心：框被画面裁掉一截时，
中心会跟着往画面内侧挪（实测偏 80 px 以上）。所以贴边的框照样返回（``clipped=True``），
但 :meth:`RingTarget.reading` 返回 ``None``——宁可说"这一帧不能对准"，也不给一个
差十几毫米、却看起来完全正常的数。

## 比例尺的顺序：针孔标定 > 靶标框（已知外径）> 配置固定值

1. ``table_depth_mm / focal_px``：做过菜单 25 的标定就用它，与 ``RegionReport`` 的毫米
   是同一套单位；
2. **靶标框**：``毫米/像素 = ring_target.outer_diameter_mm / 模型框像素长边``。相机装在
   臂上、离纸很近时 ``station.mm_per_px``（给固定相机标的值）会差好几倍，而针孔标定
   不一定做过——这一档与镜头距离无关，能先用起来。代价是模型框的松紧会抖（实测同一条
   靶标 449~580 px），所以它只适合"先算出一个能用的角度"，不是测量级的读数；
3. ``station.mm_per_px``：最后兜底；都没有就**不报毫米**。

毫米在这里还有一个用途：``grasp.plan_ring()`` 要用它和实测雅可比（毫米/脉冲）算出
"底座该转到多少度"。比例尺与雅可比必须是同一把尺（都是物理毫米），这条链才成立。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from modules.vision.config import (
    RingTargetSettings,
    StationDetectionSettings,
    StationGeometrySettings,
)

if TYPE_CHECKING:
    from modules.vision.drive.yolov8 import Detection

#: 框贴到画面边缘多少像素以内就当作"被裁掉"。
BORDER_TOLERANCE_PX = 1.0


@dataclass(frozen=True)
class RingTarget:
    """**视野里的那一个环**：环心在哪、几号、能不能直接拿去对准。"""

    #: 环心（像素）：框内同心椭圆精修之后的位置；没精修时等于框心。
    center_px: Tuple[float, float]
    #: 环的等效半径（像素）：框长边的一半。比例尺按它算，所以**不要**换成别的半径。
    radius_px: float
    #: 环号（来自 ``ring_target.classes``）。
    digit: int
    #: 模型给这个框的置信度。
    score: float
    #: 框是否被画面边缘裁掉。裁掉时框心**不是**环心。
    clipped: bool
    #: 环心在**相机系**的毫米坐标；没有比例尺时是 ``None``。
    x_mm: Optional[float]
    y_mm: Optional[float]
    #: 模型框的中心（没精修时的环心；精修过时留着便于对比"修了多少"）。
    box_center_px: Optional[Tuple[float, float]] = None
    #: 真正那一圈的外半径（像素）：精修时量到的最大圆；没精修时 ``None``。只用于画图。
    ring_radius_px: Optional[float] = None
    #: 环心是否经过框内椭圆精修（``False`` 表示用的是框心）。
    refined: bool = False
    #: 本帧实际用的比例尺（毫米/像素）与来源（``depth/focal`` / ``ring box`` /
    #: ``station.mm_per_px``）。
    scale_mm_per_px: Optional[float] = None
    scale_source: str = ""
    #: 人话说明：这一帧为什么是它、比例尺从哪来。
    detail: str = ""

    @property
    def usable(self) -> bool:
        """能不能拿去对准：框没被裁掉，而且毫米坐标算出来了。"""
        return not self.clipped and self.x_mm is not None and self.y_mm is not None

    def reading(self) -> Optional[Tuple[float, float]]:
        """相机系 ``(x_mm, y_mm)``，可直接喂给 ``align_xy(reading=...)``。"""
        if not self.usable:
            return None
        return (float(self.x_mm), float(self.y_mm))

    def summary(self) -> str:
        where = (
            f"xyz=({self.x_mm:+.1f},{self.y_mm:+.1f})mm"
            if self.x_mm is not None and self.y_mm is not None
            else "无毫米（缺比例尺）"
        )
        if self.clipped:
            shape = "**框被画面裁掉，框心不是环心**"
        elif self.refined and self.box_center_px is not None:
            shift = math.hypot(
                self.center_px[0] - self.box_center_px[0],
                self.center_px[1] - self.box_center_px[1],
            )
            shape = f"环心已精修（相对框心修正 {shift:.1f} px）"
        else:
            shape = "整框在画面里（环心用框心）"
        scale = (
            f"{self.scale_mm_per_px:.4f} mm/px（{self.scale_source}）"
            if self.scale_mm_per_px is not None
            else "无比例尺"
        )
        return (
            f"环 {self.digit}（模型 {self.score:.2f}）："
            f"@({self.center_px[0]:.0f},{self.center_px[1]:.0f}) r={self.radius_px:.0f}px "
            f"{where}，{shape}，{scale}"
        )


@dataclass(frozen=True)
class RingRow:
    """三个环排成的那条直线：倾角 + 三个环心（相机系毫米）。

    ``centers_mm`` 的 2 号是**实测**环心，1/3 号是按已知间距沿直线外推出来的；
    ``measured_span_mm`` 是两端都看得见时实测的 1↔3 跨距，用来核对"间距 150 mm"
    这条假设与比例尺对不对（``None`` = 只有一侧可见）。
    """

    #: 直线倾角（度）：画面右方为 0、下方为正，按**直线**折进 ``(-90, 90]``。
    yaw_deg: float
    #: 三个环心（相机系毫米），键就是环号 1/2/3。
    centers_mm: Dict[int, Tuple[float, float]]
    measured_span_mm: Optional[float] = None

    def summary(self) -> str:
        middle = self.centers_mm[2]
        span = (
            f"实测 1-3 跨距 {self.measured_span_mm:.0f} mm"
            if self.measured_span_mm is not None
            else "只有一侧环可见"
        )
        return (
            f"三环直线：倾角 {self.yaw_deg:+.1f}°，"
            f"2 号环心 ({middle[0]:+.1f},{middle[1]:+.1f}) mm，{span}"
        )


def derive_ring_row(
    centers_mm: "Mapping[int, Sequence[float]]", *, spacing_mm: float
) -> RingRow:
    """一帧里看到的环心（1/2/3 里至少两个）→ 三个环心 + 直线倾角。

    中间环（2 号）用**实测**环心，两侧按"相邻圆心间距 = ``spacing_mm``"沿直线外推。
    为什么不直接用 1/3 号的实测框心：相机装在臂上、镜头离纸很近，两端常常贴到画面边，
    而贴边的框心会偏出几十像素（``RingTarget.reading`` 那时干脆返回 ``None``）；三个环
    的间距是**装好的物理真值**，用实测的中间环 + 直线方向推出来比贴边框可信。

    方向取"离得最远的那一对"：两端都可见时用 1→3（同样的读数噪声摊到角度上最小，
    与 ``ring_station`` 取 1→3 是同一个理由），否则用 2 号与看得见的那一侧。

    :raises ValueError: 没有 2 号环（没有可以当锚点的实测环心）、两个相邻环都没看到
        （定不出直线）、某一对环心重合，或 ``spacing_mm`` 不是正数。
    """
    if not spacing_mm > 0:
        raise ValueError(f"spacing_mm 必须是正数（毫米），收到 {spacing_mm!r}")
    if 2 not in centers_mm:
        raise ValueError("三环直线要用 2 号环的实测环心，但这一帧没有 2 号环")
    middle = _finite_pair(centers_mm[2], "2 号环心")
    sides = [digit for digit in (1, 3) if digit in centers_mm]
    if not sides:
        raise ValueError("只看到 2 号环，定不出三环这条直线：让 1 号或 3 号也进画面")
    if len(sides) == 2:
        first = _finite_pair(centers_mm[1], "1 号环心")
        last = _finite_pair(centers_mm[3], "3 号环心")
    elif sides[0] == 3:
        first, last = middle, _finite_pair(centers_mm[3], "3 号环心")
    else:
        first, last = _finite_pair(centers_mm[1], "1 号环心"), middle

    dx, dy = last[0] - first[0], last[1] - first[1]
    span = math.hypot(dx, dy)
    if span <= 0:
        raise ValueError(f"两个环心重合（{first} 与 {last}），定不出直线")
    unit = (dx / span, dy / span)
    yaw = math.degrees(math.atan2(unit[1], unit[0]))
    return RingRow(
        # 直线没有方向：170° 与 -10° 是同一条线，折进 (-90, 90] 才能取最小转角。
        yaw_deg=(yaw + 90.0) % 180.0 - 90.0,
        centers_mm={
            1: (middle[0] - spacing_mm * unit[0], middle[1] - spacing_mm * unit[1]),
            2: middle,
            3: (middle[0] + spacing_mm * unit[0], middle[1] + spacing_mm * unit[1]),
        },
        measured_span_mm=span if len(sides) == 2 else None,
    )


def _finite_pair(value: "Sequence[float]", name: str) -> Tuple[float, float]:
    """把一个环心读成两个有限数；写错就当场报错（外推错一格就是 150 mm）。"""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} 要写成 (x, y) 两个数，收到 {value!r}")
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} 里的每一项都必须是数字，收到 {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} 里有非有限数：{item!r}")
        numbers.append(number)
    if len(numbers) != 2:
        raise ValueError(f"{name} 要写成 (x, y) 两个数，收到 {value!r}")
    return (numbers[0], numbers[1])


class RingTargetLocator:
    """一帧 BGR → **视野里那一个环**的环心（像素 + 相机系毫米）。

    本类**不拥有检测器**：一个会话里的模型只加载一次，靶标与物料共用同一个检测器
    （见 :meth:`modules.vision.runtime.Vision._ring_target_locator`）；单独跑一张图时
    由调用方自己建一个。

    选哪一个框：优先"没被画面裁掉"的（那才可能是真环心），再比置信度。

    Parameters:
        detector: 任何一个 ``detect(frame) -> list[Detection]`` 的检测器。
        settings: ``config.json`` 的 ``ring_target`` 块（类别名 → 环号、外径、精修开关）。
        geometry: 兜底比例尺（``station.mm_per_px``）。
        detection: 精修用的 Canny / 椭圆阈值；``None`` 用 ``StationDetectionSettings`` 默认值。
        focal_px / table_depth_mm: 台面的针孔标定；两个都有时优先用它换算毫米。
    """

    def __init__(
        self,
        detector,
        *,
        settings: Optional[RingTargetSettings] = None,
        geometry: Optional[StationGeometrySettings] = None,
        detection: Optional[StationDetectionSettings] = None,
        focal_px: Optional[float] = None,
        table_depth_mm: Optional[float] = None,
    ) -> None:
        if detector is None:
            raise ValueError("RingTargetLocator 需要检测器；模型只该由会话建一次")
        self.detector = detector
        self.settings = settings if settings is not None else RingTargetSettings()
        self.geometry = geometry if geometry is not None else StationGeometrySettings()
        self.detection = (
            detection if detection is not None else StationDetectionSettings()
        )
        self.focal_px = focal_px
        self.table_depth_mm = table_depth_mm
        #: 靶标图案的外径（毫米）；配了就**可以**用它当与距离无关的比例尺。
        self.diameter_mm = self.settings.outer_diameter_mm
        #: 环号表（类别名 → 环号）；构造时定死，逐帧不再回读配置。
        self.digits: Dict[str, int] = dict(self.settings.classes)
        #: 最近一次 ``detect`` 走到了哪一步（给真机入口打印"为什么没找到"）。
        self.last_diagnosis = ""
        self._check_class_table()

    def _check_class_table(self) -> None:
        """配置里的环类别必须真的在检测器的类别表里，否则当场报错。

        名字写错（``tg_1``）不会让程序挂，只会一直"画面里没有环"——那是现场最费时间的
        故障：看起来像模型没认出来，其实是配置少打了一个下划线。所以这里拒绝启动。
        """
        table = getattr(self.detector, "class_names", None)
        if not table:
            return
        missing = [name for name in self.digits if name not in set(table)]
        if missing:
            raise ValueError(
                "ring_target.classes 里的 "
                + "、".join(missing)
                + " 不在模型类别表里（"
                + "、".join(str(name) for name in table)
                + "）；检查 config.json 或模型文件的 names 元数据"
            )

    # ------------------------------------------------------------- 主流程
    def detect(
        self, frame: np.ndarray, *, digit: Optional[int] = None
    ) -> Optional[RingTarget]:
        """找出视野里最可能是"那个环"的框；一个都没有时返回 ``None``。

        ``digit`` 给了就**只看这个环号**（``tg1``/``tg2``/``tg3`` 里的那一个）：画面里
        同时看得见两个环时，模型给分高的那个未必是你要找的——校准"2 号环"的时候按分数
        挑到 3 号，后面的 ``require_digit`` 检查只会报"是 3 号不是 2 号"，而 2 号其实
        就在画面里。
        """
        if frame is None:
            raise ValueError("frame 不能是 None")
        height, width = frame.shape[:2]
        hits = [
            hit
            for hit in self.detector.detect(frame)
            if hit.name in self.digits
            and (digit is None or self.digits[hit.name] == int(digit))
        ]
        if not hits:
            wanted = (
                f"{digit} 号环（{self._names_for(digit)}）"
                if digit is not None
                else " / ".join(self.digits)
            )
            self.last_diagnosis = f"画面里没有环：模型给的框里没有 {wanted}"
            return None
        chosen = max(hits, key=lambda hit: self._rank(hit, width, height))
        return self._build(chosen, width, height, len(hits), frame)

    def detect_all(self, frame: np.ndarray) -> Dict[int, RingTarget]:
        """一帧里**每个环号各取一个**框：``{环号: RingTarget}``，模型只跑一次。

        与 :meth:`detect` 的差别只是"不再只挑一个"：三环同框时要把三个都拿到手，
        才能用环心连线的倾角把两端外推出来（见 :func:`derive_ring_row`）。同一个环号
        有多个框时仍按 :meth:`_rank` 取最好的那个；贴边 / 没比例尺的框照样返回，
        它们的 ``RingTarget.reading()`` 会如实是 ``None``。
        """
        if frame is None:
            raise ValueError("frame 不能是 None")
        height, width = frame.shape[:2]
        hits = [
            hit for hit in self.detector.detect(frame) if hit.name in self.digits
        ]
        chosen: Dict[int, "Detection"] = {}
        for hit in hits:
            digit = int(self.digits[hit.name])
            best = chosen.get(digit)
            if best is None or self._rank(hit, width, height) > self._rank(
                best, width, height
            ):
                chosen[digit] = hit
        self.last_diagnosis = (
            f"{len(hits)} 个环框，识别到 "
            + ("、".join(str(digit) for digit in sorted(chosen)) or "无")
        )
        return {
            digit: self._build(hit, width, height, len(hits), frame)
            for digit, hit in sorted(chosen.items())
        }

    def _names_for(self, digit: int) -> str:
        """环号 → 类别名（报错用；一个环号可能配了多个名字）。"""
        names = [name for name, value in self.digits.items() if int(value) == int(digit)]
        return " / ".join(names) or "未配置"

    # ------------------------------------------------------------- 内部
    @staticmethod
    def _rank(hit: "Detection", width: int, height: int) -> Tuple[int, float]:
        """先看框有没有被裁掉，再比置信度。"""
        return (0 if _touches_border(hit, width, height) else 1, float(hit.score))

    def _refine_center(
        self, frame: np.ndarray, hit: "Detection"
    ) -> Tuple[Tuple[float, float], bool, str]:
        """在框内做一次**局部椭圆拟合**，把"框心"换成"同心圆的中心"。

        模型框是回归出来的，框心只在框正好套住图案时才等于环心——实测有一帧框偏了
        82 px（框把图案左边切掉了）。而框内一圈圈同心圆的圆心是**同一个点**，取它们
        的中位数当环心：精度 1–2 px，也不受框松紧的影响。

        三条验收，任何一条不过就退回框心（``refined=False``），绝不硬凑：

        - 至少 ``refine_min_observations`` 条椭圆观测；
        - 各观测圆心到中位数的距离中位数 < ``refine_max_spread_px``（同心圆应当重合）；
        - 精修结果离框心不超过框长边的 ``refine_max_offset_ratio``——再远就说明框套住的是
          别的东西。

        Returns:
            ``(center, refined, note, outer_diameter_px)``：note 是给 ``detail`` 用的一句人话，\n            ``outer_diameter_px`` 是被当成"外圈"的那条椭圆观测的直径（没精修时 ``None``）。
        """
        from modules.vision.config import StationDetectionSettings
        from modules.vision.drive.ring_station import extract_observations

        size = max(float(hit.width), float(hit.height))
        height, width = frame.shape[:2]
        margin = self.settings.refine_margin_ratio * size
        x0 = max(0, int(round(hit.x1 - margin)))
        y0 = max(0, int(round(hit.y1 - margin)))
        x3 = min(width, int(round(hit.x2 + margin)))
        y3 = min(height, int(round(hit.y2 + margin)))
        if x3 - x0 < 16 or y3 - y0 < 16:
            return hit.center, False, "框太小，没做精修", None
        crop = frame[y0:y3, x0:x3]
        # 一帧里最大的那个环可能占满裁出来的画面，而 station 的 max_major_axis_px
        # 是按"整排三个 95 mm 环"定的——不抬上限会把真圆当杂项丢掉（老路踩过）。
        detection = replace(
            self.detection,
            max_major_axis_px=max(self.detection.max_major_axis_px, float(min(crop.shape[:2]))),
        )
        box_x, box_y = hit.center[0] - x0, hit.center[1] - y0
        observations = [
            item
            for item in extract_observations(crop, detection)
            if self.settings.refine_min_diameter_ratio * size
            <= item.diameter_px
            <= self.settings.refine_max_diameter_ratio * size
            and math.hypot(item.center[0] - box_x, item.center[1] - box_y)
            <= self.settings.refine_max_offset_ratio * size
        ]
        if len(observations) < self.settings.refine_min_observations:
            return (
                hit.center,
                False,
                f"只找到 {len(observations)} 条椭圆观测，环心用框心",
                None,
            )
        # 以**直径最大的那条观测**为种子：外圈最长、点数最多，也最不容易被数字笔画带偏。
        # 只收圆心离它足够近的观测——同心的那几圈；零碎的弧（数字、台面纹理）离得远，
        # 自然被排除。这一步比"全体中位数"稳：实测内圈会被字形拽偏 10 px 以上。
        ordered = sorted(observations, key=lambda item: item.diameter_px, reverse=True)
        seed = ordered[0]
        cluster = [
            item
            for item in ordered
            if math.hypot(
                item.center[0] - seed.center[0], item.center[1] - seed.center[1]
            )
            <= self.settings.refine_cluster_radius_px
        ]
        used = cluster if len(cluster) >= self.settings.refine_min_observations else observations
        centre_x = float(np.median([item.center[0] for item in used]))
        centre_y = float(np.median([item.center[1] for item in used]))
        spread = float(
            np.median(
                [
                    math.hypot(item.center[0] - centre_x, item.center[1] - centre_y)
                    for item in used
                ]
            )
        )
        # 观测取够了还要**互相对得上**：同心圆的圆心本该重合，散布大说明框里混着别的东西。
        if spread > self.settings.refine_max_spread_px:
            return (
                hit.center,
                False,
                f"{len(used)} 条椭圆观测的圆心散布 {spread:.1f} px，偏大，环心用框心",
                None,
            )
        offset = math.hypot(centre_x - box_x, centre_y - box_y)
        if offset > self.settings.refine_max_offset_ratio * size:  # pragma: no cover - 观测已被窗口限住
            return hit.center, False, f"精修结果离框心 {offset:.0f} px，太远，环心用框心", None
        note = (
            f"环心由框内 {len(used)} 条同心椭圆精修（最大圆直径 "
            f"{seed.diameter_px:.0f} px，散布 {spread:.1f} px，"
            f"相对框心修正 {offset:.1f} px）"
        )
        return (x0 + centre_x, y0 + centre_y), True, note, float(seed.diameter_px)

    def _build(
        self,
        hit: "Detection",
        width: int,
        height: int,
        count: int,
        frame: np.ndarray,
    ) -> RingTarget:
        clipped = _touches_border(hit, width, height)
        box_center_x, box_center_y = hit.center
        radius = max(0.0, max(hit.width, hit.height) * 0.5)
        refine_note = ""
        refined = False
        ring_diameter = None
        center_x, center_y = box_center_x, box_center_y
        # 贴边的框没有"整圈"，精修只看得到一段弧，圆心会被拽偏——那时干脆不精修。
        if self.settings.refine_center and not clipped:
            (center_x, center_y), refined, refine_note, ring_diameter = (
                self._refine_center(frame, hit)
            )
        scale, source = self._millimetres_per_pixel(radius)
        x_mm = y_mm = None
        if scale is not None:
            x_mm = (float(center_x) - width / 2.0) * float(scale)
            y_mm = (float(center_y) - height / 2.0) * float(scale)
        detail = f"模型给了 {count} 个环框"
        if refine_note:
            detail += "；" + refine_note
        if clipped:
            detail += "；选中的框贴画面边，框心不是环心，不给可对准读数"
        if scale is None:
            detail += "；没有比例尺（depth+focal / station.mm_per_px），给不出毫米"
        else:
            detail += f"；比例尺 {scale:.4f} mm/px 来自 {source}"
        self.last_diagnosis = (
            f"{count} 个环框，选中环 {self.digits[hit.name]}"
            f"（{hit.name} {float(hit.score):.2f}，"
            f"框 ({hit.x1:.0f},{hit.y1:.0f})-({hit.x2:.0f},{hit.y2:.0f})）"
            f"{'，贴边' if clipped else ('，已精修' if refined else '')}"
        )
        return RingTarget(
            center_px=(float(center_x), float(center_y)),
            box_center_px=(float(box_center_x), float(box_center_y)),
            radius_px=radius,
            ring_radius_px=None if ring_diameter is None else float(ring_diameter) / 2.0,
            digit=int(self.digits[hit.name]),
            score=float(hit.score),
            clipped=clipped,
            refined=refined,
            x_mm=x_mm,
            y_mm=y_mm,
            scale_mm_per_px=None if scale is None else float(scale),
            scale_source=source,
            detail=detail,
        )

    def _millimetres_per_pixel(self, radius_px: float) -> Tuple[Optional[float], str]:
        """本帧的比例尺与来源，优先级：针孔标定 > 靶标框（已知外径）> 配置固定值。

        为什么中间那一档存在：相机装在臂上、镜头离纸很近时，``station.mm_per_px``（给
        固定相机标的值）会差好几倍，而针孔标定不一定做过。靶标外径是**已知的物理尺寸**
        （``ring_target.outer_diameter_mm``），``毫米/像素 = 外径 / 框像素长边`` 与镜头
        距离无关——这就是老路上"已知直径当比例尺"的思路，只是直径现在从模型框量，不再
        做椭圆拟合。

        代价是精度：模型框的松紧会抖（实测同一条靶标 449~580 px），所以这个来源在计划里
        会被标出来，别拿它当测量级的毫米读数。
        """
        focal = self.focal_px
        depth = self.table_depth_mm
        if focal is not None and depth is not None and focal > 0 and depth > 0:
            return (float(depth) / float(focal), "depth/focal")
        diameter = self.diameter_mm
        if diameter is not None and diameter > 0 and radius_px > 0:
            return (float(diameter) / (2.0 * float(radius_px)), "ring box")
        if self.geometry.mm_per_px is not None:
            return (float(self.geometry.mm_per_px), "station.mm_per_px")
        return (None, "none")


def _touches_border(hit: "Detection", width: int, height: int) -> bool:
    """框有没有被画面边缘裁掉——被裁掉的框，中心已经不是环心了。"""
    return (
        hit.x1 <= BORDER_TOLERANCE_PX
        or hit.y1 <= BORDER_TOLERANCE_PX
        or hit.x2 >= width - BORDER_TOLERANCE_PX
        or hit.y2 >= height - BORDER_TOLERANCE_PX
    )


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """假检测器：验证"选哪个框、框心怎么变成读数、贴边为什么不给读数"。

    不需要模型也不需要摄像头：``FakeDetector`` 按给定框返回 ``Detection``，
    正是"一帧 → 几个框"那一层接口。
    """
    from modules.vision.drive.yolov8 import Detection

    def box(name, score, x1, y1, x2, y2):
        return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=4, name=name)

    size = (1280, 720)

    class FakeDetector:
        def __init__(self, hits, *, class_names=("red", "tg1", "tg2", "tg3")):
            self.hits = list(hits)
            self.class_names = list(class_names)
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            return list(self.hits)

    problems = []

    def run(hits, **kwargs):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        return RingTargetLocator(FakeDetector(hits), **kwargs).detect(frame)

    # 1) 画面正中、整框可见：环心在光轴上，环号来自类别，比例尺来自"已知外径/框"
    target = run([box("tg2", 0.95, 640 - 260, 360 - 258, 640 + 260, 360 + 258)],
                 geometry=StationGeometrySettings(mm_per_px=0.5))
    if target is None:
        problems.append("画面里有一个整框可见的环，应当给出环心")
    else:
        reading = target.reading()
        if reading is None or abs(reading[0]) > 0.5 or abs(reading[1]) > 0.5:
            problems.append(f"环心应当在光轴上，实际 {reading}")
        if target.digit != 2:
            problems.append(f"tg2 应当是 2 号环，实际 {target.digit}")
        if target.clipped:
            problems.append("整框在画面里，不该标成贴边")
        if target.scale_source != "ring box":
            problems.append(f"没标定时比例尺该来自模型框，实际 {target.scale_source}")
        # 框长边 520 px、外径 95 mm -> 0.1827 mm/px（压过配置里的 0.5）
        if abs(target.scale_mm_per_px - 95.0 / 520.0) > 1e-6:
            problems.append(f"框比例尺算错了：{target.scale_mm_per_px}")
        print(target.summary())

    # 2) 框贴边：照样返回，但**没有**可对准读数
    clipped = run([box("tg1", 0.9, -40.0, 60.0, 220.0, 620.0)],
                  geometry=StationGeometrySettings(mm_per_px=0.5))
    if clipped is None:
        problems.append("贴边的环也要返回（好让人看到「看到了但用不了」）")
    elif not clipped.clipped or clipped.reading() is not None:
        problems.append(f"贴边的框不该给出可对准读数：{clipped.summary()}")

    # 3) 整框可见的低分 vs 贴边的高分：选整框那个
    chosen = run(
        [
            box("tg1", 0.99, -10.0, 60.0, 240.0, 620.0),          # 贴边、分更高
            box("tg3", 0.40, 600.0, 100.0, 900.0, 400.0),          # 整框、分低
        ],
        geometry=StationGeometrySettings(mm_per_px=0.5),
    )
    if chosen is None or chosen.digit != 3:
        problems.append(f"应当选整框可见的那个环，实际 {chosen and chosen.digit}")

    # 4) 没有环框：返回 None，并把原因写进诊断
    locator = RingTargetLocator(FakeDetector([box("red", 0.9, 0.0, 0.0, 50.0, 50.0)]))
    if locator.detect(np.zeros((size[1], size[0], 3), dtype=np.uint8)) is not None:
        problems.append("只有物料框时不该报出一个环")
    if "没有环" not in locator.last_diagnosis:
        problems.append(f"没找到环时诊断要说清楚，实际 {locator.last_diagnosis!r}")

    # 5) 比例尺来源：针孔标定优先于"已知外径"与配置固定值
    target = run(
        [box("tg2", 0.9, 290.0, 260.0, 590.0, 460.0)],   # 框心 (440, 360)，偏光轴 200 px
        geometry=StationGeometrySettings(mm_per_px=0.4),
        focal_px=800.0,
        table_depth_mm=400.0,       # 0.5 mm/px
    )
    if target is None or target.scale_source != "depth/focal":
        problems.append(f"有针孔标定时应当用它，实际 {target and target.scale_source}")
    elif abs(target.reading()[0] + 100.0) > 1.0:
        problems.append(f"0.5 mm/px 下偏 200 px 应当是 -100 mm，实际 {target.reading()}")

    # 6) 关掉"已知外径"才轮得到配置固定值
    fallback = RingTargetSettings(outer_diameter_mm=None)
    target = run([box("tg2", 0.9, 290.0, 260.0, 590.0, 460.0)],
                 settings=fallback,
                 geometry=StationGeometrySettings(mm_per_px=0.5))
    if target is None or target.scale_source != "station.mm_per_px":
        problems.append(f"关掉外径后该退回配置固定值，实际 {target and target.scale_source}")
    elif abs(target.reading()[0] + 100.0) > 1.0:
        problems.append(f"0.5 mm/px 下偏 200 px 应当是 -100 mm，实际 {target.reading()}")

    # 7) 什么比例尺都没有：环照样给出，但没有毫米
    target = run([box("tg2", 0.9, 290.0, 260.0, 590.0, 460.0)],
                 settings=fallback,
                 geometry=StationGeometrySettings(mm_per_px=None))
    if target is None or target.reading() is not None:
        problems.append("没有比例尺时不该给出可对准读数")

    # 8) 类别名写错：构造时就报错，而不是一直"没找到环"
    try:
        RingTargetLocator(
            FakeDetector([]),
            settings=RingTargetSettings(classes={"tg_1": 1}),
        )
    except ValueError as error:
        if "tg_1" not in str(error):
            problems.append(f"报错要指出错的那个类别名，实际 {error}")
    else:
        problems.append("环类别名不在模型类别表里时，构造就该失败")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：选框、框心换算、贴边拒绝、类别表校验都按约定处理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

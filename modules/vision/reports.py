"""物料定位的结构化结果：一帧里的候选圆面（``RegionReport``）与整帧报告（``MaterialReport``）。

## 为什么单独一个文件

这两个类型是**唯一**的结果定义，被视觉会话（``modules.vision.runtime``）、离线图片工具
（``modules.vision.drive.material_locate``）和抓取侧的观测转换一起消费。它们原先和
``MaterialLocator`` 住在同一个模块里，于是"只要结果类型"的调用方也不得不连带导入
定位器、霍夫、颜色块和 YOLO 打包。

这里刻意**只依赖标准库**：``LockedRegion`` / ``ClassResult`` / ``FaceQuality`` /
``Detection`` / ``RingLock`` 只在类型注解里出现，放进 ``TYPE_CHECKING``。
这样 ``import modules.vision.reports`` 不会预加载任何 CLI 驱动模块，也不会让
``material_locate`` ↔ 本模块之间出现导入环。

## 字段含义（不要为了统一名字而互换）

- ``RegionReport.z_mm`` / ``distance_mm`` 是**沿光轴的深度**；``range_mm`` 才是
  圆心到光心的三维直线距离。
- ``RegionReport.region.radius`` 是半径，转成观测的 ``diameter_px`` 时要乘 2。
- ``MaterialReport.faces`` 只含通过校验的顶面，编号与 :meth:`MaterialReport.summary`
  打印的「物料 N:」一致（被丢掉的候选不进编号）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from modules.vision.drive.hough_lock import LockedRegion, RingLock
    from modules.vision.drive.material_class import ClassResult
    from modules.vision.drive.material_face import FaceQuality
    from modules.vision.drive.yolov8 import Detection


@dataclass(frozen=True)
class RegionReport:
    """一个圆面的检测结果；相机坐标单位为 mm，右/下/镜头前方为正。"""

    region: "LockedRegion"
    classification: "ClassResult"
    #: 这个圆面"像不像物料顶面"：内部均匀度与边界支撑，见 ``material_face``。
    face: "FaceQuality"
    #: 最终是否按物料顶面接受。配了物料颜色时由颜色块决定（见 ``material_blob``），
    #: 否则由 ``face.looks_like_face`` 决定。
    accepted: bool
    # 采用图像中心为主点、水平和竖直像素焦距相等的近似针孔模型。
    x_mm: float
    y_mm: float
    # 沿光轴的深度；与 z_mm 同义，保留已有调用方使用的名称。
    distance_mm: float

    @property
    def present(self) -> bool:
        return True

    @property
    def label(self) -> Optional[str]:
        return self.classification.label

    @property
    def name(self) -> Optional[str]:
        return self.classification.name

    @property
    def confidence(self) -> Optional[float]:
        """YOLO 类别置信度；没有关联检测框时为 None。"""
        return self.classification.score if self.classification.found else None

    @property
    def center_x_px(self) -> float:
        """圆面中心在原始画面中的横坐标。"""
        return self.region.center_x

    @property
    def center_y_px(self) -> float:
        """圆面中心在原始画面中的纵坐标。"""
        return self.region.center_y

    @property
    def z_mm(self) -> float:
        """圆面中心沿相机光轴的深度。"""
        return self.distance_mm

    @property
    def range_mm(self) -> float:
        """圆面中心到相机光学中心的三维直线距离。"""
        return math.sqrt(self.x_mm**2 + self.y_mm**2 + self.z_mm**2)

    def summary(self) -> str:
        """Show pixel and camera coordinates even when YOLO misses the material."""
        category = f"{self.label}（{self.name}）" if self.name else "类别未知"
        confidence = f"{self.confidence:.2f}" if self.confidence is not None else "--"
        return (
            f"@({self.region.center_x:.0f},{self.region.center_y:.0f}) "
            f"{category}，confidence={confidence}，圆直径 {2 * self.region.radius:.0f} px，"
            f"{self.face.summary()}，"
            f"xyz=({self.x_mm:.1f},{self.y_mm:.1f},{self.z_mm:.1f}) mm，"
            f"直线距离约 {self.range_mm:.1f} mm"
        )


@dataclass(frozen=True)
class MaterialReport:
    """一帧里所有检出的物料圆面。"""

    lock: "RingLock"
    hits: "tuple[Detection, ...]"
    regions: "tuple[RegionReport, ...]"
    #: 本帧实际使用的像素焦距；没有圆面、也没配深度窗口时为 ``None``。
    focal_px: Optional[float] = None
    #: 本帧实际使用的半径窗口 ``(最小, 最大)``；没配深度窗口时为 ``None``。
    radius_window_px: Optional[tuple[float, float]] = None
    #: 这一帧的候选是从哪来的：``"palette"`` 配置的调色板 / ``"model"`` 模型框里现取的
    #: 参考色 / ``"hough"`` 颜色没配上、退回霍夫的形状判据；``None`` 表示什么都没找到。
    colour_source: Optional[str] = None

    @property
    def colour_fallback(self) -> bool:
        """颜色块一个都没找到、退回霍夫形状判据时为 ``True``（多半是调色板没配这块料）。"""
        return self.colour_source == "hough"

    @property
    def present(self) -> bool:
        """Whether Hough found at least one material top face."""
        return bool(self.regions)

    @property
    def faces(self) -> "tuple[RegionReport, ...]":
        """最终按物料顶面接受的圆面。"""
        return tuple(report for report in self.regions if report.accepted)

    @property
    def rejected(self) -> "tuple[RegionReport, ...]":
        """被丢掉的候选：霍夫在夹爪圆弧、背景纹理上凑出来的圆。"""
        return tuple(report for report in self.regions if not report.accepted)

    @property
    def best(self) -> Optional[RegionReport]:
        """Prefer a classified material; otherwise the largest accepted face.

        YOLO 认出类别时按置信度取（原行为）。YOLO 一个框都没给时**不能取第一个圆**：
        霍夫在夹爪、背景上凑出来的圆半径各不相同，取错一个距离就差好几毫米，
        而下游（逆运动学）只拿到一个数。这时取**半径最大的那个已接受顶面**：
        同一个物料的顶面常常被凑出大小两个同心圆（内侧亮面 + 整块顶面），
        小的那个会把距离报远，大的才是物体真正的边界。一个都没有就返回 ``None``。
        """
        if not self.regions:
            return None
        # **只在通过校验的候选里挑**：模型框到的区域也可能是被丢掉的夹爪圆弧，
        # 拿它去算逆解就等于把"已知不对"的数当答案。
        faces = self.faces
        classified = [report for report in faces if report.classification.found]
        if classified:
            return max(classified, key=lambda report: report.classification.score)
        if not faces:
            return None
        return max(faces, key=lambda report: report.region.radius)

    def summary(self) -> str:
        """List circles left to right, including unclassified ones."""
        lines = [
            f"检出 {len(self.regions)} 个候选圆面（接受 {len(self.faces)} 个，"
            f"丢掉 {len(self.rejected)} 个），YOLO 检出 {len(self.hits)} 个框"
        ]
        if self.colour_source == "model":
            lines.append("  调色板里没有这块料：参考色是从模型框里现取的"
                         "（要固定下来就把它加进 targets.material_palette）")
        elif self.colour_source == "palette-relaxed":
            lines.append("  严格颜色容差一个色块都没找到，已按 targets.colour_relax_factor "
                         "放大后的 BGR 容差再分割一遍（光照/颜色漂移时的救回路径）")
        elif self.colour_fallback:
            lines.append("  调色板里没有这块料，模型也没认出来：已退回霍夫的形状判据，"
                         "请把它的颜色加进 targets.material_palette")
        if self.radius_window_px is not None:
            low, high = self.radius_window_px
            lines.append(f"  半径窗口 {low:.0f}–{high:.0f} px（按工作距离先验）")
        for index, report in enumerate(self.faces, start=1):
            lines.append(f"  物料 {index}: {report.summary()}")
        for report in self.rejected:
            lines.append(f"  丢掉: {report.summary()}")
        best = self.best
        lines.append(
            "  采用：" + (best.summary() if best is not None
                          else "没有通过校验的顶面，距离不可用")
        )
        return "\n".join(lines)

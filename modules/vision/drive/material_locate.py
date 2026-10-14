"""物料定位：霍夫圆面决定存在与距离，YOLO 只补充类别。

圆面实际直径固定为 30 mm。未传入像素焦距时，使用实拍样本的粗标定值。

命令行自测::

    python -m modules.vision.drive.material_locate                   # 合成场景全流程自测（不需要硬件）
    python -m modules.vision.drive.material_locate --image a.jpg     # 真模型 + 一张图片
    python -m modules.vision.drive.material_locate --camera 1        # 真模型 + 摄像头预览，按 q 退出
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from modules.vision.drive.hough_lock import HoughRingLocator, LockedRegion, RingLock
from modules.vision.drive.material_blob import ColourBlobLocator, sample_reference
from modules.vision.drive.material_class import ClassResult, MaterialClassifier
from modules.vision.drive.material_face import FaceQuality, measure_face, to_grayscale
from modules.vision.drive.material_yolo import MaterialYOLO
from modules.vision.drive.pinhole import (
    DEFAULT_FOCAL_PX_AT_1280 as DEFAULT_FOCAL_PX_AT_1280,
    PinholeCamera,
    normalize_depth_range,
    resolve_focal_px,
)
from modules.vision.drive.yolov8 import Detection


@dataclass(frozen=True)
class RegionReport:
    """一个圆面的检测结果；相机坐标单位为 mm，右/下/镜头前方为正。"""

    region: LockedRegion
    classification: ClassResult
    #: 这个圆面"像不像物料顶面"：内部均匀度与边界支撑，见 ``material_face``。
    face: FaceQuality
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
    def label(self) -> str | None:
        return self.classification.label

    @property
    def name(self) -> str | None:
        return self.classification.name

    @property
    def confidence(self) -> float | None:
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

    lock: RingLock
    hits: tuple[Detection, ...]
    regions: tuple[RegionReport, ...]
    #: 本帧实际使用的像素焦距；没有圆面、也没配深度窗口时为 ``None``。
    focal_px: float | None = None
    #: 本帧实际使用的半径窗口 ``(最小, 最大)``；没配深度窗口时为 ``None``。
    radius_window_px: tuple[float, float] | None = None
    #: 这一帧的候选是从哪来的：``"palette"`` 配置的调色板 / ``"model"`` 模型框里现取的
    #: 参考色 / ``"hough"`` 颜色没配上、退回霍夫的形状判据；``None`` 表示什么都没找到。
    colour_source: str | None = None

    @property
    def colour_fallback(self) -> bool:
        """颜色块一个都没找到、退回霍夫形状判据时为 ``True``（多半是调色板没配这块料）。"""
        return self.colour_source == "hough"

    @property
    def present(self) -> bool:
        """Whether Hough found at least one material top face."""
        return bool(self.regions)

    @property
    def faces(self) -> tuple[RegionReport, ...]:
        """最终按物料顶面接受的圆面。"""
        return tuple(report for report in self.regions if report.accepted)

    @property
    def rejected(self) -> tuple[RegionReport, ...]:
        """被丢掉的候选：霍夫在夹爪圆弧、背景纹理上凑出来的圆。"""
        return tuple(report for report in self.regions if not report.accepted)

    @property
    def best(self) -> RegionReport | None:
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


class MaterialLocator:
    """Locate top faces first, then classify YOLO boxes inside each circle.

    Parameters:
        locator: injected circle locator; defaults to :class:`HoughRingLocator`, or to
            :class:`ColourBlobLocator` when ``material_bgr`` is given.
        material_bgr: 物料顶面的参考颜色（BGR）。给了它就以**颜色块**为准
            （夹爪、背景、线材根本进不了掩膜），不再用霍夫猜圆；``None`` 表示
            不按颜色区分，走霍夫的形状判据。参见 ``material_palette``。
        material_palette: ``{"green": [165,252,114], "blue": [248,122,41]}`` 这样的
            **调色板**：换物料颜色只改配置，不用改代码。给了它就优先于
            ``material_bgr``。颜色块一个都没找到时会自动退回霍夫的形状判据
            （报告里 ``colour_fallback`` 为 True），免得"忘了配颜色"变成"什么都看不见"。
        colour_tolerance: 颜色容差（BGR 欧氏距离）。
        depth_range_mm: 相机到物料顶面的深度范围 ``(近, 远)``，单位毫米。给了它，
            每一帧的半径窗口就由针孔模型算出来（而不是按画幅猜）：近端给出大半径。
            这是"相机装在固定高度"这个物理事实的唯一去处，也是把远处的同色物体、
            夹爪圆弧这类假圆排除掉的关键。``None`` 表示不设先验（原行为）。
    """

    def __init__(
        self,
        *,
        locator=None,
        yolo: MaterialYOLO | None = None,
        classifier: MaterialClassifier | None = None,
        model_path: str | Path | None = None,
        conf_thresh: float | None = None,
        iou_thresh: float | None = None,
        region_margin: float = 0.0,
        focal_px: float | None = None,
        top_diameter_mm: float = 30.0,
        depth_range_mm: tuple[float, float] | None = None,
        material_bgr: tuple[float, float, float] | None = None,
        material_palette: Mapping[str, Sequence[float]] | None = None,
        colour_tolerance: float = 60.0,
    ) -> None:
        if focal_px is not None and focal_px <= 0:
            raise ValueError("focal_px must be positive")
        if top_diameter_mm <= 0:
            raise ValueError("top_diameter_mm must be positive")
        self.material_bgr = (
            None if material_bgr is None
            else tuple(float(channel) for channel in material_bgr)
        )
        self.material_palette = (
            None if material_palette is None
            else {str(name): tuple(float(c) for c in value)
                  for name, value in dict(material_palette).items()}
        )
        self.colour_tolerance = float(colour_tolerance)
        if locator is not None:
            self.locator = locator
        elif self.material_palette:
            self.locator = ColourBlobLocator(self.material_palette, tolerance=self.colour_tolerance)
        elif self.material_bgr is not None:
            self.locator = ColourBlobLocator(self.material_bgr, tolerance=self.colour_tolerance)
        else:
            self.locator = HoughRingLocator()
        self._fallback_locator: HoughRingLocator | None = None
        self.yolo = (
            yolo
            if yolo is not None
            else MaterialYOLO(
                model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh
            )
        )
        self.classifier = classifier if classifier is not None else MaterialClassifier()
        self.region_margin = float(region_margin)
        self.focal_px = None if focal_px is None else float(focal_px)
        self.top_diameter_mm = float(top_diameter_mm)
        self.depth_range_mm = (
            None if depth_range_mm is None else normalize_depth_range(depth_range_mm)
        )

    # ------------------------------------------------------------- 生命周期
    def close(self) -> None:
        """Release the YOLO detector."""
        self.yolo.close()

    def __enter__(self) -> "MaterialLocator":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def colour_first(self) -> bool:
        """配了物料颜色时，颜色块就是判据，不再用形状指标否决它。"""
        return (
            (self.material_palette is not None or self.material_bgr is not None)
            and isinstance(self.locator, ColourBlobLocator)
        )

    # ------------------------------------------------------------- 主流程
    def locate(self, frame: np.ndarray) -> MaterialReport:
        """从一帧 BGR 图像返回圆面、类别和近似相机坐标。"""
        height, width = frame.shape[:2]
        window = self.radius_window(width, height)
        # 灰度图**一帧只转一次**，而且要"用到才转"：measure_face 每量一个候选圆
        # 都要用同一张灰度图，而 1280×720 转一次是 12.0 ms（实测）。霍夫在真实
        # 台面图上会给 24–87 个候选，逐候选重转就是 0.3–1.0 s 白花的时间；
        # 一个候选都没有时（常见：这一帧台面上没料）则一次都不用转。
        # 两遍（严格参数 + 放松参数）共用同一张，见 _retry_relaxed。
        gray_frame = None

        def grayscale():
            """本帧的 float64 灰度图（首次调用时转换，之后共用）。"""
            nonlocal gray_frame
            if gray_frame is None:
                gray_frame = to_grayscale(frame)
            return gray_frame

        lock = self._locate_circles(frame, window, self.locator)
        qualities = self._face_qualities(grayscale, lock)
        source = "palette" if lock.regions else None
        hits: tuple[Detection, ...] = ()
        asked_yolo = False
        if not lock.regions and self.colour_first:
            # 调色板里没有这块料。先问模型：它认出来了，就从它框的中心取一次参考色，
            # 拿这个参考色再分割一遍——换一块料（红、黄……）就不必先量颜色。
            hits = tuple(self.yolo.detect(frame, materials_only=True))
            asked_yolo = True
            sampled = self._sampled_palette(frame, hits)
            if sampled:
                trial = ColourBlobLocator(sampled, tolerance=self.colour_tolerance)
                candidate = self._locate_circles(frame, window, trial)
                if candidate.regions:
                    lock = candidate
                    qualities = self._face_qualities(grayscale, candidate)
                    source = "model"
            if not lock.regions:
                # 模型也没认出来：退回霍夫的形状判据，并在报告里说清是退回来的结果。
                lock, qualities = self._locate_hough(frame, grayscale, window)
                source = "hough" if lock.regions else None
        elif not any(quality.looks_like_face for quality in qualities.values()):
            # 主定位器支持放松时有第二遍：近距离焦画面里真实圆边的梯度被摊平，
            # 严格 Canny 阈值只剩夹爪弧线。锐利画面第一遍就找到顶面，不会跑第二遍。
            lock, qualities = self._retry_relaxed(
                frame, grayscale, window, self.locator, lock, qualities
            )
        focal_px = self.resolved_focal_px(width, height, required=bool(lock.regions))
        if not lock.regions:
            return MaterialReport(
                lock=lock, hits=hits, regions=(), focal_px=focal_px,
                radius_window_px=window, colour_source=source,
            )
        if not hits and not asked_yolo:
            # 上面已经问过模型的话，这一帧就不再问第二遍：同一个 frame、同样的参数，
            # 结果必然一样，重跑一次只是白花 20 ms（ONNX 通道实测）。
            hits = tuple(self.yolo.detect(frame, materials_only=True))
        # 画幅规则与像素焦距只有一个定义，在 modules.vision.drive.pinhole 里。
        focal_px = resolve_focal_px(width, height, self.focal_px)
        reports = tuple(
            self._report_for_region(
                frame, hits, region, qualities[region], focal_px, width, height,
                colour_path=source in ("palette", "model"),
            )
            for region in lock.regions
        )
        return MaterialReport(
            lock=lock, hits=hits, regions=reports, focal_px=focal_px,
            radius_window_px=window, colour_source=source,
        )

    @staticmethod
    def _sampled_palette(frame: np.ndarray, hits: Sequence[Detection]) -> dict:
        """从模型框里现取参考色；取不出来就返回空（调用方会走别的退路）。"""
        sampled = {}
        for hit in hits:
            colour = sample_reference(frame, (hit.x1, hit.y1, hit.x2, hit.y2))
            if colour is not None:
                sampled[hit.name] = colour
        return sampled

    def _locate_hough(self, frame: np.ndarray, grayscale, window):
        """退回霍夫那一路，含"严格参数没找到顶面就放松 Canny 阈值再试一次"。"""
        locator = self._hough_fallback()
        lock = self._locate_circles(frame, window, locator)
        qualities = self._face_qualities(grayscale, lock)
        return self._retry_relaxed(frame, grayscale, window, locator, lock, qualities)

    def _retry_relaxed(self, frame: np.ndarray, grayscale, window, locator, lock, qualities):
        """严格参数没找到像顶面的圆时，用放松版定位器再跑一遍。

        放松只在这里发生（而且只在第一遍一无所获时），所以锐利画面、以及没有近距
        离焦问题的机位行为完全不变。
        """
        if any(quality.looks_like_face for quality in qualities.values()):
            return lock, qualities
        factory = getattr(locator, "relaxed", None)
        if factory is None:
            return lock, qualities
        candidate = self._locate_circles(frame, window, factory())
        candidate_qualities = self._face_qualities(grayscale, candidate)
        if any(quality.looks_like_face for quality in candidate_qualities.values()):
            return candidate, candidate_qualities
        return lock, qualities

    def _hough_fallback(self) -> HoughRingLocator:
        """颜色没配到时的退路：霍夫 + 形状判据（懒建，只有真需要时才存在）。"""
        if self._fallback_locator is None:
            self._fallback_locator = HoughRingLocator()
        return self._fallback_locator

    def _locate_circles(self, frame: np.ndarray, window, locator) -> RingLock:
        """用（可选的）半径窗口调一次圆定位器。"""
        if window is None:
            return locator.locate(frame)
        return locator.locate(frame, min_radius=window[0], max_radius=window[1])

    def _materials(self) -> tuple[str, ...]:
        """这一帧按哪些类别名算物料。

        ``MaterialYOLO`` 会在模型自己不声明 ``obj-*`` 时把所有类别都当物料
        （换了模型不必改代码）；注入的假检测器没有这条信息时退回分类器默认表。
        """
        names = getattr(self.yolo, "effective_materials", None)
        return tuple(names) if names else tuple(self.classifier.materials)

    @staticmethod
    def _face_qualities(grayscale, lock: RingLock) -> dict:
        """这一批锁定圆面各自的顶面质量。

        ``grayscale`` 是**取灰度图的可调用对象**（一帧只转一次，用到才转）：
        候选一个都没有时它一次都不会被调用，有几十个候选时也只转一次。
        """
        return {region: measure_face(grayscale(), region) for region in lock.regions}

    @staticmethod
    def _accepted(region: LockedRegion, face: FaceQuality, hits, *, colour_path: bool) -> bool:
        """最终认不认这个候选：**形状判据通过，或者模型给它盖了章**。

        两条证据互相补：顶面被光照分成亮暗两块时，颜色掩膜的边界会毛糙、形状判据
        可能不通过，但模型照样框得准；反过来模型在近距离焦画面里可能一个框都没有，
        这时只能信形状。走霍夫那一路（``colour_path=False``）时没有颜色证据，
        就只认形状判据——上一版在这里漏了一刀，把退回来的几十个假圆全当物料了。
        """
        if face.looks_like_face:
            return True
        if not colour_path:
            return False
        return any(
            region.contains(*hit.center) or _inside_box(region.center_x, region.center_y, hit)
            for hit in hits
        )

    def radius_window(self, width: int, height: int) -> tuple[float, float] | None:
        """按 ``depth_range_mm`` 先验算本帧的霍夫半径窗口；没配就返回 ``None``。"""
        if self.depth_range_mm is None:
            return None
        camera = PinholeCamera(
            focal_px=resolve_focal_px(width, height, self.focal_px),
            image_width=float(width),
            image_height=float(height),
        )
        return camera.radius_range_for_depth(self.depth_range_mm, self.top_diameter_mm)

    def resolved_focal_px(self, width: int, height: int, *, required: bool) -> float | None:
        """本帧的像素焦距；没有圆面时用不到画幅规则，可以先不报错。"""
        if not required:
            try:
                return resolve_focal_px(width, height, self.focal_px)
            except ValueError:
                return None
        return resolve_focal_px(width, height, self.focal_px)

    def _report_for_region(
        self,
        frame: np.ndarray,
        hits: Sequence[Detection],
        region: LockedRegion,
        face: FaceQuality,
        focal_px: float,
        width: int,
        height: int,
        *,
        colour_path: bool,
    ) -> RegionReport:
        """用圆面直径估深度，再将圆心投影回相机坐标系。"""
        visible = [
            hit for hit in hits
            if region.contains(*hit.center, margin=self.region_margin)
        ]
        fix = PinholeCamera(
            focal_px=focal_px,
            image_width=float(width),
            image_height=float(height),
        ).fix_from_circle(
            region.center_x,
            region.center_y,
            region.radius,
            self.top_diameter_mm,
        )
        return RegionReport(
            region=region,
            classification=self.classifier.classify(visible, materials=self._materials()),
            face=face,
            accepted=self._accepted(region, face, hits, colour_path=colour_path),
            x_mm=fix.x_mm,
            y_mm=fix.y_mm,
            distance_mm=fix.depth_mm,
        )

    # ------------------------------------------------------------- 画图
    def draw(
        self,
        frame: np.ndarray,
        report: MaterialReport,
        *,
        circle_color: tuple[int, int, int] = (0, 0, 255),
        text_color: tuple[int, int, int] = (0, 255, 255),
        best_color: tuple[int, int, int] = (0, 255, 0),
        rejected_color: tuple[int, int, int] = (128, 128, 128),
    ) -> np.ndarray:
        """Return a copy marked with material circles and their distances.

        被 ``report.best`` 选中的那个圆画成 ``best_color``（默认绿色）并加粗；
        其余通过的画成 ``circle_color``；**被丢掉的候选画成灰色细圈并标 rejected**，
        不再让夹爪圆弧看起来像物料。
        """
        import cv2

        annotated = frame.copy()
        chosen = report.best
        chosen_region = None if chosen is None else chosen.region
        accepted = {id(item.region) for item in report.faces}
        for index, region in enumerate(report.lock.regions, start=1):
            centre = (int(round(region.center_x)), int(round(region.center_y)))
            is_best = region is chosen_region
            kept = id(region) in accepted
            if is_best:
                color, thickness, tag = best_color, 3, f"material{index}*"
            elif kept:
                color, thickness, tag = circle_color, 2, f"material{index}"
            else:
                color, thickness, tag = rejected_color, 1, f"rejected{index}"
            cv2.circle(annotated, centre, int(round(region.radius)), color, thickness)
            cv2.putText(
                annotated,
                tag,
                (centre[0] - 20, centre[1] - int(region.radius) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        for region_report in report.faces:
            centre = (
                int(region_report.region.center_x),
                int(region_report.region.center_y),
            )
            # cv2.putText cannot draw Chinese; the full label is in summary().
            label = f"{region_report.name or 'material'} {region_report.distance_mm:.0f}mm"
            cv2.putText(
                annotated,
                label,
                (max(centre[0] - 40, 4), centre[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
            )
        return annotated


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """Check circle presence and 30 mm distance without a model or camera."""
    import cv2
    import numpy as np

    class EmptyYOLO:
        def detect(self, _frame, *, materials_only=False):
            return []

        def close(self):
            pass

    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 70, (220, 220, 220), 8)
    report = MaterialLocator(yolo=EmptyYOLO(), focal_px=300).locate(frame)
    print(report.summary())
    if len(report.regions) != 1:
        print("自测失败：应找到一个圆面")
        return 1
    result = report.regions[0]
    if result.name is not None or abs(result.distance_mm - 300 * 30 / (2 * result.region.radius)) > 1e-6:
        print("自测失败：YOLO 漏检后应保留圆面距离和未知类别")
        return 1
    print("自测通过：圆面决定存在与距离，YOLO 漏检只影响类别。")
    return 0


def _inside_box(x: float, y: float, hit) -> bool:
    """点是否落在检测框里（给"模型盖章"用，也不需要重叠面积那么讲究）。"""
    return hit.x1 <= x <= hit.x2 and hit.y1 <= y <= hit.y2


def parse_depth_range(text: str | None) -> tuple[float, float] | None:
    """把 ``"60,80"`` 解析成 ``(60.0, 80.0)``；空值返回 ``None``。"""
    if not text:
        return None
    parts = text.replace("，", ",").split(",")
    if len(parts) != 2:
        raise ValueError('深度范围要写成 "近,远"，例如 "60,80"（毫米）')
    return normalize_depth_range((float(parts[0]), float(parts[1])))


def parse_bgr(text) -> tuple[float, float, float] | None:
    """把 ``"165,252,114"``（或 ``[165, 252, 114]``）解析成 BGR 三元组。"""
    if text is None or text == "":
        return None
    if isinstance(text, str):
        parts = text.replace("，", ",").split(",")
        values = [float(part) for part in parts]
    else:
        values = [float(channel) for channel in text]
    if len(values) != 3:
        raise ValueError('物料颜色要写成 "B,G,R"，例如 "165,252,114"')
    if any(not 0 <= channel <= 255 for channel in values):
        raise ValueError("物料颜色的每个通道都要在 0–255 之间")
    return (values[0], values[1], values[2])


def parse_palette(text) -> dict[str, tuple[float, float, float]] | None:
    """把 ``"green:165,252,114;blue:248,122,41"`` 解析成调色板；空值返回 ``None``。

    也接受已经是 ``{名字: (B, G, R)}`` 的字典（配置读出来的就是那种）。
    """
    if text is None or text == "":
        return None
    if isinstance(text, Mapping):
        return {str(name): parse_bgr(value) for name, value in text.items()}
    palette = {}
    for entry in str(text).replace("；", ";").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name, _, colour = entry.partition(":")
        if not name.strip() or not colour.strip():
            raise ValueError('调色板要写成 "green:165,252,114;blue:248,122,41"')
        palette[name.strip()] = parse_bgr(colour)
    if not palette:
        raise ValueError('调色板至少要有一项，例如 "green:165,252,114"')
    return palette


def _run_image(args) -> int:
    import cv2

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"读不出图片：{args.image}")
        return 1
    with MaterialLocator(model_path=args.model, conf_thresh=args.conf, focal_px=args.focal,
                         depth_range_mm=args.depth_range, material_bgr=args.material_bgr,
                         colour_tolerance=args.tolerance) as locator:
        report = locator.locate(frame)
        print(report.summary())
        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save), locator.draw(frame, report))
            print(f"标注图已写入 {args.save}")
    return 0 if report.present else 1


def _run_camera(args) -> int:
    import cv2

    from modules.vision import USBCamera, close_previews, open_preview, run_preview

    window = open_preview()
    try:
        with MaterialLocator(
            model_path=args.model, conf_thresh=args.conf, focal_px=args.focal,
            depth_range_mm=args.depth_range, material_bgr=args.material_bgr,
            colour_tolerance=args.tolerance,
        ) as locator, USBCamera(args.camera, width=1280, height=720).initialize() as camera:
            print(f"摄像头 {args.camera}；窗口 {window!r}，按 q 退出")

            def overlay(annotated, report, _stats) -> None:
                best = None if report is None else report.best
                text = "no material" if best is None else f"{best.name or 'material'} {best.distance_mm:.0f}mm"
                cv2.putText(annotated, text, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            run_preview(camera, locator.locate, locator.draw, window=window, overlay=overlay)
    finally:
        close_previews()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="霍夫圆面定位 + 30 mm 测距 + YOLO 分类")
    parser.add_argument("--model", type=Path, default=None, help="YOLO 模型路径（默认按平台选）")
    parser.add_argument("--image", type=Path, help="在一张图片上跑全流程")
    parser.add_argument("--camera", type=int, help="打开摄像头实时预览")
    parser.add_argument("--save", type=Path, help="保存图片标注（只对 --image 有效）")
    parser.add_argument("--conf", type=float, default=None,
                        help="YOLO 分类阈值；默认取 config.json 的 model.conf_thresh")
    parser.add_argument("--focal", type=float, default=None,
                        help="当前画幅标定后的水平像素焦距；不传则用实拍样本粗标定")
    parser.add_argument("--depth", dest="depth_range", type=parse_depth_range, default=None,
                        help='相机到物料顶面的深度范围（毫米），例如 "60,80"；'
                             "给了它半径窗口就按针孔模型算，夹爪弧线、远处同色物体都会被排除")
    parser.add_argument("--bgr", dest="material_bgr", type=parse_bgr, default=None,
                        help='物料顶面参考颜色 "B,G,R"，例如 "165,252,114"；'
                             "给了它就走颜色块定位，不再用霍夫猜圆")
    parser.add_argument("--palette", dest="material_palette", type=parse_palette, default=None,
                        help='物料调色板 "green:165,252,114;blue:248,122,41"；'
                             "换物料颜色只改这里")
    parser.add_argument("--tolerance", type=float, default=60.0, help="颜色容差（默认 60）")
    args = parser.parse_args(argv)
    if args.camera is not None:
        return _run_camera(args)
    if args.image is not None:
        return _run_image(args)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

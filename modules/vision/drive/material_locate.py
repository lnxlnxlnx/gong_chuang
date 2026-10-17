"""物料定位：**YOLO 主算**，算不出来时才走辅助定位（颜色块 / 霍夫找顶面圆）。

## 两条路的顺序

1. **主算 = YOLO 的物料框**（:meth:`MaterialLocator.locate`）。模型给出物料框，就地用它：
   框中心当物料中心、框的等效半径 ``(w+h)/4`` 换深度。**一个像素的霍夫都不跑**——
   这是常态路径，也是快的来源（圆面那一路要给几十个候选逐个量顶面，本机实测一帧
   170–380 ms；一次 YOLO 推理约 20 ms）。
2. **辅助定位 = :meth:`MaterialLocator.locate_auxiliary`**，**只在主算一个合格物料框都
   给不出来时**执行。里面现在是"颜色块 / 霍夫找顶面圆"那套，**换算法就换这个方法**，
   主流程不用动。

## 框的几何：x/y 够用，z 别当真

框比物料顶面大——同色桌面、可见的侧面都会进框：实测 30 mm 顶面在一帧里是 253 px 直径，
模型给的框却是 401×439 px（``output/debug/now.png``），由框反推的深度因此偏近几十毫米。
**这不影响 x/y 对准**：伺服在**同一套读数**里闭环，尺度的系统偏差会被实测出来的雅可比
吸收掉；但报告里的 ``z_mm`` 不是真实距离。

圆面实际直径固定为 30 mm。未传入像素焦距时，使用实拍样本的粗标定值。

命令行自测::

    python -m modules.vision.drive.material_locate                   # 合成场景全流程自测（不需要硬件）
    python -m modules.vision.drive.material_locate --image a.jpg     # 真模型 + 一张图片
    python -m modules.vision.drive.material_locate --camera 1        # 真模型 + 摄像头预览，按 q 退出
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from modules.vision.drive.hough_lock import HoughRingLocator, LockedRegion, RingLock
from modules.vision.drive.material_blob import (
    DEFAULT_HUE_TOLERANCE,
    DEFAULT_MAX_BOUNDARY_VARIATION,
    DEFAULT_MIN_AREA_PX,
    DEFAULT_MIN_FILL_RATIO,
    DEFAULT_VALUE_RATIO,
    ColourBlobLocator,
    sample_reference,
)
from modules.vision.drive.material_class import ClassResult, MaterialClassifier
from modules.vision.drive.material_face import (
    DEFAULT_EDGE_SUPPORT_MIN,
    DEFAULT_INTERIOR_STD_MAX,
    FaceQuality,
    measure_face,
    to_grayscale,
)
from modules.vision.drive.material_yolo import MaterialYOLO
from modules.vision.drive.pinhole import (
    DEFAULT_FOCAL_PX_AT_1280 as DEFAULT_FOCAL_PX_AT_1280,
    PinholeCamera,
    normalize_depth_range,
    resolve_focal_px,
)
from modules.vision.drive.yolov8 import Detection
# 报告类型只有一份定义，住在 reports（那里顶层只依赖标准库）：
# 本模块是它的消费者，不再自己定义一份，也不做转发别名。
from modules.vision.reports import MaterialReport, RegionReport, normalize_want


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
        hue_tolerance: float = DEFAULT_HUE_TOLERANCE,
        value_ratio: float = DEFAULT_VALUE_RATIO,
        min_area_px: float = DEFAULT_MIN_AREA_PX,
        max_boundary_variation: float | None = DEFAULT_MAX_BOUNDARY_VARIATION,
        min_fill_ratio: float | None = DEFAULT_MIN_FILL_RATIO,
        interior_std_max: float = DEFAULT_INTERIOR_STD_MAX,
        edge_support_min: float = DEFAULT_EDGE_SUPPORT_MIN,
        colour_relax_factor: float = 1.5,
    ) -> None:
        if focal_px is not None and focal_px <= 0:
            raise ValueError("focal_px must be positive")
        if top_diameter_mm <= 0:
            raise ValueError("top_diameter_mm must be positive")
        if colour_relax_factor < 1.0:
            raise ValueError("colour_relax_factor must be at least 1.0")
        #: 形状判据的两个阈值：现场光照差时内部标准差会整体抬高，写死会把真顶面挡掉。
        self.interior_std_max = float(interior_std_max)
        self.edge_support_min = float(edge_support_min)
        #: 严格颜色容差一个色块都没找到时，按这个倍数再分割一遍（1.0 = 不放松）。
        self.colour_relax_factor = float(colour_relax_factor)
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
        #: 颜色块路径的全部放宽参数（默认值与 ``material_blob`` 的常量逐字相同）。
        self.blob_options = {
            "tolerance": self.colour_tolerance,
            "hue_tolerance": float(hue_tolerance),
            "value_ratio": float(value_ratio),
            "min_area_px": float(min_area_px),
            "max_boundary_variation": max_boundary_variation,
            "min_fill_ratio": min_fill_ratio,
        }
        if locator is not None:
            self.locator = locator
        elif self.material_palette:
            self.locator = ColourBlobLocator(self.material_palette, **self.blob_options)
        elif self.material_bgr is not None:
            self.locator = ColourBlobLocator(self.material_bgr, **self.blob_options)
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
    def locate(self, frame: np.ndarray, *, want=None) -> MaterialReport:
        """从一帧 BGR 图像返回物料、类别和近似相机坐标。

        ## 两条路，顺序是硬要求

        1. **主算：YOLO 的物料框。** 有框就用框，**一个像素的霍夫都不跑**——框的中心
           当物料中心、框的等效半径 ``(w+h)/4`` 换出深度。这是常态路径，也是快的来源：
           圆面那一路在真实台面图上要给几十个候选逐个量顶面（本机实测一帧 170–380 ms），
           而一次 YOLO 推理约 20 ms。
        2. **辅助定位（:meth:`locate_auxiliary`）：YOLO 一个物料框都给不出来时才跑。**
           YOLO 有结果就**不执行**它。里面现在是"颜色块 / 霍夫找顶面圆"那套，
           可以整个换掉（覆盖 :meth:`locate_auxiliary` 即可），主流程不用动。

        ``want`` 指定**要哪一个物料**：类别名（``"red"``）或从 1 开始的编号；
        它只影响 ``report.best`` 挑哪一个，整帧的候选照旧列在 ``report.regions`` 里。
        写法不合法当场报错（见 :func:`modules.vision.reports.normalize_want`）。

        ## 框的几何：x/y 够用，z 别当真

        框比物料顶面大——同色桌面、可见的侧面都会进框（实测 ``output/debug/now.png``：
        30 mm 顶面在画面里是 253 px，模型给的框是 401×439 px），所以由框反推的深度偏近。
        **这不影响 x/y 对准**：伺服是在**同一套读数**里闭环，尺度的系统偏差会被实测出来的
        雅可比吸收掉；但 ``report.best.z_mm`` 不是真实距离——要真实距离得让辅助定位去量圆面。
        """
        want = normalize_want(want)
        hits = tuple(self.yolo.detect(frame, materials_only=True))
        from_boxes = self._report_from_boxes(frame, hits, want=want)
        if from_boxes is not None:
            return from_boxes
        # 一个合格的物料框都没有：这才轮到辅助定位。框已经问过了，传下去（空元组）
        # 让它别再问第二遍——同一个 frame、同样的参数，结果必然一样，重跑只是白花 20 ms。
        return self.locate_auxiliary(frame, want=want, hits=hits)

    def _report_from_boxes(
        self, frame: np.ndarray, hits: Sequence[Detection], *, want=None
    ) -> MaterialReport | None:
        """**主算**：把每一个合格的物料框变成一个物料候选；一个都没有时返回 ``None``。

        每个框都进候选，不是只留置信度最高的那个——这样"画面里有红绿蓝三块"时
        ``locate("red")`` 才挑得到红的。**决定挑哪一个的地方只有一处**：
        :attr:`modules.vision.reports.MaterialReport.best`。
        """
        materials = self._materials()
        boxes = [hit for hit in hits if self.classifier.qualifies(hit, materials=materials)]
        if not boxes:
            return None
        height, width = frame.shape[:2]
        # 灰度图"用到才转、一帧只转一次"：每个框都要量一次"像不像顶面"，而 1280×720
        # 转一次是 12.0 ms（实测）。这个数**不参与接受与否**（框是主算，模型说它是物料
        # 它就是），只写进报告供排查——比如同色桌面那个假框，这一项会明显偏大。
        gray_frame = None

        def grayscale():
            nonlocal gray_frame
            if gray_frame is None:
                gray_frame = to_grayscale(frame)
            return gray_frame

        # 画幅规则与像素焦距只有一个定义，在 modules.vision.drive.pinhole 里。
        focal_px = resolve_focal_px(width, height, self.focal_px)
        regions = tuple(self._region_from_box(hit) for hit in boxes)
        reports = tuple(
            self._report_for_region(
                frame, boxes, region, measure_face(grayscale(), region),
                focal_px, width, height,
                # 主算路径没有"颜色路径"可言；模型框本身就是结论，不再让形状判据否决它。
                colour_path=False, force_accept=True,
            )
            for region in regions
        )
        return MaterialReport(
            lock=RingLock(regions, None), hits=hits, regions=reports,
            focal_px=focal_px, radius_window_px=self.radius_window(width, height),
            colour_source="yolo", want=want,
        )

    def locate_auxiliary(
        self, frame: np.ndarray, *, want=None, hits: Sequence[Detection] | None = None
    ) -> MaterialReport:
        """**辅助定位**：YOLO 一个物料框都给不出来时才走这条路。

        现在是"颜色块 / 霍夫找顶面圆"那套：圆的半径给出深度，模型框（``hits``）用来给
        圆面盖章、或从框里现取参考色。**它只在主算无结果时被调用**，所以"一个圆面都没通过
        校验"就是这一帧的最终结论——不会再兜第二次。

        **换检测算法就换这个方法**（覆盖它，或在子类里重写）：:meth:`locate` 不关心它内部
        怎么做，只要返回一个 :class:`~modules.vision.reports.MaterialReport`。

        :param hits: ``None`` = 还没问过模型（直接调这个方法的人），自己去问一次；
            空元组 = 主流程已经问过、一个框都没有——不再问第二遍。
        """
        want = normalize_want(want)
        # ``None`` 与空元组的区别就是"问过没有"：同一个 frame 问两遍模型是白花 20 ms。
        asked_yolo = hits is not None
        found: tuple[Detection, ...] = () if hits is None else tuple(hits)

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
        # 严格容差一个色块都没有时，先把"放大 BGR 容差"那一遍的结果**记下来**——
        # 它是一个**候补**，不是结论。**只放大 BGR 容差**：色相与亮度下限一放就会把
        # 台面收进掩膜（台面色相离绿料只差 14、亮度比 0.42），而容差放大后掩膜里
        # 出现的第一块"够大够圆"的色块完全可能是台面。实测 ``models/top_pic_2.jpg``：
        # 容差 60→90 之后先捞到的是左上角一块台面（r=197 px，正好落在半径窗口里），
        # 而它一旦成为结论，下面就再也不会去问模型——那块蓝料明明被模型框到
        # 0.96，却被这一块台面挡在门外。所以候补只在**模型也没辙**时才上场。
        relaxed_lock: RingLock | None = None
        if not lock.regions:
            relaxed = self._relaxed_locator()
            if relaxed is not None:
                candidate = self._locate_circles(frame, window, relaxed)
                if candidate.regions:
                    relaxed_lock = candidate
        if not lock.regions and self.colour_first:
            # 调色板里没有这块料。先问模型：它认出来了，就从它框的中心取一次参考色，
            # 拿这个参考色再分割一遍——换一块料（红、黄……）就不必先量颜色。
            if not asked_yolo:
                found = tuple(self.yolo.detect(frame, materials_only=True))
                asked_yolo = True
            sampled = self._sampled_palette(frame, found)
            if sampled:
                options = dict(self.blob_options)
                trial = ColourBlobLocator(sampled, **options)
                candidate = self._locate_circles(frame, window, trial)
                if candidate.regions:
                    lock = candidate
                    qualities = self._face_qualities(grayscale, candidate)
                    source = "model"
            if not lock.regions:
                if relaxed_lock is not None:
                    # 模型也没认出来，但放宽容差那一遍有东西：它仍然守着调色板的
                    # 色相与亮度下限，比霍夫的纯形状判据可信，所以排在霍夫前面。
                    lock = relaxed_lock
                    qualities = self._face_qualities(grayscale, relaxed_lock)
                    source = "palette-relaxed"
                else:
                    # 什么都没了：退回霍夫的形状判据，并在报告里说清是退回来的结果。
                    lock, qualities = self._locate_hough(frame, grayscale, window)
                    source = "hough" if lock.regions else None
        elif not any(self._looks_like_face(quality) for quality in qualities.values()):
            # 主定位器支持放松时有第二遍：近距离焦画面里真实圆边的梯度被摊平，
            # 严格 Canny 阈值只剩夹爪弧线。锐利画面第一遍就找到顶面，不会跑第二遍。
            lock, qualities = self._retry_relaxed(
                frame, grayscale, window, self.locator, lock, qualities
            )
        if not lock.regions and relaxed_lock is not None:
            # 没配颜色时不会走上面那一段，候补在这里补一次机会。
            lock = relaxed_lock
            qualities = self._face_qualities(grayscale, relaxed_lock)
            source = "palette-relaxed"
        # 圆面盖章要用模型框，所以有圆面时才需要问模型（问过就不再问）。
        if lock.regions and not asked_yolo:
            found = tuple(self.yolo.detect(frame, materials_only=True))
            asked_yolo = True

        focal_px = self.resolved_focal_px(width, height, required=bool(lock.regions))
        if not lock.regions:
            return MaterialReport(
                lock=lock, hits=found, regions=(), focal_px=focal_px,
                radius_window_px=window, colour_source=source, want=want,
            )
        # 画幅规则与像素焦距只有一个定义，在 modules.vision.drive.pinhole 里。
        focal_px = resolve_focal_px(width, height, self.focal_px)
        reports = tuple(
            self._report_for_region(
                frame, found, region, qualities[region], focal_px, width, height,
                colour_path=source in ("palette", "palette-relaxed", "model"),
            )
            for region in lock.regions
        )
        return MaterialReport(
            lock=lock, hits=found, regions=reports, focal_px=focal_px,
            radius_window_px=window, colour_source=source, want=want,
        )

    def _looks_like_face(self, quality: FaceQuality) -> bool:
        """按**配置里的阈值**判断一个候选像不像物料顶面（默认值与常量相同）。"""
        return quality.passes(
            interior_std_max=self.interior_std_max,
            edge_support_min=self.edge_support_min,
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
        if any(self._looks_like_face(quality) for quality in qualities.values()):
            return lock, qualities
        factory = getattr(locator, "relaxed", None)
        if factory is None:
            return lock, qualities
        candidate = self._locate_circles(frame, window, factory())
        candidate_qualities = self._face_qualities(grayscale, candidate)
        if any(self._looks_like_face(quality) for quality in candidate_qualities.values()):
            return candidate, candidate_qualities
        return lock, qualities

    def _hough_fallback(self) -> HoughRingLocator:
        """颜色没配到时的退路：霍夫 + 形状判据（懒建，只有真需要时才存在）。"""
        if self._fallback_locator is None:
            self._fallback_locator = HoughRingLocator()
        return self._fallback_locator

    # ------------------------------------------------------- 框 → 候选
    @staticmethod
    def _region_from_box(hit: Detection) -> LockedRegion:
        """把一个检测框**当成**一个"圆面"：中心取框中心，半径取等效半径 ``(w+h)/4``。

        框不是圆，而且比物料顶面大——同色桌面、可见的侧面都会进框：实测 30 mm 顶面在一帧
        里是 253 px 直径，模型给的框却是 401×439 px，于是由框反推的深度偏近几十毫米。
        **这不影响 x/y 对准**：伺服在**同一套读数**里闭环，尺度的系统偏差会被实测出来的
        雅可比吸收掉；但报告里的 z 不是真实距离，要真实距离得让辅助定位去量圆面。
        """
        return LockedRegion(
            center_x=(float(hit.x1) + float(hit.x2)) / 2.0,
            center_y=(float(hit.y1) + float(hit.y2)) / 2.0,
            radius=max((float(hit.width) + float(hit.height)) / 4.0, 1.0),
        )

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

    def _accepted(
        self,
        region: LockedRegion,
        face: FaceQuality,
        hits,
        *,
        colour_path: bool,
    ) -> bool:
        """最终认不认这个候选：**形状判据通过，或者模型给它盖了章**。

        两条证据互相补：顶面被光照分成亮暗两块时，颜色掩膜的边界会毛糙、形状判据
        可能不通过，但模型照样框得准；反过来模型在近距离焦画面里可能一个框都没有，
        这时只能信形状。走霍夫那一路（``colour_path=False``）时没有颜色证据，
        就只认形状判据——上一版在这里漏了一刀，把退回来的几十个假圆全当物料了。

        形状判据用的是**配置里的**阈值（``interior_std_max`` / ``edge_support_min``），
        现场光照差时内部标准差会整体抬高，写死的 12.0 会把真顶面一起挡掉。
        """
        if self._looks_like_face(face):
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

    def _relaxed_locator(self) -> "ColourBlobLocator | None":
        """严格容差没找到色块时的第二遍定位器；不需要放松时返回 ``None``。

        只在两件事同时成立时给出：调用方配了 ``colour_relax_factor > 1``，而且主定位器
        确实是颜色块定位器（霍夫那条路有自己的 ``relaxed``，不归这里管）。
        **只放大 BGR 容差**：色相与亮度下限保持不动，因为台面正好卡在那两个阈值的另一侧
        （色相差 14、亮度比 0.42），一放就会把整张台面收进掩膜。

        代价是"什么都没找到"的那一帧要多跑一遍颜色分割（实测那一遍是整帧最贵的一段），
        所以它只在**一无所获**时发生：找到料的那一帧行为与开销都不变。
        """
        if self.colour_relax_factor <= 1.0:
            return None
        if not isinstance(self.locator, ColourBlobLocator):
            return None
        options = dict(self.blob_options)
        options["tolerance"] = self.colour_tolerance * self.colour_relax_factor
        return ColourBlobLocator(self.locator.references, **options)

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
        force_accept: bool = False,
    ) -> RegionReport:
        """用圆面直径估深度，再将圆心投影回相机坐标系。

        ``force_accept=True`` 给框兜底那条路用：那个"圆面"就是模型框本身，
        "像不像顶面"与"模型有没有盖章"两条判据都不适用（它们否掉的正是这一帧）。
        """
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
            accepted=(
                True if force_accept
                else self._accepted(region, face, hits, colour_path=colour_path)
            ),
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
    """两条路各验一次，都不需要摄像头和真模型。"""
    import cv2
    import numpy as np

    class BoxYOLO:
        """主算用：一个物料框，别的什么都不给。"""

        def __init__(self, hits=()) -> None:
            self.hits = tuple(hits)

        def detect(self, _frame, *, materials_only=False):
            return list(self.hits)

        def close(self):
            pass

    # --- 主算：有框就用框，一个像素的霍夫都不跑 ---------------------------------
    blank = np.full((480, 640, 3), 80, dtype=np.uint8)
    box = Detection(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.9, class_id=0, name="red")
    primary = MaterialLocator(yolo=BoxYOLO([box]), focal_px=300).locate(blank, want="red")
    if len(primary.regions) != 1 or primary.best is None or primary.best.name != "red":
        print(f"自测失败：主算应从模型框给出 1 个 red 候选，实际 {primary.summary()}")
        return 1
    if primary.colour_source != "yolo" or not primary.from_model_box:
        print(f"自测失败：主算结果的来源应标成 yolo，实际 {primary.colour_source!r}")
        return 1
    # 框 (280,200)-(360,280)：中心 (320,240)，等效半径 (80+80)/4 = 40 px -> 112.5 mm。
    expected_mm = 300 * 30 / (2 * 40.0)
    if abs(primary.best.distance_mm - expected_mm) > 1e-6:
        print(f"自测失败：框的等效半径应给出 {expected_mm} mm，实际 {primary.best.distance_mm}")
        return 1

    class CountingAuxiliary(MaterialLocator):
        """把辅助定位换成一个计数器：主算有结果时它必须一次都不执行。"""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.calls = 0

        def locate_auxiliary(self, frame, *, want=None, hits=None):
            self.calls += 1
            return super().locate_auxiliary(frame, want=want, hits=hits)

    counting = CountingAuxiliary(yolo=BoxYOLO([box]), focal_px=300)
    counting.locate(blank, want="red")
    if counting.calls != 0:
        print(f"自测失败：主算有结果时辅助定位不该执行，实际执行了 {counting.calls} 次")
        return 1

    # --- 辅助定位：模型一个框都没有时才跑，圆面照旧给出毫米 ----------------------
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 70, (220, 220, 220), 8)
    report = MaterialLocator(yolo=BoxYOLO(), focal_px=300).locate(frame)
    if len(report.regions) != 1:
        print("自测失败：辅助定位应找到一个圆面")
        return 1
    result = report.regions[0]
    if result.name is not None or abs(result.distance_mm - 300 * 30 / (2 * result.region.radius)) > 1e-6:
        print("自测失败：模型一个框都没有时，辅助定位应保留圆面距离和未知类别")
        return 1
    if report.from_model_box:
        print("自测失败：辅助定位的结果不该标成来自模型框")
        return 1

    print(report.summary())
    print("自测通过：YOLO 主算用框给出读数；它一个框都没有时，辅助定位用圆面兜住位置与距离。")
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


def parse_want(text: str | None):
    """把命令行的 ``--want`` 解析成 :func:`~modules.vision.reports.normalize_want` 的输入。

    纯数字当**编号**（``--want 2`` 就是「物料 2」），其余当**类别名**（``--want red``）；
    空值表示不指定。不在这里做合法性判断——那些规则只有一个出处（``normalize_want``）。
    """
    if text is None or text == "":
        return None
    stripped = text.strip()
    return int(stripped) if stripped.isdigit() else stripped


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


def _overridden_settings(args):
    """把命令行覆盖落到**配置快照**上；本函数不打开任何设备。

    ``--bgr`` 与 ``--palette`` 同时给时，:class:`MaterialLocator` 里既有的优先级
    （palette 优先于单一 BGR）照旧生效，这里不重复实现一遍。
    """
    from modules.vision.runtime import _load_settings, _override_settings

    return _override_settings(
        _load_settings(),
        model_path=args.model,
        camera_id=args.camera,
        conf_thresh=args.conf,
        depth_range_mm=args.depth_range,
        material_bgr=args.material_bgr,
        material_palette=args.material_palette,
        colour_tolerance=args.tolerance,
        top_diameter_mm=args.diameter,
    )


def read_image(path) -> "np.ndarray | None":
    """读一张图片，**支持非 ASCII 路径**；读不出来返回 ``None``。

    ``cv2.imread`` 在 Windows 上按 ANSI 打开文件，路径里有中文就直接返回 ``None``
    （实测 ``models/蓝.jpg``）。所以先自己读字节再交给 ``cv2.imdecode``。
    """
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, frame) -> bool:
    """写一张图片，**支持非 ASCII 路径**；成功返回 ``True``。

    ``cv2.imwrite`` 有和 :func:`read_image` 同样的路径问题，所以先编码成内存里的
    字节再落盘。后缀决定格式（缺省 PNG）。
    """
    import cv2

    suffix = Path(path).suffix or ".png"
    try:
        ok, buffer = cv2.imencode(suffix, frame)
    except cv2.error:
        return False
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


def _run_image(args) -> int:
    """``--image``：只加载定位器处理给定图片，**不创建相机**。"""
    from modules.vision.runtime import _build_locator

    frame = read_image(args.image)
    if frame is None:
        print(f"读不出图片：{args.image}")
        return 1
    settings = _overridden_settings(args)
    # 焦距按原样直传：图片路径一直用的是显式 --focal（或粗标定），与摄像头会话一致。
    locator = _build_locator(settings, focal_px=args.focal)
    try:
        report = locator.locate(frame, want=args.want)
        print(report.summary())
        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if not write_image(args.save, locator.draw(frame, report)):
                print(f"写不出图片：{args.save}")
                return 1
            print(f"标注图已写入 {args.save}")
    finally:
        locator.close()
    return 0 if report.present else 1


def _run_camera(args) -> int:
    """``--camera``：复用与菜单 18 相同的快照和会话创建路径。"""
    import cv2

    from modules.vision.drive.preview import close_previews, open_preview
    from modules.vision.drive.stream import run_preview
    from modules.vision.runtime import Vision

    settings = _overridden_settings(args)
    window = open_preview()
    try:
        with Vision._open_config(settings, focal_px=args.focal) as vision:
            print(f"摄像头 {args.camera}；窗口 {window!r}，按 q 退出")
            if args.want is not None:
                print(f"只要 {args.want}（类别名或编号）；画面里没有它时标成 no material")

            def overlay(annotated, report, _stats) -> None:
                best = None if report is None else report.best
                text = "no material" if best is None else f"{best.name or 'material'} {best.distance_mm:.0f}mm"
                cv2.putText(annotated, text, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ``run_preview`` 只按 ``frame`` 调定位器，所以把 "要哪一个" 绑在这里。
            locate = functools.partial(vision._locator.locate, want=args.want)
            run_preview(vision._camera, locate, vision._locator.draw,
                        window=window, overlay=overlay)
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
    parser.add_argument("--diameter", type=float, default=None,
                        help="物料顶面真实直径（毫米）；默认取 config.json 的 "
                             "targets.material_diameter_mm")
    parser.add_argument("--want", default=None,
                        help='画面里有多种物料时只要哪一种：类别名（如 "red"）'
                             "或从 1 开始的编号；省略则按原规则挑"
                             "（认出来的里置信度最高的那个）")
    args = parser.parse_args(argv)
    if args.camera is not None:
        return _run_camera(args)
    if args.image is not None:
        return _run_image(args)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

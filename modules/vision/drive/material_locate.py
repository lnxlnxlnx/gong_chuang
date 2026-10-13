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
from typing import Sequence

import numpy as np

from modules.vision.drive.hough_lock import HoughRingLocator, LockedRegion, RingLock
from modules.vision.drive.material_class import ClassResult, MaterialClassifier
from modules.vision.drive.material_face import FaceQuality, measure_face
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

    @property
    def present(self) -> bool:
        """Whether Hough found at least one material top face."""
        return bool(self.regions)

    @property
    def faces(self) -> tuple[RegionReport, ...]:
        """内部均匀、边界完整，真的像物料顶面的那些圆面。"""
        return tuple(report for report in self.regions if report.face.looks_like_face)

    @property
    def best(self) -> RegionReport | None:
        """Prefer a classified material; otherwise the most face-like circle.

        YOLO 认出类别时按置信度取（原行为）。YOLO 一个框都没给时**不能取第一个圆**：
        霍夫在夹爪、背景上凑出来的圆半径各不相同，取错一个距离就差好几毫米，
        而下游（逆运动学）只拿到一个数。这时改取最像物料顶面的那个——先要求边界
        完整，再比内部谁更均匀；一个都不像就返回 ``None``，让调用方看见"没锁定"。
        """
        if not self.regions:
            return None
        classified = [report for report in self.regions if report.classification.found]
        if classified:
            return max(classified, key=lambda report: report.classification.score)
        faces = self.faces
        if not faces:
            return None
        return min(faces, key=lambda report: report.face.interior_std)

    def summary(self) -> str:
        """List circles left to right, including unclassified ones."""
        lines = [
            f"检出 {len(self.regions)} 个物料圆面（其中 {len(self.faces)} 个像顶面），"
            f"YOLO 检出 {len(self.hits)} 个框"
        ]
        if self.radius_window_px is not None:
            low, high = self.radius_window_px
            lines.append(f"  半径窗口 {low:.0f}–{high:.0f} px（按工作距离先验）")
        for index, report in enumerate(self.regions, start=1):
            lines.append(f"  物料 {index}: {report.summary()}")
        best = self.best
        lines.append(
            "  采用：" + (best.summary() if best is not None
                          else "没有像物料顶面的圆面，距离不可用")
        )
        return "\n".join(lines)


class MaterialLocator:
    """Locate top faces first, then classify YOLO boxes inside each circle.

    Parameters:
        locator: injected circle locator; defaults to :class:`HoughRingLocator`.
        depth_range_mm: 相机到物料顶面的深度范围 ``(近, 远)``，单位毫米。给了它，
            每一帧的霍夫半径窗口就由针孔模型算出来（而不是按画幅猜）：近端给出
            大半径。这是"相机装在固定高度"这个物理事实的唯一去处，也是把夹爪弧线、
            背景平坦区这类假圆排除掉的关键。``None`` 表示不设先验（原行为）。
    """

    def __init__(
        self,
        *,
        locator: HoughRingLocator | None = None,
        yolo: MaterialYOLO | None = None,
        classifier: MaterialClassifier | None = None,
        model_path: str | Path | None = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        region_margin: float = 0.0,
        focal_px: float | None = None,
        top_diameter_mm: float = 30.0,
        depth_range_mm: tuple[float, float] | None = None,
    ) -> None:
        if focal_px is not None and focal_px <= 0:
            raise ValueError("focal_px must be positive")
        if top_diameter_mm <= 0:
            raise ValueError("top_diameter_mm must be positive")
        self.locator = locator if locator is not None else HoughRingLocator()
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

    # ------------------------------------------------------------- 主流程
    def locate(self, frame: np.ndarray) -> MaterialReport:
        """从一帧 BGR 图像返回圆面、类别和近似相机坐标。"""
        height, width = frame.shape[:2]
        window = self.radius_window(width, height)
        lock = self._locate_circles(frame, window, self.locator)
        # 每个候选只量一次顶面质量，下面生成报告时直接复用（7 cm 机位下这一步最贵）。
        qualities = self._face_qualities(frame, lock)
        if not any(quality.looks_like_face for quality in qualities.values()):
            # 严格参数没找到"像顶面的圆"时才放松 Canny 阈值：近距离焦画面里真实
            # 圆边的梯度被摊平，严格参数只剩夹爪弧线。放松只在这里发生，所以锐利
            # 画面（以及没有近距问题的机位）行为不变。
            relaxed = self._relaxed_locator()
            if relaxed is not None:
                candidate = self._locate_circles(frame, window, relaxed)
                candidate_qualities = self._face_qualities(frame, candidate)
                if any(quality.looks_like_face for quality in candidate_qualities.values()):
                    lock, qualities = candidate, candidate_qualities
        focal_px = self.resolved_focal_px(width, height, required=bool(lock.regions))
        if not lock.regions:
            return MaterialReport(
                lock=lock, hits=(), regions=(),
                focal_px=focal_px, radius_window_px=window,
            )
        hits = tuple(self.yolo.detect(frame, materials_only=True))
        # 画幅规则与像素焦距只有一个定义，在 modules.vision.drive.pinhole 里。
        focal_px = resolve_focal_px(width, height, self.focal_px)
        reports = tuple(
            self._report_for_region(frame, hits, region, qualities[region], focal_px, width, height)
            for region in lock.regions
        )
        return MaterialReport(
            lock=lock, hits=hits, regions=reports,
            focal_px=focal_px, radius_window_px=window,
        )

    def _locate_circles(self, frame: np.ndarray, window, locator) -> RingLock:
        """用（可选的）半径窗口调一次圆定位器。"""
        if window is None:
            return locator.locate(frame)
        return locator.locate(frame, min_radius=window[0], max_radius=window[1])

    def _relaxed_locator(self):
        """放松版圆定位器；注入的定位器没有这个能力时返回 ``None``。"""
        factory = getattr(self.locator, "relaxed", None)
        return None if factory is None else factory()

    @staticmethod
    def _face_qualities(frame: np.ndarray, lock: RingLock) -> dict:
        """这一批锁定圆面各自的顶面质量。"""
        return {region: measure_face(frame, region) for region in lock.regions}

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
            classification=self.classifier.classify(visible),
            face=face,
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
    ) -> np.ndarray:
        """Return a copy marked with material circles and their distances.

        被 ``report.best`` 选中的那个圆画成 ``best_color``（默认绿色）并加粗：
        下游拿哪个数一眼就能对上。
        """
        import cv2

        annotated = frame.copy()
        chosen = report.best
        chosen_region = None if chosen is None else chosen.region
        for index, region in enumerate(report.lock.regions, start=1):
            centre = (int(round(region.center_x)), int(round(region.center_y)))
            is_best = region is chosen_region
            cv2.circle(
                annotated,
                centre,
                int(round(region.radius)),
                best_color if is_best else circle_color,
                3 if is_best else 2,
            )
            cv2.putText(
                annotated,
                f"material{index}" + ("*" if is_best else ""),
                (centre[0] - 20, centre[1] - int(region.radius) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                best_color if is_best else circle_color,
                1,
            )

        for region_report in report.regions:
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


def parse_depth_range(text: str | None) -> tuple[float, float] | None:
    """把 ``"60,80"`` 解析成 ``(60.0, 80.0)``；空值返回 ``None``。"""
    if not text:
        return None
    parts = text.replace("，", ",").split(",")
    if len(parts) != 2:
        raise ValueError('深度范围要写成 "近,远"，例如 "60,80"（毫米）')
    return normalize_depth_range((float(parts[0]), float(parts[1])))


def _run_image(args) -> int:
    import cv2

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"读不出图片：{args.image}")
        return 1
    with MaterialLocator(model_path=args.model, conf_thresh=args.conf, focal_px=args.focal,
                         depth_range_mm=args.depth_range) as locator:
        report = locator.locate(frame)
        print(report.summary())
        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save), locator.draw(frame, report))
            print(f"标注图已写入 {args.save}")
    return 0 if report.present else 1


def _run_camera(args) -> int:
    import cv2

    from modules.vision import USBCamera, close_previews, open_preview

    window = open_preview()
    try:
        with MaterialLocator(
            model_path=args.model, conf_thresh=args.conf, focal_px=args.focal,
            depth_range_mm=args.depth_range,
        ) as locator, USBCamera(args.camera, width=1280, height=720).initialize() as camera:
            print(f"摄像头 {args.camera}；窗口 {window!r}，按 q 退出")
            while True:
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("摄像头未返回图像")
                report = locator.locate(frame)
                cv2.imshow(window, locator.draw(frame, report))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        close_previews()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="霍夫圆面定位 + 30 mm 测距 + YOLO 分类")
    parser.add_argument("--model", type=Path, default=None, help="YOLO 模型路径（默认按平台选）")
    parser.add_argument("--image", type=Path, help="在一张图片上跑全流程")
    parser.add_argument("--camera", type=int, help="打开摄像头实时预览")
    parser.add_argument("--save", type=Path, help="保存图片标注（只对 --image 有效）")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 分类阈值")
    parser.add_argument("--focal", type=float, default=None,
                        help="当前画幅标定后的水平像素焦距；不传则用实拍样本粗标定")
    parser.add_argument("--depth", dest="depth_range", type=parse_depth_range, default=None,
                        help='相机到物料顶面的深度范围（毫米），例如 "60,80"；'
                             "给了它半径窗口就按针孔模型算，夹爪弧线等假圆会被排除")
    args = parser.parse_args(argv)
    if args.camera is not None:
        return _run_camera(args)
    if args.image is not None:
        return _run_image(args)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

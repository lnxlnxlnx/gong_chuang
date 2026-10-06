"""物料定位：霍夫圆面决定存在与距离，YOLO 只补充类别。

圆面实际直径固定为 30 mm。未传入像素焦距时，使用实拍样本的粗标定值。

命令行自测::

    python -m vision.material_locate                   # 合成场景全流程自测（不需要硬件）
    python -m vision.material_locate --image a.jpg     # 真模型 + 一张图片
    python -m vision.material_locate --camera 1        # 真模型 + 摄像头预览，按 q 退出
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vision.hough_lock import HoughRingLocator, LockedRegion, RingLock
from vision.material_class import ClassResult, MaterialClassifier
from vision.material_yolo import MaterialYOLO


# 1280x720 照片中直径约 209 px，用户实测镜头到圆面约 145 mm：145*209/30≈1012。
DEFAULT_FOCAL_PX_AT_1280 = 1012.0


@dataclass(frozen=True)
class RegionReport:
    """一个圆面的检测结果；相机坐标单位为 mm，右/下/镜头前方为正。"""

    region: LockedRegion
    classification: ClassResult
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
        """Show location even when YOLO cannot classify the material."""
        category = f"{self.label}（{self.name}）" if self.name else "类别未知"
        return (
            f"@({self.region.center_x:.0f},{self.region.center_y:.0f}) "
            f"{category}，圆直径 {2 * self.region.radius:.0f} px，"
            f"距离约 {self.distance_mm:.0f} mm"
        )


@dataclass(frozen=True)
class MaterialReport:
    """一帧里所有检出的物料圆面。"""

    lock: RingLock
    hits: tuple
    regions: tuple[RegionReport, ...]

    @property
    def present(self) -> bool:
        """Whether Hough found at least one material top face."""
        return bool(self.regions)

    @property
    def best(self) -> RegionReport | None:
        """Prefer a classified material when more than one circle is found."""
        if not self.regions:
            return None
        return max(self.regions, key=lambda report: report.classification.score)

    def summary(self) -> str:
        """List circles left to right, including unclassified ones."""
        lines = [f"检出 {len(self.regions)} 个物料圆面，YOLO 检出 {len(self.hits)} 个框"]
        for index, report in enumerate(self.regions, start=1):
            lines.append(f"  物料 {index}: {report.summary()}")
        return "\n".join(lines)


class MaterialLocator:
    """Locate top faces first, then classify YOLO boxes inside each circle."""

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

    # ------------------------------------------------------------- 生命周期
    def close(self) -> None:
        """Release the YOLO detector."""
        self.yolo.close()

    def __enter__(self) -> "MaterialLocator":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------- 主流程
    def locate(self, frame) -> MaterialReport:
        """从一帧 BGR 图像返回圆面、类别和近似相机坐标。"""
        lock = self.locator.locate(frame)
        if not lock.regions:
            return MaterialReport(lock=lock, hits=(), regions=())
        hits = tuple(self.yolo.detect(frame, materials_only=True))
        height, width = frame.shape[:2]
        if self.focal_px is None:
            if abs(width / height - 16 / 9) > 0.02:
                raise ValueError("默认焦距只适用于无裁切 16:9 画幅；请传入该画幅标定后的 focal_px")
            focal_px = DEFAULT_FOCAL_PX_AT_1280 * width / 1280
        else:
            focal_px = self.focal_px
        reports = tuple(
            self._report_for_region(hits, region, focal_px, width, height)
            for region in lock.regions
        )
        return MaterialReport(lock=lock, hits=hits, regions=reports)

    def _report_for_region(
        self, hits, region: LockedRegion, focal_px: float, width: int, height: int
    ) -> RegionReport:
        """用圆面直径估深度，再将圆心投影回相机坐标系。"""
        visible = [
            hit for hit in hits
            if region.contains(*hit.center, margin=self.region_margin)
        ]
        depth_mm = focal_px * self.top_diameter_mm / (2 * region.radius)
        return RegionReport(
            region=region,
            classification=self.classifier.classify(visible),
            x_mm=(region.center_x - width / 2) * depth_mm / focal_px,
            y_mm=(region.center_y - height / 2) * depth_mm / focal_px,
            distance_mm=depth_mm,
        )

    # ------------------------------------------------------------- 画图
    def draw(
        self,
        frame,
        report: MaterialReport,
        *,
        circle_color=(0, 0, 255),
        text_color=(0, 255, 255),
    ):
        """Return a copy marked with material circles and their distances."""
        import cv2

        annotated = frame.copy()
        for index, region in enumerate(report.lock.regions, start=1):
            centre = (int(round(region.center_x)), int(round(region.center_y)))
            cv2.circle(annotated, centre, int(round(region.radius)), circle_color, 2)
            cv2.putText(
                annotated,
                f"material{index}",
                (centre[0] - 20, centre[1] - int(region.radius) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                circle_color,
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


def _run_image(args) -> int:
    import cv2

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"读不出图片：{args.image}")
        return 1
    with MaterialLocator(model_path=args.model, conf_thresh=args.conf, focal_px=args.focal) as locator:
        report = locator.locate(frame)
        print(report.summary())
        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save), locator.draw(frame, report))
            print(f"标注图已写入 {args.save}")
    return 0 if report.present else 1


def _run_camera(args) -> int:
    import cv2

    from vision import USBCamera, close_previews, open_preview

    window = open_preview()
    try:
        with MaterialLocator(
            model_path=args.model, conf_thresh=args.conf, focal_px=args.focal
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
    args = parser.parse_args(argv)
    if args.camera is not None:
        return _run_camera(args)
    if args.image is not None:
        return _run_image(args)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

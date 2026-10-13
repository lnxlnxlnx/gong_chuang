"""YOLO 推理物料候选：把一帧（或锁定区域）变成"这里有哪些框"。

职责单一：只做 **帧 → 检测框**。不判断"有没有物料"（``modules.vision.drive.material_presence``）、
不判断"是哪一种物料"（``modules.vision.drive.material_class``）、不算距离
（``modules.vision.drive.material_distance``）。串起来用见 ``modules.vision.drive.material_locate``。

两种用法：

- :meth:`MaterialYOLO.detect`：整帧推理，再按类别（只要物料）筛选。
- :meth:`MaterialYOLO.detect_in_region`：把霍夫锁定的圆环区域裁出来推理，
  再把框坐标**映射回整帧坐标系**，下游的几何计算不用关心裁剪。

命令行自测::

    python -m modules.vision.drive.material_yolo                    # 离线自测（假检测器，不需要模型）
    python -m modules.vision.drive.material_yolo --image a.jpg      # 用真模型在一张图上跑
    python -m modules.vision.drive.material_yolo --camera 1         # 摄像头实时预览，按 q 退出
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from modules.vision.drive.yolov8 import DEFAULT_CLASS_NAMES, Detection, Detector

if TYPE_CHECKING:
    from modules.vision.drive.hough_lock import LockedRegion

#: 算作物料的类别名。当前物料模型是专用的四分类（``red`` / ``yellow`` / ``blue`` /
#: ``green``），四类都是物料，所以这里就是模型的类别表本身；换模型改
#: ``modules/vision/config.json`` 的 ``model.class_names``（和可选的 ``model.materials``），
#: 这里是没配置时的兜底。
MATERIAL_CLASS_NAMES: tuple[str, ...] = tuple(DEFAULT_CLASS_NAMES)


def is_material(name: str, materials: Iterable[str] = MATERIAL_CLASS_NAMES) -> bool:
    """Return ``True`` when ``name`` is one of the material classes."""
    return name in set(materials)


def _configured_materials() -> tuple[str, ...]:
    """``config.json`` 的 ``model.materials``；没配或读不出来时返回空元组。

    读配置失败（文件坏了、字段写错）不该把整条流水线拖死：这里是"想更精确一点"的
    增强，失败就退回内置表，让上游按老行为跑。
    """
    try:
        from modules.vision.config import load_model_settings
    except ImportError:  # 单独跑本模块的自测时也可能走到这里
        return ()
    try:
        return tuple(load_model_settings().materials)
    except (OSError, ValueError):
        return ()


def is_inside_region(
    detection: Detection, region: LockedRegion, *, margin: float = 0.0
) -> bool:
    """Return ``True`` when a detection's centre falls inside a locked region."""
    center_x, center_y = detection.center
    return region.contains(center_x, center_y, margin=margin)


class MaterialYOLO:
    """Wrap one YOLO detector behind a "frame -> detections" interface.

    Parameters:
        model_path: ONNX (Windows) or TensorRT engine (Jetson) path; defaults
            to :func:`modules.vision.default_model_path`, which reads
            ``modules/vision/config.json``.
        conf_thresh / iou_thresh / class_names: ``None`` 表示"按 ``config.json`` 的
            ``model`` 块来"。显式传值只用于临时试验，正常部署不要在代码里写死。
        materials: class names treated as materials, used by ``materials_only``;
            ``None`` 表示按配置（见 :attr:`effective_materials`）。
        detector: an already-built detector (same ``detect()`` / ``draw()`` /
            ``close()`` interface, see :class:`modules.vision.Detector`); injected by
            tests or when sharing one model between modules. When given,
            ``model_path`` is ignored.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        conf_thresh: float | None = None,
        iou_thresh: float | None = None,
        class_names: Sequence[str] | None = None,
        materials: Sequence[str] | None = None,
        detector: Detector | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        # None = "还没定，等建检测器时问配置"；显式给了就一直是它。
        self.conf_thresh = None if conf_thresh is None else float(conf_thresh)
        self.iou_thresh = None if iou_thresh is None else float(iou_thresh)
        self.class_names = list(class_names) if class_names is not None else None
        self._materials = tuple(materials) if materials is not None else None
        self._detector = detector

    # ------------------------------------------------------------- 生命周期
    @property
    def detector(self) -> Detector:
        """Lazily build the real detector the first time it is needed."""
        if self._detector is None:
            from modules.vision import create_configured_detector

            # 路径、类别表、输入边长、阈值全部可以来自 config.json；
            # 这里显式传进去的才是覆盖值（None 表示不覆盖）。
            self._detector = create_configured_detector(
                model_path=self.model_path,
                class_names=self.class_names,
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh,
            )
        return self._detector

    def close(self) -> None:
        """Release the underlying detector (safe to call when injected)."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> "MaterialYOLO":
        self.detector  # 提前建立，避免第一帧才暴露模型缺失
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------- 推理
    @property
    def materials(self) -> tuple[str, ...]:
        """算作物料的类别名；等价于 :attr:`effective_materials`，保留旧名字。"""
        return self.effective_materials

    @property
    def effective_materials(self) -> tuple[str, ...]:
        """这一帧按哪些类别名算物料。

        优先级：构造时显式传的 ``materials`` → ``config.json`` 的 ``model.materials``
        （留空表示 ``model.class_names`` 全部算物料）→ 内置表。

        最后一档是给"没配置、也读不出模型类别表"的旧场景兜底：模型的类别名跟内置表
        完全对不上时，说明它换了一套命名（例如老比赛的 ``obj-*`` 十类表），
        那就按它自己声明的全部类别算——一个专用模型不该因为改了命名就把所有框过滤掉。
        """
        if self._materials is not None:
            return self._materials
        configured = _configured_materials()
        if configured:
            return configured
        names = tuple(getattr(self.detector, "class_names", ()) or ())
        if names and not (set(names) & set(MATERIAL_CLASS_NAMES)):
            return names
        return tuple(MATERIAL_CLASS_NAMES)

    def detect(self, frame: np.ndarray, *, materials_only: bool = False) -> list[Detection]:
        """Run one inference on the whole frame."""
        hits = list(self.detector.detect(frame))
        if materials_only:
            materials = set(self.effective_materials)
            hits = [hit for hit in hits if hit.name in materials]
        return hits

    def detect_in_region(
        self,
        frame: np.ndarray,
        region: LockedRegion,
        *,
        margin: float = 0.0,
        materials_only: bool = False,
        keep_outside: bool = False,
    ) -> list[Detection]:
        """Crop a locked region, run inference there, map boxes back to the frame.

        ``margin`` grows the crop (pixels) so a block straddling the ring edge is
        still whole. ``keep_outside=False`` drops detections whose centre ends up
        outside the grown region (the crop's corners reach outside the circle).
        """
        crop_box, offset = self.crop_region(frame, region, margin=margin)
        if crop_box is None:
            return []
        x1, y1, x2, y2 = crop_box
        cropped = frame[y1:y2, x1:x2]
        hits = list(self.detector.detect(cropped))
        mapped = [
            replace(
                hit,
                x1=hit.x1 + offset[0],
                y1=hit.y1 + offset[1],
                x2=hit.x2 + offset[0],
                y2=hit.y2 + offset[1],
            )
            for hit in hits
        ]
        if materials_only:
            materials = set(self.effective_materials)
            mapped = [hit for hit in mapped if hit.name in materials]
        if not keep_outside:
            grown = region.expanded(margin) if margin else region
            mapped = [hit for hit in mapped if is_inside_region(hit, grown)]
        return mapped

    @staticmethod
    def crop_region(
        frame: np.ndarray, region: LockedRegion, *, margin: float = 0.0
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None]:
        """Return ``((x1, y1, x2, y2), (offset_x, offset_y))`` clamped to the frame.

        ``(None, None)`` when the region lies completely outside the frame.
        """
        height, width = frame.shape[:2]
        grown = region.expanded(margin) if margin else region
        x1, y1, x2, y2 = grown.bbox
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            return None, None
        return (x1, y1, x2, y2), (x1, y1)


def _fake_frame(width: int = 640, height: int = 480):
    import numpy as np

    return np.zeros((height, width, 3), dtype=np.uint8)


def _self_test() -> int:
    """Hardware-free check of the crop / coordinate-mapping logic."""
    from modules.vision.drive.hough_lock import LockedRegion

    class FakeDetector:
        """Reports one material box at ``box`` plus one non-material box."""

        def __init__(self, box: tuple[float, float, float, float] = (4.0, 6.0, 40.0, 42.0)) -> None:
            self.box = box
            self.shapes: list[tuple[int, ...]] = []

        def detect(self, frame):
            self.shapes.append(tuple(frame.shape))
            x1, y1, x2, y2 = self.box
            return [
                Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=0.9, class_id=0, name="red"),
                Detection(x1=1.0, y1=1.0, x2=9.0, y2=9.0, score=0.8, class_id=9, name="zhangai"),
            ]

        def draw(self, frame, detections, fps=None):
            """Stand in for the preview helper: return the frame unchanged."""
            return frame.copy()

        def close(self) -> None:
            pass

    frame = _fake_frame()
    region = LockedRegion(center_x=200.0, center_y=150.0, radius=60.0)
    fake = FakeDetector()
    # materials 显式传死：这个自测只验裁剪/映射/筛选，不该随 config.json 变。
    yolo = MaterialYOLO(detector=fake, materials=("red",))

    everything = yolo.detect(frame)
    materials = yolo.detect(frame, materials_only=True)
    in_region = yolo.detect_in_region(frame, region, margin=10.0, materials_only=True)

    problems: list[str] = []
    if fake.shapes[0][:2] != frame.shape[:2]:
        problems.append(f"整帧推理拿到的不是完整帧：{fake.shapes[0][:2]} != {frame.shape[:2]}")
    if len(everything) != 2:
        problems.append(f"整帧推理应返回 2 个框，实际 {len(everything)}")
    if [hit.name for hit in materials] != ["red"]:
        problems.append(f"只要物料时应只剩 red，实际 {[hit.name for hit in materials]}")

    # 锁定区域 (200,150) r=60 + margin 10 -> 裁剪框 (130,80)-(270,220)；
    # 假检测器的框在裁剪图 (4,6)-(40,42)，映射回整帧应为 (134,86)-(170,122)。
    if len(in_region) != 1:
        problems.append(f"区域推理应只剩 1 个物料框，实际 {len(in_region)}")
    else:
        hit = in_region[0]
        expected = (134.0, 86.0, 170.0, 122.0)
        got = (hit.x1, hit.y1, hit.x2, hit.y2)
        if any(abs(a - b) > 0.01 for a, b in zip(got, expected)):
            problems.append(f"坐标没有映射回整帧：{got} != {expected}")

    # 框落在裁剪图左上角时，映射回整帧后在圆外，应当被丢掉。
    corner = FakeDetector(box=(1.0, 1.0, 9.0, 9.0))
    outside = MaterialYOLO(detector=corner, materials=("red",)).detect_in_region(
        frame, LockedRegion(center_x=500.0, center_y=400.0, radius=40.0), materials_only=True
    )
    if outside:
        problems.append(f"区域外的目标不应保留，实际 {[hit.name for hit in outside]}")
    kept_outside = MaterialYOLO(detector=corner, materials=("red",)).detect_in_region(
        frame,
        LockedRegion(center_x=500.0, center_y=400.0, radius=40.0),
        materials_only=True,
        keep_outside=True,
    )
    if len(kept_outside) != 1:
        problems.append(f"keep_outside=True 时应保留 1 个框，实际 {len(kept_outside)}")

    print(f"假检测器被调用 {len(fake.shapes)} 次；整帧 {len(everything)} 框、物料 {len(materials)} 框")
    print(f"锁定区域内的物料框：{[(round(hit.x1), round(hit.y1), round(hit.x2), round(hit.y2)) for hit in in_region]}")
    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：裁剪推理、类别筛选、坐标映射回整帧、区域外过滤都正确。")
    return 0


def _run_image(model_path, image_path: Path, conf: float | None) -> int:
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"读不出图片：{image_path}")
        return 1
    with MaterialYOLO(model_path, conf_thresh=conf) as yolo:
        hits = yolo.detect(frame)
        materials = [hit for hit in hits if is_material(hit.name, yolo.materials)]
        print(f"{image_path}：检出 {len(hits)} 个框，其中物料 {len(materials)} 个")
        print(f"  输入 {yolo.detector.input_size}，类别表 {list(yolo.detector.class_names)}")
        for hit in hits:
            print(f"  {hit.name:<14} score={hit.score:.2f} box=({hit.x1:.0f},{hit.y1:.0f})-({hit.x2:.0f},{hit.y2:.0f})")
    return 0


def _run_camera(model_path, camera_id: int, conf: float | None) -> int:
    import cv2

    from modules.vision import (
        BLANK_FRAME_WARNING,
        USBCamera,
        close_previews,
        looks_blank,
        open_preview,
        warn_blank_preview,
    )

    window = open_preview()
    blank_frames = 0
    warned = False
    resolution_reported = False
    try:
        with MaterialYOLO(model_path, conf_thresh=conf) as yolo, USBCamera(camera_id, width=1280, height=720).initialize() as camera:
            print(f"摄像头 {camera_id}，窗口 {window!r}，按 q 退出")
            while True:
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("摄像头未返回图像")
                if not resolution_reported:
                    height, width = frame.shape[:2]
                    print(f"实际画面尺寸：{width}x{height}")
                    resolution_reported = True
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned:
                    warned = True
                    warn_blank_preview(camera_id)
                hits = yolo.detect(frame)
                annotated = yolo.detector.draw(frame, hits)
                for hit in hits:
                    if is_material(hit.name, yolo.materials):
                        x1, y1 = int(hit.x1), max(int(hit.y1) - 20, 12)
                        cv2.putText(annotated, "MAT", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        close_previews()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YOLO 推理物料候选")
    parser.add_argument("--model", type=Path, default=None,
                        help="模型路径；默认取 modules/vision/config.json 的 model 块"
                             "（Windows 是 ONNX，Jetson 是 TensorRT engine）")
    parser.add_argument("--image", type=Path, help="在一张图片上跑真模型")
    parser.add_argument("--camera", type=int, help="打开摄像头实时预览")
    parser.add_argument("--conf", type=float, default=None,
                        help="置信度阈值；默认取 config.json 的 model.conf_thresh")
    args = parser.parse_args(argv)

    if args.camera is not None:
        return _run_camera(args.model, args.camera, args.conf)
    if args.image is not None:
        return _run_image(args.model, args.image, args.conf)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

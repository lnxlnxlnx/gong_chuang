"""物料定位真机测试：霍夫圆面定位与测距，YOLO 补充类别。

    python test.py   # 选 18

画面里只画 ASCII（``cv2.putText`` 画不出中文），中文结论打印在终端；按 ``p`` 立刻
打印一次结论，默认每 30 帧自动打印一次，按 ``q`` 退出。未传焦距标定值时，
使用实拍样本的一次粗标定估算距离。

霍夫定位与总流程也能分别用 ``python -m modules.vision.drive.hough_lock``、
``python -m modules.vision.drive.material_locate`` 离线测试。
"""

from __future__ import annotations

MENU_ENTRY = (18, "物料定位")

import argparse
import os
from pathlib import Path

from modules.vision import (
    BLANK_FRAME_WARNING,
    MaterialLocator,
    USBCamera,
    close_previews,
    default_model_path,
    load_camera_id,
    looks_blank,
    open_preview,
    warn_blank_preview,
)
from modules.vision.config import TargetSettings, load_vision_config
from modules.vision.drive.material_locate import parse_bgr, parse_depth_range, parse_palette

#: 每这么多帧打印一次中文结论。
PRINT_EVERY = 30

#: 连续读不到这么多帧就认为摄像头掉线，退出预览。
MAX_MISSES = 30


def configured_targets():
    """读 ``config.json`` 的 ``targets`` 块；读不出来就当没配。"""
    try:
        return load_vision_config().targets
    except ValueError as error:
        print(f"config.json 读不出物料参数（{error}），这一轮按默认值跑")
        return TargetSettings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="物料圆面定位 + 30 mm 测距 + YOLO 分类")
    platform = "windows" if os.name == "nt" else "jetson"
    try:
        default_camera = load_camera_id(platform)
    except ValueError:
        default_camera = 0
    targets = configured_targets()
    default_depth = targets.working_distance_mm
    parser.add_argument("--model", type=Path, default=default_model_path(),
                        help="模型路径（Windows 默认 best.onnx，Jetson 默认 best.engine）")
    parser.add_argument("--camera", type=int, default=default_camera,
                        help=f"摄像头序号（默认读 config/vision.json：{default_camera}）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    parser.add_argument("--focal", type=float, default=None,
                        help="标定后的当前画幅水平像素焦距；不传则用实拍样本粗标定")
    parser.add_argument("--depth", dest="depth_range", type=parse_depth_range,
                        default=default_depth,
                        help='相机到物料顶面的深度范围（毫米），例如 "60,80"；'
                             "默认读 config.json 的 targets.working_distance_mm"
                             f"（当前 {default_depth}）")
    parser.add_argument("--bgr", dest="material_bgr", type=parse_bgr,
                        default=targets.material_bgr,
                        help="物料顶面参考颜色 B,G,R（例如 165,252,114）；"
                             "默认读 config.json 的 targets.material_bgr，给了它就走颜色块定位")
    parser.add_argument("--palette", dest="material_palette", type=parse_palette,
                        default=targets.material_palette or None,
                        help='物料调色板 "green:165,252,114;blue:248,122,41"；'
                             "默认读 config.json 的 targets.material_palette，"
                             "换物料颜色只改这里就行")
    parser.add_argument("--tolerance", type=float, default=targets.color_tolerance,
                        help=f"颜色容差（默认 {targets.color_tolerance}）")
    parser.add_argument("--diameter", type=float, default=targets.material_diameter_mm,
                        help="物料顶面真实直径（毫米），默认读 config.json 的 "
                             f"targets.material_diameter_mm（当前 {targets.material_diameter_mm}）")
    return parser.parse_args()


def run(
    camera_id: int,
    model_path: Path,
    *,
    focal_px: float | None = None,
    conf: float = 0.25,
    depth_range: tuple[float, float] | None = None,
    material_bgr: tuple[float, float, float] | None = None,
    material_palette: dict[str, tuple[float, float, float]] | None = None,
    colour_tolerance: float = 60.0,
    top_diameter_mm: float = 30.0,
) -> int:
    """Open the camera and print the material verdict until q is pressed."""
    import cv2

    if not Path(model_path).is_file():
        print(f"找不到模型文件：{model_path}")
        print("Windows 放 models/best.onnx，Jetson 放 models/best.engine")
        return 1

    window = open_preview()
    blank_frames = 0
    warned_blank = False
    misses = 0
    frames = 0
    try:
        with MaterialLocator(
            model_path=model_path,
            conf_thresh=conf,
            focal_px=focal_px,
            depth_range_mm=depth_range,
            material_bgr=material_bgr,
            material_palette=material_palette,
            colour_tolerance=colour_tolerance,
            top_diameter_mm=top_diameter_mm,
        ) as locator, USBCamera(camera_id, width=1280, height=720).initialize() as camera:
            print(f"摄像头 {camera_id}，窗口 {window!r}；按 q 退出，按 p 立刻打印结论")
            if depth_range is None:
                print("没有工作距离先验：半径窗口按画幅猜，远处的同色物体可能被当成物料")
            else:
                print(f"工作距离先验 {depth_range[0]:.0f}–{depth_range[1]:.0f} mm，"
                      "半径窗口按针孔模型算")
            if material_palette:
                names = "、".join(material_palette)
                print(f"物料调色板 {names}，容差 {colour_tolerance:.0f}："
                      "走颜色块定位，夹爪与背景不进掩膜")
            elif material_bgr is None:
                print("没有物料颜色：走霍夫圆 + 形状判据（夹爪圆弧可能被当成候选）")
            else:
                print(f"物料颜色 BGR={material_bgr}，容差 {colour_tolerance:.0f}："
                      "走颜色块定位，夹爪与背景不进掩膜")
            print(f"顶面真实直径按 {top_diameter_mm:.1f} mm 折算距离")
            while True:
                frame = camera.read()
                if frame is None:
                    misses += 1
                    if misses >= MAX_MISSES:
                        print(f"连续 {misses} 次取不到画面，退出")
                        return 1
                    continue
                misses = 0
                frames += 1

                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)

                report = locator.locate(frame)
                annotated = locator.draw(frame, report)
                best = report.best
                # 画面上只写 ASCII（cv2.putText 画不出中文）
                headline = "no material" if best is None else f"{best.name or 'material'} {best.distance_mm:.0f}mm"
                cv2.putText(annotated, headline, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"materials={len(report.lock)}", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return 0
                if key == ord("p") or frames % PRINT_EVERY == 0:
                    print(report.summary())
    finally:
        close_previews()
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(
        run(
            args.camera,
            args.model,
            focal_px=args.focal,
            conf=args.conf,
            depth_range=args.depth_range,
            material_bgr=args.material_bgr,
            material_palette=args.material_palette,
            colour_tolerance=args.tolerance,
            top_diameter_mm=args.diameter,
        )
    )


if __name__ == "__main__":
    main()

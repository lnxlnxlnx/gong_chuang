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

#: 每这么多帧打印一次中文结论。
PRINT_EVERY = 30

#: 连续读不到这么多帧就认为摄像头掉线，退出预览。
MAX_MISSES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="物料圆面定位 + 30 mm 测距 + YOLO 分类")
    platform = "windows" if os.name == "nt" else "jetson"
    try:
        default_camera = load_camera_id(platform)
    except ValueError:
        default_camera = 0
    parser.add_argument("--model", type=Path, default=default_model_path(),
                        help="模型路径（Windows 默认 best.onnx，Jetson 默认 best.engine）")
    parser.add_argument("--camera", type=int, default=default_camera,
                        help=f"摄像头序号（默认读 config/vision.json：{default_camera}）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    parser.add_argument("--focal", type=float, default=None,
                        help="标定后的当前画幅水平像素焦距；不传则用实拍样本粗标定")
    return parser.parse_args()


def run(
    camera_id: int,
    model_path: Path,
    *,
    focal_px: float | None = None,
    conf: float = 0.25,
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
        ) as locator, USBCamera(camera_id, width=1280, height=720).initialize() as camera:
            print(f"摄像头 {camera_id}，窗口 {window!r}；按 q 退出，按 p 立刻打印结论")
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
        )
    )


if __name__ == "__main__":
    main()

"""Vision demos: plain YOLO preview, and the full material-locating preview."""

from __future__ import annotations

import os
from pathlib import Path

from modules.vision import default_model_path, load_camera_id


def demo_vision(
    model_path: str | Path | None = None, camera_id: int | None = None
) -> None:
    """Preview detections until q is pressed in the image window."""
    model_path = Path(model_path) if model_path is not None else default_model_path()
    camera_id = (
        camera_id
        if camera_id is not None
        else load_camera_id("windows" if os.name == "nt" else "jetson")
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    import cv2

    from modules.vision import (
        BLANK_FRAME_WARNING,
        USBCamera,
        close_previews,
        create_detector,
        looks_blank,
        open_preview,
        warn_blank_preview,
    )

    # 窗口名必须是 ASCII，且要先显式建窗，见 preview.py 顶部说明。
    window = open_preview()
    blank_frames = 0
    warned_blank = False
    try:
        # 反斜杠续行而不是括号化 with：括号化多管理器是 3.9+ 语法，Jetson 的
        # Python 3.8 会直接 SyntaxError（见 tests/test_python38_compatibility.py）。
        with USBCamera(camera_id).initialize() as camera, \
                create_detector(model_path) as detector:
            print(f"摄像头编号 {camera_id}，模型 {model_path.name}；预览窗口 {window!r}，按 q 退出")
            while True:
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("摄像头未返回图像")
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)
                hits = detector.detect(frame)
                cv2.imshow(window, detector.draw(frame, hits))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        close_previews()


def demo_material(
    model_path: str | Path | None = None,
    camera_id: int | None = None,
    *,
    focal_px: float | None = None,
) -> None:
    """预览物料的像素位置和相机坐标；按 p 打印，按 q 退出。

    可用 ``python -m modules.vision.drive.hough_lock`` 单独检查圆面定位；这里
    使用 :class:`modules.vision.MaterialLocator` 跑完整流程。
    """
    model_path = Path(model_path) if model_path is not None else default_model_path()
    camera_id = (
        camera_id
        if camera_id is not None
        else load_camera_id("windows" if os.name == "nt" else "jetson")
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    import cv2

    from modules.vision import (
        BLANK_FRAME_WARNING,
        MaterialLocator,
        USBCamera,
        close_previews,
        looks_blank,
        open_preview,
        warn_blank_preview,
    )

    window = open_preview()
    blank_frames = 0
    warned_blank = False
    try:
        with MaterialLocator(
            model_path=model_path,
            focal_px=focal_px,
        ) as locator, USBCamera(camera_id, width=1280, height=720).initialize() as camera:
            print(f"摄像头编号 {camera_id}，模型 {model_path.name}；预览窗口 {window!r}，按 q 退出")
            print("画面里只画 ASCII；按 p 打印所有物料的完整结果，按 q 退出。")
            while True:
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("摄像头未返回图像")
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)

                report = locator.locate(frame)
                annotated = locator.draw(frame, report)
                cv2.putText(
                    annotated, f"present={report.present} count={len(report.regions)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                for index, item in enumerate(report.regions, start=1):
                    confidence = f"{item.confidence:.2f}" if item.confidence is not None else "--"
                    if item.name is None:
                        continue  # 只画有名字的物料，避免干扰
                    line = (
                        f"{index}: {item.name or 'unknown'} confidence={confidence} "
                        f"xyz=({item.x_mm:.0f},{item.y_mm:.0f},{item.z_mm:.0f})mm "
                        f"range={item.range_mm:.0f}mm"
                    )
                    cv2.putText(
                        annotated, line, (10, 24 + 26 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    )
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    print(report.summary())
                if key == ord("q"):
                    break
    finally:
        close_previews()

"""Standalone USB camera + YOLOv8 vision demo.

Usage from the project root (Jetson needs its TensorRT environment)::

    python test.py  # 选择 9

Press ``q`` in the preview window to quit.  Optional FPS --- and distance
readouts (when ``--real-widths`` is given) --- are drawn on the frame.
"""

from __future__ import annotations

MENU_ENTRY = (9, "视觉演示")

import argparse
import os
import time
from pathlib import Path

from modules.vision import (
    BLANK_FRAME_WARNING,
    DistanceEstimator,
    USBCamera,
    close_previews,
    create_detector,
    default_model_path,
    load_camera_id,
    looks_blank,
    open_preview,
    warn_blank_preview,
)


def parse_real_widths(text: str | None):
    """Parse ``name1=120,name2=90`` into a dict, or ``None`` if empty."""
    if not text:
        return None
    pairs = dict(item.split("=", 1) for item in text.split(","))
    return {name: float(value) for name, value in pairs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 vision demo")
    parser.add_argument("--model", type=Path, default=default_model_path(),
                        help="ONNX model on Windows, TensorRT engine on Jetson")
    parser.add_argument("--camera", type=int, default=None,
                        help="override camera index from config/vision.json")
    parser.add_argument("--width", type=int, default=640, help="camera width")
    parser.add_argument("--height", type=int, default=480, help="camera height")
    parser.add_argument("--fps", type=int, default=30, help="camera frame rate")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--real-widths", default=None,
                        help='comma-separated class widths in mm, e.g. '
                             '"obj-red=120,zhangai=80"; enables distance readout')
    parser.add_argument("--max-fps", type=float, default=30.0,
                        help="cap on the displayed FPS (default: 30)")
    args = parser.parse_args()
    camera_id = args.camera if args.camera is not None else load_camera_id(
        "windows" if os.name == "nt" else "jetson")

    if not args.model.is_file():
        parser.error(f"model file not found: {args.model}")

    import cv2

    real_widths = parse_real_widths(args.real_widths)
    focal_px = 560.0  # placeholder; see README "calibrating focal length"
    estimator = DistanceEstimator(focal_px, real_widths) if real_widths else None

    print(f"loading model: {args.model}; camera index: {camera_id} ...")
    # 窗口名必须是 ASCII，且要先显式建窗，见 vision/preview.py 顶部说明。
    window = open_preview()
    blank_frames = 0
    warned_blank = False
    try:
        with create_detector(args.model, conf_thresh=args.conf,
                            iou_thresh=args.iou) as detector, \
                USBCamera(camera_id, width=args.width, height=args.height,
                          fps=args.fps).initialize() as camera:
            print(f"preview running in window {window!r}; press q to quit")
            fps = 0.0
            previous = time.time()
            while True:
                frame = camera.read()
                if frame is None:
                    print("no frame from camera; exiting")
                    break

                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)

                detections = detector.detect(frame)
                now = time.time()
                fps = fps * 0.9 + (1.0 / (now - previous + 1e-6)) * 0.1 if previous else 0.0
                previous = now

                annotated = detector.draw(frame, detections, fps=fps)
                if estimator is not None:
                    for det in detections:
                        distance = estimator.estimate(det)
                        if distance is not None:
                            text = f"{det.name}: {distance / 1000.0:.2f} m"
                            x, y = int(det.x1), max(int(det.y1) - 24, 15)
                            cv2.putText(annotated, text, (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 255), 1)
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        close_previews()


if __name__ == "__main__":
    main()

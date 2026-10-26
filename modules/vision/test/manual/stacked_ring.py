"""叠放位识别真机测试：中环 + 上下基准线 + 工位偏角。

    python test.py   # 选 23

画面里只画 ASCII（``cv2.putText`` 画不出中文），中文结论打印在终端；按 ``p`` 立刻打印
一次结论，按 ``r`` 清掉稳定窗口（重新摆工位之后按），按 ``q`` 退出。

**两条基准线都要看到才可信**：它们本该平行，只找到一条时偏角仍然给出来，但
``line_quality`` 会低一档。两条线夹角超过 ``stacked_ring.max_line_disagreement_deg``
时结论里会写明"偏角不可信"——那时应当先查机位和光照，而不是照着这个角度去转底盘。

调参不用改代码：阈值在 ``modules/vision/config.json`` 的 ``stacked_ring`` 块里，
比例尺复用 ``station`` 块的标定。

离线自测（不需要摄像头）：``python -m modules.vision.drive.stacked_ring``
"""

from __future__ import annotations

MENU_ENTRY = (23, "叠放位识别")

import argparse
import os

from modules.vision.config import DEFAULT_CONFIG_PATH
from modules.vision.drive.camera import looks_blank
from modules.vision.drive.preview import (
    BLANK_FRAME_WARNING,
    close_previews,
    open_preview,
    warn_blank_preview,
)
from modules.vision.runtime import Vision, load_vision_settings

#: 每这么多帧打印一次中文结论。
PRINT_EVERY = 30

#: 连续读不到这么多帧就认为摄像头掉线，退出预览。
MAX_MISSES = 30


def parse_args() -> argparse.Namespace:
    """只解析命令行：**不读配置、不打开设备**。"""
    parser = argparse.ArgumentParser(description="叠放位识别：中环 + 上下基准线 + 工位偏角")
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头序号（默认读 config.json 对应平台的 camera_id）")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY,
                        help=f"每多少帧打印一次结论（默认 {PRINT_EVERY}）")
    return parser.parse_args()


def run(settings, *, print_every: int = PRINT_EVERY) -> int:
    """用**已经读好的配置快照**打开摄像头并打印叠放位结论，直到按 q 退出。"""
    import cv2

    window = open_preview()
    blank_frames = 0
    warned_blank = False
    reason = ""
    try:
        with Vision.from_settings(settings) as vision:
            camera_id = settings.camera_id("windows" if os.name == "nt" else "jetson")
            stacked = settings.stacked_ring
            geometry = settings.station_geometry
            print(f"摄像头 {camera_id}，窗口 {window!r}；"
                  "按 q 退出，按 p 立刻打印结论，按 r 清稳定窗口")
            if geometry.image_to_station_affine is not None:
                print("比例尺来自 station.image_to_station_affine")
            elif geometry.mm_per_px is not None:
                print(f"比例尺 {geometry.mm_per_px:.4f} mm/px（各向同性）")
            else:
                print("没有标定比例尺：会用中环半径临时反推，"
                      "结果的 scale_source 是 ring——**毫米读数只能当参考**")
            print(f"中环参考圆心 {stacked.middle_reference_px}，"
                  f"半径窗口 {stacked.middle_min_radius_px:.0f}–"
                  f"{stacked.middle_max_radius_px:.0f} px，"
                  f"基准线偏移 {stacked.line_offset_mm:.0f} mm")

            def watch(frame, _detection, _stats) -> None:
                nonlocal blank_frames, warned_blank
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)

            def on_key(key: int, detection, _stats) -> None:
                if key == ord("p"):
                    print(detection.summary())
                elif key == ord("r"):
                    vision._stacked_ring_detector().reset()
                    print("稳定窗口已清空")

            reason = _preview(vision, window, print_every=print_every,
                              on_frame=watch, on_key=on_key)
    finally:
        close_previews()
    return 0 if reason in ("quit", "") else 1


def _preview(vision, window: str, *, print_every: int, on_frame, on_key) -> str:
    """叠放位识别自己的预览循环。

    不复用 ``stream.run_preview`` 的理由同 ``manual/ring_station.py``：那个循环的契约
    是"推理函数返回 ``MaterialReport``"，而这里返回的是 ``StackedRingDetection``。
    骨架保持一致：丢帧、掉线计数、按键分派、``q``/``Esc`` 退出。
    """
    import cv2

    detector = vision._stacked_ring_detector()
    camera = vision._camera
    misses = 0
    frames = 0
    while True:
        frame = camera.read()
        if frame is None:
            misses += 1
            if misses >= MAX_MISSES:
                return "camera-lost"
            continue
        misses = 0
        frames += 1
        detection = detector.detect(frame)
        on_frame(frame, detection, None)
        if frames % print_every == 0:
            print(detection.summary())
        cv2.imshow(window, detector.draw(frame, detection))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            return "quit"
        if key != 255:
            on_key(key, detection, None)


def main() -> None:
    args = parse_args()
    settings = load_vision_settings()
    print(f"配置来自 {DEFAULT_CONFIG_PATH}")
    raise SystemExit(run(settings, print_every=args.print_every))


if __name__ == "__main__":
    main()

"""三圆环工位识别真机测试：圆心、偏角与毫米偏差。

    python test.py   # 选 22

画面里只画 ASCII（``cv2.putText`` 画不出中文），中文结论打印在终端；按 ``p`` 立刻打印
一次结论，按 ``r`` 清掉稳定窗口（重新摆工位之后按），按 ``q`` 退出。

调参不用改代码：所有阈值都在 ``modules/vision/config.json`` 的 ``station`` 块里。
没配比例尺（``station.mm_per_px`` 与 ``station.image_to_station_affine`` 都是 ``null``）时
只报像素，不编造毫米。

离线自测（不需要摄像头）：``python -m modules.vision.drive.ring_station``
"""

from __future__ import annotations

MENU_ENTRY = (22, "三圆环工位识别")

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
from modules.vision.runtime import Vision, _load_settings

#: 每这么多帧打印一次中文结论。
PRINT_EVERY = 30

#: 连续读不到这么多帧就认为摄像头掉线，退出预览。
MAX_MISSES = 30


def parse_args() -> argparse.Namespace:
    """只解析命令行：**不读配置、不打开设备**。"""
    parser = argparse.ArgumentParser(description="三圆环工位识别：圆心、偏角与偏差")
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头序号（默认读 config.json 对应平台的 camera_id）")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY,
                        help=f"每多少帧打印一次结论（默认 {PRINT_EVERY}）")
    return parser.parse_args()


def run(settings, *, print_every: int = PRINT_EVERY) -> int:
    """用**已经读好的配置快照**打开摄像头并打印工位结论，直到按 q 退出。"""
    import cv2

    window = open_preview()
    blank_frames = 0
    warned_blank = False
    try:
        with Vision._open_config(settings) as vision:
            camera_id = settings.camera_id("windows" if os.name == "nt" else "jetson")
            geometry = settings.station_geometry
            reference = settings.station_reference
            print(f"摄像头 {camera_id}，窗口 {window!r}；"
                  "按 q 退出，按 p 立刻打印结论，按 r 清稳定窗口")
            if geometry.image_to_station_affine is not None:
                print("比例尺来自 station.image_to_station_affine（各向异性仿射）")
            elif geometry.mm_per_px is not None:
                print(f"比例尺 {geometry.mm_per_px:.4f} mm/px（各向同性）")
            else:
                print("没有比例尺：只报像素偏差，不报毫米（"
                      "配 station.mm_per_px 或 station.image_to_station_affine）")
            print(f"参考圆心 {reference.ring_centers_px}，"
                  f"验收 ≤{reference.maximum_position_error_mm:.1f} mm / "
                  f"≤{reference.maximum_angle_error_deg:.1f}°")

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
                    vision._ring_station_detector().reset()
                    print("稳定窗口已清空")

            reason = _preview(vision, window, print_every=print_every,
                              on_frame=watch, on_key=on_key)
    finally:
        close_previews()
    # 摄像头掉线/一直没画面时仍然报错退出（与 material_locate 的取舍一致）。
    return 0 if reason in ("quit", "") else 1


def _preview(vision, window: str, *, print_every: int, on_frame, on_key) -> str:
    """工位识别自己的预览循环。

    刻意**不复用** ``modules.vision.drive.stream.run_preview``：那个循环的契约是
    "推理函数返回 ``MaterialReport``"（它要读 ``report.best`` 之类），而工位识别返回的是
    ``RingStationDetection``。硬套过去要么改 ``run_preview`` 的契约、要么在外面包一层
    适配，两者都比这十几行循环更难读。这里保留同样的骨架：丢帧、掉线计数、按键分派。
    """
    import cv2

    detector = vision._ring_station_detector()
    misses = 0
    frames = 0
    camera = vision._camera
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
    settings = _load_settings()
    print(f"配置来自 {DEFAULT_CONFIG_PATH}")
    raise SystemExit(run(settings, print_every=args.print_every))


if __name__ == "__main__":
    main()

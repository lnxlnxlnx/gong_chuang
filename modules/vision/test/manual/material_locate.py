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

from modules.vision.config import DEFAULT_CONFIG_PATH
from modules.vision.drive.camera import USBCamera, looks_blank
from modules.vision.drive.material_locate import parse_bgr, parse_depth_range, parse_palette
from modules.vision.drive.preview import (
    BLANK_FRAME_WARNING,
    close_previews,
    open_preview,
    warn_blank_preview,
)
from modules.vision.drive.stream import run_preview
from modules.vision.runtime import Vision, _load_settings, _override_settings

#: 每这么多帧打印一次中文结论。
PRINT_EVERY = 30

#: 连续读不到这么多帧就认为摄像头掉线，退出预览。
MAX_MISSES = 30


def parse_args() -> argparse.Namespace:
    """只解析命令行：**不读配置、不打开设备**。

    所有"本次覆盖"的默认值都是 ``None``，意思是"没给，按配置来"。想看到配置里的
    当前值，由 :func:`main` 加载快照之后打印，而不是在这里读一遍配置文件——
    读两次会给"改完 config.json 行为没变"这类问题留后门。
    """
    parser = argparse.ArgumentParser(description="物料圆面定位 + 30 mm 测距 + YOLO 分类")
    parser.add_argument("--model", type=Path, default=None,
                        help="模型路径（默认读 config.json 的 model 块："
                             "Windows 是 best.onnx，Jetson 是 best.engine）")
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头序号（默认读 config.json 对应平台的 camera_id）")
    parser.add_argument("--conf", type=float, default=None,
                        help="置信度阈值；默认读 config.json 的 model.conf_thresh")
    parser.add_argument("--focal", type=float, default=None,
                        help="标定后的当前画幅水平像素焦距；不传则用实拍样本粗标定")
    parser.add_argument("--depth", dest="depth_range", type=parse_depth_range, default=None,
                        help='相机到物料顶面的深度范围（毫米），例如 "60,80"；'
                             "默认读 config.json 的 targets.working_distance_mm")
    parser.add_argument("--bgr", dest="material_bgr", type=parse_bgr, default=None,
                        help="物料顶面参考颜色 B,G,R（例如 165,252,114）；"
                             "默认读 config.json 的 targets.material_bgr，给了它就走颜色块定位")
    parser.add_argument("--palette", dest="material_palette", type=parse_palette, default=None,
                        help='物料调色板 "green:165,252,114;blue:248,122,41"；'
                             "默认读 config.json 的 targets.material_palette，"
                             "换物料颜色只改这里就行")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="颜色容差；默认读 config.json 的 targets.color_tolerance")
    parser.add_argument("--diameter", type=float, default=None,
                        help="物料顶面真实直径（毫米），默认读 config.json 的 "
                             "targets.material_diameter_mm")
    return parser.parse_args()


def run(settings, *, focal_px: float | None = None) -> int:
    """用**已经读好的配置快照**打开摄像头并打印物料结论，直到按 q 退出。

    快照由 :func:`main` 读取命令行之后覆盖好再传进来；这里不读文件、不再开第二台相机。
    预览沿用 :func:`run_preview` 原来的两条线程，并且**复用本会话的相机和定位器**。
    """
    import cv2

    window = open_preview()
    blank_frames = 0
    warned_blank = False
    frames = 0
    try:
        with Vision._open_config(settings, focal_px=focal_px) as vision:
            camera_id = settings.camera_id(
                "windows" if os.name == "nt" else "jetson"
            )
            print(f"摄像头 {camera_id}，窗口 {window!r}；按 q 退出，按 p 立刻打印结论")
            targets = vision._settings.targets
            if targets.working_distance_mm is None:
                print("没有工作距离先验：半径窗口按画幅猜，远处的同色物体可能被当成物料")
            else:
                low, high = targets.working_distance_mm
                print(f"工作距离先验 {low:.0f}–{high:.0f} mm，半径窗口按针孔模型算")
            if targets.material_palette:
                names = "、".join(targets.material_palette)
                print(f"物料调色板 {names}，容差 {targets.color_tolerance:.0f}："
                      "走颜色块定位，夹爪与背景不进掩膜")
            elif targets.material_bgr is None:
                print("没有物料颜色：走霍夫圆 + 形状判据（夹爪圆弧可能被当成候选）")
            else:
                print(f"物料颜色 BGR={targets.material_bgr}，容差 {targets.color_tolerance:.0f}："
                      "走颜色块定位，夹爪与背景不进掩膜")
            print(f"顶面真实直径按 {targets.material_diameter_mm:.1f} mm 折算距离")

            def watch(frame, _report, _stats) -> None:
                nonlocal blank_frames, warned_blank
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)

            def overlay(annotated, report, _stats) -> None:
                """画面上只写 ASCII（cv2.putText 画不出中文）。"""
                nonlocal frames
                frames += 1
                best = None if report is None else report.best
                headline = (
                    "no material" if best is None
                    else f"{best.name or 'material'} {best.distance_mm:.0f}mm"
                )
                count = 0 if report is None else len(report.lock)
                cv2.putText(annotated, headline, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"materials={count}", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if report is not None and frames % PRINT_EVERY == 0:
                    print(report.summary())

            def on_key(key: int, report, _stats) -> None:
                if report is not None and key == ord("p"):
                    print(report.summary())

            stats = run_preview(
                vision._camera,
                vision._locator.locate,
                vision._locator.draw,
                window=window,
                print_every=PRINT_EVERY,
                max_misses=MAX_MISSES,
                on_frame=watch,
                overlay=overlay,
                on_key=on_key,
            )
    finally:
        close_previews()
    # 摄像头掉线/一直没画面时仍然报错退出（原来按 MAX_MISSES 退出也是返回 1）。
    return 0 if stats.reason in ("quit", "") else 1


def main() -> None:
    args = parse_args()
    # 覆盖先落到**配置快照**上，再进入唯一的真实会话创建路径：
    # 不写回文件、不在 public open 之后再关掉重开带覆盖的设备。
    settings = _override_settings(
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
    print(f"配置来自 {DEFAULT_CONFIG_PATH}")
    raise SystemExit(run(settings, focal_px=args.focal))


if __name__ == "__main__":
    main()

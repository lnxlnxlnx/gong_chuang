"""Vision demos: plain YOLO preview, and the full material-locating preview.

两个入口都走 :func:`modules.vision.run_preview`：取帧一条线程、推理一条线程，
窗口跑在摄像头帧率上，推理跟不上的帧被跳过（见 ``drive/stream.py`` 顶部说明）。
窗口左下角那行 ``cap / det / age`` 就是这两条线程各自的实际速率——
**"画面卡"和"识别慢"是两件事**，这行数把它们分开摆出来。
"""

from __future__ import annotations

import os
from pathlib import Path

from modules.vision import default_model_path, load_camera_id
from modules.vision.config import resolve_target_settings

#: 预览里每显示这么多帧往终端打一行统计（0 表示不打）。
PRINT_EVERY = 60


def _camera_id(camera_id: int | None) -> int:
    """配置里的摄像头序号（``None`` 时按平台读 ``config.json``）。"""
    return (
        camera_id
        if camera_id is not None
        else load_camera_id("windows" if os.name == "nt" else "jetson")
    )


def _require_model(model_path: str | Path | None) -> Path:
    """默认模型路径 + 不存在时说清楚该做什么。"""
    path = Path(model_path) if model_path is not None else default_model_path()
    if not path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{path}")
    return path


def demo_vision(
    model_path: str | Path | None = None, camera_id: int | None = None
) -> None:
    """Preview detections until q is pressed in the image window."""
    model_path = _require_model(model_path)
    camera_id = _camera_id(camera_id)
    import cv2

    from modules.vision import (
        BLANK_FRAME_WARNING,
        USBCamera,
        close_previews,
        create_configured_detector,
        looks_blank,
        open_preview,
        run_preview,
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
                create_configured_detector(model_path=model_path) as detector:
            print(f"摄像头编号 {camera_id}，模型 {model_path.name}；预览窗口 {window!r}，按 q 退出")

            def watch(frame, _result, _stats) -> None:
                """原始帧上的检查：纯色画面提示（被别的程序占用时最常见）。"""
                nonlocal blank_frames, warned_blank
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)

            run_preview(
                camera,
                detector.detect,
                detector.draw,
                window=window,
                print_every=PRINT_EVERY,
                on_frame=watch,
            )
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

    **物料颜色与工作距离先验来自 ``modules/vision/config.json``**（``targets`` 块：
    ``material_palette`` / ``material_bgr`` / ``working_distance_mm`` /
    ``material_diameter_mm`` / ``color_tolerance``）。这三个参数决定"用颜色块还是
    霍夫找圆"以及"半径窗口按物理先验算还是按画幅猜"——恰恰是识别速度与稳不稳的
    分水岭：真实台面图上不配它们时，霍夫会在整幅 1280×720 上凑出几十个候选圆，
    每帧要 0.8–3.3 s。菜单 18（``modules/vision/test/manual/material_locate.py``）
    一直读的就是这些值，这里与它保持一致。
    """
    model_path = _require_model(model_path)
    camera_id = _camera_id(camera_id)
    import cv2

    from modules.vision import (
        BLANK_FRAME_WARNING,
        MaterialLocator,
        USBCamera,
        close_previews,
        looks_blank,
        open_preview,
        run_preview,
        warn_blank_preview,
    )

    targets, problem = resolve_target_settings()
    if problem is not None:
        print(f"config.json 的 targets 块读不出来（{problem}），这一轮按默认值跑")
    # 配置里的值直接透传给 MaterialLocator：它自己会做类型归一与校验
    # （调色板取色、深度范围排序），这里再解析一遍只会多一处能写错的地方。
    depth_range = targets.working_distance_mm
    palette = targets.material_palette or None

    window = open_preview()
    blank_frames = 0
    warned_blank = False
    try:
        with MaterialLocator(
            model_path=model_path,
            focal_px=focal_px,
            depth_range_mm=depth_range,
            material_bgr=None if palette else targets.material_bgr,
            material_palette=palette,
            colour_tolerance=targets.color_tolerance,
            top_diameter_mm=targets.material_diameter_mm,
        ) as locator, USBCamera(camera_id, width=1280, height=720).initialize() as camera:
            print(f"摄像头编号 {camera_id}，模型 {model_path.name}；预览窗口 {window!r}，按 q 退出")
            print("画面里只画 ASCII；按 p 打印所有物料的完整结果，按 q 退出。")
            if palette or targets.material_bgr is not None:
                print(f"物料颜色取自 config.json（容差 {targets.color_tolerance:.0f}）："
                      "走颜色块定位，夹爪与背景不进掩膜")
            else:
                print("config.json 里没有物料颜色：走霍夫圆 + 形状判据"
                      "（速度与稳定性都差一截，建议配 targets.material_palette）")
            if depth_range is None:
                print("没有工作距离先验：半径窗口按画幅猜；"
                      "config.json 的 targets.working_distance_mm 可以按物理尺寸算出窗口")
            else:
                print(f"工作距离先验 {depth_range[0]:.0f}–{depth_range[1]:.0f} mm，"
                      "半径窗口按针孔模型算")

            def watch(frame, _result, _stats) -> None:
                nonlocal blank_frames, warned_blank
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)

            def overlay(annotated, report, _stats) -> None:
                """把物料结论写在画面上（``cv2.putText`` 画不出中文，只写 ASCII）。"""
                regions = () if report is None else report.regions
                present = False if report is None else report.present
                cv2.putText(
                    annotated, f"present={present} count={len(regions)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                if report is None:
                    return
                for index, item in enumerate(regions, start=1):
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

            def on_key(key: int, report, _stats) -> None:
                if key == ord("p") and report is not None:
                    print(report.summary())

            run_preview(
                camera,
                locator.locate,
                locator.draw,
                window=window,
                print_every=PRINT_EVERY,
                on_frame=watch,
                overlay=overlay,
                on_key=on_key,
            )
    finally:
        close_previews()

"""视野里那个环：环心在哪、它是几号。

    python test.py   # 选 26

相机装在臂上、镜头离纸很近时，画面里通常**只有一个环**——"同时看到三个"那种工位识别
（菜单 22）在这个取景下必然失败，而"只看到一个环"本来就不是错误。本入口就是为这种
取景准备的：

- 环心与**环号**都由模型给（``models/best.onnx`` 的 ``tg1`` / ``tg2`` / ``tg3`` 三类）；
- 环心换算成**相机系毫米**（原点在光轴、x 右、y 下），可以直接喂
  ``grasp.align_xy(reading=...)``；
- 比例尺优先用针孔标定（``table_depth_mm / focal_px``），没有就退回
  ``station.mm_per_px``；两者都没有时只报像素；
- 框被画面裁掉时**不给**可对准读数：那时框心不是环心（见
  :mod:`modules.vision.drive.numbered_ring`）。

画面里只画 ASCII（``cv2.putText`` 画不出中文），中文结论打印在终端：**每
``--print-every`` 帧打一行**（不用按键也有数据），按 ``p`` 打完整结论，
按 ``q`` 退出。

也可以直接跑一张图::

    python -m modules.vision.test.manual.numbered_ring --image models/data/ring_pic/x.jpg
"""

from __future__ import annotations

MENU_ENTRY = (26, "环内数字环心")

import argparse
import os

from modules.vision.config import DEFAULT_CONFIG_PATH
from modules.vision.drive.preview import close_previews, open_preview
from modules.vision.drive.stream import run_preview
from modules.vision.runtime import Vision, _load_settings

#: 每这么多帧打印一行环心结论（0 表示只在按 p 时打印）。
PRINT_EVERY = 30


def parse_args() -> argparse.Namespace:
    """只解析命令行：**不读配置、不打开设备、不加载模型**。"""
    parser = argparse.ArgumentParser(description="视野里那个环：环心与环号（可对准读数）")
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头序号（默认读 config.json 对应平台的 camera_id）")
    parser.add_argument("--image", default=None,
                        help="在一张图片上跑一次（不给就开摄像头预览）")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY,
                        help=f"每多少帧打印一行结论（默认 {PRINT_EVERY}；0 = 只按 p 打印）")
    return parser.parse_args()


def _unpack(result):
    """``run_preview`` 只在结果非 ``None`` 时调 ``draw``；用单元素元组把 ``None`` 也带过去。

    不这么做的话，"这一帧没找到环"就变成"画面上什么都不更新"——看起来像程序卡住了。
    """
    if isinstance(result, tuple) and len(result) == 1:
        return result[0]
    return result


def _draw(frame, result):
    """画框、环心十字与相机坐标。没有环时原样返回（``overlay`` 会写一行提示）。"""
    import cv2

    target = _unpack(result)
    canvas = frame.copy()
    if target is None:
        return canvas
    centre = (int(round(target.center_px[0])), int(round(target.center_px[1])))
    # 精修过就画真正那一圈（外圈），否则只能按框画——框比图案大一圈。
    radius = max(3, int(round(target.ring_radius_px or target.radius_px)))
    # 能拿去对准画绿，贴边/没比例尺画橙：颜色只表达"这个读数能不能直接用"。
    colour = (0, 255, 0) if target.usable else (0, 165, 255)
    cv2.circle(canvas, centre, radius, colour, 3)
    cv2.drawMarker(canvas, centre, (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    label = f"ring {target.digit} ({target.score:.2f})"
    if target.x_mm is not None and target.y_mm is not None:
        label += f" x={target.x_mm:+.1f} y={target.y_mm:+.1f} mm"
    else:
        label += " | no mm"
    if target.clipped:
        label += " | clipped"
    origin = (max(4, centre[0] - radius), max(18, centre[1] - radius - 10))
    cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return canvas


def _overlay(canvas, result, _stats) -> None:
    """没有环时在左上角说一句，免得窗口看起来像卡住了。"""
    import cv2

    if _unpack(result) is not None:
        return
    cv2.putText(canvas, "no ring in view", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)


def _status(target, locator) -> str:
    """一行结论：有环就写环号、环心与比例尺，没环就写"走到了哪一步"。"""
    if target is None:
        return f"  没有环：{locator.last_diagnosis or '检测还没跑过'}"
    reading = target.reading()
    where = "无毫米" if reading is None else f"x={reading[0]:+.1f} y={reading[1]:+.1f} mm"
    state = "贴边（不可对准）" if target.clipped else "可对准"
    scale = (
        f"{target.scale_mm_per_px:.4f} mm/px({target.scale_source})"
        if target.scale_mm_per_px is not None
        else "无比例尺"
    )
    return (
        f"  环 {target.digit}（{target.score:.2f}）"
        f"@({target.center_px[0]:.0f},{target.center_px[1]:.0f}) "
        f"r={target.radius_px:.0f}px {where}，{state}，{scale}"
    )


def _report(target, locator) -> None:
    """终端里把结论连同对准读数一起打出来（按 p）。"""
    if target is None:
        print("画面里没有环：" + (locator.last_diagnosis or "检测还没跑过"))
        print("  排查：环是否在画面里、光照是否够、模型类别表里有没有这三个 tg 类"
              "（config.json 的 model.class_names）")
        return
    print(target.summary())
    print(f"  对准读数（相机系）: {target.reading()}")
    print(f"  {locator.last_diagnosis}")
    print(f"  {target.detail}")


def _locator_for(settings, frame):
    """按一份配置快照给"单张图片"建定位器（顺带返回检测器，好让调用方关掉）。"""
    from modules.vision.detector_factory import create_configured_detector
    from modules.vision.drive.numbered_ring import RingTargetLocator
    from modules.vision.drive.pinhole import resolve_focal_px

    height, width = frame.shape[:2]
    detector = create_configured_detector(settings=settings.model)
    locator = RingTargetLocator(
        detector,
        settings=settings.ring_target,
        geometry=settings.station_geometry,
        detection=settings.station_detection,
        focal_px=resolve_focal_px(width, height, settings.focal_px),
        table_depth_mm=settings.table_depth_mm,
    )
    return detector, locator


def run_image(path: str, settings) -> int:
    """在一张图片上跑一次：要模型（环心与环号都是模型给的），但不开摄像头。"""
    from modules.vision.drive.material_locate import read_image

    frame = read_image(path)
    if frame is None:
        print(f"读不出图片：{path}")
        return 1
    detector, locator = _locator_for(settings, frame)
    try:
        target = locator.detect(frame)
        _report(target, locator)
        return 0 if target is not None else 1
    finally:
        detector.close()


def run(settings, *, print_every: int = PRINT_EVERY) -> int:
    """开摄像头预览：只找视野里那一个环，定期打印环心读数。"""
    window = open_preview()
    camera_id = settings.camera_id("windows" if os.name == "nt" else "jetson")
    shown = {"frames": 0}
    try:
        with Vision._open_config(settings) as vision:
            print(f"摄像头 {camera_id}，窗口 {window!r}；按 p 打完整结论，按 q 退出")
            print(f"环类别 {list(settings.ring_target.names)}"
                  f" → 环号 {dict(settings.ring_target.classes)}（来自 config.json）")
            if settings.table_depth_mm is not None:
                print("比例尺用针孔标定 table_depth_mm / focal_px（与物料读数同一套单位）")
            else:
                print("没有 table_depth_mm：比例尺退回 station.mm_per_px")
            locator = vision._ring_target_locator()

            def pipeline(frame):
                # 单元素元组：即使没找到环，draw/on_key 也能拿到"这一帧的结果"。
                return (locator.detect(frame),)

            def on_frame(_frame, result, _stats) -> None:
                shown["frames"] += 1
                if print_every and shown["frames"] % print_every == 0:
                    print(_status(_unpack(result), locator))

            def on_key(key: int, result, _stats) -> None:
                if key == ord("p"):
                    _report(_unpack(result), locator)

            stats = run_preview(
                vision._camera,
                pipeline,
                _draw,
                window=window,
                print_every=0,
                on_frame=on_frame,
                on_key=on_key,
                overlay=_overlay,
            )
            print("  " + stats.line())
    finally:
        close_previews()
    return 0


def main() -> None:
    args = parse_args()
    settings = _load_settings()
    print(f"配置来自 {DEFAULT_CONFIG_PATH}")
    if args.image is not None:
        raise SystemExit(run_image(args.image, settings))
    raise SystemExit(run(settings, print_every=args.print_every))


if __name__ == "__main__":
    main()

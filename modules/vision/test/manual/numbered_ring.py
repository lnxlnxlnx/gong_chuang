"""视野里那个环：环心在哪、能不能直接拿去对准。

    python test.py   # 选 26

相机装在臂上、镜头离纸很近时，画面里通常**只有一个环**——"同时看到三个"那种检测
（菜单 22 / ``Vision.detect_numbered_rings``）在这个取景下必然失败，而"只看到一个环"
本来就不是错误。本入口就是为这种取景准备的：

- 找出画面里最像环的那一个（圆心在画面内 > 支撑观测数 > 直径带数 > 离光轴近）；
- 环心换算成**相机系毫米**（原点在光轴、x 右、y 下），可以直接喂
  ``grasp.align_xy(reading=...)``；
- 比例尺优先用**已知的环外圈直径**（``ring_target.diameter_mm``）：
  ``毫米/像素 = 直径 / 实测外圈直径像素``，与镜头距离无关；
- 顺带核验环里有没有数字（墨迹占比 + 字形高度），但**不影响**环心读数。

画面里只画 ASCII（``cv2.putText`` 画不出中文），中文结论打印在终端：**每
``--print-every`` 帧打一行**（不用按键也有数据），按 ``p`` 打完整结论，
按 ``q`` 退出。

也可以直接跑一张图（不需要摄像头、不加载模型）::

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
    """只解析命令行：**不读配置、不打开设备**。"""
    parser = argparse.ArgumentParser(description="视野里那个环：环心相机坐标（可对准读数）")
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
    """画环、环心十字与相机坐标。没有环时原样返回（``overlay`` 会写一行提示）。"""
    import cv2

    target = _unpack(result)
    canvas = frame.copy()
    if target is None:
        return canvas
    centre = (int(round(target.center_px[0])), int(round(target.center_px[1])))
    radius = max(3, int(round(target.radius_px)))
    # 有数字画绿、空环画橙：颜色只表达"这个环里有没有数字"，不表达准不准。
    colour = (0, 255, 0) if target.digit_present else (0, 165, 255)
    cv2.circle(canvas, centre, radius, colour, 3)
    cv2.drawMarker(canvas, centre, (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    if target.usable:
        label = f"centre x={target.x_mm:+.1f} y={target.y_mm:+.1f} mm"
    else:
        label = "centre: no mm"
    if not target.digit_present:
        label += " | no digit"
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
    """一行结论：有环就写环心与比例尺，没环就写"走到了哪一步"。"""
    if target is None:
        return f"  没有环：{locator.last_diagnosis or '检测还没跑过'}"
    reading = target.reading()
    where = "无毫米" if reading is None else f"x={reading[0]:+.1f} y={reading[1]:+.1f} mm"
    digit = "有数字" if target.digit_present else "空环"
    scale = (
        f"{target.scale_mm_per_px:.4f} mm/px({target.scale_source})"
        if target.scale_mm_per_px is not None
        else "无比例尺"
    )
    return (
        f"  环心 @({target.center_px[0]:.0f},{target.center_px[1]:.0f}) "
        f"r={target.radius_px:.0f}px 基准圈 {target.reference_diameter_px:.0f}px "
        f"{where}，{digit}，{scale}"
    )


def _report(target, locator) -> None:
    """终端里把结论连同对准读数一起打出来（按 p）。"""
    if target is None:
        print("画面里没有环：" + (locator.last_diagnosis or "检测还没跑过"))
        print("  排查：环是否在画面里、光照是否够（Canny 找不到边缘时不成环）、"
              "环的直径是否超过画面短边")
        return
    print(target.summary())
    print(f"  对准读数（相机系）: {target.reading()}")
    print(f"  {locator.last_diagnosis}")
    print(f"  {target.detail}")


def run_image(path: str, settings) -> int:
    """在一张图片上跑一次：不建会话（会话会加载模型），环检测是纯经典 CV。"""
    from modules.vision.drive.material_locate import read_image
    from modules.vision.drive.numbered_ring import RingTargetLocator

    frame = read_image(path)
    if frame is None:
        print(f"读不出图片：{path}")
        return 1
    locator = RingTargetLocator(
        geometry=settings.station_geometry,
        digits=settings.ring_digits,
        detection=settings.station_detection,
        diameter_mm=settings.ring_target.diameter_mm,
    )
    target = locator.detect(frame)
    _report(target, locator)
    return 0 if target is not None else 1


def run(settings, *, print_every: int = PRINT_EVERY) -> int:
    """开摄像头预览：只找视野里那一个环，定期打印环心读数。"""
    window = open_preview()
    camera_id = settings.camera_id("windows" if os.name == "nt" else "jetson")
    shown = {"frames": 0}
    try:
        with Vision._open_config(settings) as vision:
            print(f"摄像头 {camera_id}，窗口 {window!r}；按 p 打完整结论，按 q 退出")
            diameter = settings.ring_target.diameter_mm
            if diameter is not None:
                print(f"比例尺用环外圈直径 {diameter:.1f} mm 自标定："
                      "毫米/像素 = 直径 / 实测外圈像素（与镜头距离无关）")
            else:
                print("没有配 ring_target.diameter_mm：退回 depth/focal 或 station.mm_per_px")
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

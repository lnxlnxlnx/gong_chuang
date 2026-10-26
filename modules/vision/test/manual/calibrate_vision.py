"""视觉标定工具：焦距、台面深度，以及标定文件的现状检查。

    python test.py   # 选 25

**这个工具补的是当前最大的实际短板**：``modules/vision/calibration.json`` 里
``focal_px``、``table_depth_mm``、``plane`` 全是 ``null``，所以所有毫米读数（``z_mm``、
两点测量的毫米值、叠放位的毫米位移）用的都是粗标定兜底值，**不是真实距离**。
README 自己也写了"由框反推的深度偏近"。

## 怎么标

把一块**已知直径的圆片**（例如 30 mm 的物料顶面）平放在台面上、离镜头已知距离处，
让它在画面里尽量正对镜头，然后：

1. 按 ``f`` 进入焦距标定；
2. 左键点圆片**左边缘**、再点**右边缘**——量出像素直径；
3. 终端会问真实直径与到镜头的距离（回车用默认值）；
4. 工具反解出焦距，并在 ``calibration.json`` 里更新它。

焦距反解公式是 ``f = 距离 × 像素直径 / 真实直径``，与
:func:`modules.vision.drive.pixel_measure.focal_px_from_known_length` 同一处实现。
多测几次更稳：每次测完都会把样本列出来并给平均值，按 ``s`` 保存当前平均值。

按 ``d`` 直接设置台面深度（镜头到台面的距离，毫米）——它是"离台面高度"换算的必需项，
用卷尺量一次即可。按 ``c`` 打印标定文件现状与还缺什么。按 ``q`` 退出。

**只写 ``calibration.json``，不碰 ``config.json``**：两者生命周期不同，
标定产物不该顺手改动手改的配置。
"""

from __future__ import annotations

MENU_ENTRY = (25, "视觉标定（焦距/台面深度）")

import argparse

from modules.vision.config import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CONFIG_PATH,
    load_vision_config,
    save_vision_calibration,
)
from modules.vision.drive.camera import looks_blank
from modules.vision.drive.pixel_measure import (
    DEFAULT_REFERENCE_MM,
    focal_px_from_known_length,
    mean_focal_px,
    pixel_distance,
)
from modules.vision.drive.preview import (
    BLANK_FRAME_WARNING,
    close_previews,
    open_preview,
    warn_blank_preview,
)
from modules.vision.runtime import Vision, load_vision_settings

#: 连续读不到这么多帧就认为摄像头掉线，退出预览。
MAX_MISSES = 30

#: 默认的"到镜头距离"（毫米）：现场常用机位，作为提问时的默认值。
DEFAULT_DISTANCE_MM = 200.0


def parse_args() -> argparse.Namespace:
    """只解析命令行：**不读配置、不打开设备**。"""
    parser = argparse.ArgumentParser(description="视觉标定：焦距与台面深度")
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头序号（默认读 config.json 对应平台的 camera_id）")
    parser.add_argument("--reference-mm", type=float, default=DEFAULT_REFERENCE_MM,
                        help=f"标定圆片的真实直径（毫米），默认 {DEFAULT_REFERENCE_MM:.0f}")
    return parser.parse_args()


class FocalCalibration:
    """收集多次焦距样本，给出平均值；只负责算，不负责存。"""

    def __init__(self) -> None:
        self.samples = []
        self.pending = []          # 这一次测量的两个边缘点
        self.last_measurement = None

    def add_edge(self, x: float, y: float, image_size):
        """放一个边缘点；两个都放好后返回这次量到的 :class:`PixelMeasurement`。"""
        self.pending.append((float(x), float(y)))
        if len(self.pending) < 2:
            return None
        measured = pixel_distance(self.pending[0], self.pending[1], image_size=image_size)
        self.pending = []
        self.last_measurement = measured
        return measured

    def record(self, pixel_length: float, real_length_mm: float, distance_mm: float) -> float:
        """记一次样本并返回它反解出的焦距。"""
        focal = focal_px_from_known_length(
            pixel_length=pixel_length, real_length_mm=real_length_mm, distance_mm=distance_mm
        )
        self.samples.append(focal)
        return focal

    def clear(self) -> None:
        self.samples = []
        self.pending = []
        self.last_measurement = None

    @property
    def mean(self):
        """样本平均焦距；一个样本都没有时是 ``None``。"""
        return None if not self.samples else mean_focal_px(self.samples)

    def summary(self) -> str:
        if not self.samples:
            return "还没有焦距样本"
        listed = "、".join(f"{value:.1f}" for value in self.samples)
        return f"{len(self.samples)} 个焦距样本：{listed}；平均 {self.mean:.1f} px"


def run(settings, *, reference_mm: float = DEFAULT_REFERENCE_MM) -> int:
    """打开摄像头并进入标定循环，直到按 q 退出。"""
    import cv2
    import os

    window = open_preview()
    calibration = FocalCalibration()
    blank_frames = 0
    warned_blank = False
    reason = ""
    try:
        with Vision.from_settings(settings) as vision:
            camera_id = settings.camera_id("windows" if os.name == "nt" else "jetson")
            frame_size = vision._frame_size()
            print(f"摄像头 {camera_id}，窗口 {window!r}")
            _print_status(settings)
            print("按键：f 焦距标定（左键点圆片左右边缘）｜d 设置台面深度｜"
                  "s 保存焦距平均值｜c 打印标定现状｜q 退出")

            current = [None]

            def on_mouse(event, x, y, _flags, _param) -> None:
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                measured = calibration.add_edge(x, y, frame_size)
                if measured is None:
                    print("放下第一个边缘点，再点另一侧边缘")
                    return
                print(f"像素直径 {measured.length_px:.2f} px"
                      f"（ΔX={measured.dx_px:+.1f}，ΔY={measured.dy_px:+.1f}）")
                _ask_and_record(calibration, measured, reference_mm, vision)

            cv2.setMouseCallback(window, on_mouse)

            while True:
                frame = vision._camera.read()
                if frame is None:
                    blank_frames += 1
                    if blank_frames >= MAX_MISSES:
                        reason = "camera-lost"
                        break
                    continue
                blank_frames = 0
                current[0] = frame
                if looks_blank(frame) and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)
                cv2.imshow(window, _draw(frame, calibration))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    reason = "quit"
                    break
                if key == ord("f"):
                    calibration.pending = []
                    print("焦距标定：左键点圆片的左边缘，再点右边缘")
                elif key == ord("d"):
                    _ask_table_depth(vision)
                elif key == ord("s"):
                    _save_focal(calibration)
                elif key == ord("c"):
                    _print_status(load_vision_settings())
    finally:
        close_previews()
    return 0 if reason in ("quit", "") else 1


def _ask_and_record(calibration, measured, reference_mm, vision) -> None:
    """问真实直径与距离，记一个样本，并提示当前平均值。"""
    real = _ask_float("圆片真实直径（毫米）", reference_mm)
    distance = _ask_float("圆片到镜头的距离（毫米）", DEFAULT_DISTANCE_MM)
    focal = calibration.record(measured.length_px, real, distance)
    print(f"这次反解出焦距 {focal:.1f} px；{calibration.summary()}")
    print("再测一次更稳（换一个距离或挪一下位置），按 s 保存平均值")


def _ask_table_depth(vision) -> None:
    """问一个台面深度并写盘。"""
    current = vision._settings.table_depth_mm
    default = DEFAULT_DISTANCE_MM if current is None else float(current)
    depth = _ask_float("镜头到台面的距离（毫米）", default)
    if depth <= 0:
        print("台面深度必须是正数，没有写盘")
        return
    save_vision_calibration(table_depth_mm=depth)
    print(f"已写入 {DEFAULT_CALIBRATION_PATH}：table_depth_mm = {depth:.1f}")


def _save_focal(calibration) -> None:
    """把当前焦距平均值写盘。"""
    if calibration.mean is None:
        print("还没有样本可存：按 f 先量一次圆片")
        return
    save_vision_calibration(focal_px=calibration.mean)
    print(f"已写入 {DEFAULT_CALIBRATION_PATH}：focal_px = {calibration.mean:.1f}")


def _ask_float(question: str, default: float) -> float:
    """问一个浮点数；空回答用默认值，解析失败也说清楚并回默认值。"""
    text = input(f"{question} [{default:g}]: ").strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        print(f"无法识别 {text!r}，改用 {default:g}")
        return float(default)


def _print_status(settings) -> None:
    """打印标定文件的现状与还缺什么——这是这个工具最常用的一项。"""
    missing = []
    if settings.focal_px is None:
        missing.append("focal_px（按 f 标定）")
    if settings.table_depth_mm is None:
        missing.append("table_depth_mm（按 d 设置）")
    if settings.plane is None:
        missing.append("plane（实验 A 平面标定，见 modules/grasp/docs/抓取坐标链路.md）")
    print(f"标定文件 {DEFAULT_CALIBRATION_PATH}")
    print(f"  focal_px       = {settings.focal_px}")
    print(f"  table_depth_mm = {settings.table_depth_mm}")
    print(f"  plane          = {'已标定' if settings.plane is not None else None}")
    if missing:
        print("  还缺：" + "；".join(missing))
        print("  缺项期间毫米读数用的是粗标定兜底值，**不是真实距离**")
    else:
        print("  标定齐全")


def _draw(frame, calibration):
    """画已放下的边缘点与上一次的测量线。**只写 ASCII**。"""
    import cv2

    annotated = frame.copy()
    for index, (x, y) in enumerate(calibration.pending, start=1):
        point = (int(round(x)), int(round(y)))
        cv2.drawMarker(annotated, point, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(annotated, f"E{index}", (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    measured = calibration.last_measurement
    if measured is not None:
        cv2.line(
            annotated,
            (int(round(measured.start[0])), int(round(measured.start[1]))),
            (int(round(measured.end[0])), int(round(measured.end[1]))),
            (255, 128, 0),
            2,
        )
    headline = (
        f"d={measured.length_px:.1f}px" if measured is not None else "f: click both edges"
    )
    cv2.putText(annotated, headline, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    mean = calibration.mean
    if mean is not None:
        cv2.putText(annotated, f"f={mean:.0f}px n={len(calibration.samples)}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return annotated


def main() -> None:
    args = parse_args()
    settings = load_vision_settings()
    print(f"配置来自 {DEFAULT_CONFIG_PATH}")
    raise SystemExit(run(settings, reference_mm=args.reference_mm))


if __name__ == "__main__":
    main()

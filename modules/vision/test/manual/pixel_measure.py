"""两点像素测量真机测试：鼠标点两个点，量 ΔX / ΔY / 直线距离。

    python test.py   # 选 24

操作：

- **第一次左键**冻结当前画面并放下起点；
- **第二次左键**放下终点，显示 ΔX / ΔY / 距离；
- **第三次左键**清空结果并恢复实时画面（画面不再冻结）。

按 ``m`` 在像素与毫米之间切换显示（毫米要有焦距与工作距离，缺哪一项就说缺哪一项，
不编一个数出来）；按 ``p`` 把当前结论打到终端；按 ``q`` / ``Esc`` 退出。

画面里只画 ASCII（``cv2.putText`` 画不出中文），中文结论打印在终端。

**为什么冻结画面**：不冻结的话，从第一次点击到第二次点击之间画面一直在动，
两次点击量的就不是同一个场景里的两个点，读数没有意义。这是测量工具的常规做法，
不是偷懒。

离线自测（不需要摄像头）：``python -m modules.vision.drive.pixel_measure``
"""

from __future__ import annotations

MENU_ENTRY = (24, "两点像素测量")

import argparse
import os

from modules.vision.config import DEFAULT_CONFIG_PATH
from modules.vision.drive.camera import looks_blank
from modules.vision.drive.pixel_measure import pixel_distance, to_millimetres
from modules.vision.drive.preview import (
    BLANK_FRAME_WARNING,
    close_previews,
    open_preview,
    warn_blank_preview,
)
from modules.vision.runtime import Vision, load_vision_settings

#: 连续读不到这么多帧就认为摄像头掉线，退出预览。
MAX_MISSES = 30

#: 测点的画法：十字丝大小与线宽。
MARKER_SIZE = 14
MARKER_THICKNESS = 2


def parse_args() -> argparse.Namespace:
    """只解析命令行：**不读配置、不打开设备**。"""
    parser = argparse.ArgumentParser(description="两点像素测量：ΔX / ΔY / 直线距离")
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头序号（默认读 config.json 对应平台的 camera_id）")
    parser.add_argument("--image", type=str, default=None,
                        help="在一张图片上量（不给就开摄像头）")
    parser.add_argument("--focal", type=float, default=None,
                        help="像素焦距；不给就用 calibration.json 的标定值或粗标定兜底")
    parser.add_argument("--depth", type=float, default=None,
                        help="被测平面到镜头的距离（毫米）；不给就用 calibration.json 的 "
                             "table_depth_mm，两者都没有时不显示毫米")
    return parser.parse_args()


class MeasurementSession:
    """三点循环的测量状态机：0 个点（实时）→ 1 个点（冻结）→ 2 个点（冻结）→ 清空。

    状态单独一个类，是为了让"哪一次点击该干什么"只有一处实现——散在鼠标回调里
    会让"第三次点击到底该清空还是该移动终点"变成看代码猜。
    """

    def __init__(self, image_size) -> None:
        self.image_size = image_size
        self.points = []
        self.frozen = None

    @property
    def complete(self) -> bool:
        return len(self.points) == 2

    def add_point(self, x: float, y: float, frame):
        """放一个点；返回 True 表示这一次点击应当重新冻结画面。"""
        if len(self.points) >= 2:
            # 第三次点击：清空，回到实时画面。
            self.points = []
            self.frozen = None
            return False
        self.points.append((float(x), float(y)))
        if len(self.points) == 1:
            self.frozen = frame.copy()
            return True
        return False

    def clear(self) -> None:
        self.points = []
        self.frozen = None

    def measurement(self):
        """两个点都放好了就返回测量结果，否则 ``None``。"""
        if not self.complete:
            return None
        return pixel_distance(self.points[0], self.points[1], image_size=self.image_size)


def run(settings, *, image_path=None, focal_px=None, depth_mm=None) -> int:
    """打开摄像头（或读一张图）并进入测量循环，直到按 q 退出。"""
    import cv2

    if image_path is not None:
        return _run_image(image_path, focal_px=focal_px, depth_mm=depth_mm)

    window = open_preview()
    session = None
    blank_frames = 0
    warned_blank = False
    show_mm = False
    reason = ""
    try:
        with Vision.from_settings(settings) as vision:
            camera_id = settings.camera_id("windows" if os.name == "nt" else "jetson")
            frame_size = vision._frame_size()
            session = MeasurementSession(frame_size)
            scale = _resolve_scale(vision, focal_px, depth_mm)
            print(f"摄像头 {camera_id}，窗口 {window!r}")
            print("左键：第 1 次放下起点（冻结画面）、第 2 次放下终点、第 3 次清空")
            print("按键：m 切换像素/毫米，p 打印结论，q 退出")
            _report_scale(scale)
            if scale is None:
                print("（缺焦距或工作距离，毫米显示不可用；按 m 只会提示缺什么）")

            # ``current`` 必须在 ``setMouseCallback`` **之前**建好：回调是 OpenCV
            # 从别的线程调的，第一次点击可能早于下面那句赋值。
            current = [None]

            def on_mouse(event, x, y, _flags, _param) -> None:
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                session.add_point(x, y, current[0])
                measured = session.measurement()
                if measured is not None:
                    print(measured.summary())
                    if show_mm:
                        _print_mm(measured, scale)

            cv2.setMouseCallback(window, on_mouse)

            while True:
                if session.frozen is not None:
                    frame = session.frozen
                else:
                    frame = vision._camera.read()
                    if frame is None:
                        blank_frames += 1
                        if blank_frames >= MAX_MISSES:
                            reason = "camera-lost"
                            break
                        continue
                    blank_frames = 0
                    current[0] = frame
                    if looks_blank(frame) and blank_frames == 0 and not warned_blank:
                        warned_blank = True
                        warn_blank_preview(camera_id)
                if session.frozen is None:
                    current[0] = frame
                annotated = _draw(frame, session, show_mm=show_mm, scale=scale)
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    reason = "quit"
                    break
                if key == ord("m"):
                    show_mm = not show_mm
                    measured = session.measurement()
                    if show_mm and measured is not None:
                        _print_mm(measured, scale)
                    elif show_mm:
                        print("切换到毫米显示：放满两个点之后会一起给出毫米")
                elif key == ord("p"):
                    measured = session.measurement()
                    if measured is None:
                        print(f"已经放了 {len(session.points)} 个点，还差 "
                              f"{2 - len(session.points)} 个")
                    else:
                        print(measured.summary())
                        if show_mm:
                            _print_mm(measured, scale)
    finally:
        close_previews()
    return 0 if reason in ("quit", "") else 1


def _resolve_scale(vision, focal_px, depth_mm):
    """定出 ``(焦距, 工作距离)``；缺哪一项就返回 ``None`` 并在终端说清楚缺什么。"""
    try:
        focal = vision._effective_focal_px(focal_px)
    except ValueError:
        return None
    depth = vision._settings.table_depth_mm if depth_mm is None else float(depth_mm)
    if depth is None or depth <= 0:
        return None
    return (focal, depth)


def _report_scale(scale) -> None:
    if scale is None:
        return
    focal, depth = scale
    print(f"毫米换算：焦距 {focal:.0f} px、工作距离 {depth:.0f} mm"
          f"（比例尺 {depth / focal:.4f} mm/px）")


def _print_mm(measured, scale) -> None:
    if scale is None:
        print("没有毫米读数：缺焦距或工作距离。用 --focal / --depth 传，"
              "或先跑 python -m modules.vision.test.manual.calibrate_vision")
        return
    focal, depth = scale
    print(to_millimetres(measured, focal_px=focal, working_distance_mm=depth).summary())


def _draw(frame, session: MeasurementSession, *, show_mm: bool, scale):
    """画十字丝、连线与结论。**只写 ASCII**，中文走终端。"""
    import cv2

    annotated = frame.copy()
    for index, (x, y) in enumerate(session.points, start=1):
        point = (int(round(x)), int(round(y)))
        cv2.drawMarker(annotated, point, (0, 255, 255), cv2.MARKER_CROSS,
                       MARKER_SIZE, MARKER_THICKNESS)
        cv2.putText(annotated, f"P{index}", (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), MARKER_THICKNESS)

    measured = session.measurement()
    if measured is None:
        hint = "click 2 points" if not session.points else "click the 2nd point"
        cv2.putText(annotated, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), MARKER_THICKNESS)
        return annotated

    first = (int(round(measured.start[0])), int(round(measured.start[1])))
    second = (int(round(measured.end[0])), int(round(measured.end[1])))
    cv2.line(annotated, first, second, (255, 128, 0), MARKER_THICKNESS)
    lines = [
        f"dX={measured.dx_px:+.1f} dY={measured.dy_px:+.1f} px",
        f"dist={measured.length_px:.1f} px",
    ]
    if show_mm and scale is not None:
        focal, depth = scale
        mm = to_millimetres(measured, focal_px=focal, working_distance_mm=depth)
        lines.append(f"dX={mm.dx_mm:+.2f} dY={mm.dy_mm:+.2f} mm")
        lines.append(f"dist={mm.length_mm:.2f} mm")
    for row, text in enumerate(lines):
        cv2.putText(annotated, text, (10, 24 + row * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), MARKER_THICKNESS)
    return annotated


def _run_image(image_path, *, focal_px, depth_mm) -> int:
    """在一张图片上量：读出图片、让用户点两个点。

    非 ASCII 路径必须走 ``read_image``：``cv2.imread`` 在 Windows 按 ANSI 打开文件，
    ``models/蓝.jpg`` 会直接返回 ``None``（见 ``material_locate`` 里的同一处处理）。
    """
    import cv2

    from modules.vision.drive.material_locate import read_image

    frame = read_image(image_path)
    if frame is None:
        print(f"读不出图片：{image_path}")
        return 1
    window = open_preview()
    session = MeasurementSession((frame.shape[1], frame.shape[0]))
    session.frozen = frame.copy()
    scale = None
    if focal_px is not None and depth_mm is not None:
        scale = (float(focal_px), float(depth_mm))
        _report_scale(scale)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(session.points) >= 2:
            session.clear()
            session.frozen = frame.copy()
            return
        session.add_point(x, y, frame)
        session.frozen = frame.copy()
        measured = session.measurement()
        if measured is not None:
            print(measured.summary())
            if scale is not None:
                _print_mm(measured, scale)

    try:
        cv2.setMouseCallback(window, on_mouse)
        print(f"图片 {image_path}；左键点两个点，第 3 次清空，按 q 退出")
        while True:
            cv2.imshow(window, _draw(session.frozen, session, show_mm=scale is not None,
                                     scale=scale))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        close_previews()
    return 0


def main() -> None:
    args = parse_args()
    settings = load_vision_settings()
    print(f"配置来自 {DEFAULT_CONFIG_PATH}")
    raise SystemExit(run(settings, image_path=args.image, focal_px=args.focal,
                         depth_mm=args.depth))


if __name__ == "__main__":
    main()

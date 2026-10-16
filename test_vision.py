"""实时看视觉识别结果：画面 + x/y/z + YOLO 框 + 置信度 + 每道门槛的判定。

和 ``main.py`` / ``test.py`` 一样是项目根目录下的入口，区别是它**只看不动**：
不开机械臂、不开串口，只开摄像头跑识别。所以光照不对时可以反复跑它调参数，
不会碰机械臂。

用法::

    python test_vision.py                    # 实时预览，按 q 或 ESC 退出
    python test_vision.py --once             # 只跑一帧，结果打到终端后退出
    python test_vision.py --camera 0         # 指定摄像头序号（默认读 config.json）
    python test_vision.py --save             # 预览时按 s 存图；--once --save 则立即存
    python test_vision.py --async            # 推理放后台线程（画面更顺，框会滞后几帧）

现场调参（**只对本次运行生效，不写回文件**）::

    python test_vision.py --tolerance 90         # targets.color_tolerance（BGR 距离）
    python test_vision.py --hue-tolerance 12     # targets.hue_tolerance（台面色相差 14，别超 12）
    python test_vision.py --value-ratio 0.55     # targets.value_ratio（台面亮度比 0.42，别低于它）
    python test_vision.py --min-area 300         # targets.min_area_px
    python test_vision.py --std-max 18           # targets.interior_std_max（内部灰度标准差上限）
    python test_vision.py --edge-min 0.4         # targets.edge_support_min（边界支撑下限）
    python test_vision.py --relax 1.0            # targets.colour_relax_factor（1.0 = 不放松）
    python test_vision.py --window 30 400        # targets.working_distance_mm（半径窗口的出处）
    python test_vision.py --conf 0.15            # model.conf_thresh（让 YOLO 多出框）

找到合适的值就写进 ``modules/vision/config.json``；每一项的**代价**都写在
``modules/vision/config.py`` 的字段注释里（比如色相容差一超过 12，台面就会整片进掩膜）。

画面上标了什么：

- **绿粗圈** = 通过校验的物料顶面，旁边是 ``x / y / z``（相机系毫米，z 是镜头到顶面的距离）
  与置信度；
- **灰细圈** = 被丢掉的候选，旁边是它没过的判据（内部 σ / 边界支撑）；
- **黄框** = YOLO 框（类别 + 置信度）；
- **左上角** = 这一帧的来源（palette / palette-relaxed / model / hough）、半径窗口、
  候选与接受数，以及当前生效的阈值；
- **左下角** = 取帧与推理速率、框滞后多少毫秒（``run_preview`` 的标准 HUD）。

预览默认走**同帧**推理（``async_detect=False``）：框和画面一定是同一帧，调参时看到的
就是算法实际看到的。``--async`` 更顺，但框会比画面旧几帧（左下角会写滞后多少毫秒）。
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from modules.vision.config import TargetSettings, VisionConfig
from modules.vision.drive.preview import close_previews
from modules.vision.drive.stream import PreviewStats, run_preview
from modules.vision.reports import MaterialReport, RegionReport
from modules.vision.runtime import Vision, _load_settings, _override_settings
from project_paths import PROJECT_ROOT

#: 窗口名必须是 ASCII：``cv2.putText`` / HighGUI 都画不出中文。
WINDOW = "test_vision"

#: 存图目录（项目根下）。
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

#: 每显示这么多帧往终端打一行统计；0 表示不打印。
PRINT_EVERY = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="实时看视觉识别结果（只看不动：不开机械臂、不开串口）"
    )
    parser.add_argument("--camera", type=int, default=None, help="摄像头序号；省略则读配置")
    parser.add_argument("--once", action="store_true", help="只跑一帧，结果打到终端后退出")
    parser.add_argument("--save", action="store_true", help="预览时按 s 存图；配合 --once 立即存")
    parser.add_argument("--out", default=None, help=f"存图目录；默认 {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--async", dest="async_detect", action="store_true",
                        help="推理放后台线程（画面更顺，框会滞后几帧）")
    parser.add_argument("--focal", type=float, default=None, help="覆盖像素焦距（本画幅专用）")
    # ---- 现场调参：全部只覆盖这一次 ----
    parser.add_argument("--tolerance", type=float, default=None, dest="colour_tolerance",
                        help="targets.color_tolerance：颜色容差（BGR 欧氏距离）")
    parser.add_argument("--hue-tolerance", type=float, default=None, dest="hue_tolerance",
                        help="targets.hue_tolerance：色相容差（台面差 14，别超过 12）")
    parser.add_argument("--value-ratio", type=float, default=None, dest="value_ratio",
                        help="targets.value_ratio：亮度下限倍数（台面比 0.42，别低于它）")
    parser.add_argument("--min-area", type=float, default=None, dest="min_area_px",
                        help="targets.min_area_px：色块最小面积（像素）")
    parser.add_argument("--std-max", type=float, default=None, dest="interior_std_max",
                        help="targets.interior_std_max：内部灰度标准差上限")
    parser.add_argument("--edge-min", type=float, default=None, dest="edge_support_min",
                        help="targets.edge_support_min：圆周边界支撑下限（0–1）")
    parser.add_argument("--relax", type=float, default=None, dest="colour_relax_factor",
                        help="targets.colour_relax_factor：找不到色块时放大 BGR 容差的倍数")
    parser.add_argument("--window", type=float, nargs=2, default=None, dest="working_distance_mm",
                        metavar=("NEAR", "FAR"), help="targets.working_distance_mm：工作距离窗口（毫米）")
    parser.add_argument("--conf", type=float, default=None, help="model.conf_thresh：YOLO 置信度阈值")
    return parser


def overridden_settings(arguments) -> VisionConfig:
    """把命令行覆盖落到**配置快照**上（复用 vision 模块自己的校验，不写回文件）。"""
    return _override_settings(
        _load_settings(),
        camera_id=arguments.camera,
        conf_thresh=arguments.conf,
        colour_tolerance=arguments.colour_tolerance,
        hue_tolerance=arguments.hue_tolerance,
        value_ratio=arguments.value_ratio,
        min_area_px=arguments.min_area_px,
        interior_std_max=arguments.interior_std_max,
        edge_support_min=arguments.edge_support_min,
        colour_relax_factor=arguments.colour_relax_factor,
        depth_range_mm=(
            None if arguments.working_distance_mm is None
            else tuple(arguments.working_distance_mm)
        ),
    )


# ------------------------------------------------------------------ 画面标注
def _confidence(region: RegionReport) -> str:
    return "--" if region.confidence is None else f"{region.confidence:.2f}"


def _label_for(region: RegionReport) -> str:
    """一个候选圆面旁边的那行字（ASCII：``cv2.putText`` 画不出中文）。"""
    if region.accepted:
        return (
            f"{region.name or 'material'} x={region.x_mm:+.1f} y={region.y_mm:+.1f} "
            f"z={region.z_mm:.1f}mm r={region.region.radius:.0f}px c={_confidence(region)}"
        )
    return (
        f"rejected std={region.face.interior_std:.1f} "
        f"edge={region.face.edge_support:.2f}"
    )


def matching_box(region: RegionReport, report: MaterialReport):
    """这个圆面落在哪个 YOLO 框里（取最小的那个；没有就返回 ``None``）。

    用来把**圆拟合出来的半径**和**模型框的尺寸**摆在一起看：顶面是个圆，
    框得住它的时候 ``(宽 + 高) / 4`` 就是半径的另一个独立估计。两者接近说明
    这一帧的圆心和半径互相印证；差得远说明框是松的（模型把台面/夹爪一起框进去了），
    这时只该信圆。实测两个接近的例子差 0.6 px 与 1.4 px；松框能差到 40 px 以上。
    """
    boxes = [
        hit for hit in report.hits
        if hit.x1 <= region.center_x_px <= hit.x2 and hit.y1 <= region.center_y_px <= hit.y2
    ]
    if not boxes:
        return None
    return min(boxes, key=lambda hit: hit.width * hit.height)


def draw_report(frame: np.ndarray, report: MaterialReport) -> np.ndarray:
    """画 YOLO 框 + 每个候选圆 + 它的 x/y/z 或没过的判据。"""
    import cv2

    canvas = frame.copy()
    for hit in report.hits:
        top_left = (int(hit.x1), int(hit.y1))
        bottom_right = (int(hit.x2), int(hit.y2))
        cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"{hit.name} {hit.score:.2f}",
            (top_left[0], max(top_left[1] - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
    for region in report.regions:
        centre = (int(round(region.center_x_px)), int(round(region.center_y_px)))
        radius = max(2, int(round(region.region.radius)))
        if region.accepted:
            cv2.circle(canvas, centre, radius, (0, 255, 0), 3)
            colour = (0, 255, 0)
        else:
            cv2.circle(canvas, centre, radius, (140, 140, 140), 1)
            colour = (200, 200, 200)
        text = _label_for(region)
        origin = (max(centre[0] - radius, 4), min(centre[1] + radius + 18, canvas.shape[0] - 6))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    return canvas


def hud_lines(report: Optional[MaterialReport], targets: TargetSettings) -> Sequence[str]:
    """左上角那几行：来源、窗口、计数、当前生效的阈值。"""
    source = "none" if report is None else (report.colour_source or "none")
    faces = 0 if report is None else len(report.faces)
    regions = 0 if report is None else len(report.regions)
    hits = 0 if report is None else len(report.hits)
    window = "none" if report is None or report.radius_window_px is None else (
        f"{report.radius_window_px[0]:.0f}-{report.radius_window_px[1]:.0f}px"
    )
    return (
        f"source={source}  faces={faces}/{regions}  yolo={hits}  window={window}",
        f"tol={targets.color_tolerance:.0f} hue={targets.hue_tolerance:.0f} "
        f"V>={targets.value_ratio:.2f} area>={targets.min_area_px:.0f}",
        f"face std<={targets.interior_std_max:.1f} edge>={targets.edge_support_min:.2f} "
        f"relax={targets.colour_relax_factor:.1f}",
        "q/ESC quit   s save   p print",
    )


def make_overlay(targets: TargetSettings):
    """生成 ``run_preview`` 要的 overlay 回调（在标准 HUD 之后画自己的几行）。"""

    def overlay(canvas: np.ndarray, report: Optional[MaterialReport], _stats: PreviewStats) -> None:
        import cv2

        for index, text in enumerate(hud_lines(report, targets)):
            origin = (10, 24 + index * 26)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return overlay


# ------------------------------------------------------------------ 终端输出
def describe(report: MaterialReport, *, focal_px: Optional[float] = None) -> str:
    """把一帧的结果写成终端可读的多行文本（含每个候选的判据数字）。"""
    lines = [report.summary()]
    if focal_px is not None:
        lines.append(f"  像素焦距 {focal_px:.1f} px")
    best = report.best
    if best is not None:
        lines.append(
            f"  采用：{best.name or '未知'} 相机系 xyz=({best.x_mm:+.1f}, {best.y_mm:+.1f}, "
            f"{best.z_mm:.1f}) mm，置信度 {_confidence(best)}"
        )
    for index, region in enumerate(report.regions, start=1):
        verdict = "接受" if region.accepted else "丢掉"
        box = matching_box(region, report)
        if box is None:
            box_text = "｜YOLO 框：无"
        else:
            implied = (box.width + box.height) / 4.0
            gap = implied - region.region.radius
            box_text = (
                f"｜YOLO 框 {box.width:.0f}x{box.height:.0f} → 半径约 {implied:.1f} px"
                f"（与圆差 {gap:+.1f} px）"
            )
        lines.append(
            f"  候选{index} {verdict}：圆心 ({region.center_x_px:.0f}, {region.center_y_px:.0f}) "
            f"半径 {region.region.radius:.1f} px｜内部σ {region.face.interior_std:.1f}、"
            f"边界支撑 {region.face.edge_support:.2f}｜类别 {region.name or '未知'}"
            f"（{_confidence(region)}）{box_text}"
        )
    return "\n".join(lines)


def save_frame(frame: np.ndarray, directory: Path) -> Path:
    """把一帧（含标注）存成 PNG，返回写出的路径。"""
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"写不出图片：{path}")
    return path


def run_once(vision: Vision, *, save_dir: Optional[Path]) -> int:
    """只跑一帧：取几帧丢掉（摄像头刚开的那几帧常常是黑的），再定位一次。"""
    camera = vision._camera
    frame = None
    for _ in range(5):
        candidate = camera.read()
        if candidate is not None:
            frame = candidate
        time.sleep(0.05)
    if frame is None:
        print("读不到摄像头画面：确认序号、连接，以及没有被别的程序占用")
        return 1
    report = vision._locator.locate(frame)
    print(describe(report, focal_px=report.focal_px))
    if save_dir is not None:
        path = save_frame(draw_report(frame, report), save_dir)
        print(f"已存图：{path}")
    return 0


def run_preview_session(
    vision: Vision, *, targets: TargetSettings, save_dir: Optional[Path], async_detect: bool
) -> int:
    """实时预览；返回进程退出码。

    按键：``q``/``ESC`` 退出、``p`` 把当前结果打到终端、``s`` 存图（需要启动时给 ``--save``）。
    ``run_preview`` 的 ``on_frame`` 给的是**未标注**的原始帧，所以这里自己留一份，
    按 ``s`` 时再按同一份结果画一次——存下来的图和窗口里看到的是同一帧。
    """
    latest = {"frame": None, "report": None}

    def on_frame(frame: np.ndarray, report: Optional[MaterialReport], _stats: PreviewStats) -> None:
        latest["frame"], latest["report"] = frame, report

    def on_key(key: int, report: Optional[MaterialReport], _stats: PreviewStats) -> None:
        if key == ord("p") and report is not None:
            print(describe(report, focal_px=report.focal_px))
        if key == ord("s"):
            frame, current = latest["frame"], latest["report"]
            if save_dir is None:
                print("按 s 存图要先在启动时给 --save（例如 python test_vision.py --save）")
            elif frame is None or current is None:
                print("这一帧还没有识别结果，没存图")
            else:
                print(f"已存图：{save_frame(draw_report(frame, current), save_dir)}")

    stats = run_preview(
        vision._camera,
        vision._locator.locate,
        draw_report,
        window=WINDOW,
        async_detect=async_detect,
        print_every=PRINT_EVERY,
        on_key=on_key,
        on_frame=on_frame,
        overlay=make_overlay(targets),
    )
    print("  " + stats.line())
    return 0 if stats.reason in ("quit", "") else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        settings = overridden_settings(arguments)
    except ValueError as problem:
        print(f"参数不合法：{problem}")
        return 1
    save_dir = None
    if arguments.save:
        save_dir = DEFAULT_OUTPUT_DIR if arguments.out is None else Path(arguments.out)
    targets = settings.targets
    print(f"阈值：容差 {targets.color_tolerance:.0f}｜色相 {targets.hue_tolerance:.0f}｜"
          f"亮度比 {targets.value_ratio:.2f}｜最小面积 {targets.min_area_px:.0f}｜"
          f"内部σ≤{targets.interior_std_max:.1f}｜边界≥{targets.edge_support_min:.2f}｜"
          f"放松 {targets.colour_relax_factor:.1f}")
    print("这些值来自 modules/vision/config.json；命令行覆盖只对本次运行生效。")
    try:
        with Vision._open_config(settings, focal_px=arguments.focal) as vision:
            if arguments.once:
                return run_once(vision, save_dir=save_dir)
            return run_preview_session(
                vision,
                targets=targets,
                save_dir=save_dir,
                async_detect=arguments.async_detect,
            )
    finally:
        close_previews()


if __name__ == "__main__":
    raise SystemExit(main())

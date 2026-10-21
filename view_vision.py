"""上位机全功能查看器：一个窗口看遍视觉模块的每一类识别结果。

``test_vision.py`` 只盯**物料定位**（它要调颜色阈值、看每道判据，画面越干净越好）。
本文件补的是另一半：把视觉模块**其余的识别能力**也摆到同一个窗口里，用数字键切换，
不用为了看工位识别再开一个脚本、再等一次模型加载。

## 七个模式（对应 ``modules/vision`` 的七类能力）

============  ==============================================  ==========================
键            模式                                            背后的 API
============  ==============================================  ==========================
``1``         物料定位（圆面 + YOLO 框 + 每道判据）            ``MaterialLocator.locate``
``2``         YOLO 原始检测（模型看到了什么）                  ``Detector.detect``
``3``         三圆环工位（圆心、偏角、毫米偏差）               ``Vision.detect_ring_station``
``4``         叠放位（中环 + 上下基准线）                      ``Vision.detect_stacked_ring``
``5``         同帧多目标拆分（marker / material）              ``CellTargetDetector.detect``
``6``         两点像素测量（鼠标点两个点，毫米可选）           ``Vision.pixel_distance*``
``7``         视野里那一个环（环心 → 相机系毫米）              ``Vision.locate_ring``
============  ==============================================  ==========================

## 按键

- ``1``–``7``：切换模式（切换会清掉上一次的融合结论，避免看着旧值做判断）；
- ``m``：把**最近 N 次定位**做一次稳健融合（N 由 ``--frames`` 决定），结论留在画面上；
- ``t``：把这一帧翻成闭环观测（``frame()`` 那种 ``MaterialSighting``）打到终端；
- ``p``：把当前模式的结果完整打到终端；
- ``r``：清掉工位/叠放位的跨帧稳定窗口（重新摆工位之后按）；
- ``s``：存一张标注图（启动时给了 ``--save`` 才生效）；
- ``h``：把这份按键表打到终端；``q`` / ``Esc``：退出。

## 画面里全是 ASCII（``cv2.putText`` 画不出中文）

和仓库里其它预览入口一致：画面上只写数字与英文，中文结论打印在终端。窗口名也必须是
ASCII，所以叫 ``view_vision``。

## 多帧融合：预览采样 vs 真实连拍（**两者的判据不同，别混用**）

``Vision.measure()`` 自己连拍：它在一次调用里连续读 7 帧（约 1.2 s），所以那批读数的
时间跨度是可控的，:func:`modules.vision.drive.sequence.fuse` 的"在动 / 已停稳"判据才
成立。本窗口的预览链路做不到这一点——推理只有 0.3–3 fps，相邻两次定位可能隔好几秒，
拿它们当"连拍"喂给 ``fuse`` 会得到一个时间跨度虚高、``moving`` 判错的结论。

所以这里刻意分成两条路，并且都摆出来：

- 按 ``m`` 是**预览采样融合**：用的是显示链路上最近 N 次成功定位，画面上会写明
  ``sampled``。它回答的是"这几帧看到的还是不是同一个位置"，**不是**"转盘停稳了没有"；
- `--measure` 是**真实连拍**：进预览前调一次 ``Vision.measure()``（配 ``--settle`` 则调
  ``measure_settled``），把严格的融合结论打到终端，然后照常进预览。

要边看画面边等停稳，用 ``python test_vision.py --measure --settle``：那个入口本来就是
为"停下来再采"设计的。

## 用法

::

    python view_vision.py                    # 默认从物料定位看起
    python view_vision.py --mode rings       # 直接从三圆环工位看起
    python view_vision.py --once             # 不开窗口：一帧逐个模式跑一遍并打印
    python view_vision.py --once --save      # 顺便把每个模式的标注图存到 output/
    python view_vision.py --measure          # 先做一次真实连拍融合，再进预览
    python view_vision.py --sync             # 串行推理：结果与画面严格同帧，但更卡

现场调阈值仍走 ``test_vision.py`` 的那组参数（本文件不重复提供，避免两套覆盖入口）。

## 导入本模块不产生副作用

不读配置、不开设备、不建窗口、不 import cv2：``main()`` 才做那些事，``--once`` 也只
跑一帧。
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from modules.vision.drive.preview import (
    BLANK_FRAME_WARNING,
    close_previews,
    open_preview,
    warn_blank_preview,
)
from modules.vision.drive.sequence import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_STILL_MM,
    TimedPoint,
    fuse,
)
from modules.vision.drive.stream import PreviewStats, run_preview
from modules.vision.runtime import (
    DEFAULT_MEASURE_FRAMES,
    DEFAULT_SETTLE_TIMEOUT_SECONDS,
    Vision,
    _load_settings,
    _override_settings,
    _sightings_from_report,
)
from project_paths import PROJECT_ROOT

#: 窗口名必须是 ASCII：``cv2.putText`` / HighGUI 都画不出中文（与其它预览入口一致）。
WINDOW = "view_vision"

#: 存图目录（项目根下，git 忽略）。
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

#: 每显示这么多帧往终端打一行统计；0 表示不打印。
PRINT_EVERY = 60

#: 测点十字丝的尺寸与线宽（与 ``test.manual.pixel_measure`` 同一套画法）。
MARKER_SIZE = 14
MARKER_THICKNESS = 2

#: 结果里"这一帧没有结果"用的哨兵。``None`` 是合法结果（预览还没算出东西），
#: 不能拿它当标志——``run_preview`` 正是用 ``result is not None`` 决定要不要调 ``draw``。
_MEASURE_PENDING = object()

# ------------------------------------------------------------------ 模式
MODE_MATERIAL = "material"
MODE_YOLO = "yolo"
MODE_RINGS = "rings"
MODE_STACKED = "stacked"
MODE_CELLS = "cells"
MODE_MEASURE = "measure"
MODE_RING = "ring"

#: 模式顺序就是数字键的顺序。
MODES: tuple[str, ...] = (
    MODE_MATERIAL,
    MODE_YOLO,
    MODE_RINGS,
    MODE_STACKED,
    MODE_CELLS,
    MODE_MEASURE,
    MODE_RING,
)

#: 画面上用的短标题（ASCII）。
MODE_TITLES = {
    MODE_MATERIAL: "material",
    MODE_YOLO: "yolo",
    MODE_RINGS: "rings",
    MODE_STACKED: "stacked",
    MODE_CELLS: "cells",
    MODE_MEASURE: "measure",
    MODE_RING: "ring",
}

#: 画面上那一行操作提示。
MODE_HINTS = {
    MODE_MATERIAL: "m fuses the last few locates; t prints closed-loop sightings",
    MODE_YOLO: "raw model boxes, no colour gate",
    MODE_RINGS: "r clears the cross-frame stability window",
    MODE_STACKED: "middle ring + top/bottom reference lines",
    MODE_CELLS: "marker / material split inside one frame",
    MODE_MEASURE: "click 2 points, 3rd click clears; frame freezes on 1st click",
    MODE_RING: "one ring in view: x/y is the align_xy reading; p prints it",
}

#: 数字键 → 模式。
MODE_KEYS = {ord(str(index)): mode for index, mode in enumerate(MODES, start=1)}

KEY_HELP = (ord("h"),)
KEY_FUSE = (ord("m"),)
KEY_SIGHTING = (ord("t"),)
KEY_PRINT = (ord("p"),)
KEY_RESET = (ord("r"),)
KEY_SAVE = (ord("s"),)


@dataclass(frozen=True)
class ModeResult:
    """一次推理的结果，**带着它是哪个模式的**。

    为什么带模式：异步预览里"用到一半结果"是常态——切到模式 3 的瞬间，工作线程可能
    还在算模式 1 的那一帧。结果自带模式，画图就只按结果自己的模式画，不会把
    ``RingStationDetection`` 塞进物料定位的画笔里。
    """

    mode: str
    value: object


@dataclass(frozen=True)
class Failure:
    """一次推理失败：把异常留在结果里显示，而不是让整条预览链路崩掉。"""

    mode: str
    message: str


class MeasureSession:
    """两点测量的三次点击状态机：0 点（实时）→ 1 点（冻结）→ 2 点（冻结）→ 清空。

    语义与 ``modules.vision.test.manual.pixel_measure`` 完全相同（第 3 次点击清空），
    单独一个类是为了让"这一次点击到底该干什么"只有一处实现。

    冻结画面不是偷懒：不冻结的话，两次点击之间画面一直在动，量的就不是同一个场景里的
    两个点，读数没有意义。
    """

    def __init__(self) -> None:
        self.points: list[tuple[float, float]] = []
        self.frozen: Optional[np.ndarray] = None
        #: 两点的像素测量结果（``PixelMeasurement``），点满两个才有。
        self.measured = None
        #: 两点的毫米测量结果（``MillimetreMeasurement``）；缺焦距/工作距离时为 ``None``。
        self.millimetres = None
        #: 毫米不可用的原因（原样来自 ``Vision.pixel_distance_mm`` 的报错）。
        self.note = ""

    @property
    def complete(self) -> bool:
        return len(self.points) == 2

    def add_point(self, x: float, y: float, frame: Optional[np.ndarray]) -> None:
        """放一个点；点满两个之后再点一下就是清空。"""
        if len(self.points) >= 2:
            self.clear()
            return
        self.points.append((float(x), float(y)))
        if len(self.points) == 1 and frame is not None:
            self.frozen = frame.copy()

    def clear(self) -> None:
        self.points = []
        self.frozen = None
        self.measured = None
        self.millimetres = None
        self.note = ""


# ------------------------------------------------------------------ 画图小工具
def _put_lines(
    canvas: np.ndarray,
    lines: Sequence[str],
    *,
    origin: tuple[int, int] = (10, 24),
    step: int = 26,
    scale: float = 0.6,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """画几行带黑色描边的 ASCII 文字（描边是为了压在浅色台面上也看得清）。"""
    import cv2

    for index, text in enumerate(lines):
        point = (origin[0], origin[1] + index * step)
        cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thickness + 2)
        cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _draw_boxes(canvas: np.ndarray, report) -> np.ndarray:
    """把 ``MaterialReport`` 里的 YOLO 框画上去（圆面由 ``MaterialLocator.draw`` 负责）。

    框和圆放在同一张图上才看得出"模型框得住顶面"还是"框把台面一起吞了"——
    后者是 ``report.from_model_box`` 那条主算路径最容易骗人的地方。
    """
    import cv2

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
    return canvas


def save_frame(frame: np.ndarray, directory: Path, *, mode: str) -> Path:
    """把一帧（含标注）存成 PNG，返回写出的路径。"""
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"view_{mode}_{stamp}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"写不出图片：{path}")
    return path


# ------------------------------------------------------------------ 查看器
class Viewer:
    """一个会话 + 六种画笔：模式切换只换"算哪一步、用哪支笔"，设备不重开。

    刻意不复用 ``run_preview`` 的"推理函数返回 ``MaterialReport``"契约（那是它给物料
    定位定的），而是让 ``pipeline`` 返回 :class:`ModeResult`——模式跟着结果走，
    切模式时不会出现"用新画笔去画旧模式的数据"。
    """

    def __init__(
        self,
        vision: Vision,
        settings,
        *,
        focal_px: Optional[float] = None,
        depth_mm: Optional[float] = None,
        frames: int = DEFAULT_MEASURE_FRAMES,
    ) -> None:
        self.vision = vision
        self.settings = settings
        self.focal_px = focal_px
        self.depth_mm = depth_mm
        #: 融合窗口：最近几次定位。少于 3 个的窗口在 ``fuse`` 里没有意义，所以抬到 3。
        self.frames = max(DEFAULT_MIN_SAMPLES, int(frames))
        self.mode = MODE_MATERIAL
        self.measure = MeasureSession()
        #: 最近几次定位的观测（``None`` = 那一帧没看到），供 ``m`` 做预览采样融合。
        self.samples: "deque[Optional[TimedPoint]]" = deque(maxlen=self.frames)
        #: 上一次 ``m`` 的结论与它的口径说明。
        self.fusion = None
        self.fusion_note = ""
        self.last_frame: Optional[np.ndarray] = None
        self.last_result: Optional[ModeResult] = None
        self.last_stamp = 0.0
        self.blank_frames = 0
        self.warned_blank = False
        self.saved = 0
        #: 存图目录；``main`` 按 ``--save`` 赋值，``None`` 表示这次不许存图。
        self.directory: Optional[Path] = None
        self._samples_lock = threading.Lock()
        self._cells = None
        self._cells_error: Optional[str] = None

    # ------------------------------------------------------------- 推理
    def pipeline(self, frame):
        """``run_preview`` 要的推理函数：按**调用开始时**的模式算一帧。"""
        mode = self.mode
        try:
            value = self._compute(mode, frame)
        except Exception as error:  # 单帧算错不该让预览整个崩掉
            return ModeResult(mode, Failure(mode, f"{type(error).__name__}: {error}"))
        return ModeResult(mode, value)

    def _compute(self, mode: str, frame):
        if mode == MODE_MATERIAL:
            return self._locate(frame)
        if mode == MODE_YOLO:
            return self.vision._locator.yolo.detector.detect(frame)
        if mode == MODE_RINGS:
            return self._ring_detector().detect(frame)
        if mode == MODE_STACKED:
            return self._stacked_detector().detect(frame)
        if mode == MODE_CELLS:
            return self._cell_detector().detect(frame)
        if mode == MODE_RING:
            return self.vision.locate_ring()
        # measure 模式不推理，但要给 run_preview 一个非 None 的结果，
        # 否则它走 ``current.copy()`` 分支，draw 根本不会被调用——画面就没有测点。
        return _MEASURE_PENDING

    def _locate(self, frame):
        """物料定位：复用会话自己的定位器，顺便记一个融合样本。"""
        stamp = time.monotonic()
        report = self.vision._locator.locate(frame)
        best = report.best
        sample = None if best is None else TimedPoint(
            stamp=stamp,
            x_mm=float(best.x_mm),
            y_mm=float(best.y_mm),
            z_mm=float(best.distance_mm),
            diameter_px=float(2 * best.region.radius),
            color=best.name,
        )
        with self._samples_lock:
            self.samples.append(sample)
        self.last_stamp = stamp
        return report

    def _ring_detector(self):
        """三圆环检测器：按会话缓存（配置在 ``open`` 时定死，稳定窗口要连续）。"""
        return self.vision._ring_station_detector()

    def _stacked_detector(self):
        return self.vision._stacked_ring_detector()

    def _cell_detector(self):
        """同帧多目标拆分：需要一个``PinholeCamera``，焦距取本会话实际用的那个。

        这条路径**不在** ``Vision`` 的公开方法里（它是抓取侧 ``perception`` 在用的
        低层能力），所以这里自己装配；失败原因缓存下来，免得每帧重复构造。
        """
        if self._cells is not None:
            return self._cells
        if self._cells_error is not None:
            raise RuntimeError(self._cells_error)
        from modules.vision.drive.cell_targets import CellTargetDetector
        from modules.vision.drive.pinhole import PinholeCamera

        try:
            width, height = self.vision._frame_size()
            camera = PinholeCamera.from_shape(
                (height, width), focal_px=self.vision._effective_focal_px(self.focal_px)
            )
            self._cells = CellTargetDetector(camera=camera, targets=self.settings.targets)
        except ValueError as error:
            self._cells_error = str(error)
            raise
        return self._cells

    # ------------------------------------------------------------- 画图
    def draw(self, frame: np.ndarray, result) -> np.ndarray:
        """``run_preview`` 要的画图函数：按**结果自己的模式**画。"""
        if result is None:
            return self._waiting_card(frame)
        try:
            return self._draw(page=result.mode, frame=frame, value=result.value)
        except Exception as error:  # 画错一帧也不该让窗口消失
            return self._failure_card(frame, Failure(result.mode, f"draw: {error}"))

    def _draw(self, *, page: str, frame: np.ndarray, value) -> np.ndarray:
        if isinstance(value, Failure):
            return self._failure_card(frame, value)
        if page == MODE_MATERIAL:
            canvas = self.vision._locator.draw(frame, value)
            return _draw_boxes(canvas, value)
        if page == MODE_YOLO:
            detector = self.vision._locator.yolo.detector
            return detector.draw(frame, value)
        if page == MODE_RINGS:
            return self._ring_detector().draw(frame, value)
        if page == MODE_STACKED:
            return self._stacked_detector().draw(frame, value)
        if page == MODE_CELLS:
            return self._cell_detector().draw(frame, value)
        if page == MODE_RING:
            return self._draw_ring(frame, value)
        return self._draw_measure(frame)

    def _waiting_card(self, frame: np.ndarray) -> np.ndarray:
        canvas = frame.copy()
        _put_lines(canvas, [f"{MODE_TITLES.get(self.mode, self.mode)}: waiting for the first result"])
        return canvas

    def _failure_card(self, frame: np.ndarray, failure: Failure) -> np.ndarray:
        canvas = frame.copy()
        _put_lines(
            canvas,
            [
                f"{MODE_TITLES.get(failure.mode, failure.mode)}: FAILED",
                failure.message[:96],
            ],
            color=(0, 200, 255),
        )
        return canvas

    def _draw_ring(self, frame: np.ndarray, target) -> np.ndarray:
        """视野里那个环：画框、环心十字与相机坐标（可对准画绿，贴边/无毫米画橙）。"""
        import cv2

        canvas = frame.copy()
        if target is None:
            _put_lines(canvas, ["no ring in view"])
            return canvas
        for item in (target,):
            centre = (int(round(item.center_px[0])), int(round(item.center_px[1])))
            # 精修过就画真正那一圈（外圈），否则只能按框画——框比图案大一圈。
            radius = max(3, int(round(item.ring_radius_px or item.radius_px)))
            colour = (0, 255, 0) if item.usable else (0, 165, 255)
            cv2.circle(canvas, centre, radius, colour, 3)
            cv2.drawMarker(canvas, centre, (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
            label = f"ring {item.digit} ({item.score:.2f})"
            if item.x_mm is None or item.y_mm is None:
                label += ": no mm"
            else:
                label += f": x={item.x_mm:+.1f} y={item.y_mm:+.1f}mm"
            if item.clipped:
                label += " | clipped"
            origin = (max(4, centre[0] - radius), max(18, centre[1] - radius - 10))
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return canvas

    def _draw_measure(self, frame: np.ndarray) -> np.ndarray:
        """两点测量：画十字丝、连线与结论。

        冻结时就画冻结的那一帧——和显示的那一帧是同源，量出来的距离才对得上。
        """
        import cv2

        source = frame if self.measure.frozen is None else self.measure.frozen
        canvas = source.copy()
        for index, (x, y) in enumerate(self.measure.points, start=1):
            point = (int(round(x)), int(round(y)))
            cv2.drawMarker(canvas, point, (0, 255, 255), cv2.MARKER_CROSS, MARKER_SIZE,
                           MARKER_THICKNESS)
            cv2.putText(canvas, f"P{index}", (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), MARKER_THICKNESS)

        measured = self.measure.measured
        if measured is None:
            hint = "click 2 points" if not self.measure.points else "click the 2nd point"
            _put_lines(canvas, [hint])
            return canvas

        first = (int(round(measured.start[0])), int(round(measured.start[1])))
        second = (int(round(measured.end[0])), int(round(measured.end[1])))
        cv2.line(canvas, first, second, (255, 128, 0), MARKER_THICKNESS)
        lines = [
            f"dX={measured.dx_px:+.1f} dY={measured.dy_px:+.1f} px",
            f"dist={measured.length_px:.1f} px",
        ]
        millimetres = self.measure.millimetres
        if millimetres is not None:
            lines.append(f"dX={millimetres.dx_mm:+.2f} dY={millimetres.dy_mm:+.2f} mm")
            lines.append(f"dist={millimetres.length_mm:.2f} mm")
        else:
            lines.append("mm: unavailable (focal / working distance missing)")
        _put_lines(canvas, lines)
        return canvas

    # ------------------------------------------------------------- 叠加层
    def overlay(self, canvas: np.ndarray, result, stats: PreviewStats) -> None:
        """左上角：模式、这一帧的结论、按键；左下角由 ``run_preview`` 自己画速率。"""
        _put_lines(canvas, self.head_lines(result), step=24, scale=0.55)

    def head_lines(self, result) -> Sequence[str]:
        index = MODES.index(self.mode) + 1 if self.mode in MODES else 0
        lines = [
            f"[{index}/{len(MODES)}] {MODE_TITLES.get(self.mode, self.mode)}  "
            f"{MODE_HINTS.get(self.mode, '')}"
        ]
        if isinstance(result, ModeResult) and result.mode != self.mode:
            # 切模式的瞬间，工作线程可能还在算上一模式的帧：说清楚，别让人以为切错了。
            lines.append(f"  (last result: {MODE_TITLES.get(result.mode, result.mode)}, switching)")
        lines.extend(self._state_lines(result))
        lines.append("1-7 mode | m fuse | t sighting | p print | r reset | s save | q quit")
        return lines

    def _state_lines(self, result) -> "list[str]":
        if not isinstance(result, ModeResult):
            return ["  no result yet"]
        value = result.value
        if isinstance(value, Failure):
            return [f"  FAILED: {value.message[:80]}"]
        if result.mode == MODE_MATERIAL:
            return self._material_lines(value)
        if result.mode == MODE_YOLO:
            return self._yolo_lines(value)
        if result.mode == MODE_RINGS:
            return self._rings_lines(value)
        if result.mode == MODE_STACKED:
            return self._stacked_lines(value)
        if result.mode == MODE_CELLS:
            return self._cells_lines(value)
        if result.mode == MODE_RING:
            return self._ring_lines(value)
        return self._measure_lines()

    def _fusion_lines(self) -> "list[str]":
        fusion = self.fusion
        if fusion is None:
            return []
        state = "moving" if fusion.moving else "settled"
        return [
            f"  fusion(sampled) {state} xyz=("
            f"{fusion.point.x_mm:+.1f},{fusion.point.y_mm:+.1f},{fusion.point.z_mm:.1f}) "
            f"inliers={fusion.samples}/{fusion.frames} spread={fusion.spread_mm:.2f}mm "
            f"v={fusion.speed_mm_s:.1f}mm/s confident={fusion.confident}",
        ]

    def _material_lines(self, report) -> "list[str]":
        best = report.best
        source = report.colour_source or "none"
        window = (
            "none" if report.radius_window_px is None
            else f"{report.radius_window_px[0]:.0f}-{report.radius_window_px[1]:.0f}px"
        )
        lines = [
            f"  source={source} faces={len(report.faces)}/{len(report.regions)} "
            f"yolo={len(report.hits)} window={window} want={report.want}",
        ]
        if best is None:
            lines.append("  best=none (no accepted top face)")
        else:
            confidence = "--" if best.confidence is None else f"{best.confidence:.2f}"
            lines.append(
                f"  best={best.name or 'unknown'} xyz=("
                f"{best.x_mm:+.1f},{best.y_mm:+.1f},{best.z_mm:.1f})mm "
                f"r={best.region.radius:.0f}px conf={confidence}"
            )
        return lines + self._fusion_lines()

    def _yolo_lines(self, detections) -> "list[str]":
        lines = [f"  boxes={len(detections)}"]
        for hit in list(detections)[:3]:
            lines.append(f"  {hit.name} {hit.score:.2f} ({hit.width:.0f}x{hit.height:.0f})")
        return lines

    def _rings_lines(self, detection) -> "list[str]":
        if not detection.detected:
            return [f"  detected=false candidates={detection.candidates}",
                    f"  detail={detection.detail[:70]}"]
        lines = [
            f"  detected=true conf={detection.confidence:.2f} "
            f"stable={detection.stable} within={detection.within_tolerance} "
            f"cand={detection.candidates} {detection.processing_ms:.0f}ms",
            f"  yaw={_value(detection.station_yaw_deg, 'deg')} "
            f"angle_err={_value(detection.angle_error_deg, 'deg')}",
            f"  err=({_value(detection.error_x_mm, 'mm')}, {_value(detection.error_y_mm, 'mm')}) "
            f"|err|={_value(detection.position_error_mm, 'mm')} "
            f"jitter={_value(detection.center_jitter_px, 'px')}",
        ]
        return lines

    def _stacked_lines(self, detection) -> "list[str]":
        if not detection.detected:
            return [f"  detected=false detail={detection.detail[:70]}"]
        lines = [
            f"  detected=true conf={detection.confidence:.2f} lines={detection.line_count} "
            f"stable={detection.stable} {detection.processing_ms:.0f}ms",
            f"  yaw={_value(detection.station_yaw_deg, 'deg')} "
            f"disagree={_value(detection.line_disagreement_deg, 'deg')} "
            f"line_q={detection.line_quality:.2f}",
            f"  err=({_value(detection.error_x_mm, 'mm')}, {_value(detection.error_y_mm, 'mm')}) "
            f"|err|={_value(detection.position_error_mm, 'mm')} "
            f"scale={_value(detection.scale_mm_per_px, 'mm/px')} ({detection.scale_source})",
        ]
        return lines

    def _cells_lines(self, targets) -> "list[str]":
        lines = [
            f"  markers={len(targets.markers)} materials={len(targets.materials)} "
            f"unmatched={len(targets.unmatched)}",
        ]
        for target in list(targets.markers)[:1] + list(targets.materials)[:1]:
            lines.append(
                f"  {target.role} center=({target.region.center_x:.0f},{target.region.center_y:.0f}) "
                f"r={target.region.radius:.0f}px z={target.fix.depth_mm:.0f}mm"
            )
        return lines

    def _ring_lines(self, target) -> "list[str]":
        if target is None:
            return ["  no ring in view"]
        where = (
            "--" if target.x_mm is None
            else f"({target.x_mm:+.1f},{target.y_mm:+.1f})mm"
        )
        if target.usable:
            state = "ok"
        elif target.clipped:
            state = "clipped"
        else:
            state = "no-mm"
        scale = (
            f"{target.scale_mm_per_px:.4f} mm/px {target.scale_source}"
            if target.scale_mm_per_px is not None
            else "no scale"
        )
        return [
            f"  ring {target.digit} centre {where} r={target.radius_px:.0f}px {state}",
            f"  score={target.score:.2f} {scale}",
        ]

    def _measure_lines(self) -> "list[str]":
        lines = [f"  points={len(self.measure.points)} frozen={self.measure.frozen is not None}"]
        measured = self.measure.measured
        if measured is not None:
            lines.append(f"  dX={measured.dx_px:+.1f} dY={measured.dy_px:+.1f} "
                         f"dist={measured.length_px:.1f} px")
            if self.measure.millimetres is not None:
                millimetres = self.measure.millimetres
                lines.append(f"  dX={millimetres.dx_mm:+.2f} dY={millimetres.dy_mm:+.2f} "
                             f"dist={millimetres.length_mm:.2f} mm")
            elif self.measure.note:
                lines.append(f"  mm: {self.measure.note[:70]}")
        return lines

    # ------------------------------------------------------------- 回调
    def on_frame(self, frame: np.ndarray, result, _stats: PreviewStats) -> None:
        """每显示一帧：留一份原始帧（存图/测点要用），并检查纯色画面。"""
        self.last_frame = frame
        self.last_result = result if isinstance(result, ModeResult) else self.last_result
        self.blank_frames = self.blank_frames + 1 if _looks_blank(frame) else 0
        if self.blank_frames >= BLANK_FRAME_WARNING and not self.warned_blank:
            self.warned_blank = True
            warn_blank_preview(self.settings.camera_id(_platform()))

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        """鼠标回调：只在测量模式下收左键。"""
        import cv2

        if event != cv2.EVENT_LBUTTONDOWN or self.mode != MODE_MEASURE:
            return
        frame = self.measure.frozen if self.measure.frozen is not None else self.last_frame
        if frame is None:
            return
        self.measure.add_point(x, y, frame)
        if not self.measure.complete:
            if not self.measure.points:
                print("测量：已清空，回到实时画面")
            else:
                print(f"测量：起点 ({x}, {y})，画面已冻结，再点终点")
            return
        self._resolve_measure()

    def _resolve_measure(self) -> None:
        """点满两个点：算像素，能算毫米就一起算出来（缺哪项就说缺哪项）。"""
        start, end = self.measure.points
        session = self.measure
        session.measured = self.vision.pixel_distance(start, end)
        try:
            session.millimetres = self.vision.pixel_distance_mm(
                start, end, working_distance_mm=self.depth_mm, focal_px=self.focal_px
            )
            session.note = ""
        except ValueError as problem:
            # 没标定就是没标定，不拿粗焦距凑一个"看起来正常"的毫米数。
            session.millimetres = None
            session.note = str(problem)
        print(session.measured.summary())
        if session.millimetres is not None:
            print(session.millimetres.summary())
        else:
            print(f"  毫米不可用：{session.note}")

    def on_key(self, key: int, result, _stats: PreviewStats) -> None:
        """按键分派。``q`` 与 ``Esc`` 由调用方/``wait_key`` 处理，这里不管。"""
        if key in MODE_KEYS:
            self.switch_mode(MODE_KEYS[key])
            return
        if key in KEY_FUSE:
            self.fuse_samples()
        elif key in KEY_SIGHTING:
            self.print_sightings(result)
        elif key in KEY_PRINT:
            print(self.describe(result))
        elif key in KEY_RESET:
            self.reset_stability()
        elif key in KEY_SAVE:
            self.save_current(result)
        elif key in KEY_HELP:
            print_help()

    # ------------------------------------------------------------- 动作
    def switch_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        # 换模式就丢掉上一次的融合结论：那是**另一类读数**的结论，留在画面上只会误导。
        self.fusion = None
        self.fusion_note = ""
        if mode != MODE_MEASURE:
            self.measure.clear()
        print(f"模式 → {MODE_TITLES[mode]}（{MODE_HINTS[mode]}）")

    def fuse_samples(self) -> None:
        """``m``：把最近 N 次定位融合一次（预览采样口径，不是连拍）。"""
        with self._samples_lock:
            samples = list(self.samples)
        if len(samples) < DEFAULT_MIN_SAMPLES:
            print(
                f"融合还差数据：只有 {len(samples)} 次定位，至少要 {DEFAULT_MIN_SAMPLES} 次。"
                "先在物料定位模式多看几帧。"
            )
            return
        fusion = fuse(samples, still_mm=DEFAULT_STILL_MM,
                      min_samples=DEFAULT_MIN_SAMPLES)
        if fusion is None:
            self.fusion = None
            print(
                f"这 {len(samples)} 次定位里有效读数不够（少于 {DEFAULT_MIN_SAMPLES} 次看到物料），"
                "没有结论。"
            )
            return
        self.fusion = fusion
        self.fusion_note = "sampled"
        print(_describe_fusion(fusion))
        print(
            "  口径：这是**预览采样**（相邻两次定位的间隔由推理速度决定），"
            "不是 Vision.measure() 的连拍；"
            "\n  要严格的连拍结论用 python test_vision.py --measure [--settle]。"
        )

    def print_sightings(self, result) -> None:
        """``t``：把这一帧翻成闭环观测（``frame()`` 返回的那种元组）打出来。"""
        if not isinstance(result, ModeResult) or result.mode != MODE_MATERIAL:
            print("观测只在物料定位模式下可取：先按 1。")
            return
        if isinstance(result.value, Failure):
            print(f"这一帧没有结果：{result.value.message}")
            return
        # 用 runtime 里那份换算，而不是在这里再写一遍：像素半径 ×2、边界支撑 ×100
        # 这两个换算一旦分叉，界面显示的观测就和抓取真的收到的不一样了。
        sightings = _sightings_from_report(result.value, received_at=self.last_stamp)
        if not sightings:
            print("这一帧没有可交给闭环的观测（通过校验的顶面为空）。")
            return
        print(f"闭环观测 {len(sightings)} 条：")
        for sighting in sightings:
            print("  " + sighting.summary())

    def reset_stability(self) -> None:
        """``r``：清掉工位/叠放位的跨帧稳定窗口（重新摆工位之后按）。"""
        self._ring_detector().reset()
        self._stacked_detector().reset()
        print("稳定窗口已清空（三圆环与叠放位都会重新累计）")

    def save_current(self, result) -> None:
        """``s``：把**当前这一帧 + 当前结果**画一次再存，存的就是窗口里看到的。"""
        if self.directory is None:
            print("按 s 存图要先在启动时给 --save（例如 python view_vision.py --save）")
            return
        frame = self.last_frame
        if frame is None or result is None:
            print("还没有画面或结果，没存图")
            return
        annotated = self.draw(frame, result)
        path = save_frame(annotated, self.directory, mode=self.mode)
        self.saved += 1
        print(f"已存图：{path}")

    # ------------------------------------------------------------- 终端输出
    def describe(self, result) -> str:
        """把当前结果写成终端可读的多行文本（``p`` 用）。"""
        if not isinstance(result, ModeResult):
            return "还没有结果（工作线程还在算第一帧）"
        value = result.value
        mode = result.mode
        if isinstance(value, Failure):
            return f"[{MODE_TITLES.get(mode, mode)}] 失败：{value.message}"
        if mode == MODE_MATERIAL:
            return value.summary()
        if mode == MODE_YOLO:
            lines = [f"YOLO 检出 {len(value)} 个框"]
            for hit in value:
                lines.append(
                    f"  {hit.name} {hit.score:.2f} "
                    f"({hit.x1:.0f},{hit.y1:.0f})-({hit.x2:.0f},{hit.y2:.0f}) "
                    f"{hit.width:.0f}x{hit.height:.0f}"
                )
            return "\n".join(lines)
        if mode == MODE_RING:
            if value is None:
                return "画面里没有环（把环放到视野中间一点，或者检查光照）"
            return value.summary() + f"\n  对准读数（相机系）: {value.reading()}\n  {value.detail}"
        if mode in (MODE_RINGS, MODE_STACKED, MODE_CELLS):
            return value.summary()
        if self.measure.measured is None:
            return f"测量：已经放了 {len(self.measure.points)} 个点（还要放 {2 - len(self.measure.points)} 个）"
        lines = [self.measure.measured.summary()]
        if self.measure.millimetres is not None:
            lines.append(self.measure.millimetres.summary())
        else:
            lines.append(f"毫米不可用：{self.measure.note}")
        return "\n".join(lines)


# ------------------------------------------------------------------ 小工具
def _value(number, unit: str) -> str:
    """可选数值的统一写法：没有就写 ``--``，不写 0（0 是一个有效读数）。"""
    return "--" if number is None else f"{number:.2f}{unit}"


def _platform() -> str:
    import os

    return "windows" if os.name == "nt" else "jetson"


def _looks_blank(frame) -> bool:
    from modules.vision.drive.camera import looks_blank

    return looks_blank(frame)


def _escape_as_quit(delay: int = 1) -> int:
    """把 ``Esc`` 翻成 ``q``：``run_preview`` 只认 ``q``，而现场习惯按 Esc。"""
    import cv2

    key = int(cv2.waitKey(delay) & 0xFF)
    return ord("q") if key == 27 else key


def _describe_fusion(fusion) -> str:
    """一次融合结果的终端摘要（比 ``test_vision.describe_fusion`` 短一档）。"""
    lines = [fusion.summary()]
    lines.append(
        f"  值：相机系 xyz=({fusion.point.x_mm:+.2f}, {fusion.point.y_mm:+.2f}, "
        f"{fusion.point.z_mm:.2f}) mm（参考时刻 t={fusion.point.stamp:.3f}）"
    )
    if fusion.moving:
        lines.append(
            f"  还在动：速度 {fusion.speed_mm_s:.2f} mm/s —— 上面那个值是外推到参考时刻的"
        )
    if len(fusion.colors) > 1:
        lines.append(f"  注意：这一批里出现了 {'、'.join(fusion.colors)}，可能换了目标")
    if not fusion.confident:
        lines.append("  结论：不算一次可信测量（内点不够 / 还在动 / 野值过半）")
    return "\n".join(lines)


def _first_frame(vision: Vision, *, tries: int = 5, delay: float = 0.05):
    """取一帧可用画面（摄像头刚开的那几帧常常是黑的，所以要连读几次）。"""
    camera = vision._camera
    frame = None
    for _ in range(max(1, tries)):
        candidate = camera.read()
        if candidate is not None:
            frame = candidate
        time.sleep(delay)
    return frame


# ------------------------------------------------------------------ 一次跑完
def run_once(viewer: Viewer, modes: Sequence[str], *, save_dir: Optional[Path]) -> int:
    """不开窗口：对**同一帧**逐个模式跑一遍，把结论打到终端（可顺手存图）。"""
    frame = _first_frame(viewer.vision)
    if frame is None:
        print("读不到摄像头画面：确认序号、连接，以及没有被别的程序占用")
        return 1
    print(f"一帧 {frame.shape[1]}x{frame.shape[0]}，逐个模式跑一遍：")
    for mode in modes:
        viewer.mode = mode
        result = viewer.pipeline(frame)
        print(f"\n[{MODE_TITLES[mode]}] {MODE_HINTS[mode]}")
        print(viewer.describe(result))
        if save_dir is not None:
            annotated = viewer.draw(frame, result)
            print(f"  已存图：{save_frame(annotated, save_dir, mode=mode)}")
    return 0


def _measure_once(
    vision: Vision, *, frames: int, settled: bool, timeout: float
) -> int:
    """进预览前做一次**真实连拍**融合（``Vision.measure`` / ``measure_settled``）。"""
    fusion = (
        vision.measure_settled(timeout=timeout, frames=frames)
        if settled
        else vision.measure(frames=frames)
    )
    if fusion is None:
        print(
            f"连拍 {frames} 帧里有效读数不够（少于 3 帧看到物料）："
            "运动模糊、爪子挡住、画面里没料都会这样。"
        )
        return 1
    print(_describe_fusion(fusion))
    return 0 if fusion.confident else 1


def print_help() -> None:
    """把按键表打到终端（``h``）。"""
    print("\n按键：")
    for index, mode in enumerate(MODES, start=1):
        print(f"  {index}. {MODE_TITLES[mode]:8s} {MODE_HINTS[mode]}")
    print("  m  融合最近几次定位（预览采样口径）")
    print("  t  把这一帧翻成闭环观测（MaterialSighting）")
    print("  p  打印当前模式的完整结果")
    print("  r  清空工位/叠放位的稳定窗口")
    print("  s  存一张标注图（需要 --save）")
    print("  q / Esc  退出")


# ------------------------------------------------------------------ 入口
def build_parser() -> argparse.ArgumentParser:
    """只解析命令行：不读配置、不开设备、不建窗口。"""
    parser = argparse.ArgumentParser(
        description="上位机全功能查看器：物料定位 / YOLO / 三圆环 / 叠放位 / 多目标 / 两点测量"
    )
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头序号；省略则读 config.json 对应平台的 camera_id")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="启动时先看哪个模式（默认 material；配合 --once 时只跑这一个）")
    parser.add_argument("--focal", type=float, default=None,
                        help="覆盖像素焦距（影响毫米读数与半径窗口，不写回文件）")
    parser.add_argument("--depth", type=float, default=None,
                        help="两点测量用的工作距离（毫米）；省略则用 calibration.json 的 "
                             "table_depth_mm，两者都没有时不显示毫米")
    parser.add_argument("--frames", type=int, default=DEFAULT_MEASURE_FRAMES,
                        help=f"m 融合用最近几次定位（默认 {DEFAULT_MEASURE_FRAMES}）")
    parser.add_argument("--measure", action="store_true",
                        help="进预览前先做一次真实连拍融合（会阻塞约 1 秒）")
    parser.add_argument("--settle", action="store_true",
                        help="配合 --measure：反复连拍直到停稳（超时返回最后一批）")
    parser.add_argument("--settle-timeout", type=float, default=DEFAULT_SETTLE_TIMEOUT_SECONDS,
                        help=f"--settle 的耐心上限，秒（默认 {DEFAULT_SETTLE_TIMEOUT_SECONDS:.0f}）")
    parser.add_argument("--once", action="store_true",
                        help="不开窗口：对一帧逐个模式跑一遍并打印")
    parser.add_argument("--save", action="store_true",
                        help="按 s 存图；配合 --once 立刻存")
    parser.add_argument("--out", default=None, help=f"存图目录；默认 {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--sync", dest="async_detect", action="store_false",
                        help="串行推理：结果与画面严格同帧，但显示会跟着推理一起变慢")
    parser.set_defaults(async_detect=True)
    return parser


def _banner(settings, initial: str, *, frames: int) -> None:
    """启动时把"这次看的是什么配置"写清楚（现场最容易错的就是以为改了配置）。"""
    camera_id = settings.camera_id(_platform())
    print(f"摄像头 {camera_id}；窗口 {WINDOW!r}，按 h 看按键，按 q / Esc 退出")
    print(f"模式：{', '.join(f'{index}={mode}' for index, mode in enumerate(MODES, start=1))}"
          f"（当前 {initial}）")
    print(f"m 融合窗口 {frames} 次定位（预览采样口径；严格连拍用 --measure）")
    if settings.table_depth_mm is None and settings.focal_px is None:
        print("calibration.json 里没有 focal_px / table_depth_mm：两点测量的毫米读数不可用，"
              "只能看像素")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        settings = _override_settings(_load_settings(), camera_id=arguments.camera)
    except ValueError as problem:
        print(f"配置或参数不合法：{problem}")
        return 1

    save_dir = None
    if arguments.save:
        save_dir = DEFAULT_OUTPUT_DIR if arguments.out is None else Path(arguments.out)
    modes = MODES if arguments.mode is None else (arguments.mode,)
    initial = MODE_MATERIAL if arguments.mode is None else arguments.mode
    frames = max(DEFAULT_MIN_SAMPLES, int(arguments.frames))

    try:
        with Vision._open_config(settings, focal_px=arguments.focal) as vision:
            viewer = Viewer(
                vision, settings, focal_px=arguments.focal,
                depth_mm=arguments.depth, frames=frames,
            )
            # ``save_current`` 读的是这个目录：给了 --save 才有。
            viewer.directory = save_dir
            _banner(settings, initial, frames=frames)

            if arguments.measure:
                _measure_once(
                    vision, frames=frames, settled=arguments.settle,
                    timeout=arguments.settle_timeout,
                )

            if arguments.once:
                printable = tuple(mode for mode in modes if mode != MODE_MEASURE)
                if not printable:
                    print("--once 不跑测量模式（它要鼠标点两个点）；去掉 --mode measure 再试")
                    return 1
                return run_once(viewer, printable, save_dir=save_dir)

            viewer.mode = initial
            window = open_preview(WINDOW)
            import cv2

            # 鼠标回调必须在窗口建好之后、进循环之前挂上（与 manual.pixel_measure 一致）。
            cv2.setMouseCallback(window, viewer.on_mouse)
            stats = run_preview(
                vision._camera,
                viewer.pipeline,
                viewer.draw,
                window=window,
                async_detect=arguments.async_detect,
                print_every=PRINT_EVERY,
                on_key=viewer.on_key,
                on_frame=viewer.on_frame,
                overlay=viewer.overlay,
                wait_key=_escape_as_quit,
            )
            print("  " + stats.line())
            return 0 if stats.reason in ("quit", "") else 1
    finally:
        close_previews()


if __name__ == "__main__":
    raise SystemExit(main())

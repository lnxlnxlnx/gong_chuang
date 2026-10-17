"""预览两条线程：一条只管取帧，一条只管推理，显示永远只画最新的那一帧。

**为什么需要它**（本机实测，1280×720）：原来的预览是一个串行循环——``read()``
阻塞等一帧 → 推理阻塞算完 → ``imshow`` → ``waitKey(1)``。显示帧率因此等于
``1 /（取帧 + 推理 + 画图 + 显示）``，任何一段慢，整条链路一起慢；而且**取帧从
来不停**，驱动缓冲区里越积越多：画面看着卡，延迟还越来越大（手在镜头前动一下，
窗口里要过一秒才动）。

实测各段耗时：摄像头 27–30 fps（不是瓶颈）、YOLO 推理约 20 ms、
``MaterialLocator.locate`` 在真实台面图上 0.5–3.3 s。也就是说
**显示帧率被最慢的那一段（经典 CV，不是 YOLO）拖到了 0.3–2 fps。**

这里把三件事拆开：

- :class:`FrameGrabber`：后台线程不停 ``read()``，**只留最新的一帧**，旧帧直接
  丢掉。缓冲区不会积压，所以显示的是"现在"，不是"刚才"。
- :class:`DetectionWorker`：后台线程对最新的一帧跑推理。忙的时候新帧**顶掉**
  待处理的那一帧（不是排队），推理因此永远算最新的画面，不会一路追赶旧帧。
- :func:`run_preview`：显示循环只做三件事——拿最新帧、用最近一次结果标注、
  ``imshow`` + ``waitKey(1)``。窗口跑在**摄像头帧率**上，推理按自己的速度跑；
  两者各自的实际速率都画在窗口左下角（``cap`` / ``det`` / ``age``），
  不会让人误以为"画面卡"就是"识别慢"。

**标注的是最新帧，结果可能略旧**：画面上写着结果年龄 ``age``（毫秒）。台面上的
物料几乎不动，这点差看不出来；但它是有意画出来的，不藏在代码里。需要"结果必须
和画面严格同帧"的地方（抓取控制链路）不要用本模块，直接同步调 ``locate()``。

用法::

    from modules.vision.drive.camera import USBCamera
    from modules.vision.drive.preview import open_preview

    with USBCamera(0).initialize() as camera:
        run_preview(camera, locator.locate, locator.draw, window=open_preview())

命令行自测（不需要摄像头和模型：假摄像头按 60 fps 交帧、假推理固定 50 ms，
验证显示仍跑在取帧速率上）::

    python -m modules.vision.drive.stream
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

#: 取帧线程连续读失败时的等待时间（秒）。摄像头掉线时靠它避免空转烧 CPU。
DEFAULT_IDLE_SLEEP = 0.002

#: 显示循环没有新帧可画时，最多等这么久再去处理按键（秒）。
#:
#: 有帧时 :meth:`FrameGrabber.wait_newer` 立刻返回，这个值只在"摄像头没动静"时
#: 生效：既不能空转烧 CPU（``waitKey(1)`` 在假实现/某些后端上并不阻塞），
#: 也不能让按键迟钝（20 ms 人感觉不到）。
DEFAULT_KEY_POLL = 0.02

#: 连续这么多次取不到画面就认为摄像头掉线（等第一帧、运行中判掉线都用它）。
#:
#: 摄像头每次打开都会先丢几帧，所以不能读一次失败就退出；但真掉线时也不该等到
#: 用户按 q：30 次失败在 30 fps 下不到一秒，掉线时则立刻返回。
DEFAULT_MAX_MISSES = 30

#: 等第一帧/第一次推理的最长时间（秒）。
DEFAULT_FIRST_FRAME_TIMEOUT = 2.0

#: 速率计的指数滑动平均系数：越小平滑越强。与仓库里原有的 FPS 显示一致（0.1）。
DEFAULT_RATE_ALPHA = 0.1


def _cv2():
    """Import and return ``cv2`` lazily (keeps module import free of OpenCV)."""
    import cv2
    return cv2


def _imshow(window: str, frame) -> None:
    """默认的显示回调；单独抽出来是为了自测能替换掉它，不必开窗口。"""
    _cv2().imshow(window, frame)


def _wait_key(delay: int = 1) -> int:
    """默认的按键回调（``cv2.waitKey``）；同样是为了自测可替换。"""
    return int(_cv2().waitKey(delay))


class RateMeter:
    """指数滑动平均的速率计（次/秒）。

    用 EMA 而不是"最近 N 帧求平均"：预览要的是**当前**速率，滑动窗口在停顿之后
    会报一个虚高的值，而 EMA 会立刻掉下来。仓库里原有的 FPS 显示也是这个算法
    （``fps * 0.9 + 当前帧率 * 0.1``），保持一致，两处读数不会互相打架。
    """

    def __init__(self, alpha: float = DEFAULT_RATE_ALPHA) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be within (0, 1]")
        self.alpha = float(alpha)
        self._rate = 0.0
        self._previous: Optional[float] = None
        self.count = 0

    def tick(self, now: Optional[float] = None) -> float:
        """记一次事件，返回更新后的速率（次/秒）。"""
        now = time.perf_counter() if now is None else float(now)
        self.count += 1
        if self._previous is not None:
            interval = now - self._previous
            if interval > 0:
                instant = 1.0 / interval
                self._rate = (
                    instant if self._rate <= 0.0
                    else self._rate * (1.0 - self.alpha) + instant * self.alpha
                )
        self._previous = now
        return self._rate

    @property
    def rate(self) -> float:
        """当前速率（次/秒）；还没测出间隔时为 0。"""
        return self._rate

    def reset(self) -> None:
        """丢掉历史重新开始测（换了数据源时用）。"""
        self._rate = 0.0
        self._previous = None
        self.count = 0


@dataclass(frozen=True)
class GrabbedFrame:
    """取帧线程交出来的一帧：图像 + 序号 + 取到它的时刻。

    ``sequence`` 从 1 开始单调递增。显示循环靠它判断"有没有新帧"，
    不必比较像素；推理线程靠它知道结果是从哪一帧算出来的。
    """

    frame: object
    sequence: int
    stamp: float

    def age(self, now: Optional[float] = None) -> float:
        """这一帧从取到到现在过了多少秒。"""
        now = time.perf_counter() if now is None else float(now)
        return max(0.0, now - self.stamp)


class FrameGrabber:
    """后台线程不停取帧，只保留最新一帧（旧帧丢掉，不排队）。

    ``camera`` 是任何有 ``read()`` 的对象（:class:`modules.vision.USBCamera`
    或测试里的假摄像头都行），本类不关心它是 V4L2 还是 MSMF。
    """

    def __init__(self, camera, *, name: str = "vision-capture",
                 idle_sleep: float = DEFAULT_IDLE_SLEEP) -> None:
        if not hasattr(camera, "read"):
            raise ValueError("camera must provide read()")
        if idle_sleep <= 0:
            raise ValueError("idle_sleep must be positive")
        self.camera = camera
        self.name = name
        self.idle_sleep = float(idle_sleep)
        self.misses = 0
        self.failures = 0
        self.last_error: Optional[BaseException] = None
        self.rate = RateMeter()
        self._latest: Optional[GrabbedFrame] = None
        self._sequence = 0
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- 生命周期
    def start(self) -> "FrameGrabber":
        """Start the capture thread (safe to call twice)."""
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        """Ask the thread to finish and wait for it (safe to call twice).

        ``read()`` 本身可能正卡在驱动里，所以 ``join`` 带超时；线程是 daemon，
        最坏情况下也不会拖住进程退出。
        """
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def _loop(self) -> None:
        """取帧线程主体：唯一的出口是 ``_stop``。"""
        while not self._stop.is_set():
            try:
                frame = self.camera.read()
            except Exception as error:  # 摄像头拔掉/驱动报错都不该带走整个进程
                self.failures += 1
                self.last_error = error
                frame = None
            if frame is None:
                self.misses += 1
                if self._stop.wait(self.idle_sleep):
                    return
                continue
            self.misses = 0
            with self._condition:
                self._sequence += 1
                self._latest = GrabbedFrame(frame, self._sequence, time.perf_counter())
                self._condition.notify_all()
            self.rate.tick()

    # ------------------------------------------------------------- 取用
    def latest(self) -> Optional[GrabbedFrame]:
        """最新一帧；还没取到时返回 ``None``。"""
        with self._condition:
            return self._latest

    def wait_newer(self, sequence: int, timeout: float) -> Optional[GrabbedFrame]:
        """等一帧比 ``sequence`` 新的画面，最多等 ``timeout`` 秒。

        比"睡一小会儿再看一眼"好在**有帧时立刻返回**：60 fps 的摄像头下显示循环
        就是 60 Hz，中间不掺任何固定延时；没帧时才退化成定时轮询。
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                if self._latest is not None and self._latest.sequence > sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def wait_first(self, timeout: float = DEFAULT_FIRST_FRAME_TIMEOUT,
                   max_misses: int = DEFAULT_MAX_MISSES) -> Optional[GrabbedFrame]:
        """等第一帧；一直取不到就返回 ``None``。

        两个退出条件：超过 ``timeout``，或者连续失败 ``max_misses`` 次——后者让
        "摄像头根本没插"这种情况立刻返回，而不是干等两秒。
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            item = self.latest()
            if item is not None:
                return item
            if self.misses >= max_misses:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.wait_newer(0, min(remaining, 0.01))

    def __enter__(self) -> "FrameGrabber":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


@dataclass(frozen=True)
class DetectionResult:
    """一次推理的结果，连同它是从哪一帧、什么时候算出来的。

    ``age()`` 就是预览要画的那个数：画面上的框可能比当前帧旧这么多毫秒。
    """

    value: object
    frame_sequence: int
    started_at: float
    finished_at: float
    seconds: float

    def age(self, now: Optional[float] = None) -> float:
        """结果算完到现在过了多少秒。"""
        now = time.perf_counter() if now is None else float(now)
        return max(0.0, now - self.finished_at)

    @property
    def milliseconds(self) -> float:
        """这次推理花了多少毫秒。"""
        return self.seconds * 1000.0


class DetectionWorker:
    """后台线程跑推理；忙的时候新帧顶掉待处理的那一帧。

    **顶掉**而不是排队是有意的：排队会让推理一路追赶旧画面，越追越旧；顶掉之后
    每次算的都是当下最新的一帧，代价只是中间那些帧没被推理过——对预览来说这正好
    是想要的那一半（显示流畅），另一半（结果新鲜度）由 :meth:`latest` 的年龄
    如实显示出来。
    """

    def __init__(self, compute: Callable[[object], object], *,
                 name: str = "vision-detect") -> None:
        if not callable(compute):
            raise ValueError("compute must be callable")
        self.compute = compute
        self.name = name
        self.results = 0
        self.failures = 0
        self.last_error: Optional[BaseException] = None
        self.rate = RateMeter()
        self._latest: Optional[DetectionResult] = None
        self._pending: Optional[GrabbedFrame] = None
        self._condition = threading.Condition()
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- 生命周期
    def start(self) -> "DetectionWorker":
        """Start the worker thread (safe to call twice)."""
        if self._thread is not None:
            return self
        with self._condition:
            self._stop = False
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker and wait for the in-flight inference to finish.

        推理可能正卡在一次 2 秒的霍夫上，所以 ``join`` 给足超时；线程是 daemon，
        超时也不会把进程吊住。
        """
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def _loop(self) -> None:
        """推理线程主体。"""
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait(0.1)
                if self._stop and self._pending is None:
                    return
                item, self._pending = self._pending, None
            started = time.perf_counter()
            try:
                value = self.compute(item.frame)
            except Exception as error:  # 单帧算错不该让预览整个崩掉
                self.failures += 1
                self.last_error = error
                with self._condition:
                    self._condition.notify_all()
                continue
            finished = time.perf_counter()
            with self._condition:
                self._latest = DetectionResult(
                    value=value, frame_sequence=item.sequence,
                    started_at=started, finished_at=finished,
                    seconds=finished - started,
                )
                self.results += 1
                self._condition.notify_all()
            self.rate.tick(finished)

    # ------------------------------------------------------------- 取用
    def submit(self, frame, sequence: int) -> None:
        """把一帧交给推理线程；上一帧还没开始算的话直接顶掉它。"""
        with self._condition:
            self._pending = GrabbedFrame(frame, int(sequence), time.perf_counter())
            self._condition.notify()

    def latest(self) -> Optional[DetectionResult]:
        """最近一次算完的结果；一次都还没算完时为 ``None``。"""
        with self._condition:
            return self._latest

    def wait_first(self, timeout: float = DEFAULT_FIRST_FRAME_TIMEOUT) -> Optional[DetectionResult]:
        """等第一次推理完成；超时返回 ``None``（画面照常显示，只是还没框）。"""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._latest is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._latest

    def __enter__(self) -> "DetectionWorker":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


@dataclass
class PreviewStats:
    """一次预览的统计：取了多少帧、推理了多少次、画了多少帧。"""

    #: 摄像头实际交出来的帧数（取帧线程数的）。
    captured: int = 0
    #: 真正跑过推理的帧数。
    detected: int = 0
    #: 画到窗口上的帧数（同一帧不重复画）。
    shown: int = 0
    #: 取帧线程的速率（次/秒）——就是摄像头实际给出来的帧率。
    capture_fps: float = 0.0
    #: 推理线程的速率（次/秒）——识别真正跑得有多快。
    detect_fps: float = 0.0
    #: 最近一次推理耗时（毫秒）。
    detect_ms: float = 0.0
    #: 画面上那组框的年龄（毫秒）：框比当前画面旧这么多。
    age_ms: float = 0.0
    #: 因为推理跟不上而**没被推理过**的帧数（被新帧顶掉的）。
    skipped: int = 0
    #: 预览为什么结束：``"quit"``（按了 q）/ ``"no-frame"``（一直取不到画面）/
    #: ``"camera-lost"``（跑到一半掉线）/ ``"pipeline-error"``（推理抛异常）。
    #: 调用方靠它决定退出码——原来看摄像头掉线是要报错的，不能都当成正常退出。
    reason: str = ""

    def line(self) -> str:
        """一行中文摘要，供终端定期打印。"""
        return (
            f"取帧 {self.captured} 帧（{self.capture_fps:.1f} fps），"
            f"推理 {self.detected} 次（{self.detect_fps:.1f} fps，"
            f"最近一次 {self.detect_ms:.0f} ms），显示 {self.shown} 帧，"
            f"跳过 {self.skipped} 帧，框滞后 {self.age_ms:.0f} ms"
        )

    def hud(self) -> str:
        """画在窗口上的一行（只有 ASCII，``cv2.putText`` 画不出中文）。"""
        return (
            f"cap {self.capture_fps:4.1f} fps | det {self.detect_fps:4.1f} fps "
            f"{self.detect_ms:5.0f} ms | age {self.age_ms:4.0f} ms"
        )


def run_preview(
    camera,
    pipeline: Callable[[object], object],
    draw: Callable[[object, object], object],
    *,
    window: str,
    async_detect: bool = True,
    first=None,
    first_frame_timeout: float = DEFAULT_FIRST_FRAME_TIMEOUT,
    key_poll: float = DEFAULT_KEY_POLL,
    max_misses: int = DEFAULT_MAX_MISSES,
    print_every: int = 0,
    on_key: Optional[Callable[[int, object, PreviewStats], None]] = None,
    on_frame: Optional[Callable[[object, object, PreviewStats], None]] = None,
    overlay: Optional[Callable[[object, object, PreviewStats], None]] = None,
    printer: Callable[[str], None] = print,
    show: Callable[[str, object], None] = _imshow,
    wait_key: Callable[[int], int] = _wait_key,
) -> PreviewStats:
    """预览循环：显示跑在摄像头帧率上，推理按自己的速度跑。

    Parameters:
        camera: 任何有 ``read()`` 的对象。
        pipeline: ``frame -> 结果`` 的推理函数（``locator.locate`` /
            ``detector.detect`` 都符合）。
        draw: ``(frame, 结果) -> 标注图`` 的画图函数。
        window: HighGUI 窗口名（必须是 ASCII，见 ``preview.py``）。
        async_detect: ``False`` 时退化成原来的串行循环（取一帧算一帧），
            抓取那种"结果必须和画面同帧"的场合用。
        first: 调用方已经取到的第一帧。摄像头"先丢几帧"是常态，有的入口要先用
            自己的重试策略拿到画面（顺便打印实际画幅），拿到之后再进来，
            免得两处各写一套等待逻辑。``None`` 表示由本函数等第一帧。
        print_every: 每显示这么多帧往终端打一行 :meth:`PreviewStats.line`；
            0 表示不打印。
        on_key: ``(按键, 最近结果, 统计)``：按键处理交给调用方（``q`` 由本函数处理）。
        on_frame: ``(原始帧, 最近结果, 统计)``：每显示一帧调一次。给的是**未标注**的
            原始帧——纯色画面检查、打印实际画幅这类事要看原始像素。
        overlay: ``(标注图, 最近结果, 统计)``：在标准 HUD 之后调用，用来往画面上写
            自己的东西。标准 HUD 是取帧/推理速率与框的年龄——正是这次改动想让人
            看见的两件事，所以由本函数负责画，调用方的 ``overlay`` 画别的地方。

    Returns:
        本次预览的 :class:`PreviewStats`（按 ``q``、摄像头掉线、推理报错都会返回）。
    """
    stats = PreviewStats()
    grabber = FrameGrabber(camera).start()
    worker = DetectionWorker(pipeline).start() if async_detect else None
    try:
        current = first
        sequence = 0  # 0 表示这一帧不是取帧线程给的（调用方自己取的）
        if current is None:
            grabbed = grabber.wait_first(first_frame_timeout, max_misses)
            if grabbed is None:
                printer(
                    "摄像头没有给出画面：可能被别的程序占用、序号不对，或者设备没插好"
                    "（测试菜单里有“摄像头自检”，可以挨个序号试）"
                )
                stats.reason = "no-frame"
                return stats
            current, sequence = grabbed.frame, grabbed.sequence

        result = None
        if worker is not None:
            worker.submit(current, sequence)
            first_result = worker.wait_first(first_frame_timeout)
            result = None if first_result is None else first_result.value
        else:
            # 串行模式：第一帧也要先算完再画，与原来的"取一帧算一帧"完全一致。
            result = _compute(pipeline, current, stats, printer)
            if result is _FAILED:
                stats.reason = "pipeline-error"
                return stats

        submitted = sequence
        shown_sequence = -1  # 还没画过任何一帧，第一帧一定要画出来
        while True:
            if submitted <= shown_sequence:
                # 上一帧已经画过了：等一帧新的，最多等 key_poll 秒就去处理按键。
                fresh = grabber.wait_newer(submitted, key_poll)
                if fresh is not None:
                    submitted, current = fresh.sequence, fresh.frame
                    if worker is not None:
                        worker.submit(current, submitted)
                    else:
                        result = _compute(pipeline, current, stats, printer)
                        if result is _FAILED:
                            stats.reason = "pipeline-error"
                            return stats

            now = time.perf_counter()
            stats.captured = grabber.rate.count
            stats.capture_fps = grabber.rate.rate
            if worker is not None:
                latest = worker.latest()
                if latest is not None:
                    # 回调拿到的永远是**推理结果本身**（report / list[Detection]），
                    # 不是内层的 DetectionResult 包装；包装只用来算耗时和年龄。
                    result = latest.value
                    stats.detect_ms = latest.milliseconds
                    stats.age_ms = latest.age(now) * 1000.0
                stats.detected = worker.results
                stats.detect_fps = worker.rate.rate
                if worker.failures == 1 and worker.last_error is not None:
                    printer(f"推理第 1 次失败：{worker.last_error!r}")
            stats.skipped = max(0, stats.captured - stats.detected)

            annotated = draw(current, result) if result is not None else current.copy()
            _draw_hud(annotated, stats)
            if on_frame is not None:
                on_frame(current, result, stats)
            if overlay is not None:
                overlay(annotated, result, stats)
            show(window, annotated)
            shown_sequence = submitted
            stats.shown += 1
            if print_every and stats.shown % print_every == 0:
                printer("  " + stats.line())

            key = wait_key(1) & 0xFF
            if on_key is not None:
                on_key(key, result, stats)
            if key == ord("q"):
                stats.reason = "quit"
                return stats
            if grabber.misses >= max_misses:
                printer(f"连续 {grabber.misses} 次取不到画面，退出预览")
                stats.reason = "camera-lost"
                return stats
    finally:
        if worker is not None:
            worker.stop()
        grabber.stop()


#: 串行分支里"这一帧算错了"的哨兵（``None`` 是合法结果，不能拿来当错误标志）。
_FAILED = object()


def _compute(pipeline, frame, stats: PreviewStats, printer: Callable[[str], None]):
    """串行分支：直接算一帧，出错就报告并返回哨兵。"""
    try:
        value = pipeline(frame)
    except Exception as error:
        printer(f"推理失败：{type(error).__name__}: {error}")
        return _FAILED
    stats.detected += 1
    return value


def _draw_hud(annotated, stats: PreviewStats) -> None:
    """把取帧/推理速率与框的年龄画在左下角。

    画在左下角是因为各入口的标题都在左上角（第 24、52 行那两个位置）。
    """
    cv2 = _cv2()
    height = annotated.shape[0]
    cv2.putText(
        annotated,
        stats.hud(),
        (10, max(20, height - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """假摄像头 + 假推理验证两条线程的配合（不需要摄像头，也不开窗口）。

    假摄像头按 60 fps 交帧，假推理固定花 50 ms（相当于 20 fps）：
    两条线程下显示应当仍在 60 fps 附近、推理 20 fps；串行对照则会塌到 ~20 fps。
    """
    import numpy as np

    class FakeCamera:
        """按 ``interval`` 秒一帧的假摄像头（帧上写着序号）。"""

        def __init__(self, interval: float = 1.0 / 60.0) -> None:
            self.interval = interval
            self.served = 0

        def read(self):
            time.sleep(self.interval)
            self.served += 1
            frame = np.full((48, 64, 3), self.served % 251, dtype=np.uint8)
            frame[0, 0, 0] = self.served % 256
            return frame

    def slow_pipeline(_frame):
        time.sleep(0.05)
        return "result"

    class Harness:
        """假窗口：数显示了哪些帧、第 ``quit_after`` 次按键时返回 q。"""

        def __init__(self, quit_after: int = 60) -> None:
            self.quit_after = quit_after
            self.shown: list = []
            self.keys = 0

        def show(self, _window, frame) -> None:
            self.shown.append(int(frame[0, 0, 0]))

        def wait_key(self, _delay: int = 1) -> int:
            self.keys += 1
            if len(self.shown) >= self.quit_after or self.keys > 20000:
                return ord("q")
            return -1

    def preview(harness: Harness, *, async_detect: bool = True, seconds: float = 0.0):
        started = time.perf_counter()
        stats = run_preview(
            FakeCamera(),
            slow_pipeline,
            lambda frame, _result: frame,
            window="vision-preview",
            async_detect=async_detect,
            print_every=0,
            printer=lambda _message: None,
            show=harness.show,
            wait_key=harness.wait_key,
        )
        return stats, time.perf_counter() - started

    problems: list[str] = []
    harness = Harness(quit_after=60)
    stats, elapsed = preview(harness)
    print(f"两条线程：显示 {stats.shown} 帧，取帧 {stats.capture_fps:.1f} fps，"
          f"推理 {stats.detect_fps:.1f} fps，跳过 {stats.skipped} 帧，"
          f"{stats.shown / max(elapsed, 1e-6):.1f} fps 显示速率，耗时 {elapsed:.2f} s")
    if stats.shown < 50:
        problems.append(f"显示帧数太少（{stats.shown}），两条线程没跑起来")
    if stats.capture_fps < 40:
        problems.append(f"取帧速率只有 {stats.capture_fps:.1f} fps，取帧线程被推理拖住了")
    if not 10 <= stats.detect_fps <= 25:
        problems.append(f"推理速率 {stats.detect_fps:.1f} fps，与 50 ms 的假推理不符")
    if stats.skipped <= 0:
        problems.append("推理比显示慢，应当有帧被跳过")
    if stats.shown / max(elapsed, 1e-6) < 40:
        problems.append("显示速率没有跟上取帧速率：显示循环还在等推理")

    serial_harness = Harness(quit_after=30)
    serial, serial_elapsed = preview(serial_harness, async_detect=False)
    serial_fps = serial.shown / max(serial_elapsed, 1e-6)
    print(f"串行对照：显示 {serial.shown} 帧，{serial_fps:.1f} fps（应当接近 20，"
          f"也就是推理速率）")
    if serial_fps > 30:
        problems.append(f"串行对照跑了 {serial_fps:.1f} fps，自测没有测到该测的东西")
    if serial_elapsed < 1.0:
        problems.append("串行对照太快结束，样本不足")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：显示跑在取帧速率上，推理按自己的速度跑，"
          "跟不上的帧被跳过而不是拖慢显示。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

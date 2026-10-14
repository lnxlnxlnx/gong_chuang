"""Offline checks for the two-thread preview (``modules.vision.drive.stream``).

摄像头和窗口都用假的：这里验的是**线程配合**——取帧只留最新一帧、推理忙时新帧顶掉
待处理帧、显示跑在取帧速率上、退出路径（按 q / 摄像头掉线 / 一直没画面）都能返回，
而且不会留下还在跑的线程。

真机上的现象（画面卡、延迟越积越大）不会在没有摄像头的 CI 里复现，所以这里用
"假摄像头按固定间隔交帧 + 假推理固定耗时"把这套时序造出来。
"""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from modules.vision.drive.stream import (
    DEFAULT_MAX_MISSES,
    DetectionWorker,
    FrameGrabber,
    PreviewStats,
    RateMeter,
    run_preview,
)


def frame(value: int = 7, size: tuple = (24, 32)):
    """A tiny frame whose first pixel encodes ``value`` (so tests can track frames)."""
    picture = np.full((size[0], size[1], 3), value % 251, dtype=np.uint8)
    picture[0, 0, 0] = value % 256
    return picture


def value_of(picture) -> int:
    """Read back the marker written by :func:`frame`."""
    return int(picture[0, 0, 0])


class FakeCamera:
    """A camera that hands out numbered frames every ``interval`` seconds."""

    def __init__(self, interval: float = 0.0, frames: int | None = None) -> None:
        self.interval = interval
        self.limit = frames
        self.served = 0
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.limit is not None and self.served >= self.limit:
            return None
        if self.interval:
            time.sleep(self.interval)
        self.served += 1
        return frame(self.served)


class SilentCamera:
    """A camera that opens but never delivers a frame."""

    def __init__(self) -> None:
        self.reads = 0

    def read(self):
        self.reads += 1
        return None


class BrokenCamera:
    """A camera whose ``read`` raises (cable pulled mid-run)."""

    def __init__(self) -> None:
        self.reads = 0

    def read(self):
        self.reads += 1
        raise OSError("device disconnected")


def quiet(*_args, **_kwargs) -> None:
    """Swallow printer output in tests."""


class Harness:
    """A fake window: records shown frames and quits after ``quit_after`` of them.

    ``poll_limit`` 是给"本来就不该画出东西"的用例兜底的：一个永远返回 -1 的假
    ``waitKey`` 会让预览循环一直转下去，测试套件就会挂住而不是失败——那比失败更难查。
    """

    def __init__(self, quit_after: int = 20, keys: list | None = None,
                 poll_limit: int = 2000) -> None:
        self.quit_after = quit_after
        self.shown: list = []
        self.key_calls = 0
        self.poll_limit = poll_limit
        self.scripted = list(keys or [])

    def show(self, _window, picture) -> None:
        self.shown.append(value_of(picture))

    def wait_key(self, _delay: int = 1) -> int:
        self.key_calls += 1
        if self.scripted:
            return self.scripted.pop(0)
        if self.key_calls >= self.poll_limit:
            return ord("q")
        return ord("q") if len(self.shown) >= self.quit_after else -1


class RateMeterTests(unittest.TestCase):
    def test_first_tick_has_no_rate_yet(self) -> None:
        meter = RateMeter()
        self.assertEqual(meter.tick(now=100.0), 0.0)
        self.assertEqual(meter.count, 1)

    def test_steady_ticks_approach_the_real_rate(self) -> None:
        meter = RateMeter()
        now = 0.0
        for _ in range(200):
            now += 0.01  # 100 Hz
            meter.tick(now=now)
        self.assertAlmostEqual(meter.rate, 100.0, delta=5.0)

    def test_a_pause_pulls_the_rate_down(self) -> None:
        meter = RateMeter()
        now = 0.0
        for _ in range(50):
            now += 0.01
            meter.tick(now=now)
        steady = meter.rate
        now += 1.0  # 停了一秒
        meter.tick(now=now)
        self.assertLess(meter.rate, steady, "停顿后速率必须往下走，不能还报 100")
        for _ in range(60):   # EMA 收敛慢是它的特性：这里只验它确实收敛到 1 Hz 附近
            now += 1.0
            meter.tick(now=now)
        self.assertLess(meter.rate, 5.0, "持续 1 Hz 之后应当收敛到 1 Hz 附近")

    def test_rejects_a_bad_alpha(self) -> None:
        for alpha in (0.0, -0.1, 1.5):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                RateMeter(alpha)


class FrameGrabberTests(unittest.TestCase):
    def test_it_keeps_only_the_newest_frame(self) -> None:
        grabber = FrameGrabber(FakeCamera(interval=0.001)).start()
        try:
            first = grabber.wait_first(2.0)
            self.assertIsNotNone(first)
            time.sleep(0.05)
            latest = grabber.latest()
            self.assertGreater(latest.sequence, first.sequence,
                               "取帧线程应当一直在推进序号")
            self.assertGreater(value_of(latest.frame), value_of(first.frame))
        finally:
            grabber.stop()

    def test_wait_newer_returns_only_newer_frames(self) -> None:
        # 只交一帧的摄像头：拿到第一帧以后不会再有新帧，正好验"不重复交同一帧"。
        grabber = FrameGrabber(FakeCamera(interval=0.0, frames=1)).start()
        try:
            first = grabber.wait_first(2.0)
            self.assertIsNotNone(first)
            self.assertIsNone(grabber.wait_newer(first.sequence, 0.05),
                              "没有比它更新的帧时应当等到超时返回 None")
        finally:
            grabber.stop()

    def test_wait_newer_hands_over_each_sequence_once(self) -> None:
        grabber = FrameGrabber(FakeCamera(interval=0.001)).start()
        try:
            first = grabber.wait_first(2.0)
            newer = grabber.wait_newer(first.sequence, 2.0)
            self.assertIsNotNone(newer)
            self.assertGreater(newer.sequence, first.sequence)
            self.assertGreater(value_of(newer.frame), value_of(first.frame))
        finally:
            grabber.stop()

    def test_stop_joins_the_thread_and_is_idempotent(self) -> None:
        grabber = FrameGrabber(FakeCamera(interval=0.001)).start()
        self.assertIsNotNone(grabber.wait_first(2.0))
        grabber.stop()
        grabber.stop()
        self.assertEqual([t for t in threading.enumerate() if t.name == "vision-capture"], [])

    def test_a_silent_camera_gives_up_quickly(self) -> None:
        camera = SilentCamera()
        grabber = FrameGrabber(camera).start()
        try:
            started = time.perf_counter()
            self.assertIsNone(grabber.wait_first(5.0, max_misses=DEFAULT_MAX_MISSES))
            self.assertLess(time.perf_counter() - started, 3.0,
                            "连续读失败应当立刻返回，而不是干等超时")
            self.assertGreaterEqual(camera.reads, DEFAULT_MAX_MISSES)
        finally:
            grabber.stop()

    def test_a_raising_camera_is_counted_and_the_thread_survives(self) -> None:
        camera = BrokenCamera()
        grabber = FrameGrabber(camera).start()
        try:
            self.assertIsNone(grabber.wait_first(1.0, max_misses=5))
            self.assertGreaterEqual(grabber.failures, 5)
            self.assertIsInstance(grabber.last_error, OSError)
            self.assertTrue(grabber._thread.is_alive(), "单次读失败不该结束取帧线程")
        finally:
            grabber.stop()

    def test_it_rejects_objects_without_read(self) -> None:
        with self.assertRaises(ValueError):
            FrameGrabber(object())


class DetectionWorkerTests(unittest.TestCase):
    def test_it_returns_the_computed_value(self) -> None:
        worker = DetectionWorker(lambda picture: value_of(picture)).start()
        try:
            worker.submit(frame(11), 1)
            result = worker.wait_first(2.0)
            self.assertIsNotNone(result)
            self.assertEqual(result.value, 11)
            self.assertEqual(result.frame_sequence, 1)
            self.assertGreater(result.seconds, 0.0)
            self.assertGreater(worker.rate.rate if worker.results > 1 else 1.0, 0.0)
        finally:
            worker.stop()

    def test_a_busy_worker_drops_pending_frames_instead_of_queueing(self) -> None:
        seen: list = []

        def slow(picture):
            time.sleep(0.02)
            seen.append(value_of(picture))
            return value_of(picture)

        worker = DetectionWorker(slow).start()
        try:
            for sequence in range(1, 41):  # 40 帧灌进去，每帧推理 20 ms
                worker.submit(frame(sequence), sequence)
                time.sleep(0.002)
            time.sleep(0.1)
            self.assertLess(len(seen), 40,
                            "推理跟不上时应当有帧被顶掉，而不是排成一队全算一遍")
            self.assertGreater(len(seen), 0)
            # 最后算的应当是接近最新的那一帧，而不是最早那几帧
            self.assertGreater(seen[-1], 20)
        finally:
            worker.stop()

    def test_a_failing_computation_is_counted_and_the_worker_keeps_going(self) -> None:
        def failing(picture):
            if value_of(picture) == 1:
                raise RuntimeError("boom")
            return value_of(picture)

        worker = DetectionWorker(failing).start()
        try:
            worker.submit(frame(1), 1)
            time.sleep(0.05)
            worker.submit(frame(2), 2)
            result = worker.wait_first(2.0)
            self.assertIsNotNone(result)
            self.assertEqual(result.value, 2)
            self.assertEqual(worker.failures, 1)
            self.assertIsInstance(worker.last_error, RuntimeError)
        finally:
            worker.stop()

    def test_stop_is_idempotent_and_leaves_no_thread(self) -> None:
        worker = DetectionWorker(lambda picture: picture).start()
        worker.stop()
        worker.stop()
        self.assertEqual([t for t in threading.enumerate() if t.name == "vision-detect"], [])

    def test_it_rejects_a_non_callable(self) -> None:
        with self.assertRaises(ValueError):
            DetectionWorker("not callable")


class RunPreviewTests(unittest.TestCase):
    def test_the_window_runs_at_the_capture_rate_while_detection_is_slow(self) -> None:
        """这次改动的全部意义：显示跟取帧走，不跟推理走。"""
        harness = Harness(quit_after=15)
        started = time.perf_counter()
        stats = run_preview(
            FakeCamera(interval=0.002),          # ~500 fps 的假摄像头
            lambda picture: time.sleep(0.03),    # 33 ms 的假推理（远慢于取帧）
            lambda picture, _result: picture,
            window="vision-preview",
            printer=quiet,
            show=harness.show,
            wait_key=harness.wait_key,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(stats.reason, "quit")
        self.assertEqual(stats.shown, 15)
        self.assertGreater(stats.skipped, 0, "推理比取帧慢，应当有帧被跳过")
        self.assertLess(stats.detect_fps, stats.capture_fps)
        # 15 帧显示若被推理串起来要 0.45 s 以上；这里应当紧跟着取帧走
        self.assertLess(elapsed, 0.4)

    def test_serial_mode_blocks_the_display_on_every_inference(self) -> None:
        harness = Harness(quit_after=5)
        started = time.perf_counter()
        stats = run_preview(
            FakeCamera(interval=0.002),
            lambda picture: time.sleep(0.03),
            lambda picture, _result: picture,
            window="vision-preview",
            async_detect=False,
            printer=quiet,
            show=harness.show,
            wait_key=harness.wait_key,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(stats.shown, 5)
        self.assertGreater(elapsed, 0.15, "串行模式每帧都要等推理，应当明显更慢")

    def test_callbacks_receive_the_unwrapped_result_and_the_raw_frame(self) -> None:
        seen = {"keys": [], "frames": [], "overlays": []}

        def pipeline(_picture):
            return "the-report"

        def overlay(annotated, result, stats):
            seen["overlays"].append((result, stats.shown))

        def on_key(key, result, _stats):
            seen["keys"].append((key, result))

        def on_frame(picture, result, _stats):
            seen["frames"].append((value_of(picture), result))

        harness = Harness(quit_after=3)
        stats = run_preview(
            FakeCamera(interval=0.002),
            pipeline,
            lambda picture, result: picture.copy(),
            window="vision-preview",
            printer=quiet,
            show=harness.show,
            wait_key=harness.wait_key,
            on_key=on_key,
            on_frame=on_frame,
            overlay=overlay,
        )
        self.assertEqual(stats.shown, 3)
        # 回调拿到的是推理结果本身，不是内层的 DetectionResult 包装
        self.assertTrue(all(result == "the-report" for _key, result in seen["keys"]))
        self.assertTrue(all(result == "the-report" for _value, result in seen["frames"]))
        self.assertEqual(len(seen["overlays"]), 3)

    def test_p_does_not_stop_the_preview_and_q_does(self) -> None:
        harness = Harness(keys=[ord("p"), ord("p"), ord("q")])
        stats = run_preview(
            FakeCamera(interval=0.002),
            lambda picture: "report",
            lambda picture, _result: picture,
            window="vision-preview",
            printer=quiet,
            show=harness.show,
            wait_key=harness.wait_key,
        )
        self.assertEqual(stats.reason, "quit")
        self.assertEqual(stats.shown, 3)

    def test_a_camera_that_never_delivers_reports_and_returns_without_a_window(self) -> None:
        printed: list = []
        harness = Harness()
        stats = run_preview(
            SilentCamera(),
            lambda picture: "report",
            lambda picture, _result: picture,
            window="vision-preview",
            printer=printed.append,
            show=harness.show,
            wait_key=harness.wait_key,
        )
        self.assertEqual(stats.reason, "no-frame")
        self.assertEqual(stats.shown, 0)
        self.assertEqual(harness.shown, [])
        self.assertTrue(any("画面" in line for line in printed))

    def test_a_camera_lost_mid_run_stops_the_preview(self) -> None:
        class DyingCamera:
            """两帧之后再也不给画面。"""

            def __init__(self) -> None:
                self.served = 0

            def read(self):
                self.served += 1
                return frame(self.served) if self.served <= 2 else None

        printed: list = []
        harness = Harness(quit_after=10_000)
        stats = run_preview(
            DyingCamera(),
            lambda picture: "report",
            lambda picture, _result: picture,
            window="vision-preview",
            max_misses=5,
            printer=printed.append,
            show=harness.show,
            wait_key=harness.wait_key,
        )
        self.assertEqual(stats.reason, "camera-lost")
        self.assertTrue(any("取不到画面" in line for line in printed))

    def test_a_failing_pipeline_in_serial_mode_returns_with_a_reason(self) -> None:
        def exploding(_picture):
            raise RuntimeError("model exploded")

        printed: list = []
        stats = run_preview(
            FakeCamera(interval=0.002),
            exploding,
            lambda picture, _result: picture,
            window="vision-preview",
            async_detect=False,
            printer=printed.append,
            show=Harness().show,
            wait_key=Harness().wait_key,
        )
        self.assertEqual(stats.reason, "pipeline-error")
        self.assertTrue(any("推理失败" in line for line in printed))

    def test_the_first_frame_is_always_shown_even_without_a_newer_one(self) -> None:
        """调用方自己取到的第一帧也不能被跳过（摄像头打开后第一帧往往最要紧）。"""
        harness = Harness(quit_after=1)
        stats = run_preview(
            FakeCamera(interval=0.5),   # 第一帧之后很久才有下一帧
            lambda picture: "report",
            lambda picture, _result: picture,
            window="vision-preview",
            first=frame(99),
            printer=quiet,
            show=harness.show,
            wait_key=harness.wait_key,
        )
        self.assertEqual(stats.shown, 1)
        self.assertEqual(harness.shown, [99])

    def test_print_every_writes_the_summary_line(self) -> None:
        printed: list = []
        harness = Harness(quit_after=4)   # show 与 wait_key 必须共用同一个假窗口
        stats = run_preview(
            FakeCamera(interval=0.002),
            lambda picture: "report",
            lambda picture, _result: picture,
            window="vision-preview",
            print_every=2,
            printer=printed.append,
            show=harness.show,
            wait_key=harness.wait_key,
        )
        self.assertEqual(stats.shown, 4)
        self.assertTrue(any("取帧" in line and "推理" in line for line in printed))


class PreviewStatsTests(unittest.TestCase):
    def test_line_and_hud_report_both_rates(self) -> None:
        stats = PreviewStats(captured=120, detected=30, shown=60,
                             capture_fps=30.0, detect_fps=7.5, detect_ms=133.0,
                             age_ms=40.0, skipped=90)
        self.assertIn("30.0 fps", stats.line())
        self.assertIn("7.5 fps", stats.line())
        self.assertIn("133", stats.line())
        self.assertIn("90", stats.line())
        hud = stats.hud()
        self.assertIn("cap", hud)
        self.assertIn("det", hud)
        self.assertIn("age", hud)
        self.assertTrue(hud.isascii(), "画在窗口上的那行必须只有 ASCII")


if __name__ == "__main__":
    unittest.main()

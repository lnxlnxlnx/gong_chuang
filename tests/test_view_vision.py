"""``view_vision`` 的离线检查：不需要摄像头、模型或窗口。

这个入口的价值在于"六种结果都能被正确分派、画错一帧也不会让窗口消失"。所以这里用
假会话把六条路都走一遍，并钉住三件容易在改动中丢掉的事：

1. 结果**自带模式**（异步预览里切模式时，画笔不会拿旧数据去画）;
2. 单帧算错/画错变成画面上的一张失败卡，而不是把整条预览链路带崩;
3. 画到窗口上的字**全部是 ASCII**——``cv2.putText`` 画不出中文，漏一个字就是一片方块。

``--once`` 那条路也覆盖了：它不建窗口，所以可以在无显示环境里跑。
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import view_vision
from modules.vision.drive.cell_targets import CellTargets
from modules.vision.drive.hough_lock import LockedRegion, RingLock
from modules.vision.drive.material_class import ClassResult
from modules.vision.drive.material_face import FaceQuality
from modules.vision.drive.numbered_ring import RingTarget
from modules.vision.drive.ring_station import RingStationDetection
from modules.vision.drive.stacked_ring import StackedRingDetection
from modules.vision.reports import MaterialReport, RegionReport
from view_vision import (
    MODE_CELLS,
    MODE_KEYS,
    MODE_MEASURE,
    MODE_MATERIAL,
    MODE_RING,
    MODE_RINGS,
    MODE_STACKED,
    MODE_YOLO,
    MODES,
    Failure,
    MeasureSession,
    ModeResult,
    Viewer,
    build_parser,
    run_once,
)


# ------------------------------------------------------------------ 假对象
class FakeCamera:
    """每次 ``read()`` 都交同一帧（帧内容随机，避免被判成"纯色画面"）。"""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.frame


class FakeLocator:
    def __init__(self, report: MaterialReport) -> None:
        self.report = report
        self.calls = 0

    def locate(self, frame, *, want=None):
        self.calls += 1
        return self.report

    def draw(self, frame, report, **kwargs):
        return frame.copy()


class FakeDetector:
    """够用的假检测器：``detect`` 返回固定值（或抛固定异常），``draw`` 原样返回一份。"""

    def __init__(self, value=None, *, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.value

    def draw(self, frame, value, **kwargs):
        return frame.copy()


def synthetic_frame() -> np.ndarray:
    """一张 240x320 的"有纹理"画面：纯色会被 ``looks_blank`` 判成摄像头被占用。"""
    generator = np.random.default_rng(7)
    return generator.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def material_report() -> MaterialReport:
    """一个通过校验的顶面：够画出圆、也够翻成闭环观测。"""
    region = LockedRegion(center_x=160.0, center_y=120.0, radius=30.0)
    classification = ClassResult(found=True, name="red", label="物料", score=0.9)
    face = FaceQuality(interior_std=5.0, edge_support=0.8, sampled_rays=32)
    report_region = RegionReport(
        region=region,
        classification=classification,
        face=face,
        accepted=True,
        x_mm=1.5,
        y_mm=-2.5,
        distance_mm=97.0,
    )
    return MaterialReport(lock=RingLock(regions=(region,)), hits=(), regions=(report_region,))


def ring_target() -> RingTarget:
    """视野里那一个环：中心在光轴上、整框可见、可对准（环号与置信度来自模型）。"""
    return RingTarget(
        center_px=(480.0, 120.0),
        radius_px=96.0,
        digit=2,
        score=0.93,
        clipped=False,
        x_mm=0.0,
        y_mm=0.0,
        scale_mm_per_px=0.27,
        scale_source="depth/focal",
        detail="模型给了 1 个环框",
    )


class FakeVision:
    """把 ``Viewer`` 用到的会话接口都换成假实现（一台设备也不开）。"""

    def __init__(self, frame: np.ndarray, *, yolo_error: Exception | None = None) -> None:
        self.frame = frame
        self.report = material_report()
        self._camera = FakeCamera(frame)
        self.locator = FakeLocator(self.report)
        self.yolo_detector = FakeDetector((), error=yolo_error)
        self._locator = SimpleNamespace(
            locate=self.locator.locate,
            draw=self.locator.draw,
            yolo=SimpleNamespace(detector=self.yolo_detector),
        )
        self.ring_detector = FakeDetector(RingStationDetection(detected=True, confidence=0.95))
        # 半径是必需字段：``StackedRingDetection.summary`` 会直接格式化它（没有就 TypeError）。
        self.stacked_detector = FakeDetector(
            StackedRingDetection(detected=True, confidence=0.8, middle_radius_px=70.0)
        )
        self.ring_target = ring_target()
        self.ring_target_calls = 0

    def locate_ring(self, *, timeout=None, digit=None):
        self.ring_target_calls += 1
        return self.ring_target

    def _frame_size(self):
        return (self.frame.shape[1], self.frame.shape[0])

    def _effective_focal_px(self, override=None):
        return float(override) if override is not None else 800.0

    def pixel_distance(self, start, end):
        from modules.vision.drive.pixel_measure import pixel_distance

        return pixel_distance(start, end, image_size=self._frame_size())

    def pixel_distance_mm(self, start, end, *, working_distance_mm=None, focal_px=None):
        raise ValueError("测试里没有标定：本轮只验证报错被显示出来而不是抛出去")

    def _ring_station_detector(self):
        return self.ring_detector

    def _stacked_ring_detector(self):
        return self.stacked_detector


def fake_settings():
    return SimpleNamespace(
        targets=SimpleNamespace(),
        camera_id=lambda platform: 0,
        table_depth_mm=None,
        focal_px=None,
    )


def make_viewer(*, yolo_error: Exception | None = None) -> tuple[Viewer, FakeVision, np.ndarray]:
    frame = synthetic_frame()
    vision = FakeVision(frame, yolo_error=yolo_error)
    return Viewer(vision, fake_settings(), frames=5), vision, frame


# ------------------------------------------------------------------ 用例
class ParserTests(unittest.TestCase):
    def test_defaults_open_the_material_view_asynchronously(self) -> None:
        arguments = build_parser().parse_args([])
        self.assertIsNone(arguments.mode)
        self.assertIsNone(arguments.camera)
        self.assertIsNone(arguments.focal)
        self.assertEqual(arguments.frames, view_vision.DEFAULT_MEASURE_FRAMES)
        self.assertTrue(arguments.async_detect)
        self.assertFalse(arguments.once)
        self.assertFalse(arguments.save)

    def test_sync_flag_switches_off_the_worker_thread(self) -> None:
        self.assertFalse(build_parser().parse_args(["--sync"]).async_detect)

    def test_mode_choices_are_exactly_the_six_modes(self) -> None:
        arguments = build_parser().parse_args(["--mode", MODE_RINGS])
        self.assertEqual(arguments.mode, MODE_RINGS)
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--mode", "nope"])

    def test_number_keys_follow_the_mode_order(self) -> None:
        self.assertEqual(
            [MODE_KEYS[ord(str(index))] for index in range(1, len(MODES) + 1)], list(MODES)
        )


class PipelineTests(unittest.TestCase):
    def test_material_result_carries_its_mode_and_records_a_sample(self) -> None:
        viewer, vision, frame = make_viewer()
        result = viewer.pipeline(frame)
        self.assertIsInstance(result, ModeResult)
        self.assertEqual(result.mode, MODE_MATERIAL)
        self.assertIs(result.value, vision.report)
        self.assertEqual(len(viewer.samples), 1)
        sample = viewer.samples[0]
        self.assertAlmostEqual(sample.x_mm, 1.5)
        self.assertAlmostEqual(sample.y_mm, -2.5)
        self.assertAlmostEqual(sample.z_mm, 97.0)
        # diameter_px 是半径 ×2：这里的换算只有一处（与 runtime 的观测转换同源）。
        self.assertAlmostEqual(sample.diameter_px, 60.0)
        self.assertEqual(sample.color, "red")

    def test_each_mode_runs_its_own_detector(self) -> None:
        viewer, vision, frame = make_viewer()
        for mode in (MODE_YOLO, MODE_RINGS, MODE_STACKED, MODE_RING):
            with self.subTest(mode=mode):
                viewer.mode = mode
                result = viewer.pipeline(frame)
                self.assertEqual(result.mode, mode)
                self.assertNotIsInstance(result.value, Failure)
        self.assertEqual(vision.ring_detector.calls, 1)
        self.assertEqual(vision.stacked_detector.calls, 1)
        self.assertEqual(vision.ring_target_calls, 1)

    def test_measure_mode_returns_a_marker_so_draw_still_runs(self) -> None:
        """``run_preview`` 只在结果非 None 时调 draw；测量模式必须给一个非 None 的结果。"""
        viewer, _vision, frame = make_viewer()
        viewer.mode = MODE_MEASURE
        result = viewer.pipeline(frame)
        self.assertIsInstance(result, ModeResult)
        self.assertEqual(result.mode, MODE_MEASURE)
        self.assertIsNotNone(result.value)

    def test_a_raising_detector_becomes_a_result_not_an_exception(self) -> None:
        viewer, _vision, frame = make_viewer(yolo_error=RuntimeError("模型掉了"))
        viewer.mode = MODE_YOLO
        result = viewer.pipeline(frame)
        self.assertIsInstance(result.value, Failure)
        self.assertIn("模型掉了", result.value.message)
        # 画成失败卡：不抛异常，尺寸与原帧一致。
        annotated = viewer.draw(frame, result)
        self.assertEqual(annotated.shape, frame.shape)

    def test_fusion_needs_at_least_three_samples(self) -> None:
        viewer, _vision, frame = make_viewer()
        with redirect_stdout(io.StringIO()) as output:
            viewer.fuse_samples()
        self.assertIn("至少要 3 次", output.getvalue())
        for _ in range(3):
            viewer.pipeline(frame)
        with redirect_stdout(io.StringIO()) as output:
            viewer.fuse_samples()
        self.assertIsNotNone(viewer.fusion, output.getvalue())
        self.assertEqual(viewer.fusion_note, "sampled")


class DrawTests(unittest.TestCase):
    def test_draw_dispatches_on_the_result_mode_not_the_current_mode(self) -> None:
        """切模式瞬间旧结果还在：画的一定是结果自己的模式。"""
        viewer, _vision, frame = make_viewer()
        viewer.mode = MODE_RINGS
        material_result = ModeResult(MODE_MATERIAL, material_report())
        annotated = viewer.draw(frame, material_result)
        self.assertEqual(annotated.shape, frame.shape)

    def test_drawing_never_leaves_the_canvas_size(self) -> None:
        viewer, _vision, frame = make_viewer()
        cases = {
            MODE_MATERIAL: material_report(),
            MODE_YOLO: (),
            MODE_RINGS: RingStationDetection(detected=True),
            MODE_STACKED: StackedRingDetection(detected=True, middle_radius_px=70.0),
            MODE_CELLS: CellTargets(markers=(), materials=(), unmatched=()),
            MODE_RING: ring_target(),
        }
        for mode, value in cases.items():
            with self.subTest(mode=mode):
                viewer.mode = mode
                with patch.object(Viewer, "_cell_detector", lambda _self: FakeDetector(value)):
                    annotated = viewer.draw(frame, ModeResult(mode, value))
                self.assertEqual(annotated.shape, frame.shape)

    def test_none_result_still_returns_a_canvas(self) -> None:
        viewer, _vision, frame = make_viewer()
        self.assertEqual(viewer.draw(frame, None).shape, frame.shape)


class AsciiTests(unittest.TestCase):
    def test_every_overlay_line_is_ascii(self) -> None:
        """``cv2.putText`` 画不出中文：一条含中文的 HUD 就是窗口里一排方块。"""
        viewer, _vision, _frame = make_viewer()
        results = {
            MODE_MATERIAL: material_report(),
            MODE_YOLO: (),
            MODE_RINGS: RingStationDetection(detected=True, confidence=0.9),
            MODE_STACKED: StackedRingDetection(detected=True, confidence=0.9, middle_radius_px=70.0),
            MODE_CELLS: CellTargets(markers=(), materials=(), unmatched=()),
            MODE_MEASURE: object(),
            MODE_RING: ring_target(),
        }
        for mode, value in results.items():
            viewer.mode = mode
            result = ModeResult(mode, value)
            for line in viewer.head_lines(result):
                with self.subTest(mode=mode, line=line):
                    self.assertIsInstance(line, str)
                    self.assertEqual(line.encode("ascii").decode("ascii"), line)

    def test_failure_line_is_ascii_too(self) -> None:
        viewer, _vision, _frame = make_viewer()
        for line in viewer.head_lines(ModeResult(MODE_MATERIAL, Failure(MODE_MATERIAL, "boom"))):
            self.assertEqual(line.encode("ascii").decode("ascii"), line)


class MeasureSessionTests(unittest.TestCase):
    def test_third_click_clears_and_unfreezes(self) -> None:
        session = MeasureSession()
        frame = synthetic_frame()
        session.add_point(10, 20, frame)
        self.assertEqual(len(session.points), 1)
        self.assertIsNotNone(session.frozen)
        session.add_point(30, 40, frame)
        self.assertTrue(session.complete)
        session.add_point(50, 60, frame)
        self.assertEqual(session.points, [])
        self.assertIsNone(session.frozen)
        self.assertIsNone(session.measured)


class PreviewLoopTests(unittest.TestCase):
    def test_preview_loop_switches_modes_and_draws_every_frame(self) -> None:
        """真跑一遍 ``run_preview``：假摄像头 + 假窗口，按 3、4、1 再退出。

        这条覆盖的是"窗口里真的会显示东西"：``pipeline`` 返回 :class:`ModeResult`、
        ``draw`` 按结果模式分派、``overlay`` 只写 ASCII、``on_key`` 能把模式切过去。
        """
        from modules.vision.drive.stream import run_preview

        viewer, vision, frame = make_viewer()
        keys = iter([ord("3"), ord("4"), ord("1"), ord("q")])
        shown: list[np.ndarray] = []

        def wait_key(_delay: int = 1) -> int:
            return next(keys, ord("q"))

        stats = run_preview(
            vision._camera,
            viewer.pipeline,
            viewer.draw,
            window="test",
            async_detect=False,
            print_every=0,
            on_key=viewer.on_key,
            on_frame=viewer.on_frame,
            overlay=viewer.overlay,
            show=lambda _window, canvas: shown.append(canvas),
            wait_key=wait_key,
            printer=lambda _message: None,
        )
        self.assertEqual(stats.reason, "quit")
        self.assertGreaterEqual(len(shown), 4)
        for canvas in shown:
            self.assertEqual(canvas.shape, frame.shape)
        # 3 → rings、4 → stacked、1 → material：最后停在按 1 切回来的那个模式。
        self.assertEqual(viewer.mode, MODE_MATERIAL)
        # 只有物料模式会记融合样本：开局一帧 + 切回 material 之后一帧。
        self.assertEqual(len(viewer.samples), 2)


class OnceTests(unittest.TestCase):
    def test_once_runs_every_mode_without_a_window(self) -> None:
        viewer, _vision, frame = make_viewer()
        modes = tuple(mode for mode in MODES if mode != MODE_MEASURE)
        with patch.object(view_vision, "_first_frame", lambda vision: frame):
            with redirect_stdout(io.StringIO()) as output:
                with patch.object(Viewer, "_cell_detector",
                                  lambda _self: FakeDetector(CellTargets((), (), ()))):
                    code = run_once(viewer, modes, save_dir=None)
        self.assertEqual(code, 0)
        text = output.getvalue()
        for mode in modes:
            self.assertIn(f"[{view_vision.MODE_TITLES[mode]}]", text)

    def test_once_reports_a_lost_camera_instead_of_crashing(self) -> None:
        viewer, _vision, _frame = make_viewer()
        with patch.object(view_vision, "_first_frame", lambda _vision: None):
            with redirect_stdout(io.StringIO()) as output:
                code = run_once(viewer, (MODE_MATERIAL,), save_dir=None)
        self.assertEqual(code, 1)
        self.assertIn("读不到摄像头画面", output.getvalue())

    def test_once_saves_one_image_per_mode(self) -> None:
        """存图**调用点**：每个模式画一次、存一次。

        真正的写盘交给 ``save_frame``，这里只钉住"每个模式都会存、文件名带模式"；
        写盘本身由 ``cv2.imwrite`` 负责，不该让一个离线用例依赖磁盘权限。
        """
        viewer, _vision, frame = make_viewer()
        calls: list[tuple[Path, str]] = []

        def fake_save(canvas, directory, *, mode):
            calls.append((Path(directory), mode))
            self.assertEqual(canvas.shape, frame.shape)
            return Path(directory) / f"view_{mode}.png"

        with patch.object(view_vision, "_first_frame", lambda vision: frame):
            with patch.object(view_vision, "save_frame", fake_save):
                with redirect_stdout(io.StringIO()):
                    code = run_once(
                        viewer, (MODE_MATERIAL, MODE_RINGS), save_dir=Path("output")
                    )
        self.assertEqual(code, 0)
        self.assertEqual([mode for _directory, mode in calls], [MODE_MATERIAL, MODE_RINGS])


if __name__ == "__main__":
    unittest.main()

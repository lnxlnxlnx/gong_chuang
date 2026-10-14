"""Checks for the circle-first material report."""

from __future__ import annotations

import unittest
import math
import numpy as np

from modules.vision.drive.hough_lock import HoughRingLocator, LockedRegion, RingLock, synthetic_rings
from modules.vision.drive.material_locate import MaterialLocator, parse_depth_range
from modules.vision.drive.yolov8 import Detection


class FakeYOLO:
    def __init__(self, hits=()):
        self.hits = tuple(hits)

    def detect(self, _frame, *, materials_only=False):
        return list(self.hits)

    def close(self):
        pass


class FixedCircleLocator:
    """Return a known circle so camera-coordinate checks have exact values."""

    def locate(self, _frame):
        return RingLock((LockedRegion(400.0, 200.0, 60.0),))


class RecordingLocator:
    """Record the window the locator was called with."""

    def __init__(self, regions=()):
        self.regions = tuple(regions)
        self.calls = []

    def locate(self, _frame, **kwargs):
        self.calls.append(kwargs)
        return RingLock(self.regions)


def hit(name, score, x):
    return Detection(x1=x - 20, y1=220, x2=x + 20, y2=260,
                     score=score, class_id=0, name=name)


class MaterialLocatorTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_rings()
        self.locator = MaterialLocator(
            yolo=FakeYOLO([hit("red", 0.9, 160), hit("blue", 0.7, 480)]),
            focal_px=300,
        )
        self.report = self.locator.locate(self.frame)

    def test_each_circle_is_a_material_even_when_yolo_misses_one(self):
        self.assertEqual(len(self.report.regions), 3)
        self.assertTrue(self.report.present)
        self.assertEqual([region.name for region in self.report.regions],
                         ["red", None, "blue"])
        self.assertTrue(all(region.present for region in self.report.regions))

    def test_distance_uses_circle_diameter(self):
        for region in self.report.regions:
            self.assertAlmostEqual(region.distance_mm,
                                   300 * 30 / (2 * region.region.radius))

    def test_best_prefers_the_highest_classification_score(self):
        self.assertEqual(self.report.best.name, "red")

    def test_summary_keeps_the_unknown_category_and_distance(self):
        summary = self.report.summary()
        self.assertIn("red", summary)
        self.assertIn("blue", summary)
        self.assertIn("mm", summary)
        self.assertIn("3", summary)
        self.assertIn("xyz=", summary)
        self.assertIn("confidence=0.90", summary)

    def test_draw_marks_circles_without_changing_input(self):
        before = self.frame.copy()
        marked = self.locator.draw(self.frame, self.report)
        self.assertTrue(np.array_equal(self.frame, before))
        self.assertFalse(np.array_equal(marked, before))

    def test_yolo_miss_keeps_all_circles(self):
        report = MaterialLocator(yolo=FakeYOLO(), focal_px=300).locate(self.frame)
        self.assertEqual(len(report.regions), 3)
        self.assertTrue(all(region.name is None for region in report.regions))

    def test_no_circle_means_no_material_even_with_a_yolo_box(self):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        report = self.locator.locate(blank)
        self.assertFalse(report.present)
        self.assertEqual(report.regions, ())
        self.assertIsNone(report.best)

    def test_invalid_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            MaterialLocator(yolo=FakeYOLO(), focal_px=0)
        with self.assertRaises(ValueError):
            MaterialLocator(yolo=FakeYOLO(), top_diameter_mm=0)

    def test_public_result_has_camera_coordinates_and_confidence(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        locator = MaterialLocator(
            locator=FixedCircleLocator(),
            yolo=FakeYOLO([Detection(380, 180, 420, 220, 0.8, 2, "blue")]),
            focal_px=300,
        )

        report = locator.locate(frame)
        item = report.regions[0]
        self.assertTrue(report.present)
        self.assertEqual(item.name, "blue")
        self.assertAlmostEqual(item.confidence, 0.8)
        self.assertEqual((item.center_x_px, item.center_y_px), (400.0, 200.0))
        self.assertAlmostEqual(item.x_mm, 20.0)
        self.assertAlmostEqual(item.y_mm, -10.0)
        self.assertAlmostEqual(item.z_mm, 75.0)
        self.assertAlmostEqual(item.distance_mm, 75.0)
        self.assertAlmostEqual(item.range_mm, math.sqrt(20**2 + 10**2 + 75**2))

    def test_yolo_miss_keeps_position_but_has_no_confidence(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        report = MaterialLocator(
            locator=FixedCircleLocator(), yolo=FakeYOLO(), focal_px=300
        ).locate(frame)

        item = report.regions[0]
        self.assertTrue(report.present)
        self.assertIsNone(item.name)
        self.assertIsNone(item.confidence)
        self.assertEqual((item.x_mm, item.y_mm, item.z_mm), (20.0, -10.0, 75.0))


class PerFrameRedundancyTests(unittest.TestCase):
    """一帧里的重复计算：它们曾经是这条流水线上最贵的一笔，不能再长回来。

    实测（1280×720 真实台面图）：``to_grayscale`` 转一次 12.0 ms，而霍夫会给
    24–87 个候选圆，逐个候选重转一次就是 0.3–1.0 s；模型那一侧则是同一个 frame、
    同样的参数被问了两遍，白花约 20 ms。
    """

    class CountingYOLO:
        def __init__(self, hits=()):
            self.hits = tuple(hits)
            self.calls = 0

        def detect(self, _frame, *, materials_only=False):
            self.calls += 1
            return list(self.hits)

        def close(self):
            pass

    def test_the_model_is_asked_once_per_frame(self) -> None:
        for label, kwargs in (
            # 品红在这个合成画面里根本不存在 → 调色板一个都不命中 → 问模型 →
            # 模型也没认出 → 退回霍夫。**重复调用的 bug 只在这条分支上。**
            ("colour path (palette misses, hough finds)",
             {"material_palette": {"magenta": (255.0, 0.0, 255.0)}}),
            ("hough path", {}),
        ):
            with self.subTest(path=label):
                yolo = self.CountingYOLO()
                locator = MaterialLocator(
                    yolo=yolo, focal_px=300, colour_tolerance=1.0, **kwargs
                )
                report = locator.locate(synthetic_rings())
                self.assertEqual(yolo.calls, 1,
                                 f"{label}：一帧只该问模型一次")
                if label.startswith("colour"):
                    # 必须先确认真的走过了"调色板没找到 → 问模型 → 退回霍夫"这条
                    # 分支（重复调用的 bug 只在这条分支上），否则这个用例是空转的。
                    self.assertEqual(report.colour_source, "hough")
                    self.assertTrue(report.regions)

    def test_the_grayscale_is_built_once_per_frame(self) -> None:
        from modules.vision.drive import material_locate as module

        calls = {"count": 0}
        original = module.to_grayscale

        def counting(frame):
            calls["count"] += 1
            return original(frame)

        locator = MaterialLocator(yolo=self.CountingYOLO(), focal_px=300,
                                  locator=RecordingLocator(
                                      regions=tuple(LockedRegion(160.0 + 160.0 * index, 240.0, 40.0)
                                                    for index in range(6))))
        module.to_grayscale = counting
        try:
            report = locator.locate(synthetic_rings())
        finally:
            module.to_grayscale = original

        self.assertEqual(len(report.regions), 6)
        self.assertEqual(calls["count"], 1,
                         "6 个候选圆也只该转一次灰度图（每转一次 12 ms）")

    def test_the_grayscale_is_not_built_when_there_is_nothing_to_measure(self) -> None:
        from modules.vision.drive import material_locate as module

        calls = {"count": 0}
        original = module.to_grayscale

        def counting(frame):
            calls["count"] += 1
            return original(frame)

        locator = MaterialLocator(yolo=self.CountingYOLO(), focal_px=300,
                                  locator=RecordingLocator(regions=()))
        module.to_grayscale = counting
        try:
            locator.locate(synthetic_rings())
        finally:
            module.to_grayscale = original
        self.assertEqual(calls["count"], 0,
                         "一个候选都没有时不该白转一张 1280×720 的灰度图")


class WorkingDistanceWindowTests(unittest.TestCase):
    """工作距离先验 → 霍夫半径窗口；没配就先验为空，行为与以前一致。"""
    def test_working_distance_is_turned_into_a_radius_window(self) -> None:
        locator = RecordingLocator()
        report = MaterialLocator(
            locator=locator, yolo=FakeYOLO(), focal_px=1012.0, depth_range_mm=(60.0, 80.0)
        ).locate(np.zeros((720, 1280, 3), dtype=np.uint8))

        window = locator.calls[0]
        self.assertAlmostEqual(window["min_radius"], 1012.0 * 30 / 160.0)
        self.assertAlmostEqual(window["max_radius"], 1012.0 * 30 / 120.0)
        self.assertEqual(report.radius_window_px, (window["min_radius"], window["max_radius"]))
        self.assertEqual(report.focal_px, 1012.0)

    def test_without_a_prior_the_locator_keeps_its_own_window(self) -> None:
        locator = RecordingLocator()
        report = MaterialLocator(locator=locator, yolo=FakeYOLO(), focal_px=500.0).locate(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        self.assertEqual(locator.calls, [{}])
        self.assertIsNone(report.radius_window_px)

    def test_the_window_scales_with_the_frame_width(self) -> None:
        locator = RecordingLocator()
        MaterialLocator(
            locator=locator, yolo=FakeYOLO(), depth_range_mm=(60.0, 80.0)
        ).locate(np.zeros((360, 640, 3), dtype=np.uint8))
        window = locator.calls[0]
        self.assertAlmostEqual(window["min_radius"], (1012.0 / 2) * 30 / 160.0)
        self.assertAlmostEqual(window["max_radius"], (1012.0 / 2) * 30 / 120.0)

    def test_invalid_depth_ranges_are_rejected(self) -> None:
        for depth in ((80.0, 60.0), (70.0, 70.0), (0.0, 80.0), 70.0):
            with self.subTest(depth=depth):
                with self.assertRaises(ValueError):
                    MaterialLocator(yolo=FakeYOLO(), focal_px=1012.0, depth_range_mm=depth)

    def test_parse_depth_range_reads_the_command_line_form(self) -> None:
        self.assertEqual(parse_depth_range("60,85"), (60.0, 85.0))
        self.assertEqual(parse_depth_range("60，85"), (60.0, 85.0))
        self.assertIsNone(parse_depth_range(None))
        self.assertIsNone(parse_depth_range(""))
        for bad in ("60", "85,60", "a,b"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_depth_range(bad)


class BestCircleTests(unittest.TestCase):
    """YOLO 沉默时不能随便挑一个圆：那正是距离跳变的来源。"""

    @staticmethod
    def scene():
        """一帧里有：单色亮圆面、内部不匀的干扰圆、平坦背景。"""
        frame = np.full((480, 640), 60.0)
        ys, xs = np.ogrid[:480, :640]
        frame[240:250, 120:520] = 250.0
        frame[(xs - 320) ** 2 + (ys - 240) ** 2 <= 70**2] = 200.0
        return np.repeat(frame.astype(np.uint8)[:, :, None], 3, axis=2), (
            LockedRegion(center_x=320.0, center_y=240.0, radius=70.0),
            LockedRegion(center_x=320.0, center_y=245.0, radius=140.0),
            LockedRegion(center_x=560.0, center_y=400.0, radius=60.0),
        )

    def test_best_picks_the_most_face_like_circle_when_yolo_is_silent(self) -> None:
        frame, regions = self.scene()
        report = MaterialLocator(
            locator=RecordingLocator(regions), yolo=FakeYOLO(), focal_px=500.0
        ).locate(frame)

        self.assertEqual(len(report.regions), 3)
        self.assertEqual(len(report.faces), 1)
        self.assertEqual(report.best.region, regions[0])
        self.assertAlmostEqual(report.best.distance_mm, 500.0 * 30 / 140.0)

    def test_best_is_none_when_nothing_looks_like_a_top_face(self) -> None:
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        report = MaterialLocator(
            locator=RecordingLocator((LockedRegion(320.0, 240.0, 70.0),)),
            yolo=FakeYOLO(),
            focal_px=500.0,
        ).locate(frame)

        self.assertTrue(report.present)
        self.assertEqual(report.faces, ())
        self.assertIsNone(report.best)
        self.assertIn("距离不可用", report.summary())

    def test_classification_still_wins_over_uniformity(self) -> None:
        frame, regions = self.scene()
        hit = Detection(x1=530, y1=370, x2=590, y2=430, score=0.8, class_id=2, name="blue")
        report = MaterialLocator(
            locator=RecordingLocator(regions), yolo=FakeYOLO([hit]), focal_px=500.0
        ).locate(frame)

        self.assertEqual(report.best.region, regions[2])
        self.assertEqual(report.best.name, "blue")

    def test_summary_shows_the_window_and_the_chosen_circle(self) -> None:
        frame, regions = self.scene()
        report = MaterialLocator(
            locator=RecordingLocator(regions), yolo=FakeYOLO(), focal_px=500.0,
            depth_range_mm=(60.0, 80.0),
        ).locate(frame)
        summary = report.summary()
        self.assertIn("半径窗口", summary)
        self.assertIn("采用：", summary)
        self.assertIn("接受 1 个", summary)
        self.assertIn("丢掉 2 个", summary)


class RelaxedFallbackTests(unittest.TestCase):
    """离焦画面：严格霍夫参数找不到顶面时，自动放松阈值再来一次。"""

    @staticmethod
    def soft_frame():
        import cv2

        frame = np.full((480, 640), 70, dtype=np.float32)
        ys, xs = np.ogrid[:480, :640]
        frame[(xs - 320) ** 2 + (ys - 240) ** 2 <= 90**2] = 210
        blurred = cv2.GaussianBlur(frame, (0, 0), 8.0)
        return cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def test_a_soft_frame_is_locked_through_the_relaxed_locator(self) -> None:
        frame = self.soft_frame()
        strict = HoughRingLocator(param1=100.0, min_radius=60, max_radius=130, min_distance=100)
        self.assertEqual(len(strict.locate(frame)), 0)

        report = MaterialLocator(locator=strict, yolo=FakeYOLO(), focal_px=500.0).locate(frame)
        self.assertEqual(len(report.regions), 1)
        self.assertIsNotNone(report.best)
        self.assertAlmostEqual(report.best.distance_mm, 500.0 * 30 / 180.0, delta=10.0)

    def test_a_sharp_frame_never_needs_the_fallback(self) -> None:
        frame = synthetic_rings()
        recorder = RecordingLocator(RingLock(()).regions)
        MaterialLocator(locator=recorder, yolo=FakeYOLO(), focal_px=300.0).locate(frame)
        self.assertEqual(recorder.calls, [{}])

    def test_an_injected_locator_without_relaxed_is_left_alone(self) -> None:
        report = MaterialLocator(
            locator=FixedCircleLocator(), yolo=FakeYOLO(), focal_px=300.0
        ).locate(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(len(report.regions), 1)


if __name__ == "__main__":
    unittest.main()
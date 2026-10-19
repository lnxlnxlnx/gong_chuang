"""叠放位识别：中环定位、双基准线偏角、平行性与毫米位移。

这一组用 :func:`modules.vision.drive.stacked_ring.synthetic_stacked` 生成的合成帧，
**不开摄像头、不读图片**。

重点钉住三件事：

1. **偏角的符号**。画面 y 轴向下，"右端抬高为正"要求取反号；早先合成图与检测器
   用了相反的约定，测出来是 -4.57° 而不是 +5°——这条用例就是那次修正的回归。
2. **两条线不平行时结论不可信**。它们本该平行，夹角超限说明至少有一条认错了。
3. **没标定时尺度来源要如实标注**。从环半径反推的尺度能用但不可信，
   结果里必须写明 ``scale_source == "ring"``。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from modules.vision.config import StackedRingSettings, StationGeometrySettings
from modules.vision.drive.stacked_ring import (
    SCALE_FROM_MM_PER_PX,
    SCALE_FROM_RING,
    SCALE_UNKNOWN,
    ReferenceLine,
    StackedRingDetector,
    _combine_lines,
    _line_quality,
    edge_support,
    synthetic_stacked,
)

#: 合成叠放位的中环圆心。
CENTER = (640.0, 360.0)

#: 合成图与比例尺：0.5 mm/px，于是 75 mm 的基准线偏移正好是 150 px。
MM_PER_PX = 0.5


def stacked_settings(**overrides) -> StackedRingSettings:
    values = {
        "middle_reference_px": CENTER,
        "middle_search_radius_px": 260.0,
    }
    values.update(overrides)
    return StackedRingSettings(**values)


def detector(**overrides) -> StackedRingDetector:
    return StackedRingDetector(
        stacked=stacked_settings(**overrides),
        geometry=StationGeometrySettings(mm_per_px=MM_PER_PX),
    )


def line(angle_deg: float, points: int = 100, span: float = 500.0) -> ReferenceLine:
    """造一条合成基准线。"""
    return ReferenceLine(
        point=(0.0, 0.0),
        direction=(1.0, 0.0),
        angle_deg=angle_deg,
        points=points,
        x_span_px=span,
    )


class CombineLinesTests(unittest.TestCase):
    def test_no_lines_gives_no_yaw(self) -> None:
        self.assertEqual(_combine_lines(None, None), (None, None))

    def test_two_lines_are_averaged(self) -> None:
        yaw, disagreement = _combine_lines(line(2.0), line(4.0))
        self.assertAlmostEqual(yaw, 3.0, delta=1e-9)
        self.assertAlmostEqual(disagreement, 2.0, delta=1e-9)

    def test_one_line_is_used_directly_and_has_no_disagreement(self) -> None:
        """只有一条线时夹角必须是 ``None``（不是 0）——"只有一条"和"两条完全平行"是两回事。"""
        yaw, disagreement = _combine_lines(line(2.0), None)
        self.assertAlmostEqual(yaw, 2.0, delta=1e-9)
        self.assertIsNone(disagreement)
        yaw, disagreement = _combine_lines(None, line(-3.0))
        self.assertAlmostEqual(yaw, -3.0, delta=1e-9)
        self.assertIsNone(disagreement)


class LineQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = stacked_settings()

    def test_no_lines_score_zero(self) -> None:
        self.assertEqual(_line_quality(None, None, self.settings), 0.0)

    def test_one_line_scores_a_half(self) -> None:
        self.assertAlmostEqual(_line_quality(line(0.0), None, self.settings), 0.5, delta=1e-9)

    def test_perfectly_parallel_lines_score_one(self) -> None:
        self.assertAlmostEqual(_line_quality(line(3.0), line(3.0), self.settings), 1.0, delta=1e-9)

    def test_two_lines_never_score_below_one_line(self) -> None:
        """**连续性**：夹角越大质量越低，但永远不比"只有一条线"更差。

        早先的写法在夹角刚好等于上限时给 0.1，比只有一条线的 0.5 还低——
        而"两条都快不平行了"和"只看到一条"本来就是一回事，不该有这种断崖。
        """
        limit = self.settings.max_line_disagreement_deg
        for angle in (0.0, limit / 2, limit, limit * 1.5, limit * 3):
            with self.subTest(angle=angle):
                self.assertGreaterEqual(
                    _line_quality(line(0.0), line(angle), self.settings), 0.5 - 1e-9
                )

    def test_the_quality_decreases_with_the_disagreement(self) -> None:
        low = _line_quality(line(0.0), line(0.5), self.settings)
        high = _line_quality(line(0.0), line(1.5), self.settings)
        self.assertGreater(low, high)


class EdgeSupportTests(unittest.TestCase):
    def test_a_perfect_ring_has_full_support(self) -> None:
        """画一个圆环，它自己的边缘图上的支撑率应当是 1.0。"""
        import cv2

        frame = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(frame, (200, 200), 60, 255, 2)
        edges = cv2.Canny(frame, 50.0, 150.0)
        support = edge_support(edges, (200.0, 200.0), 60.0)
        self.assertGreater(support, 0.95)

    def test_a_wrong_radius_has_low_support(self) -> None:
        """半径差得远时支撑率应当很低——这正是它能把假圆筛掉的原因。"""
        import cv2

        frame = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(frame, (200, 200), 60, 255, 2)
        edges = cv2.Canny(frame, 50.0, 150.0)
        self.assertLess(edge_support(edges, (200.0, 200.0), 120.0), 0.2)

    def test_a_degenerate_circle_scores_zero(self) -> None:
        edges = np.zeros((100, 100), dtype=np.uint8)
        self.assertEqual(edge_support(edges, (50.0, 50.0), 0.0), 0.0)

    def test_an_empty_edge_image_scores_zero(self) -> None:
        self.assertEqual(edge_support(np.zeros((0, 0), dtype=np.uint8), (0.0, 0.0), 10.0), 0.0)


class DetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = detector()
        self.frame = synthetic_stacked(center=CENTER)

    def test_a_synthetic_station_is_detected(self) -> None:
        detection = self.detector.detect(self.frame)
        self.assertTrue(detection.detected, detection.detail)
        self.assertIsNotNone(detection.middle_center_px)
        self.assertAlmostEqual(detection.middle_center_px[0], CENTER[0], delta=4.0)
        self.assertAlmostEqual(detection.middle_center_px[1], CENTER[1], delta=4.0)

    def test_both_reference_lines_are_found(self) -> None:
        detection = self.detector.detect(self.frame)
        self.assertEqual(detection.line_count, 2)
        self.assertIsNotNone(detection.top_line)
        self.assertIsNotNone(detection.bottom_line)

    def test_a_level_station_reports_a_zero_yaw(self) -> None:
        detection = self.detector.detect(self.frame)
        self.assertIsNotNone(detection.station_yaw_deg)
        self.assertAlmostEqual(detection.station_yaw_deg, 0.0, delta=0.6)

    def test_a_tilt_reports_a_positive_yaw_when_the_right_end_rises(self) -> None:
        """**符号回归**：右端抬高为正。改之前这里读出的是 -4.57°。"""
        tilted = synthetic_stacked(center=CENTER, yaw_deg=5.0)
        detection = self.detector.detect(tilted)
        self.assertTrue(detection.detected, detection.detail)
        self.assertIsNotNone(detection.station_yaw_deg)
        self.assertGreater(detection.station_yaw_deg, 4.0)
        self.assertLess(detection.station_yaw_deg, 6.0)

    def test_a_negative_tilt_reports_a_negative_yaw(self) -> None:
        tilted = synthetic_stacked(center=CENTER, yaw_deg=-5.0)
        detection = self.detector.detect(tilted)
        self.assertTrue(detection.detected, detection.detail)
        self.assertLess(detection.station_yaw_deg, -4.0)

    def test_the_scale_comes_from_mm_per_px_when_configured(self) -> None:
        detection = self.detector.detect(self.frame)
        self.assertEqual(detection.scale_source, SCALE_FROM_MM_PER_PX)
        self.assertTrue(detection.scale_trustworthy)
        self.assertAlmostEqual(detection.scale_mm_per_px, MM_PER_PX, delta=1e-9)

    def test_a_translation_shows_up_in_the_millimetre_errors(self) -> None:
        shifted = synthetic_stacked(center=(CENTER[0] + 20.0, CENTER[1] + 15.0))
        detection = self.detector.detect(shifted)
        self.assertTrue(detection.detected, detection.detail)
        self.assertGreater(detection.error_x_mm, 0.0, "右移应当给出正的 error_x_mm")
        self.assertGreater(detection.error_y_mm, 0.0, "下移应当给出正的 error_y_mm（y 轴向下）")
        # (20, 15) px @ 0.5 mm/px = (10, 7.5) mm，模约 12.5 mm。
        self.assertAlmostEqual(detection.position_error_mm, 12.5, delta=1.5)

    def test_a_ring_without_lines_cannot_give_a_direction(self) -> None:
        """没有基准线就定不了方向——必须报未检出，但仍然报出找到的中环便于排查。"""
        ring_only = synthetic_stacked(center=CENTER, draw_lines=False)
        detection = self.detector.detect(ring_only)
        self.assertFalse(detection.detected)
        self.assertIsNotNone(detection.middle_center_px, "仍然应当报出找到的中环")
        self.assertIn("基准线", detection.detail)

    def test_a_blank_frame_reports_not_detected_with_a_reason(self) -> None:
        detection = self.detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertFalse(detection.detected)
        self.assertIsNone(detection.middle_center_px)
        self.assertTrue(detection.detail)

    def test_the_input_frame_is_not_modified(self) -> None:
        before = self.frame.copy()
        detection = self.detector.detect(self.frame)
        marked = self.detector.draw(self.frame, detection)
        self.assertTrue(np.array_equal(self.frame, before), "draw 不该改输入帧")
        self.assertFalse(np.array_equal(marked, before))

    def test_drawing_a_miss_does_not_raise(self) -> None:
        detection = self.detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(self.detector.draw(self.frame, detection).shape, self.frame.shape)

    def test_empty_and_wrong_channel_frames_are_rejected(self) -> None:
        for bad, name in (
            (None, "None"),
            (np.zeros((0, 0, 3), dtype=np.uint8), "空数组"),
            (np.zeros((10, 10), dtype=np.uint8), "单通道"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.detector.detect(bad)

    def test_stability_needs_the_full_window(self) -> None:
        self.assertFalse(self.detector.detect(self.frame).stable)
        for _ in range(stacked_settings().stable_frame_count):
            last = self.detector.detect(self.frame)
        self.assertTrue(last.stable)

    def test_reset_clears_the_stability_window(self) -> None:
        for _ in range(stacked_settings().stable_frame_count):
            self.detector.detect(self.frame)
        self.detector.reset()
        self.assertFalse(self.detector.detect(self.frame).stable)

    def test_processing_time_is_recorded(self) -> None:
        self.assertGreater(self.detector.detect(self.frame).processing_ms, 0.0)

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs, name in (
            ({"middle_min_radius_px": 90.0, "middle_max_radius_px": 60.0}, "中环半径窗口颠倒"),
            ({"middle_min_radius_px": 60.0, "middle_max_radius_px": 60.0}, "中环半径窗口相等"),
            ({"line_offset_mm": 0.0}, "基准线偏移为 0"),
            ({"stable_frame_count": 0}, "稳定帧数为 0"),
            ({"min_line_points": 1}, "直线点数下限低于 2"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    StackedRingDetector(stacked=stacked_settings(**kwargs))


class UncalibratedScaleTests(unittest.TestCase):
    """没标定时的取舍：能用，但**必须如实标注不可信**。"""

    def test_the_scale_is_derived_from_the_ring_and_flagged(self) -> None:
        """从环半径反推的尺度：来源写 ``ring``，并且 ``scale_trustworthy`` 为假。

        这里直接测 :meth:`_resolve_scale`，**不测"能不能检出"**：合成图的两条基准线是按
        0.5 mm/px 摆在 150 px 处的，而从环反推的尺度（约 0.32 mm/px）会把搜索位置算到
        234 px 去——那两条线本来就该找不到。这恰好说明"没标定的尺度不可信"不是一句
        空话，所以断言的是来源与可信标志，不是检出与否。
        """
        uncalibrated = StackedRingDetector(
            stacked=stacked_settings(),
            geometry=StationGeometrySettings(mm_per_px=None, image_to_station_affine=None),
        )
        scale, source = uncalibrated._resolve_scale(70.0)
        self.assertEqual(source, SCALE_FROM_RING)
        self.assertIsNotNone(scale)
        # 外径 95 mm -> 半径 47.5 mm；除以实测 70 px 就是临时尺度。
        self.assertAlmostEqual(scale, 47.5 / 70.0, delta=1e-9)

        detection = uncalibrated.detect(synthetic_stacked(center=CENTER))
        self.assertEqual(detection.scale_source, SCALE_FROM_RING)
        self.assertTrue(detection.scale_known, "从环反推的尺度仍然算有尺度")
        self.assertFalse(detection.scale_trustworthy, "但它不可信，必须标出来")

    def test_a_calibrated_detector_marks_its_scale_trustworthy(self) -> None:
        """配了比例尺的检测器必须把尺度标成可信——否则上面那条断言没有对照。"""
        detection = detector().detect(synthetic_stacked(center=CENTER))
        self.assertTrue(detection.detected, detection.detail)
        self.assertTrue(detection.scale_trustworthy)

    def test_an_affine_scale_is_preferred_over_mm_per_px(self) -> None:
        """仿射矩阵优先：工位被斜着看时横竖尺度不一样，只有它能如实换算。"""
        from modules.vision.drive.stacked_ring import SCALE_FROM_AFFINE

        affine = StackedRingDetector(
            stacked=stacked_settings(),
            geometry=StationGeometrySettings(
                mm_per_px=MM_PER_PX, image_to_station_affine=((1.0, 0.0), (0.0, 0.5))
            ),
        )
        detection = affine.detect(synthetic_stacked(center=CENTER))
        self.assertEqual(detection.scale_source, SCALE_FROM_AFFINE)
        self.assertTrue(detection.scale_trustworthy)

    def test_no_scale_at_all_means_no_millimetres(self) -> None:
        """环半径也拿不到时（半径 0）尺度来源是 none，毫米字段必须是 None。"""
        detector_instance = StackedRingDetector(
            stacked=stacked_settings(),
            geometry=StationGeometrySettings(mm_per_px=None, image_to_station_affine=None),
        )
        scale, source = detector_instance._resolve_scale(0.0)
        self.assertIsNone(scale)
        self.assertEqual(source, SCALE_UNKNOWN)


class SummaryTests(unittest.TestCase):
    def test_the_summary_of_a_miss_explains_why(self) -> None:
        detection = detector().detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertIn("未识别到", detection.summary())
        self.assertIn(detection.detail, detection.summary())

    def test_the_summary_of_a_hit_carries_the_numbers(self) -> None:
        detection = detector().detect(synthetic_stacked(center=CENTER))
        summary = detection.summary()
        self.assertIn("叠放位", summary)
        self.assertIn("偏角", summary)
        self.assertIn("基准线 2 条", summary)

    def test_a_single_line_is_reported_as_such(self) -> None:
        """只有一条线时 ``line_count`` 是 1，夹角是 ``None``（不是 0）。"""
        from modules.vision.drive.stacked_ring import StackedRingDetection

        detection = StackedRingDetection(detected=True, top_line=line(0.0))
        self.assertEqual(detection.line_count, 1)
        self.assertIsNone(detection.line_disagreement_deg)


if __name__ == "__main__":
    unittest.main()

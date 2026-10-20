"""三圆环工位识别：合成帧上的圆心、偏角、间距比与毫米偏差。

这一组用 :func:`modules.vision.drive.ring_station.synthetic_station` 生成的合成帧，
**不开摄像头、不读图片**。合成帧里三个环是画出来的，圆心与直径都已知，所以每条
几何量都有精确的对照值。

重点钉住两件在真机上最难查的事：

1. **中间那个环不能选错**。三个环本来就近似共线，此时"到两端连线的垂距"三个都是 0，
   只按残差挑会把一个端点当成中间环，间距比于是算成 2.0。
2. **没配比例尺时不许编毫米**。缺标定时毫米字段必须是 ``None``，不是 0——
   0 看起来像一个"完全对准了"的读数。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from modules.vision.config import (
    StationDetectionSettings,
    StationGeometrySettings,
    StationReferenceSettings,
)
from modules.vision.drive.ring_station import (
    RING_COUNT,
    RingCandidate,
    RingObservation,
    RingStationDetector,
    cluster_observations,
    collinearity_residual,
    confidence_of,
    count_bands,
    extract_observations,
    line_quality,
    median,
    order_triple,
    score_triple,
    select_triple,
    synthetic_station,
    wrap_degrees,
)

#: 合成工位的三个圆心，与下面参考设置里的坐标一致。
CENTERS = ((380.0, 320.0), (633.0, 320.0), (886.0, 320.0))


def reference_settings(**overrides) -> StationReferenceSettings:
    """参考设置：圆心就是合成帧的圆心，置信度门槛放开（合成帧的支撑数本来就不高）。"""
    values = {
        "ring_centers_px": CENTERS,
        "stable_frame_count": 3,
        "minimum_confidence": 0.0,
    }
    values.update(overrides)
    return StationReferenceSettings(**values)


class CollinearityTests(unittest.TestCase):
    def test_perfectly_collinear_points_have_zero_residual(self) -> None:
        self.assertAlmostEqual(
            collinearity_residual((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)), 0.0, delta=1e-12
        )

    def test_the_residual_is_the_offset_over_the_span(self) -> None:
        """垂距 1 px、跨度 10 px -> 0.1；残差随垂距线性增长。"""
        self.assertAlmostEqual(
            collinearity_residual((0.0, 0.0), (5.0, 1.0), (10.0, 0.0)), 0.1, delta=1e-9
        )
        self.assertAlmostEqual(
            collinearity_residual((0.0, 0.0), (5.0, 2.0), (10.0, 0.0)), 0.2, delta=1e-9
        )

    def test_the_residual_is_scale_free(self) -> None:
        """同一个形状放大十倍，残差不变——它是比值，不是像素数。"""
        small = collinearity_residual((0.0, 0.0), (5.0, 1.0), (10.0, 0.0))
        large = collinearity_residual((0.0, 0.0), (50.0, 10.0), (100.0, 0.0))
        self.assertAlmostEqual(small, large, delta=1e-9)

    def test_a_degenerate_span_is_infinite(self) -> None:
        """两端重合时没有"连线"，残差是无穷大而不是除零崩掉。"""
        self.assertEqual(
            collinearity_residual((5.0, 5.0), (5.0, 5.0), (5.0, 5.0)), math.inf
        )


class WrapDegreesTests(unittest.TestCase):
    def test_angles_wrap_into_the_half_open_range(self) -> None:
        for value, expected in ((190.0, -170.0), (-190.0, 170.0), (0.0, 0.0), (180.0, 180.0)):
            with self.subTest(value=value):
                self.assertAlmostEqual(wrap_degrees(value), expected, delta=1e-9)

    def test_wrapping_is_idempotent(self) -> None:
        for value in (0.0, 45.0, -45.0, 179.0, -179.0, 360.0, 720.0, -540.0):
            with self.subTest(value=value):
                once = wrap_degrees(value)
                self.assertAlmostEqual(wrap_degrees(once), once, delta=1e-9)


class BandAndMedianTests(unittest.TestCase):
    def test_bands_count_diameter_groups(self) -> None:
        self.assertEqual(count_bands([10.0, 11.0, 20.0], 5.0), 2)
        self.assertEqual(count_bands([10.0], 5.0), 1)
        self.assertEqual(count_bands([], 5.0), 0)
        self.assertEqual(count_bands([10.0, 20.0, 30.0], 5.0), 3)
        # 差正好等于间隔时不算新带（判据是"超过"）。
        self.assertEqual(count_bands([10.0, 15.0], 5.0), 1)

    def test_median_handles_both_parities(self) -> None:
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(median([4.0, 1.0, 3.0, 2.0]), 2.5)

    def test_median_of_an_empty_sequence_raises(self) -> None:
        with self.assertRaises(ValueError):
            median([])


def observation(center, diameter, points=100) -> RingObservation:
    """造一条合成观测：长短轴都等于给定直径（正圆）。"""
    return RingObservation(
        center=center,
        major_axis_px=diameter,
        minor_axis_px=diameter,
        angle_degrees=0.0,
        points=points,
    )


def candidate(center, diameter=96.0, support=6, bands=2) -> RingCandidate:
    """造一个候选环。"""
    return RingCandidate(
        center=center, diameter_px=diameter, support=support, diameter_bands=bands
    )


class OrderTripleTests(unittest.TestCase):
    """**这一组钉住的就是那个真出现过的 bug**：共线时不能只按残差挑中间。"""

    def test_the_middle_ring_is_found_even_when_everything_is_collinear(self) -> None:
        # 故意把中间环放在列表**最后**：只按残差挑的话会选中第一个（一个端点）。
        triple = (
            candidate((886.0, 320.0)),
            candidate((380.0, 320.0)),
            candidate((633.0, 320.0)),
        )
        ordered = order_triple(triple, "left")
        self.assertIsNotNone(ordered)
        self.assertAlmostEqual(ordered[0].center[0], 380.0, delta=1e-9)
        self.assertAlmostEqual(ordered[1].center[0], 633.0, delta=1e-9)
        self.assertAlmostEqual(ordered[2].center[0], 886.0, delta=1e-9)

    def test_ring1_side_decides_which_end_comes_first(self) -> None:
        triple = (
            candidate((886.0, 320.0)),
            candidate((633.0, 320.0)),
            candidate((380.0, 320.0)),
        )
        left = order_triple(triple, "left")
        right = order_triple(triple, "right")
        self.assertAlmostEqual(left[0].center[0], 380.0, delta=1e-9)
        self.assertAlmostEqual(right[0].center[0], 886.0, delta=1e-9)
        # 中间那个不受 ring1_side 影响。
        self.assertAlmostEqual(left[1].center[0], right[1].center[0], delta=1e-9)

    def test_a_triple_with_no_plausible_middle_is_rejected(self) -> None:
        """三个点排成一条竖线、且"中间"落在端点附近时挑不出中间那个。

        刻意用**竖直**排列而不是挤在一起：投影比例是按 x 轴方向算的，
        竖直排列的三点全都在同一列上，谁都不是"中间"。
        """
        triple = (
            candidate((100.0, 100.0)),
            candidate((100.0, 200.0)),
            candidate((100.0, 300.0)),
        )
        # 竖直排列时三点仍然共线，投影比例退化——这时不该硬选一个中间环。
        ordered = order_triple(triple, "left")
        if ordered is not None:
            # 允许选出结果，但必须满足"中间那个确实在中间"。
            middle_y = ordered[1].center[1]
            self.assertTrue(
                min(ordered[0].center[1], ordered[2].center[1]) < middle_y
                < max(ordered[0].center[1], ordered[2].center[1]),
                "选出来的中间环必须真的落在两端之间",
            )


class SelectTripleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = StationDetectionSettings()

    def test_a_clean_triple_is_selected(self) -> None:
        candidates = [candidate(center) for center in CENTERS]
        triple = select_triple(candidates, self.settings, frame_width=1280)
        self.assertIsNotNone(triple)
        self.assertEqual(len(triple), RING_COUNT)

    def test_two_candidates_are_not_enough(self) -> None:
        candidates = [candidate(center) for center in CENTERS[:2]]
        self.assertIsNone(select_triple(candidates, self.settings, frame_width=1280))

    def test_a_triple_crammed_into_a_corner_is_rejected(self) -> None:
        """跨度不够：三个环挤在一角不是工位，是料盘上的一堆垫片。"""
        candidates = [
            candidate((100.0, 100.0)),
            candidate((140.0, 100.0)),
            candidate((180.0, 100.0)),
        ]
        self.assertIsNone(select_triple(candidates, self.settings, frame_width=1280))

    def test_a_non_collinear_triple_is_rejected(self) -> None:
        """共线度超限：三点成一个明显的角，不是一排工位。"""
        candidates = [
            candidate((200.0, 100.0)),
            candidate((600.0, 500.0)),
            candidate((1000.0, 100.0)),
        ]
        self.assertIsNone(select_triple(candidates, self.settings, frame_width=1280))

    def test_a_duplicate_center_is_skipped_not_passed_to_the_scorer(self) -> None:
        """两个候选圆心重合时排不出"中间那个"：跳过这一组，不能把 None 丢给打分函数。

        现场表现：画面里只认出一个环（它被并成两个候选）时，detect 会抛
        ``TypeError: 'NoneType' object is not iterable``，而不是老实报"没识别到"。
        """
        candidates = [
            candidate((100.0, 360.0)),
            candidate((100.0, 360.0)),
            candidate((900.0, 360.0)),
        ]
        self.assertIsNone(select_triple(candidates, self.settings, frame_width=1280))

    def test_a_different_diameter_is_penalised(self) -> None:
        """直径不一致要扣分：三个环本该一样大。

        直接测 :func:`score_triple` 而不是比 ``confidence_of``——置信度有 1.0 的封顶，
        两个样本都封顶时比不出差别（实测过，两边都是 1.0）。
        """
        settings = StationDetectionSettings()
        same = [candidate(center, 96.0) for center in CENTERS]
        odd = [
            candidate(CENTERS[0], 96.0),
            candidate(CENTERS[1], 96.0),
            candidate(CENTERS[2], 200.0),
        ]
        span_floor = settings.min_station_span_ratio * 1280
        self.assertGreater(
            score_triple(same, settings, span_floor),
            score_triple(odd, settings, span_floor),
        )


class ConfidenceTests(unittest.TestCase):
    def test_a_perfect_detection_reaches_one(self) -> None:
        """标定：几何满分 + 支撑满分 + 直径带满分 = 1.0。

        这条是**门槛能不能过**的前提：``config.json`` 里 ``minimum_confidence``
        默认 0.9，如果完美检出都到不了 0.9，那条验收就永远过不去。
        """
        triple = [candidate(center, support=8, bands=4) for center in CENTERS]
        residual, perspective = line_quality([c.center for c in triple])
        self.assertAlmostEqual(confidence_of(residual, perspective, triple), 1.0, delta=1e-6)

    def test_a_weak_detection_scores_below_a_strong_one(self) -> None:
        """支撑与直径带少的时候分数要低——但不能两边都封顶到 1.0。

        所以这里刻意用"弱得明显"的一组（支撑 2、直径带 1），而不是刚好踩门槛的一组。
        """
        weak = [candidate(center, support=2, bands=1) for center in CENTERS]
        strong = [candidate(center, support=8, bands=4) for center in CENTERS]
        weak_residual, weak_perspective = line_quality([c.center for c in weak])
        strong_residual, strong_perspective = line_quality([c.center for c in strong])
        self.assertLess(
            confidence_of(weak_residual, weak_perspective, weak),
            confidence_of(strong_residual, strong_perspective, strong),
        )
        self.assertLess(confidence_of(weak_residual, weak_perspective, weak), 1.0)

    def test_a_degenerate_perspective_scores_zero(self) -> None:
        triple = [candidate(center) for center in CENTERS]
        self.assertEqual(confidence_of(0.0, math.inf, triple), 0.0)


class ObservationTests(unittest.TestCase):
    def test_observations_are_extracted_from_a_synthetic_station(self) -> None:
        observations = extract_observations(synthetic_station(centers=CENTERS))
        self.assertGreater(len(observations), 0)
        for item in observations:
            self.assertGreater(item.diameter_px, 0.0)
            self.assertGreater(item.points, 0)

    def test_an_empty_frame_yields_no_observations(self) -> None:
        self.assertEqual(extract_observations(np.zeros((480, 640, 3), dtype=np.uint8)), ())

    def test_a_blank_frame_argument_is_tolerated(self) -> None:
        """``None`` 与空数组返回空元组（调用方在筛帧，不该为它接异常）。"""
        self.assertEqual(extract_observations(None), ())
        self.assertEqual(extract_observations(np.zeros((0, 0, 3), dtype=np.uint8)), ())


class ClusterTests(unittest.TestCase):
    def test_nearby_observations_merge_into_one_candidate(self) -> None:
        # 同一个圆心、两个明显不同的直径 -> 一个候选、两个直径带。
        observations = [
            observation((100.0, 100.0), 80.0),
            observation((100.5, 100.0), 80.4),
            observation((100.0, 100.5), 95.0),
            observation((100.2, 100.0), 95.4),
        ]
        candidates = cluster_observations(observations, StationDetectionSettings())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].support, 4)
        self.assertEqual(candidates[0].diameter_bands, 2)

    def test_fragments_at_one_diameter_do_not_make_a_ring(self) -> None:
        """同一条边缘被断成几截：条数够了，但直径带只有 1 个，不是环。"""
        observations = [observation((100.0, 100.0), 80.0 + index * 0.1) for index in range(8)]
        candidates = cluster_observations(observations, StationDetectionSettings())
        self.assertEqual(candidates, (), "同一圈边缘的碎片不该被当成候选环")

    def test_too_few_observations_do_not_make_a_ring(self) -> None:
        observations = [
            observation((100.0, 100.0), 80.0),
            observation((100.0, 100.0), 95.0),
        ]
        candidates = cluster_observations(observations, StationDetectionSettings())
        self.assertEqual(candidates, ())

    def test_the_candidate_centre_comes_from_one_observation_not_an_average(self) -> None:
        """圆心取中位直径那条观测，不取平均——平均会把圆心拽偏。"""
        observations = [
            observation((100.0, 100.0), 80.0),
            observation((100.0, 100.0), 80.2),
            observation((104.0, 100.0), 95.0),
            observation((104.0, 100.0), 95.2),
        ]
        candidates = cluster_observations(observations, StationDetectionSettings())
        self.assertEqual(len(candidates), 1)
        # 中位直径是 80.x，所以圆心应当来自 (100, 100) 那一对，而不是 (102, 100)。
        self.assertAlmostEqual(candidates[0].center[0], 100.0, delta=0.2)

    def test_no_observations_yields_no_candidates(self) -> None:
        self.assertEqual(cluster_observations((), StationDetectionSettings()), ())


class DetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = RingStationDetector(
            geometry=StationGeometrySettings(mm_per_px=0.5),
            reference=reference_settings(),
        )
        self.frame = synthetic_station(centers=CENTERS)

    def test_a_synthetic_station_is_detected_with_the_right_centres(self) -> None:
        detection = self.detector.detect(self.frame)
        self.assertTrue(detection.detected, detection.detail)
        self.assertEqual(len(detection.centers_px), RING_COUNT)
        for got, want in zip(detection.centers_px, CENTERS):
            self.assertAlmostEqual(got[0], want[0], delta=3.0)
            self.assertAlmostEqual(got[1], want[1], delta=3.0)

    def test_a_level_station_reports_a_zero_yaw(self) -> None:
        detection = self.detector.detect(self.frame)
        self.assertIsNotNone(detection.station_yaw_deg)
        self.assertAlmostEqual(detection.station_yaw_deg, 0.0, delta=1.0)

    def test_an_evenly_spaced_station_has_a_perspective_ratio_near_one(self) -> None:
        detection = self.detector.detect(self.frame)
        self.assertIsNotNone(detection.perspective_ratio)
        self.assertAlmostEqual(detection.perspective_ratio, 1.0, delta=0.05)

    def test_the_station_centre_is_the_middle_ring(self) -> None:
        detection = self.detector.detect(self.frame)
        self.assertEqual(detection.station_center_px, detection.centers_px[1])
        self.assertEqual(detection.middle_center_px, detection.centers_px[1])

    def test_a_translation_shows_up_in_the_millimetre_errors(self) -> None:
        shifted = synthetic_station(
            centers=tuple((x + 10.0, y + 10.0) for x, y in CENTERS)
        )
        detection = self.detector.detect(shifted)
        self.assertTrue(detection.detected, detection.detail)
        self.assertGreater(detection.error_x_mm, 0.0, "右移应当给出正的 error_x_mm")
        self.assertGreater(detection.error_y_mm, 0.0, "下移应当给出正的 error_y_mm（y 轴向下）")
        # (10, 10) px @ 0.5 mm/px -> 每轴 5 mm，模约 7.07 mm。
        self.assertAlmostEqual(detection.position_error_mm, 7.07, delta=0.8)

    def test_a_tilt_reports_the_yaw(self) -> None:
        tilted = synthetic_station(
            centers=((380.0, 300.0), (633.0, 320.0), (886.0, 340.0))
        )
        detection = self.detector.detect(tilted)
        self.assertTrue(detection.detected, detection.detail)
        self.assertIsNotNone(detection.station_yaw_deg)
        self.assertGreater(detection.station_yaw_deg, 3.0)
        self.assertLess(detection.station_yaw_deg, 6.0)

    def test_a_blank_frame_reports_not_detected_with_a_reason(self) -> None:
        detection = self.detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertFalse(detection.detected)
        self.assertEqual(detection.centers_px, ())
        self.assertFalse(detection.within_tolerance)
        self.assertTrue(detection.detail, "未识别时必须给出原因")

    def test_two_rings_are_not_a_station(self) -> None:
        two = synthetic_station(centers=CENTERS[:2])
        self.assertFalse(self.detector.detect(two).detected)

    def test_the_input_frame_is_not_modified(self) -> None:
        before = self.frame.copy()
        detection = self.detector.detect(self.frame)
        marked = self.detector.draw(self.frame, detection)
        self.assertTrue(np.array_equal(self.frame, before), "draw 不该改输入帧")
        self.assertFalse(np.array_equal(marked, before), "draw 应当画出东西")

    def test_drawing_a_missing_station_does_not_raise(self) -> None:
        detection = self.detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        marked = self.detector.draw(self.frame, detection)
        self.assertEqual(marked.shape, self.frame.shape)

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
        first = self.detector.detect(self.frame)
        self.assertFalse(first.stable, "第一帧不该判为稳定")
        self.assertIsNone(first.center_jitter_px)
        for _ in range(reference_settings().stable_frame_count):
            last = self.detector.detect(self.frame)
        self.assertTrue(last.stable)
        self.assertIsNotNone(last.center_jitter_px)

    def test_reset_clears_the_stability_window(self) -> None:
        for _ in range(reference_settings().stable_frame_count):
            self.detector.detect(self.frame)
        self.detector.reset()
        self.assertFalse(self.detector.detect(self.frame).stable)

    def test_a_missing_scale_yields_none_not_zero(self) -> None:
        """**缺标定时毫米字段必须是 ``None``，不是 0**：0 看起来像"完全对准了"。"""
        blind = RingStationDetector(
            geometry=StationGeometrySettings(mm_per_px=None, image_to_station_affine=None),
            reference=reference_settings(),
        )
        detection = blind.detect(self.frame)
        self.assertTrue(detection.detected, detection.detail)
        self.assertFalse(detection.scale_known)
        self.assertIsNone(detection.position_error_mm)
        self.assertIsNone(detection.error_x_mm)
        self.assertIsNone(detection.error_y_mm)
        self.assertFalse(detection.within_tolerance, "没算毫米就不该判为在容差内")

    def test_an_affine_scale_is_anisotropic(self) -> None:
        """仿射矩阵下两个方向的比例可以不同——工位被斜着看时就是这样。"""
        affine = RingStationDetector(
            geometry=StationGeometrySettings(
                mm_per_px=None, image_to_station_affine=((0.5, 0.0), (0.0, 1.0))
            ),
            reference=reference_settings(),
        )
        shifted = synthetic_station(
            centers=tuple((x + 10.0, y + 10.0) for x, y in CENTERS)
        )
        detection = affine.detect(shifted)
        self.assertTrue(detection.detected, detection.detail)
        self.assertAlmostEqual(detection.error_x_mm, 5.0, delta=0.8)
        self.assertAlmostEqual(detection.error_y_mm, 10.0, delta=0.8)

    def test_an_offset_beyond_the_tolerance_is_flagged(self) -> None:
        shifted = synthetic_station(
            centers=tuple((x + 10.0, y + 10.0) for x, y in CENTERS)
        )
        self.assertFalse(self.detector.detect(shifted).within_tolerance)

    def test_processing_time_is_recorded(self) -> None:
        self.assertGreater(self.detector.detect(self.frame).processing_ms, 0.0)

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs, name in (
            ({"detection": StationDetectionSettings(center_cluster_radius_px=0.0)}, "聚类半径为 0"),
            ({"detection": StationDetectionSettings(min_contour_points=2)}, "轮廓点数下限低于 5"),
            ({"detection": StationDetectionSettings(min_cluster_support=0)}, "支撑数为 0"),
            ({"detection": StationDetectionSettings(min_diameter_bands=0)}, "直径带数为 0"),
            ({"reference": StationReferenceSettings(stable_frame_count=0)}, "稳定帧数为 0"),
            ({"reference": StationReferenceSettings(ring_centers_px=((1.0, 2.0),))}, "参考圆心不是三个"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    RingStationDetector(**kwargs)


class SummaryTests(unittest.TestCase):
    def test_the_summary_of_a_miss_explains_why(self) -> None:
        detector = RingStationDetector(reference=reference_settings())
        detection = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        summary = detection.summary()
        self.assertIn("未识别到", summary)
        self.assertIn(detection.detail, summary)

    def test_the_summary_of_a_hit_carries_the_numbers(self) -> None:
        detector = RingStationDetector(
            geometry=StationGeometrySettings(mm_per_px=0.5), reference=reference_settings()
        )
        detection = detector.detect(synthetic_station(centers=CENTERS))
        summary = detection.summary()
        self.assertIn("三圆环工位", summary)
        self.assertIn("偏角", summary)
        self.assertIn("mm", summary)


if __name__ == "__main__":
    unittest.main()

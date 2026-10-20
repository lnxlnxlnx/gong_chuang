"""环内数字与环心：顺序映射、相机系毫米、空环与缺比例尺的取舍。

这个模块的价值不在"能算"，而在**不猜**：数字靠纸上的顺序（可配置），但"环里到底有
没有印数字"必须单独验一次；验不过就不给可对准的读数，而不是按顺序硬报一个号。
所以这里的用例围绕三件事：

1. 左→右是否稳定映射成 1/2/3（顺序来自配置，不是代码常量）；
2. 毫米换算是否与 ``RegionReport`` 同一套坐标（原点在光轴、x 右、y 下）；
3. 空环、缺比例尺、没检出三种情况下，读数是不是 ``None`` 而不是一个像样的错数。
"""

from __future__ import annotations

import unittest

import numpy as np

from modules.vision.config import (
    RingDigitSettings,
    StationGeometrySettings,
    StationReferenceSettings,
    VisionConfig,
    _ring_digits,
    load_vision_config,
)
from modules.vision.drive.numbered_ring import (
    NumberedRingLocator,
    normalize_digit,
)
from modules.vision.drive.ring_station import RingStationDetector, synthetic_station

#: 合成工位：三个环横向排开、中间那个在光轴上。
CENTERS = ((300.0, 360.0), (640.0, 360.0), (980.0, 360.0))
SIZE = (1280, 720)

#: 合成帧里的环直径（像素）。
DIAMETER_PX = 120.0


def station_frame(*, digits: str = "123") -> np.ndarray:
    """白纸黑环的三环帧，环心画上 ``digits`` 里的字（``""`` 表示空环）。"""
    frame = synthetic_station(
        size=SIZE,
        centers=CENTERS,
        diameter_px=DIAMETER_PX,
        thickness=6,
        background=255,
        foreground=0,
    )
    if digits:
        import cv2

        for (center_x, center_y), digit in zip(CENTERS, digits):
            cv2.putText(
                frame,
                digit,
                (int(center_x) - 18, int(center_y) + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 0, 0),
                4,
            )
    return frame


def make_locator(*, mm_per_px: float | None = 0.5, digits=None) -> NumberedRingLocator:
    geometry = StationGeometrySettings(mm_per_px=mm_per_px)
    return NumberedRingLocator(
        station=RingStationDetector(
            geometry=geometry,
            reference=StationReferenceSettings(ring_centers_px=CENTERS),
        ),
        geometry=geometry,
        digits=digits,
    )


class OrderTests(unittest.TestCase):
    def test_left_to_right_is_one_two_three(self) -> None:
        result = make_locator().detect(station_frame())
        self.assertTrue(result.detected, result.detail)
        self.assertEqual([item.digit for item in result.rings], [1, 2, 3])
        self.assertEqual(result.order, (1, 2, 3))

    def test_digits_come_from_config_not_from_the_code(self) -> None:
        """换一张纸（顺序反了）只改配置：左端变成 3 号。"""
        result = make_locator(digits=RingDigitSettings(order=(3, 2, 1))).detect(station_frame())
        self.assertEqual([item.digit for item in result.rings], [3, 2, 1])
        # 检测器给出的圆心差零点几像素是正常的（它取中位直径那条观测）。
        self.assertAlmostEqual(result.rings[0].center_px[0], CENTERS[0][0], delta=1.0)

    def test_short_order_is_rejected_instead_of_inventing_a_digit(self) -> None:
        """配置少给一个数字时不许自己补：补出来的号看起来完全正常。"""
        with self.assertRaises(ValueError):
            make_locator(digits=RingDigitSettings(order=(1, 2))).detect(station_frame())

    def test_ring_index_keeps_the_detector_order(self) -> None:
        result = make_locator().detect(station_frame())
        self.assertEqual([item.ring_index for item in result.rings], [1, 2, 3])


class ReadingTests(unittest.TestCase):
    def test_camera_millimetres_use_the_image_centre_as_optical_axis(self) -> None:
        result = make_locator().detect(station_frame())
        # 中间环在光轴上，左右各 340 px × 0.5 mm/px = ±170 mm。
        middle = result.reading(2)
        self.assertIsNotNone(middle)
        self.assertAlmostEqual(middle[0], 0.0, delta=0.5)
        self.assertAlmostEqual(middle[1], 0.0, delta=0.5)
        left = result.reading(1)
        right = result.reading(3)
        self.assertAlmostEqual(left[0], -170.0, delta=0.5)
        self.assertAlmostEqual(right[0], +170.0, delta=0.5)

    def test_scale_is_reported_so_callers_can_tell_where_it_came_from(self) -> None:
        result = make_locator().detect(station_frame())
        self.assertTrue(result.scale_known)
        self.assertAlmostEqual(result.scale_mm_per_px, 0.5)

    def test_no_scale_means_no_reading_at_all(self) -> None:
        result = make_locator(mm_per_px=None).detect(station_frame())
        self.assertTrue(result.detected, "环还是认得的，只是没有毫米")
        self.assertFalse(result.scale_known)
        self.assertIsNone(result.rings[0].x_mm)
        self.assertIsNone(result.reading(2), "没有比例尺时不能给出可对准的读数")

    def test_unknown_digit_raises_before_anything_is_returned(self) -> None:
        result = make_locator().detect(station_frame())
        for bad in (0, -1, "1", 1.0, True):
            with self.subTest(digit=bad):
                with self.assertRaises(ValueError):
                    result.reading(bad)

    def test_missing_digit_is_never_reported_as_a_number(self) -> None:
        """空环：顺序上它是 2 号，但纸上没印数字，就不该给出可对准的读数。"""
        result = make_locator().detect(station_frame(digits=""))
        self.assertTrue(result.detected)
        self.assertFalse(any(item.digit_present for item in result.rings))
        self.assertIsNone(result.reading(2))
        self.assertIn("没有数字", result.summary())


class DegenerateTests(unittest.TestCase):
    def test_no_rings_is_a_normal_result(self) -> None:
        blank = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        result = make_locator().detect(blank)
        self.assertFalse(result.detected)
        self.assertEqual(result.rings, ())
        self.assertIn("未识别", result.summary())

    def test_none_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_locator().detect(None)

    def test_normalize_digit_rejects_bool_and_non_positive(self) -> None:
        self.assertEqual(normalize_digit(3), 3)
        for bad in (True, 0, -2, "2", 2.0):
            with self.subTest(digit=bad):
                with self.assertRaises(ValueError):
                    normalize_digit(bad)


class ConfigTests(unittest.TestCase):
    def test_shipped_config_carries_the_order(self) -> None:
        settings = load_vision_config()
        self.assertEqual(settings.ring_digits.order, (1, 2, 3))
        self.assertGreater(settings.ring_digits.inner_ratio, 0.0)

    def test_missing_block_falls_back_to_defaults(self) -> None:
        self.assertEqual(_ring_digits(None), RingDigitSettings())

    def test_duplicate_numbers_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ring_digits({"order": [1, 1, 3]})

    def test_bad_inner_ratio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ring_digits({"inner_ratio": 1.5})

    def test_bad_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ring_digits({"threshold": 300})


class RingTargetTests(unittest.TestCase):
    """窄视野：画面里只有一个环，也照样给出可对准的环心。"""

    CENTRE = (640.0, 360.0)

    def single_frame(
        self, *, digit: str = "2", diameter: float = 200.0, centre=None
    ) -> np.ndarray:
        centre = self.CENTRE if centre is None else centre
        frame = synthetic_station(
            size=SIZE,
            centers=(centre,),
            diameter_px=diameter,
            thickness=8,
            background=255,
            foreground=0,
        )
        if digit:
            import cv2

            cv2.putText(
                frame, digit, (int(centre[0]) - 18, int(centre[1]) + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4,
            )
        return frame

    def locator(self, *, mm_per_px: float | None = 0.5):
        from modules.vision.drive.numbered_ring import RingTargetLocator

        return RingTargetLocator(geometry=StationGeometrySettings(mm_per_px=mm_per_px))

    def test_one_ring_in_view_gives_its_centre(self) -> None:
        target = self.locator().detect(self.single_frame())
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.center_px[0], self.CENTRE[0], delta=2.0)
        self.assertAlmostEqual(target.center_px[1], self.CENTRE[1], delta=2.0)
        reading = target.reading()
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading[0], 0.0, delta=1.0)
        self.assertAlmostEqual(reading[1], 0.0, delta=1.0)
        self.assertFalse(target.partial)

    def test_the_digit_is_reported_but_does_not_change_the_centre(self) -> None:
        with_digit = self.locator().detect(self.single_frame(digit="3"))
        blank = self.locator().detect(self.single_frame(digit=""))
        self.assertTrue(with_digit.digit_present)
        self.assertFalse(blank.digit_present)
        # 空环的圆心照样给：机械臂可能只是要跟住那个圆圈；由调用方决定用不用。
        self.assertIsNotNone(blank.reading())
        self.assertAlmostEqual(blank.center_px[0], with_digit.center_px[0], delta=2.0)

    def test_no_ring_in_view_is_none_not_an_exception(self) -> None:
        blank = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        self.assertIsNone(self.locator().detect(blank))

    def test_no_scale_means_no_readable_centre(self) -> None:
        target = self.locator(mm_per_px=None).detect(self.single_frame())
        self.assertIsNotNone(target, "环还是认得的，只是没有毫米")
        self.assertIsNone(target.reading())
        self.assertIn("mm_per_px", target.detail)

    def test_known_diameter_drives_the_scale(self) -> None:
        """配了环直径就用它：毫米/像素 = 直径 / 实测外圈像素，与距离无关。"""
        from modules.vision.drive.numbered_ring import RingTargetLocator

        locator = RingTargetLocator(diameter_mm=53.0)
        target = locator.detect(self.single_frame(centre=(440.0, 360.0)))
        reading = target.reading()
        self.assertIsNotNone(reading)
        # 外圈像素 ≈ 208（直径 200 + 笔画），53/208 ≈ 0.255 mm/px；偏 200 px ≈ -51 mm。
        self.assertAlmostEqual(
            reading[0], -200.0 * 53.0 / target.reference_diameter_px, delta=2.0
        )
        self.assertEqual(target.scale_source, "ring diameter")

    def test_no_diameter_falls_back_to_the_configured_scale(self) -> None:
        target = self.locator(mm_per_px=0.5).detect(self.single_frame(centre=(440.0, 360.0)))
        self.assertEqual(target.scale_source, "station.mm_per_px")
        self.assertAlmostEqual(target.reading()[0], -100.0, delta=2.0)

    def test_pinhole_calibration_wins_over_the_fixed_scale(self) -> None:
        """针孔标定在时用 depth/focal：环心读数必须与物料的毫米同一套单位。"""
        from modules.vision.drive.numbered_ring import RingTargetLocator

        locator = RingTargetLocator(
            geometry=StationGeometrySettings(mm_per_px=0.4),
            focal_px=800.0,
            table_depth_mm=400.0,   # 0.5 mm/px，而不是配置里的 0.4
        )
        target = locator.detect(self.single_frame(centre=(440.0, 360.0)))
        reading = target.reading()
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading[0], -100.0, delta=2.0)
        self.assertIn("depth/focal", target.detail)

    def test_none_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.locator().detect(None)

    def test_two_rings_in_view_still_returns_a_single_target(self) -> None:
        """两个环都露出来时只给一个（支撑最多的那个），不猜也不报错。"""
        frame = synthetic_station(
            size=SIZE,
            centers=((420.0, 360.0), (900.0, 360.0)),
            diameter_px=200.0,
            thickness=8,
            background=255,
            foreground=0,
        )
        target = self.locator().detect(frame)
        self.assertIsNotNone(target)
        self.assertIn(round(target.center_px[0]), (420, 900))


class SessionTests(unittest.TestCase):
    """会话入口：一帧 → ``detect_numbered_rings`` / ``ring_reading``。"""

    def make_session(self, frame: np.ndarray | None = None):
        from modules.vision.runtime import Vision

        settings = VisionConfig(
            camera_ids={"windows": 0, "jetson": 0},
            station_geometry=StationGeometrySettings(mm_per_px=0.5),
            station_reference=StationReferenceSettings(ring_centers_px=CENTERS),
        )
        vision = Vision(settings)
        frame = station_frame() if frame is None else frame
        # 只替换取帧：定位器仍走真实实现，这样测的是"会话怎么把两段接起来"。
        vision._read_frame = lambda timeout=None: frame
        return vision

    def test_detect_numbered_rings_returns_the_three_rings(self) -> None:
        vision = self.make_session()
        result = vision.detect_numbered_rings()
        self.assertTrue(result.detected, result.detail)
        self.assertEqual([item.digit for item in result.rings], [1, 2, 3])

    def test_ring_reading_returns_a_servo_ready_pair(self) -> None:
        vision = self.make_session()
        reading = vision.ring_reading(3)
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading[1], 0.0, delta=0.5)
        self.assertGreater(reading[0], 160.0)

    def test_the_locator_is_cached_like_the_other_detectors(self) -> None:
        vision = self.make_session()
        first = vision._numbered_ring_locator()
        self.assertIs(vision._numbered_ring_locator(), first)
        # 与工位识别共用同一个三圆环检测器：两套稳定窗口会让结论互相矛盾。
        self.assertIs(first.station, vision._ring_station_detector())

    def test_locate_ring_finds_the_single_ring(self) -> None:
        frame = RingTargetTests().single_frame()
        vision = self.make_session(frame)
        target = vision.locate_ring()
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.center_px[0], 640.0, delta=2.0)

    def test_ring_center_returns_the_align_xy_reading(self) -> None:
        frame = RingTargetTests().single_frame()
        vision = self.make_session(frame)
        centre = vision.ring_center()
        self.assertIsNotNone(centre)
        self.assertAlmostEqual(centre[0], 0.0, delta=1.0)
        self.assertAlmostEqual(centre[1], 0.0, delta=1.0)

    def test_ring_target_locator_is_cached(self) -> None:
        vision = self.make_session()
        self.assertIs(vision._ring_target_locator(), vision._ring_target_locator())

    def test_effective_focal_px_needs_only_the_frame_size(self) -> None:
        """没有显式焦距时也要算得出来：``resolve_focal_px`` 要画幅参数。"""
        vision = self.make_session()
        self.assertGreater(vision._effective_focal_px(), 0.0)


if __name__ == "__main__":
    unittest.main()

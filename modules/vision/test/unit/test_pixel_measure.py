"""两点像素测量：有符号分量、直线距离、方向角与毫米换算。

这一组刻意**全部是纯计算**：不读图片、不开摄像头、不导入 cv2。
两点测量这条链路上真正会出错的只有"哪个是哪一个点"——第一击与第二击的顺序决定
ΔX / ΔY 的正负号，而正负号在"判断工件偏左还是偏右"时就是结论本身。所以这里
重点钉住符号，而不是长度。
"""

from __future__ import annotations

import unittest

from modules.vision.drive.pixel_measure import (
    focal_px_from_known_length,
    mean_focal_px,
    pixel_distance,
    to_millimetres,
)


class PixelDistanceTests(unittest.TestCase):
    def test_a_3_4_5_triangle_measures_exactly(self) -> None:
        measured = pixel_distance((100.0, 200.0), (400.0, 600.0))
        self.assertEqual(measured.dx_px, 300.0)
        self.assertEqual(measured.dy_px, 400.0)
        self.assertAlmostEqual(measured.length_px, 500.0, delta=1e-9)

    def test_the_signs_follow_the_argument_order(self) -> None:
        """符号是主输出：反过来量一次必须整体取反，长度不变。"""
        forward = pixel_distance((100.0, 200.0), (400.0, 600.0))
        backward = pixel_distance((400.0, 600.0), (100.0, 200.0))
        self.assertEqual((backward.dx_px, backward.dy_px), (-300.0, -400.0))
        self.assertAlmostEqual(backward.length_px, forward.length_px, delta=1e-9)

    def test_integer_and_float_points_agree(self) -> None:
        """坐标是整数还是浮点不该改变结果——鼠标给的是整数，别的检测器给的是浮点。"""
        from_ints = pixel_distance((10, 20), (40, 60))
        from_floats = pixel_distance((10.0, 20.0), (40.0, 60.0))
        self.assertEqual(from_ints, from_floats)

    def test_image_size_is_optional_and_reported_as_none(self) -> None:
        """没给画面尺寸时 ``inside`` 是 ``None``（表示没查），**不是** ``False``。"""
        unqueried = pixel_distance((100.0, 200.0), (400.0, 600.0))
        self.assertIsNone(unqueried.start_inside)
        self.assertIsNone(unqueried.end_inside)

        queried = pixel_distance((100.0, 200.0), (400.0, 600.0), image_size=(1280, 720))
        self.assertIs(queried.start_inside, True)
        self.assertIs(queried.end_inside, True)

    def test_a_point_outside_the_frame_is_reported(self) -> None:
        outside = pixel_distance((100.0, 200.0), (1400.0, 600.0), image_size=(1280, 720))
        self.assertIs(outside.start_inside, True)
        self.assertIs(outside.end_inside, False)

    def test_the_frame_edge_counts_as_inside(self) -> None:
        """闭区间：正好落在最后一列/行上算在内（鼠标能点到那个像素）。"""
        edge = pixel_distance((0.0, 0.0), (1280.0, 720.0), image_size=(1280, 720))
        self.assertIs(edge.start_inside, True)
        self.assertIs(edge.end_inside, True)

    def test_the_angle_uses_screen_axes(self) -> None:
        """画面右方 0°、下方 +90°、左方 ±180°（y 轴向下）。"""
        self.assertAlmostEqual(pixel_distance((0.0, 0.0), (10.0, 0.0)).angle_degrees, 0.0, delta=1e-9)
        self.assertAlmostEqual(pixel_distance((0.0, 0.0), (0.0, 10.0)).angle_degrees, 90.0, delta=1e-9)
        self.assertAlmostEqual(pixel_distance((0.0, 0.0), (-10.0, 0.0)).angle_degrees, 180.0, delta=1e-9)
        self.assertAlmostEqual(pixel_distance((0.0, 0.0), (10.0, 10.0)).angle_degrees, 45.0, delta=1e-9)

    def test_a_degenerate_measurement_has_no_direction_but_does_not_raise(self) -> None:
        """退化成一个点：没有方向，但让调用方为了打印一行字去接异常是过度设计。"""
        degenerate = pixel_distance((5.0, 5.0), (5.0, 5.0))
        self.assertEqual(degenerate.length_px, 0.0)
        self.assertEqual(degenerate.angle_degrees, 0.0)

    def test_the_summary_carries_the_signed_components(self) -> None:
        summary = pixel_distance((100.0, 200.0), (400.0, 600.0), image_size=(1280, 720)).summary()
        self.assertIn("ΔX=+300.0", summary)
        self.assertIn("ΔY=+400.0", summary)
        self.assertIn("500.00", summary)
        self.assertIn("1280×720", summary)

    def test_invalid_points_are_rejected(self) -> None:
        """写法不对要当场报错，而不是静默补 0 变成一个看着正常的数。"""
        for start, end, name in (
            ((1.0,), (2.0, 3.0), "起点少一个分量"),
            ((1.0, 2.0), (2.0,), "终点少一个分量"),
            (None, (2.0, 3.0), "起点是 None"),
            ((1.0, 2.0), None, "终点是 None"),
            (("a", 1.0), (2.0, 3.0), "分量不是数字"),
            ((True, 1.0), (2.0, 3.0), "分量是布尔值"),
            ((1.0, 2.0, 3.0), (2.0, 3.0), "起点是三元组"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    pixel_distance(start, end)


class MillimetreConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.measured = pixel_distance((100.0, 200.0), (400.0, 600.0))

    def test_the_scale_is_distance_over_focal_length(self) -> None:
        millimetres = to_millimetres(self.measured, focal_px=500.0, working_distance_mm=100.0)
        self.assertAlmostEqual(millimetres.scale_mm_per_px, 0.2, delta=1e-9)
        self.assertAlmostEqual(millimetres.length_mm, 100.0, delta=1e-9)
        self.assertAlmostEqual(millimetres.dx_mm, 60.0, delta=1e-9)
        self.assertAlmostEqual(millimetres.dy_mm, 80.0, delta=1e-9)

    def test_conversion_is_isotropic(self) -> None:
        """尺度各向同性：换算不该改变 ΔX:ΔY 的比值。"""
        millimetres = to_millimetres(self.measured, focal_px=500.0, working_distance_mm=100.0)
        self.assertAlmostEqual(millimetres.dx_mm / millimetres.dy_mm, 0.75, delta=1e-9)

    def test_doubling_the_distance_doubles_every_millimetre(self) -> None:
        near = to_millimetres(self.measured, focal_px=500.0, working_distance_mm=100.0)
        far = to_millimetres(self.measured, focal_px=500.0, working_distance_mm=200.0)
        self.assertAlmostEqual(far.length_mm, 2.0 * near.length_mm, delta=1e-9)

    def test_the_summary_names_the_scale_it_used(self) -> None:
        """比例尺必须跟读数一起报出来——换机位它就变了，藏在配置里会让人猜。"""
        summary = to_millimetres(self.measured, focal_px=500.0, working_distance_mm=100.0).summary()
        self.assertIn("0.2000 mm/px", summary)
        self.assertIn("距离=100.00 mm", summary)
        self.assertIn("工作距离 100 mm", summary)

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs, name in (
            ({"focal_px": 0.0, "working_distance_mm": 100.0}, "焦距为 0"),
            ({"focal_px": -1.0, "working_distance_mm": 100.0}, "焦距为负"),
            ({"focal_px": 500.0, "working_distance_mm": 0.0}, "工作距离为 0"),
            ({"focal_px": 500.0, "working_distance_mm": -1.0}, "工作距离为负"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    to_millimetres(self.measured, **kwargs)


class FocalLengthTests(unittest.TestCase):
    def test_focal_length_is_recovered_from_a_known_disc(self) -> None:
        """f = 距离 × 像素长度 / 真实长度。"""
        focal = focal_px_from_known_length(pixel_length=250.0, real_length_mm=30.0, distance_mm=60.0)
        self.assertAlmostEqual(focal, 500.0, delta=1e-9)

    def test_the_formula_round_trips(self) -> None:
        """反解出的焦距拿去换算，应当还原出真实长度。"""
        focal = focal_px_from_known_length(pixel_length=250.0, real_length_mm=30.0, distance_mm=60.0)
        measured = pixel_distance((0.0, 0.0), (250.0, 0.0))
        millimetres = to_millimetres(measured, focal_px=focal, working_distance_mm=60.0)
        self.assertAlmostEqual(millimetres.length_mm, 30.0, delta=1e-9)

    def test_the_mean_is_a_plain_average(self) -> None:
        self.assertAlmostEqual(mean_focal_px([480.0, 500.0, 520.0]), 500.0, delta=1e-9)

    def test_the_mean_does_not_reject_outliers(self) -> None:
        """刻意不做野值剔除：焦距样本差得离谱通常意味着某次填错了，应当让人看见散布。"""
        self.assertAlmostEqual(mean_focal_px([500.0, 500.0, 1000.0]), 666.666666, delta=1e-4)

    def test_invalid_inputs_are_rejected(self) -> None:
        for call, name in (
            (lambda: focal_px_from_known_length(pixel_length=0.0, real_length_mm=30.0, distance_mm=60.0), "像素长为 0"),
            (lambda: focal_px_from_known_length(pixel_length=250.0, real_length_mm=0.0, distance_mm=60.0), "真实长为 0"),
            (lambda: focal_px_from_known_length(pixel_length=250.0, real_length_mm=30.0, distance_mm=0.0), "距离为 0"),
            (lambda: mean_focal_px([]), "样本为空"),
            (lambda: mean_focal_px([500.0, 0.0]), "样本里有 0"),
            (lambda: mean_focal_px([500.0, -1.0]), "样本里有负数"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()

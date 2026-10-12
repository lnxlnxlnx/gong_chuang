"""像素与相机系毫米之间的换算，以及焦距的选择规则。"""

from __future__ import annotations

import unittest

from modules.vision.drive.pinhole import (
    DEFAULT_FOCAL_PX_AT_1280,
    CameraFix,
    PinholeCamera,
    focal_px_from_known_target,
    resolve_focal_px,
)


class PinholeDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = PinholeCamera(
            focal_px=DEFAULT_FOCAL_PX_AT_1280, image_width=1280, image_height=720
        )

    def test_documented_sample_returns_the_measured_distance(self) -> None:
        """文档样本：209 px 的 30 mm 圆面拍自约 145 mm。"""
        depth = self.camera.depth_from_radius(209 / 2.0, 30.0)
        self.assertAlmostEqual(depth, 145.2, delta=1.0)

    def test_larger_pixel_radius_means_closer(self) -> None:
        self.assertLess(
            self.camera.depth_from_radius(120.0, 30.0),
            self.camera.depth_from_radius(60.0, 30.0),
        )

    def test_zero_or_negative_inputs_are_rejected(self) -> None:
        for radius, diameter in ((0.0, 30.0), (-1.0, 30.0), (10.0, 0.0), (10.0, -3.0)):
            with self.subTest(radius=radius, diameter=diameter):
                with self.assertRaises(ValueError):
                    self.camera.depth_from_radius(radius, diameter)


class PinholeProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = PinholeCamera(focal_px=1000.0, image_width=1280, image_height=720)

    def test_principal_point_maps_to_zero_horizontal_offset(self) -> None:
        fix = self.camera.fix_from_circle(640.0, 360.0, 100.0, 30.0)
        self.assertAlmostEqual(fix.x_mm, 0.0, places=9)
        self.assertAlmostEqual(fix.y_mm, 0.0, places=9)
        self.assertAlmostEqual(fix.depth_mm, 1000.0 * 30 / 200.0, places=9)

    def test_offsets_scale_with_depth_over_focal(self) -> None:
        fix = self.camera.fix_from_circle(740.0, 260.0, 100.0, 30.0)
        depth = 1000.0 * 30 / 200.0
        self.assertAlmostEqual(fix.x_mm, 100.0 * depth / 1000.0, places=9)
        self.assertAlmostEqual(fix.y_mm, -100.0 * depth / 1000.0, places=9)

    def test_signs_follow_screen_right_and_down(self) -> None:
        right = self.camera.fix_from_circle(700.0, 360.0, 100.0, 30.0)
        down = self.camera.fix_from_circle(640.0, 400.0, 100.0, 30.0)
        self.assertGreater(right.x_mm, 0.0)
        self.assertGreater(down.y_mm, 0.0)

    def test_range_is_the_three_dimensional_distance(self) -> None:
        fix = CameraFix(x_mm=3.0, y_mm=4.0, depth_mm=12.0)
        self.assertAlmostEqual(fix.range_mm, 13.0, places=9)


class PinholeFrameRuleTests(unittest.TestCase):
    def test_default_focal_scales_with_an_uncropped_width(self) -> None:
        self.assertAlmostEqual(resolve_focal_px(1280, 720), DEFAULT_FOCAL_PX_AT_1280, places=9)
        self.assertAlmostEqual(resolve_focal_px(640, 360), DEFAULT_FOCAL_PX_AT_1280 / 2, places=9)

    def test_cropped_frame_must_supply_its_own_focal(self) -> None:
        with self.assertRaisesRegex(ValueError, "16:9"):
            resolve_focal_px(640, 480)

    def test_explicit_focal_wins_over_the_frame_rule(self) -> None:
        self.assertAlmostEqual(resolve_focal_px(640, 480, 555.0), 555.0, places=9)

    def test_non_positive_focal_is_rejected(self) -> None:
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_focal_px(1280, 720, value)

    def test_from_shape_builds_the_model_from_a_frame_shape(self) -> None:
        camera = PinholeCamera.from_shape((720, 1280, 3))
        self.assertEqual((camera.image_width, camera.image_height), (1280.0, 720.0))
        self.assertAlmostEqual(camera.focal_px, DEFAULT_FOCAL_PX_AT_1280, places=9)

    def test_empty_image_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PinholeCamera(focal_px=1000.0, image_width=0, image_height=720)


class FocalCalibrationTests(unittest.TestCase):
    def test_known_target_at_a_known_distance_gives_the_focal(self) -> None:
        focal = focal_px_from_known_target(
            distance_mm=145.0, pixel_diameter=209.0, real_diameter_mm=30.0
        )
        self.assertAlmostEqual(focal, 1010.17, delta=0.5)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            focal_px_from_known_target(distance_mm=0.0, pixel_diameter=10.0, real_diameter_mm=5.0)
        with self.assertRaises(ValueError):
            focal_px_from_known_target(distance_mm=10.0, pixel_diameter=0.0, real_diameter_mm=5.0)
        with self.assertRaises(ValueError):
            focal_px_from_known_target(distance_mm=10.0, pixel_diameter=10.0, real_diameter_mm=0.0)


if __name__ == "__main__":
    unittest.main()

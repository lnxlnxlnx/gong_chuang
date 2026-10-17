"""平面标定：从「相机测到的靶标位置 + 当时命令的关节量」解出圆心与 θ 零点。"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from modules.vision.drive.calibration import (
    CalibrationSample,
    PlaneCalibration,
    fit_plane_calibration,
)


def synthetic_samples(
    *,
    center_x_mm: float = 120.0,
    center_y_mm: float = -45.0,
    rotation_degrees: float = 37.0,
    radial_offset_mm: float = 0.0,
    count: int = 16,
    noise_mm: float = 0.0,
) -> list[CalibrationSample]:
    """由真值反推「相机应该测到什么」，可选注入噪声与命令起点偏移。"""
    samples = []
    for index in range(count):
        base_degrees = -160.0 + index * 20.0
        radial_mm = 60.0 + index * 5.0
        radians = math.radians(base_degrees - rotation_degrees)
        offset_x = 0.0 if noise_mm == 0.0 else noise_mm * math.cos(index)
        offset_y = 0.0 if noise_mm == 0.0 else noise_mm * math.sin(index)
        samples.append(
            CalibrationSample(
                x_mm=center_x_mm + radial_mm * math.cos(radians) + offset_x,
                y_mm=center_y_mm + radial_mm * math.sin(radians) + offset_y,
                base_degrees=base_degrees,
                radial_mm=radial_mm + radial_offset_mm,
            )
        )
    return samples


class PlaneCalibrationFitTests(unittest.TestCase):
    def test_exact_samples_recover_the_ground_truth(self) -> None:
        fitted = fit_plane_calibration(synthetic_samples())
        self.assertAlmostEqual(fitted.center_x_mm, 120.0, places=6)
        self.assertAlmostEqual(fitted.center_y_mm, -45.0, places=6)
        self.assertAlmostEqual(fitted.rotation_degrees, 37.0, places=6)
        self.assertLess(fitted.rms_mm, 1e-6)
        self.assertEqual(fitted.samples, 16)

    def test_unknown_radial_start_offset_does_not_drag_the_centre(self) -> None:
        """只有相对位置模式，命令径向含一个未知常数偏移，圆心仍必须解对。"""
        fitted = fit_plane_calibration(synthetic_samples(radial_offset_mm=133.0))
        self.assertAlmostEqual(fitted.center_x_mm, 120.0, places=6)
        self.assertAlmostEqual(fitted.center_y_mm, -45.0, places=6)
        self.assertAlmostEqual(fitted.rotation_degrees, 37.0, places=6)

    def test_rotation_is_the_theta_zero_offset_not_a_scale(self) -> None:
        for rotation in (0.0, 37.0, -128.0, 179.0):
            with self.subTest(rotation=rotation):
                fitted = fit_plane_calibration(synthetic_samples(rotation_degrees=rotation))
                self.assertAlmostEqual(fitted.rotation_degrees, rotation, places=6)

    def test_noise_shows_up_as_a_non_zero_residual(self) -> None:
        clean = fit_plane_calibration(synthetic_samples())
        noisy = fit_plane_calibration(synthetic_samples(noise_mm=2.0))
        self.assertLess(clean.rms_mm, 1e-6)
        self.assertGreater(noisy.rms_mm, 0.5)
        # 噪声下仍应落在真值附近，而不是被判成另一个解。
        self.assertAlmostEqual(noisy.center_x_mm, 120.0, delta=3.0)
        self.assertAlmostEqual(noisy.center_y_mm, -45.0, delta=3.0)

    def test_too_few_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "3"):
            fit_plane_calibration(synthetic_samples(count=2))

    def test_collinear_samples_are_rejected(self) -> None:
        collinear = [
            CalibrationSample(x_mm=float(index), y_mm=0.0, base_degrees=0.0, radial_mm=10.0)
            for index in range(8)
        ]
        with self.assertRaises(ValueError):
            fit_plane_calibration(collinear)

    def test_angles_that_cancel_out_are_rejected(self) -> None:
        """样本足够但角度观测量两两抵消时，θ 零点无法确定。"""
        samples = [
            CalibrationSample(x_mm=0.0, y_mm=10.0, base_degrees=0.0, radial_mm=10.0),
            CalibrationSample(x_mm=0.0, y_mm=10.0, base_degrees=180.0, radial_mm=10.0),
            CalibrationSample(x_mm=0.0, y_mm=10.0, base_degrees=90.0, radial_mm=10.0),
            CalibrationSample(x_mm=0.0, y_mm=10.0, base_degrees=-90.0, radial_mm=10.0),
        ]
        with self.assertRaises(ValueError):
            fit_plane_calibration(samples)


class PlaneCalibrationProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = PlaneCalibration(
            center_x_mm=100.0, center_y_mm=50.0, rotation_degrees=30.0
        )

    def test_the_centre_maps_to_the_origin(self) -> None:
        self.assertEqual(self.calibration.to_base_plane(100.0, 50.0), (0.0, 0.0))

    def test_polar_output_is_in_the_documented_half_open_range(self) -> None:
        for degrees in (0.0, 90.0, 179.0, -179.0):
            with self.subTest(degrees=degrees):
                radians = math.radians(degrees)
                base_degrees, radial_mm = self.calibration.to_polar(
                    100.0 + 80.0 * math.cos(radians), 50.0 + 80.0 * math.sin(radians)
                )
                self.assertAlmostEqual(radial_mm, 80.0, places=9)
                expected = (degrees + 30.0 + 180.0) % 360.0 - 180.0
                self.assertAlmostEqual(base_degrees, expected, places=9)

    def test_residual_is_zero_on_its_own_generating_data(self) -> None:
        # 平面角 = 相机角 + 标定转角，所以 20° 的相机点对应 50° 的 base_degrees。
        sample = CalibrationSample(
            x_mm=100.0 + 70.0 * math.cos(math.radians(20.0)),
            y_mm=50.0 + 70.0 * math.sin(math.radians(20.0)),
            base_degrees=50.0,
            radial_mm=70.0,
        )
        self.assertLess(self.calibration.residual_mm(sample), 1e-9)

    def test_summary_reports_the_residual_and_sample_count(self) -> None:
        described = replace(self.calibration, rms_mm=1.25, samples=9).summary()
        self.assertIn("1.25", described)
        self.assertIn("9", described)


if __name__ == "__main__":
    unittest.main()

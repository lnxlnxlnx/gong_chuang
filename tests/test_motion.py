"""Verify the shared motion timing vocabulary used by the arm and the chassis."""

from __future__ import annotations

import unittest

from modules.motor.drive.motion import (
    DEFAULT_SETTLE_SECONDS,
    MAX_ACCELERATION,
    cruise_seconds,
    estimate_move_seconds,
    ramp_seconds,
)


class RampSecondsTests(unittest.TestCase):
    def test_follows_the_datasheet_slope(self) -> None:
        # 每档加速度让转速 +1 RPM 需要 (256 - 档位) * 50µs。
        self.assertAlmostEqual(ramp_seconds(100, 6), 100 * 250 * 50e-6, places=9)

    def test_zero_acceleration_means_no_curve(self) -> None:
        self.assertEqual(ramp_seconds(3000, 0), 0.0)

    def test_zero_rpm_needs_no_ramp(self) -> None:
        self.assertEqual(ramp_seconds(0, 10), 0.0)

    def test_higher_gear_is_faster(self) -> None:
        self.assertLess(ramp_seconds(100, 200), ramp_seconds(100, 10))

    def test_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            ramp_seconds(3001, 10)
        with self.assertRaises(ValueError):
            ramp_seconds(100, MAX_ACCELERATION + 1)
        with self.assertRaises(ValueError):
            ramp_seconds(100.0, 10)  # type: ignore[arg-type]


class CruiseSecondsTests(unittest.TestCase):
    def test_one_revolution_at_60_rpm_takes_one_second(self) -> None:
        self.assertAlmostEqual(cruise_seconds(3200, 60, 3200), 1.0, places=9)

    def test_zero_rpm_cannot_be_estimated(self) -> None:
        with self.assertRaises(ValueError):
            cruise_seconds(100, 0, 3200)


class EstimateMoveSecondsTests(unittest.TestCase):
    def test_is_ramp_plus_cruise_plus_settle(self) -> None:
        expected = ramp_seconds(10, 5) + cruise_seconds(1000, 10, 3200) + DEFAULT_SETTLE_SECONDS
        self.assertAlmostEqual(estimate_move_seconds(1000, 10, 3200, 5), expected, places=9)

    def test_longer_move_takes_longer(self) -> None:
        short = estimate_move_seconds(400, 30, 3200, 5)
        long = estimate_move_seconds(1600, 30, 3200, 5)
        self.assertLess(short, long)

    def test_settle_margin_can_be_disabled(self) -> None:
        self.assertAlmostEqual(
            estimate_move_seconds(3200, 60, 3200, 5, settle_seconds=0.0),
            ramp_seconds(60, 5) + 1.0,
            places=9,
        )

    def test_rejects_negative_settle_margin(self) -> None:
        with self.assertRaises(ValueError):
            estimate_move_seconds(3200, 60, 3200, 5, settle_seconds=-0.1)


if __name__ == "__main__":
    unittest.main()

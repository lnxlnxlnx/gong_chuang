"""``center_chassis_on_ring``：用车横向把环心对到光轴上。

整条链都不碰真机：假底盘按"每脉冲挪多少毫米"改环心 x，假视觉按当前 x 造一条
``RingTarget``。守三件事——第一步先试方向、之后按实测响应比一步到位、方向反了当场停。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import app.gripper as app_gripper
from modules.chassis.drive.mecanum import WHEEL_ORDER
from modules.vision.drive.numbered_ring import RingTarget


class FakePlant:
    """一辆只会横移的假车：收到底盘方向命令后按 ``mm_per_pulse`` 改环心 x。"""

    def __init__(self, x_mm: float, mm_per_pulse: float = 0.02) -> None:
        self.x_mm = x_mm
        self.mm_per_pulse = mm_per_pulse
        self.moves: list[int] = []      # 带符号的横移脉冲：正 = 左移

    def strafe_left(self, *, rpm=None, pulses=0, **_kwargs):
        self.moves.append(int(pulses))
        self.x_mm += pulses * self.mm_per_pulse
        return [pulses] * 4

    def strafe_right(self, *, rpm=None, pulses=0, **_kwargs):
        self.moves.append(-int(pulses))
        self.x_mm -= pulses * self.mm_per_pulse
        return [-pulses] * 4


class FakeVision:
    """只"看得见" ``digit`` 号环；要给别的号就返回 ``None``（和真视觉一样按号筛）。"""

    def __init__(self, plant: FakePlant, *, digit: int = 2, blanks: int = 0) -> None:
        self.plant = plant
        self.digit = digit
        self.blanks = blanks          # 前几帧"没看见环"（运动模糊 / 爪影扫过）
        self.calls = 0

    def locate_ring(self, *, digit=None, **_kwargs):
        self.calls += 1
        if self.blanks > 0:
            self.blanks -= 1
            return None
        if digit is not None and int(digit) != self.digit:
            return None
        return RingTarget(
            center_px=(640.0, 360.0),
            radius_px=250.0,
            digit=self.digit,
            score=0.95,
            clipped=False,
            x_mm=self.plant.x_mm,
            y_mm=0.0,
            scale_mm_per_px=0.5,
            scale_source="ring box",
        )


def make_robot(plant: FakePlant, **kwargs) -> Mock:
    robot = Mock()
    robot.chassis = plant
    robot.vision = FakeVision(plant, **kwargs)
    return robot


class CenterChassisOnRingTests(unittest.TestCase):
    def run_it(self, plant: FakePlant, **kwargs):
        robot = make_robot(plant)
        with patch("time.sleep"):
            return app_gripper.center_chassis_on_ring(robot, **kwargs)

    def test_it_walks_to_the_ring_from_the_wrong_side(self) -> None:
        plant = FakePlant(x_mm=12.0, mm_per_pulse=0.05)

        residual = self.run_it(plant)

        self.assertLessEqual(abs(residual), app_gripper.RING_LATERAL_TOLERANCE_MM)
        # 12 mm 在右边：第一步往右挪（负脉冲），试走 300 脉冲就够越过中心。
        self.assertEqual(len(plant.moves), 1)
        self.assertLess(plant.moves[0], 0)

    def test_it_learns_the_response_and_finishes_in_two_steps(self) -> None:
        """门槛设得很紧时第二步是牛顿步：按上一步实测的"毫米/脉冲"一次算到位。"""
        plant = FakePlant(x_mm=12.0, mm_per_pulse=0.02)

        residual = self.run_it(plant, tolerance_mm=0.5)

        self.assertLessEqual(abs(residual), 0.5)
        self.assertEqual(len(plant.moves), 2)
        # 第一步试走 300 脉冲（挪 6 mm），第二步按学到的响应比补齐剩下那 6 mm。
        self.assertEqual(abs(plant.moves[0]), app_gripper.LATERAL_TRIAL_PULSES)
        self.assertAlmostEqual(plant.moves[1], -300, delta=1)

    def test_a_chassis_that_does_not_move_is_reported(self) -> None:
        plant = FakePlant(x_mm=12.0, mm_per_pulse=0.0)

        with self.assertRaisesRegex(RuntimeError, "反方向"):
            self.run_it(plant, max_iterations=3)

    def test_it_refuses_a_wanted_ring_that_is_not_in_view(self) -> None:
        """只要看得见 3 号，要 2 号就报错——绝不拿 3 号横向对齐。"""
        plant = FakePlant(x_mm=12.0)
        robot = make_robot(plant, digit=3)

        with patch("time.sleep"), self.assertRaisesRegex(RuntimeError, "2 号环"):
            app_gripper.center_chassis_on_ring(robot, require_digit=2)
        self.assertEqual(plant.moves, [])

    def test_a_blurred_frame_is_retried_before_giving_up(self) -> None:
        plant = FakePlant(x_mm=12.0, mm_per_pulse=0.05)
        robot = make_robot(plant)
        robot.vision.blanks = 2          # 前两帧没看见环

        with patch("time.sleep"):
            residual = app_gripper.center_chassis_on_ring(robot)

        self.assertLessEqual(abs(residual), app_gripper.RING_LATERAL_TOLERANCE_MM)
        self.assertGreaterEqual(robot.vision.calls, 3)

    def test_a_missing_ring_raises_before_any_wheel_turns(self) -> None:
        plant = FakePlant(x_mm=0.0)
        robot = make_robot(plant)
        robot.vision = Mock()
        robot.vision.locate_ring.return_value = None

        with patch("time.sleep"), self.assertRaisesRegex(RuntimeError, "没有环"):
            app_gripper.center_chassis_on_ring(robot)
        self.assertEqual(plant.moves, [])
        self.assertEqual(robot.vision.locate_ring.call_count, app_gripper.RING_READ_ATTEMPTS)

    def test_a_refused_frame_says_why(self) -> None:
        """视觉因为"同一号出现在两个环上"而作废时，报错要带出那句原因，别让人去查相机。"""
        plant = FakePlant(x_mm=0.0)
        robot = make_robot(plant)
        robot.vision = Mock()
        robot.vision.locate_ring.return_value = None
        robot.vision.last_ring_diagnosis = "同一帧里 1 号环出现了 2 次"

        with patch("time.sleep"), self.assertRaisesRegex(RuntimeError, "出现了 2 次"):
            app_gripper.center_chassis_on_ring(robot)
        self.assertEqual(plant.moves, [])


if __name__ == "__main__":
    unittest.main()

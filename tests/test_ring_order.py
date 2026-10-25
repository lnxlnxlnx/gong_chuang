"""环号 → 四轮绝对位置的那几步算术：不需要真机，也不开串口。

守的是两件容易静默错掉的事：

- **符号**：``motion_transform="calibrated"`` 下"前进"是四轮三负一正，按"四轮同号"
  估一遍车会拧着走，而动作看起来完全正常；
- **量纲**：``0x36`` 读数是 1/65536 电机圈，位置命令是 ``pulses_per_revolution``
  脉冲一圈，直接用会差 20 倍。
"""

from __future__ import annotations

import unittest

from app import gripper as app_gripper
from modules.chassis.drive.mecanum import WHEEL_ORDER
from modules.motor.drive.emm42 import POSITION_UNITS_PER_REVOLUTION


class _Wheel:
    def __init__(self, position: int) -> None:
        self.position = position

    def read_position(self) -> int:
        return self.position


class _Drive:
    pulses_per_revolution = 3200
    wheel_signs = {wheel: 1 for wheel in WHEEL_ORDER}

    @staticmethod
    def _kinematics(vx: float, vy: float, omega: float) -> list:
        # 实车标定矩阵的前进列：FL/FR/RR 反转，RL 正转。
        return [-1.0, -1.0, 1.0, -1.0]


class _Chassis:
    def __init__(self, positions) -> None:
        self.drive = _Drive()
        self.wheels = {
            wheel: _Wheel(position) for wheel, position in zip(WHEEL_ORDER, positions)
        }

    def wheel(self, wheel):
        return self.wheels[wheel]


def _base_pulses(units: int) -> int:
    """``0x36`` 读数换算成命令脉冲，与 ``_wheel_positions`` 同一把尺。"""
    return round(units * _Drive.pulses_per_revolution / POSITION_UNITS_PER_REVOLUTION)


class RingTargetTests(unittest.TestCase):
    def test_one_grid_uses_the_chassis_sign_vector(self) -> None:
        counts = app_gripper._grid_counts(_Chassis([0, 0, 0, 0]), 1950)
        self.assertEqual(counts, [-1950, -1950, 1950, -1950])

    def test_positions_are_converted_from_0x36_units(self) -> None:
        units = 3200 // 4  # 800 个 0x36 单位
        positions = app_gripper._wheel_positions(_Chassis([units] * 4))
        self.assertEqual(positions, [_base_pulses(units)] * 4)

    def test_targets_are_absolute_and_follow_the_number(self) -> None:
        units = 3200 // 4
        base = _base_pulses(units)
        targets = app_gripper._ring_targets(_Chassis([units] * 4), start_digit=1)
        self.assertEqual(targets[1], [base] * 4)
        self.assertEqual(
            targets[3], [base - 3900, base - 3900, base + 3900, base - 3900]
        )

    def test_grab_starts_at_three_and_walks_backward(self) -> None:
        targets = app_gripper._ring_targets(_Chassis([0, 0, 0, 0]), start_digit=3)
        self.assertEqual(targets[2], [1950, 1950, -1950, 1950])
        self.assertEqual(targets[1], [3900, 3900, -3900, 3900])

    def test_a_typo_in_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            app_gripper._check_order((1, 4))

    def test_order_accepts_any_repeatable_sequence(self) -> None:
        self.assertEqual(app_gripper._check_order([2, 1, 3]), (2, 1, 3))
        self.assertEqual(app_gripper._check_order((1, 1)), (1, 1))


if __name__ == "__main__":
    unittest.main()

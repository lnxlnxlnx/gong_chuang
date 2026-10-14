"""菜单 21 的 z 段（``--align-z``）：顺序、前置条件，以及"失败就不发动作"。

这一段是**开环**的——升降下降、伸缩后缩，走完之前画面里就看不到物块了，
所以真机上出错没有第二次机会。能在这里钉死的就是三条：

1. 顺序是"先下降、再后缩"；
2. 两个脉冲数来自配置，命令行的覆盖只作用于这一次；
3. 平面没对准就**一个动作帧都不发**（脉冲数只保证往下走了那么多，不保证下面有物块）。
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from modules.grasp.drive.eye_in_hand import (
    LiftScale,
    ObjectSelector,
    ServoError,
    ServoJacobian,
    ServoResult,
    ServoTarget,
    ZDescentResult,
    load_servo_config,
)
from modules.grasp.drive.eye_in_hand import GraspReadyResult
from modules.grasp.test.manual import align_xy as menu
from modules.grasp.test.manual.align_xy import run_descend_z, run_pick_ready
from modules.vision.drive.sighting import parse_sighting


def sighting(x: float, y: float):
    return parse_sighting(f"物料 4: green物料（green），xyz=({x:.6f},{y:.6f},113.2) mm")


class FakeArm:
    """只记动作：升降与伸缩各记一条，够断言顺序和脉冲数。"""

    def __init__(self) -> None:
        self.calls: list = []

    def move_down(self, pulses, **_kwargs):
        self.calls.append(("down", pulses))

    def move_up(self, pulses, **_kwargs):
        self.calls.append(("up", pulses))

    def move_backward(self, pulses, **_kwargs):
        self.calls.append(("back", pulses))

    def move_forward(self, pulses, **_kwargs):
        self.calls.append(("forward", pulses))


class FakeSource:
    """画面静止的假镜头：连续两次读数一致，取帧立刻返回。"""

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        self.error = None

    def frame(self, **_kwargs):
        return (sighting(self.x, self.y),)


class DescendZTests(unittest.TestCase):
    def run_step(self, *, x: float = 1.0, y: float = -1.0, **overrides):
        """跑一次菜单 21 的 z 段，返回 (退出码, 假臂, 打出来的字)。"""
        arm = FakeArm()
        printed = io.StringIO()
        with redirect_stdout(printed):
            status = run_descend_z(
                arm=arm,
                source=FakeSource(x, y),
                selector=ObjectSelector(),
                config=load_servo_config(),
                **overrides,
            )
        return status, arm, printed.getvalue()

    def test_the_z_step_descends_first_and_then_retracts(self) -> None:
        config = load_servo_config()
        status, arm, printed = self.run_step()
        self.assertEqual(status, 0)
        # 脉冲数跟着配置走（现场会改这两个数），不写死具体值。
        self.assertEqual(
            arm.calls,
            [("down", config.z_descend_pulses), ("back", config.extension_retract_pulses)],
        )
        # 两个数都要打在屏幕上：现场就是照着这两行核对配置改对没有。
        self.assertIn(str(config.z_descend_pulses), printed)
        self.assertIn(str(config.extension_retract_pulses), printed)

    def test_the_overrides_replace_both_counts_for_this_run_only(self) -> None:
        before = load_servo_config()
        status, arm, _ = self.run_step(z_pulses=800, retract_pulses=20)
        self.assertEqual(status, 0)
        self.assertEqual(arm.calls, [("down", 800), ("back", 20)])
        # 覆盖不写回配置：下一次还是配置文件里的值。
        after = load_servo_config()
        self.assertEqual(after.z_descend_pulses, before.z_descend_pulses)
        self.assertEqual(after.extension_retract_pulses, before.extension_retract_pulses)

    def test_a_plane_that_is_not_aligned_refuses_without_moving(self) -> None:
        status, arm, printed = self.run_step(x=20.0, y=0.0)
        self.assertEqual(status, 1)
        self.assertEqual(arm.calls, [])
        self.assertIn("没有发出动作", printed)

    def test_a_nonsense_override_stops_before_moving(self) -> None:
        status, arm, _ = self.run_step(z_pulses=0)
        self.assertEqual(status, 1)
        self.assertEqual(arm.calls, [])


class PickReadyTests(unittest.TestCase):
    """一键就位（``--align-and-descend``）：四步都交给 ``align_and_descend``。"""

    def canned(self) -> GraspReadyResult:
        return GraspReadyResult(
            raised_pulses=250,
            aligned=ServoResult(
                target=ServoTarget(x_mm=0.0, y_mm=0.0),
                readings=((0.3, -0.2),),
                error_x=0.3,
                error_y=-0.2,
                base_pulses=8,
                radial_pulses=-3,
                iterations=1,
                tolerance_mm=2.0,
            ),
            descent=ZDescentResult(
                pulses=2200, planar_mm=0.36, start_z_mm=100.0, seconds=1.2
            ),
            retracted_pulses=150,
            lift_seconds=1.4,
            align_seconds=1.1,
            descend_seconds=1.6,
            retract_seconds=0.2,
        )

    def run_it(self, **kwargs):
        printed = io.StringIO()
        with redirect_stdout(printed):
            status = run_pick_ready(
                arm=object(),
                source=object(),
                selector=ObjectSelector(),
                target=ServoTarget(x_mm=0.0, y_mm=0.0),
                jacobian=ServoJacobian(
                    base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091
                ),
                config=load_servo_config(),
                lift_scale=LiftScale(dz_per_pulse=0.0121),
                **kwargs,
            )
        return status, printed.getvalue()

    def test_a_run_reports_every_phase_and_forwards_the_overrides(self) -> None:
        seen: dict = {}

        def fake(*_args, **kwargs):
            seen.update(kwargs)
            return self.canned()

        with patch.object(menu, "align_and_descend", side_effect=fake):
            status, printed = self.run_it(z_pulses=800, retract_pulses=20)

        self.assertEqual(status, 0)
        self.assertEqual((seen["z_pulses"], seen["retract_pulses"]), (800, 20))
        self.assertEqual(seen["lift_scale"], LiftScale(dz_per_pulse=0.0121))
        self.assertIn("就位完成", printed)
        self.assertIn("共 4.3 s", printed)
        self.assertIn("下一步：合上夹爪", printed)

    def test_a_failed_run_returns_one_and_prints_the_reason(self) -> None:
        with patch.object(
            menu, "align_and_descend", side_effect=ServoError("平面还没对准就下降")
        ):
            status, printed = self.run_it()

        self.assertEqual(status, 1)
        self.assertIn("就位失败", printed)
        self.assertIn("平面还没对准", printed)


if __name__ == "__main__":
    unittest.main()

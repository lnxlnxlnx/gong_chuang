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

from modules.grasp.drive.eye_in_hand import ObjectSelector, load_servo_config
from modules.grasp.test.manual.align_xy import run_descend_z
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
        status, arm, printed = self.run_step()
        self.assertEqual(status, 0)
        self.assertEqual(arm.calls, [("down", 1700), ("back", 150)])
        # 两个数都要打在屏幕上：现场就是照着这两行核对配置改对没有。
        self.assertIn("1700", printed)
        self.assertIn("150", printed)

    def test_the_overrides_replace_both_counts_for_this_run_only(self) -> None:
        status, arm, _ = self.run_step(z_pulses=800, retract_pulses=20)
        self.assertEqual(status, 0)
        self.assertEqual(arm.calls, [("down", 800), ("back", 20)])
        # 覆盖不写回配置：下一次还是配置文件里的值。
        config = load_servo_config()
        self.assertEqual(config.z_descend_pulses, 1700)
        self.assertEqual(config.extension_retract_pulses, 150)

    def test_a_plane_that_is_not_aligned_refuses_without_moving(self) -> None:
        status, arm, printed = self.run_step(x=20.0, y=0.0)
        self.assertEqual(status, 1)
        self.assertEqual(arm.calls, [])
        self.assertIn("没有发出动作", printed)

    def test_a_nonsense_override_stops_before_moving(self) -> None:
        status, arm, _ = self.run_step(z_pulses=0)
        self.assertEqual(status, 1)
        self.assertEqual(arm.calls, [])


if __name__ == "__main__":
    unittest.main()

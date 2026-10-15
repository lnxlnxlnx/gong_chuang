"""Wheel demos must release the chassis when a route is interrupted."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.chassis.app import demo
from modules.chassis.app import map as wheel_map


class FakeChassis:
    """假底盘：``_from_settings`` 返回它，``close`` 记一笔（收尾由上下文负责）。"""

    def __init__(self) -> None:
        self.calls = []

    def forward(self, **_kwargs):
        self.calls.append("forward")
        raise KeyboardInterrupt

    def stop(self):
        self.calls.append("stop")

    def disable_all(self):
        self.calls.append("disable_all")

    def close(self):
        self.calls.append("close")

    def __enter__(self):
        self.calls.append("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


class WheelAppTests(unittest.TestCase):
    def test_chassis_demo_stops_and_closes_on_interrupt(self) -> None:
        chassis = FakeChassis()
        with patch.object(demo.MecanumChassis, "_from_settings", return_value=chassis):
            with patch("builtins.input", return_value=""):
                with self.assertRaises(KeyboardInterrupt):
                    demo.demo_chassis(SimpleNamespace(chassis_rpm=30, chassis_duration=1.0))
        # 收尾只有一处：上下文调用 close（它内部才是 停 → 失能 → 释放驱动）。
        self.assertEqual(chassis.calls, ["enter", "forward", "close"])

    def test_route_runner_stops_and_closes_on_interrupt(self) -> None:
        chassis = FakeChassis()
        with patch.object(wheel_map.MecanumChassis, "_from_settings", return_value=chassis):
            with patch("builtins.input", return_value=""):
                with self.assertRaises(KeyboardInterrupt):
                    wheel_map.run_map_by_func(
                        SimpleNamespace(chassis_rpm=30, chassis_duration=1.0),
                        lambda car, _profile: car.forward(rpm=30, duration=1),
                    )
        self.assertEqual(chassis.calls, ["enter", "forward", "close"])

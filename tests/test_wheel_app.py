"""Wheel demos must release the chassis when a route is interrupted."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.wheel import demo, map as wheel_map


class FakeChassis:
    def __init__(self) -> None:
        self.calls = []

    def initialize(self):
        self.calls.append("initialize")
        return self

    def enable_all(self):
        self.calls.append("enable_all")

    def forward(self, **_kwargs):
        self.calls.append("forward")
        raise KeyboardInterrupt

    def stop(self):
        self.calls.append("stop")

    def disable_all(self):
        self.calls.append("disable_all")

    def close(self):
        self.calls.append("close")


class WheelAppTests(unittest.TestCase):
    def test_chassis_demo_stops_and_closes_on_interrupt(self) -> None:
        chassis = FakeChassis()
        with patch.object(demo, "create_chassis", return_value=chassis):
            with patch("builtins.input", return_value=""):
                with self.assertRaises(KeyboardInterrupt):
                    demo.demo_chassis(SimpleNamespace(chassis_rpm=30, chassis_duration=1.0))
        self.assertEqual(
            chassis.calls,
            ["initialize", "enable_all", "forward", "stop", "disable_all", "close"],
        )

    def test_route_runner_stops_and_closes_on_interrupt(self) -> None:
        chassis = FakeChassis()
        with patch.object(wheel_map, "create_chassis", return_value=chassis):
            with patch("builtins.input", return_value=""):
                with self.assertRaises(KeyboardInterrupt):
                    wheel_map.run_map_by_func(
                        SimpleNamespace(chassis_rpm=30, chassis_duration=1.0),
                        lambda car, _profile: car.forward(rpm=30, duration=1),
                    )
        self.assertEqual(
            chassis.calls,
            ["initialize", "enable_all", "forward", "stop", "disable_all", "close"],
        )

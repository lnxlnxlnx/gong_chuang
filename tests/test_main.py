"""The application must release actuators when a demo is interrupted."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


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


class MainDemoTests(unittest.TestCase):
    def test_chassis_demo_stops_and_closes_on_interrupt(self) -> None:
        chassis = FakeChassis()
        with patch.object(main, "create_chassis", return_value=chassis):
            with patch("builtins.input", return_value=""):
                with self.assertRaises(KeyboardInterrupt):
                    main.demo_chassis(SimpleNamespace(chassis_rpm=30, chassis_duration=1.0))
        self.assertEqual(
            chassis.calls,
            ["initialize", "enable_all", "forward", "stop", "disable_all", "close"],
        )


if __name__ == "__main__":
    unittest.main()

"""Verify hardware-test command sequences without a connected vehicle."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from hardware_tests.mecanum import chassis_motion, individual_wheels
from mecanum import WHEEL_ORDER, WheelPosition


class FakeMecanumDrive:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def initialize(self) -> "FakeMecanumDrive":
        self.calls.append(("initialize",))
        return self

    def enable_all(self) -> None:
        self.calls.append(("enable_all",))

    def disable_all(self) -> None:
        self.calls.append(("disable_all",))

    def enable_wheel(self, wheel: WheelPosition, enabled: bool) -> None:
        self.calls.append(("enable_wheel", wheel, enabled))

    def set_wheel_speed(self, wheel: WheelPosition, rpm: int) -> None:
        self.calls.append(("set_wheel_speed", wheel, rpm))

    def stop_wheel(self, wheel: WheelPosition) -> None:
        self.calls.append(("stop_wheel", wheel))

    def stop(self) -> None:
        self.calls.append(("stop",))

    def close(self) -> None:
        self.calls.append(("close",))

    def forward(self, rpm: int) -> None:
        self.calls.append(("forward", rpm))

    def backward(self, rpm: int) -> None:
        self.calls.append(("backward", rpm))

    def strafe_left(self, rpm: int) -> None:
        self.calls.append(("strafe_left", rpm))

    def strafe_right(self, rpm: int) -> None:
        self.calls.append(("strafe_right", rpm))

    def rotate_cw(self, rpm: int) -> None:
        self.calls.append(("rotate_cw", rpm))

    def rotate_ccw(self, rpm: int) -> None:
        self.calls.append(("rotate_ccw", rpm))

    def forward_left(self, rpm: int) -> None:
        self.calls.append(("forward_left", rpm))

    def forward_right(self, rpm: int) -> None:
        self.calls.append(("forward_right", rpm))

    def backward_left(self, rpm: int) -> None:
        self.calls.append(("backward_left", rpm))

    def backward_right(self, rpm: int) -> None:
        self.calls.append(("backward_right", rpm))


class FailingInitializeDrive(FakeMecanumDrive):
    def initialize(self) -> "FailingInitializeDrive":
        self.calls.append(("initialize",))
        raise RuntimeError("serial port unavailable")


class IndividualWheelScriptTests(unittest.TestCase):
    def test_each_wheel_runs_forward_then_reverse_in_fixed_order(self) -> None:
        car = FakeMecanumDrive()
        confirmations: list[str] = []
        sleeps: list[float] = []

        individual_wheels.run_sequence(
            car,
            rpm=20,
            duration=0.5,
            confirm=confirmations.append,
            sleep=sleeps.append,
        )

        expected: list[tuple[object, ...]] = []
        for wheel in WHEEL_ORDER:
            expected.extend(
                [
                    ("enable_wheel", wheel, True),
                    ("set_wheel_speed", wheel, 20),
                    ("stop_wheel", wheel),
                    ("set_wheel_speed", wheel, -20),
                    ("stop_wheel", wheel),
                    ("enable_wheel", wheel, False),
                ]
            )
        self.assertEqual(car.calls, expected)
        self.assertEqual(confirmations, [wheel.name for wheel in WHEEL_ORDER])
        self.assertEqual(sleeps, [0.5] * 8)

    def test_each_wheel_reports_acknowledged_motion_states(self) -> None:
        car = FakeMecanumDrive()
        output = io.StringIO()

        with redirect_stdout(output):
            individual_wheels.run_sequence(
                car,
                rpm=20,
                duration=0.5,
                confirm=lambda _: None,
                sleep=lambda _: None,
            )

        expected: list[str] = []
        for wheel in WHEEL_ORDER:
            expected.extend(
                [
                    f"[{wheel.name}] enabled",
                    f"[{wheel.name}] forward 20 RPM acknowledged",
                    f"[{wheel.name}] stopped",
                    f"[{wheel.name}] reverse 20 RPM acknowledged",
                    f"[{wheel.name}] stopped",
                    f"[{wheel.name}] disabled",
                ]
            )
        self.assertEqual(output.getvalue().splitlines(), expected)

    def test_individual_run_cleans_up_after_an_interrupted_sequence(self) -> None:
        car = FakeMecanumDrive()

        def interrupted_sleep(_: float) -> None:
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            individual_wheels.run_hardware_test(
                car,
                rpm=20,
                duration=0.5,
                confirm=lambda _: None,
                sleep=interrupted_sleep,
            )

        self.assertEqual(car.calls[-3:], [("stop",), ("disable_all",), ("close",)])

    def test_individual_run_cleans_up_when_initialization_fails(self) -> None:
        car = FailingInitializeDrive()

        with self.assertRaisesRegex(RuntimeError, "serial port unavailable"):
            individual_wheels.run_hardware_test(
                car,
                rpm=20,
                duration=0.5,
                confirm=lambda _: None,
            )

        self.assertEqual(car.calls, [("initialize",), ("stop",), ("disable_all",), ("close",)])


class ChassisMotionScriptTests(unittest.TestCase):
    def test_chassis_sequence_runs_each_motion_and_stops_between_steps(self) -> None:
        car = FakeMecanumDrive()
        sleeps: list[float] = []
        motions = (
            "forward",
            "backward",
            "strafe_left",
            "strafe_right",
            "forward_left",
            "forward_right",
            "backward_left",
            "backward_right",
            "rotate_cw",
            "rotate_ccw",
        )
        prompts = iter(["" if i % 2 == 0 else "n" for i in range(len(motions) * 2)])

        chassis_motion.run_sequence(
            car,
            rpm=30,
            duration=1.0,
            pause=0.25,
            prompt=lambda _: next(prompts, "q"),
            sleep=sleeps.append,
        )

        expected = []
        for name in motions:
            expected.extend([(name, 30), ("stop",)])
        self.assertEqual(car.calls, expected)
        self.assertEqual(sleeps, [1.0, 0.25] * 10)

    def test_chassis_sequence_accepts_enter_to_repeat_current_motion(self) -> None:
        car = FakeMecanumDrive()
        sleeps: list[float] = []
        prompts = iter(["", "", "", "n", "q"])  # forward 重做三次后进入下一个动作

        chassis_motion.run_sequence(
            car,
            rpm=30,
            duration=1.0,
            pause=0.25,
            prompt=lambda _: next(prompts, "q"),
            sleep=sleeps.append,
        )

        self.assertEqual(car.calls, [("forward", 30), ("stop",)] * 3)
        self.assertEqual(sleeps, [1.0, 1.0, 1.0, 0.25])

    def test_chassis_sequence_quits_early_when_asked(self) -> None:
        car = FakeMecanumDrive()

        chassis_motion.run_sequence(
            car,
            rpm=30,
            duration=1.0,
            pause=0.25,
            prompt=lambda _: "q",
        )

        self.assertEqual(car.calls, [])

    def test_chassis_run_cleans_up_after_an_interrupted_sequence(self) -> None:
        car = FakeMecanumDrive()

        def interrupted_sleep(_: float) -> None:
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            chassis_motion.run_hardware_test(
                car,
                rpm=30,
                duration=1.0,
                pause=0.25,
                prompt=lambda _: "n",
                sleep=interrupted_sleep,
            )

        self.assertEqual(car.calls[:2], [("initialize",), ("enable_all",)])
        self.assertEqual(car.calls[-3:], [("stop",), ("disable_all",), ("close",)])

    def test_chassis_run_cleans_up_when_initialization_fails(self) -> None:
        car = FailingInitializeDrive()

        with self.assertRaisesRegex(RuntimeError, "serial port unavailable"):
            chassis_motion.run_hardware_test(
                car,
                rpm=30,
                duration=1.0,
                pause=0.25,
                prompt=lambda _: "n",
            )

        self.assertEqual(car.calls, [("initialize",), ("stop",), ("disable_all",), ("close",)])


if __name__ == "__main__":
    unittest.main()

"""Run each mecanum wheel forward and reverse at a safe low speed.

Run on the vehicle from the project root:

    python -m hardware_tests.mecanum.individual_wheels
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Protocol

from mecanum import MecanumDrive, WHEEL_ORDER, WheelPosition


class IndividualWheelDrive(Protocol):
    """The narrow controller interface used by this test sequence."""

    def initialize(self) -> "IndividualWheelDrive": ...

    def enable_wheel(self, wheel: WheelPosition, enabled: bool) -> None: ...

    def set_wheel_speed(self, wheel: WheelPosition, signed_rpm: int) -> None: ...

    def stop_wheel(self, wheel: WheelPosition) -> None: ...

    def stop(self) -> None: ...

    def disable_all(self) -> None: ...

    def close(self) -> None: ...


def run_sequence(
    car: IndividualWheelDrive,
    *,
    rpm: int,
    duration: float,
    confirm: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Test FL, FR, RL, RR one at a time with a confirmation before each wheel."""
    for wheel in WHEEL_ORDER:
        confirm(wheel.name)
        car.enable_wheel(wheel, True)
        car.set_wheel_speed(wheel, rpm)
        sleep(duration)
        car.stop_wheel(wheel)
        car.set_wheel_speed(wheel, -rpm)
        sleep(duration)
        car.stop_wheel(wheel)
        car.enable_wheel(wheel, False)


def run_hardware_test(
    car: IndividualWheelDrive,
    *,
    rpm: int,
    duration: float,
    confirm: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run the sequence and always leave the chassis stopped, disabled, and closed."""
    try:
        car.initialize()
        run_sequence(car, rpm=rpm, duration=duration, confirm=confirm, sleep=sleep)
    finally:
        car.stop()
        car.disable_all()
        car.close()


def _confirm(wheel_name: str) -> None:
    input(f"准备测试 {wheel_name}。确认轮子悬空且周围安全后按 Enter 继续: ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐个测试四个麦轮的正反转方向")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="四轮共用的 TTL 串口")
    parser.add_argument("--rpm", type=int, default=20, help="测试转速，默认 20 RPM")
    parser.add_argument("--duration", type=float, default=0.75, help="每个方向运行秒数，默认 0.75")
    parser.add_argument("--acceleration", type=int, default=5, help="加速度档，默认 5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    car = MecanumDrive(args.port, acceleration=args.acceleration)
    run_hardware_test(car, rpm=args.rpm, duration=args.duration, confirm=_confirm)


if __name__ == "__main__":
    main()

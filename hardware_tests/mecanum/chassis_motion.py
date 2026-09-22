"""Run timed whole-chassis mecanum movement checks.

Run on the vehicle from the project root:

    python -m hardware_tests.mecanum.chassis_motion
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Protocol

from mecanum import MecanumDrive


class ChassisDrive(Protocol):
    """The narrow controller interface used by this test sequence."""

    def initialize(self) -> "ChassisDrive": ...

    def enable_all(self) -> None: ...

    def disable_all(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def forward(self, rpm: int) -> None: ...

    def backward(self, rpm: int) -> None: ...

    def strafe_left(self, rpm: int) -> None: ...

    def strafe_right(self, rpm: int) -> None: ...

    def rotate_cw(self, rpm: int) -> None: ...

    def rotate_ccw(self, rpm: int) -> None: ...


MOTION_METHODS = (
    ("forward", "forward"),
    ("backward", "backward"),
    ("strafe_left", "strafe_left"),
    ("strafe_right", "strafe_right"),
    ("rotate_cw", "rotate_cw"),
    ("rotate_ccw", "rotate_ccw"),
)


def run_sequence(
    car: ChassisDrive,
    *,
    rpm: int,
    duration: float,
    pause: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run each whole-chassis movement for a bounded interval and stop between steps."""
    for label, method_name in MOTION_METHODS:
        print(f"测试动作: {label}")
        getattr(car, method_name)(rpm)
        sleep(duration)
        car.stop()
        sleep(pause)


def run_hardware_test(
    car: ChassisDrive,
    *,
    rpm: int,
    duration: float,
    pause: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run the sequence and always leave the chassis stopped, disabled, and closed."""
    try:
        car.initialize()
        car.enable_all()
        run_sequence(car, rpm=rpm, duration=duration, pause=pause, sleep=sleep)
    finally:
        car.stop()
        car.disable_all()
        car.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试麦轮小车的整车运动")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="四轮共用的 TTL 串口")
    parser.add_argument("--rpm", type=int, default=30, help="测试转速，默认 30 RPM")
    parser.add_argument("--duration", type=float, default=1.0, help="每个动作运行秒数，默认 1.0")
    parser.add_argument("--pause", type=float, default=0.5, help="动作之间的停止等待秒数，默认 0.5")
    parser.add_argument("--acceleration", type=int, default=5, help="加速度档，默认 5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.pause < 0:
        raise ValueError("duration must be positive and pause must not be negative")
    car = MecanumDrive(args.port, acceleration=args.acceleration)
    run_hardware_test(car, rpm=args.rpm, duration=args.duration, pause=args.pause)


if __name__ == "__main__":
    main()

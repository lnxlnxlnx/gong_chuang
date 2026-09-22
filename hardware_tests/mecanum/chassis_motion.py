"""Run timed whole-chassis mecanum movement checks.

Run on the vehicle from the project root:

    python -m hardware_tests.mecanum.chassis_motion
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .config import DEFAULT_CONFIG_PATH, create_drive, load_profile


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
    parser.add_argument("--profile", default="jetson", help="配置档名，默认 jetson")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="JSON 配置文件路径")
    parser.add_argument("--port", help="本次运行覆盖配置中的 TTL 串口")
    parser.add_argument("--rpm", type=int, help="本次运行覆盖配置中的测试转速")
    parser.add_argument("--duration", type=float, help="本次运行覆盖配置中的单动作时长")
    parser.add_argument("--pause", type=float, help="本次运行覆盖配置中的动作间停止时间")
    parser.add_argument("--acceleration", type=int, help="本次运行覆盖配置中的加速度档")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile, args.config)
    duration = profile.chassis_duration if args.duration is None else args.duration
    pause = profile.chassis_pause if args.pause is None else args.pause
    if duration <= 0 or pause < 0:
        raise ValueError("duration must be positive and pause must not be negative")
    car = create_drive(profile, port=args.port, acceleration=args.acceleration)
    run_hardware_test(
        car,
        rpm=profile.chassis_rpm if args.rpm is None else args.rpm,
        duration=duration,
        pause=pause,
    )


if __name__ == "__main__":
    main()

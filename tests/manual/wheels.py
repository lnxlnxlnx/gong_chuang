"""Run each mecanum wheel forward and reverse at a safe low speed.

Run on the vehicle from the project root:

    python test.py  # 选择 3
"""

from __future__ import annotations

MENU_ENTRY = (3, "麦轮逐轮")

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from mecanum import WHEEL_ORDER, WheelPosition

from mecanum_config import DEFAULT_CONFIG_PATH, create_drive, load_profile


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
        print(f"[{wheel.name}] enabled")
        car.set_wheel_speed(wheel, rpm)
        print(f"[{wheel.name}] forward {rpm} RPM acknowledged")
        sleep(duration)
        car.stop_wheel(wheel)
        print(f"[{wheel.name}] stopped")
        car.set_wheel_speed(wheel, -rpm)  # 反向转动
        print(f"[{wheel.name}] reverse {rpm} RPM acknowledged")
        sleep(duration)
        car.stop_wheel(wheel)
        print(f"[{wheel.name}] stopped")
        car.enable_wheel(wheel, False)  # disable to avoid heating up the motor
        print(f"[{wheel.name}] disabled")


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
    default_profile = "windows" if os.name == "nt" else "jetson"
    parser.add_argument(
        "--profile", default=default_profile, help=f"配置档名，默认 {default_profile}"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="JSON 配置文件路径"
    )
    parser.add_argument("--port", help="本次运行覆盖配置中的 TTL 串口")
    parser.add_argument("--rpm", type=int, help="本次运行覆盖配置中的测试转速")
    parser.add_argument("--duration", type=float, help="本次运行覆盖配置中的单方向时长")
    parser.add_argument("--acceleration", type=int, help="本次运行覆盖配置中的加速度档")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile, args.config)
    duration = profile.individual_duration if args.duration is None else args.duration
    if duration <= 0:
        raise ValueError("duration must be positive")
    car = create_drive(profile, port=args.port, acceleration=args.acceleration)
    run_hardware_test(
        car,
        rpm=profile.individual_rpm if args.rpm is None else args.rpm,
        duration=duration,
        confirm=_confirm,
    )


if __name__ == "__main__":
    main()

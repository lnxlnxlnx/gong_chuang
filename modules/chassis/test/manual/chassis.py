"""Run timed whole-chassis mecanum movement checks interactively.

Run on the vehicle from the project root:

    python test.py  # 选择 4

交互: 每个动作按 Enter 执行并可在当前动作上反复重做，
输入 n 进入下一个动作，输入 q 结束并停止整个底盘。
"""

from __future__ import annotations

MENU_ENTRY = (4, "麦轮底盘")

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from modules.chassis.config import DEFAULT_CONFIG_PATH, HardwareProfile, load_profile
from modules.chassis.drive.mecanum import MecanumDrive


def _create_drive(
    profile: HardwareProfile,
    *,
    port: str | None = None,
    acceleration: int | None = None,
) -> MecanumDrive:
    """这个工具直接驱动四轮控制器（不看整车门面），所以自己建 ``MecanumDrive``。

    原来是 ``modules.chassis.config.create_drive``；那个工厂已退出——装配现在只有
    :meth:`modules.chassis.runtime.MecanumChassis.open` 一处，而本工具要的是**驱动**而不是
    整车门面，所以按同一份 profile 自己建，端口照旧来自 motor 配置。
    """
    from modules.motor.config import load_bus_config

    return MecanumDrive(
        port if port is not None else load_bus_config().port,
        motor_ids=profile.motor_ids,
        wheel_signs=profile.wheel_signs,
        acceleration=profile.acceleration if acceleration is None else acceleration,
        pattern=profile.pattern,
        rotation_gain=profile.rotation_gain,
        motion_transform=profile.motion_transform,
    )


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

    def forward_left(self, rpm: int) -> None: ...

    def forward_right(self, rpm: int) -> None: ...

    def backward_left(self, rpm: int) -> None: ...

    def backward_right(self, rpm: int) -> None: ...


MOTION_METHODS = (
    ("forward", "forward"),
    ("backward", "backward"),
    ("strafe_left", "strafe_left"),
    ("strafe_right", "strafe_right"),
    ("forward_left", "forward_left"),
    ("forward_right", "forward_right"),
    ("backward_left", "backward_left"),
    ("backward_right", "backward_right"),
    ("rotate_cw", "rotate_cw"),
    ("rotate_ccw", "rotate_ccw"),
)


def run_sequence(
    car: ChassisDrive,
    *,
    rpm: int,
    duration: float,
    pause: float,
    prompt: Callable[[str], str],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run each whole-chassis movement for a bounded interval, confirmed one at a time.

    ``prompt`` 在每次执行前被调用，返回用户输入并去除首尾空白;
    返回 "n" 表示跳过当前动作进入下一个，返回 "q" 表示结束整个序列，
    其余输入（直接回车等）都会执行当前动作；执行完毕后再次等待输入，
    因此同一个动作可以反复重做，直到输入 n/q 为止。
    """
    for label, method_name in MOTION_METHODS:
        print(f"准备测试动作: {label}")
        while True:
            choice = prompt(f"[{label}] 执行本动作? 回车=执行/重做, n=下一个, q=退出: ").strip()
            if choice == "q":
                return
            if choice == "n":
                break
            getattr(car, method_name)(rpm)
            sleep(duration)
            car.stop()
            print(f"[{label}] 已停止")
        sleep(pause)


def run_hardware_test(
    car: ChassisDrive,
    *,
    rpm: int,
    duration: float,
    pause: float,
    prompt: Callable[[str], str],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run the sequence and always leave the chassis stopped, disabled, and closed."""
    try:
        car.initialize()
        car.enable_all()
        run_sequence(car, rpm=rpm, duration=duration, pause=pause, prompt=prompt, sleep=sleep)
    finally:
        car.stop()
        car.disable_all()
        car.close()


def _prompt(message: str) -> str:
    return input(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试麦轮小车的整车运动")
    default_profile = "windows" if os.name == "nt" else "jetson"
    parser.add_argument("--profile", default=default_profile, help=f"配置档名，默认 {default_profile}")
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
    car = _create_drive(profile, port=args.port, acceleration=args.acceleration)
    run_hardware_test(
        car,
        rpm=profile.chassis_rpm if args.rpm is None else args.rpm,
        duration=duration,
        pause=pause,
        prompt=_prompt,
    )


if __name__ == "__main__":
    main()

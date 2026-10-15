"""Recorded wheel routes and their hardware lifecycle."""

from __future__ import annotations

import time
from collections.abc import Callable

from modules.chassis.config import HardwareProfile
from modules.chassis.runtime import MecanumChassis


def _port() -> str:
    """步进总线的端口来自 motor 配置（与 arm 共用同一根线）。"""
    from modules.motor.config import load_bus_config

    return load_bus_config().port


def _confirmed(label: str) -> bool:
    """Ask the operator before moving the chassis."""
    return input(f"准备{label}，回车执行，输入 q 跳过：").strip().lower() != "q"


def demo_run_one_map(profile: HardwareProfile) -> None:
    """Run the first recorded route with the configured wheel calibration."""
    with MecanumChassis._from_settings(profile=profile, port=_port()) as car:
        if _confirmed("运行地图"):
            car.forward_left(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)
            car.forward(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
            car.left(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
            car.backward(rpm=profile.chassis_rpm, duration=15 * profile.chassis_duration)
            car.right(rpm=profile.chassis_rpm, duration=16 * profile.chassis_duration)
            car.backward_right(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)


def func_run_circle_map(car, profile: HardwareProfile) -> None:
    """Drive the recorded full-circle route on an already enabled chassis."""
    car.forward_left(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
    car.left(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
    car.backward_right(rpm=profile.chassis_rpm, duration=profile.chassis_duration)
    car.backward(rpm=profile.chassis_rpm, duration=15 * profile.chassis_duration)
    car.right(rpm=profile.chassis_rpm, duration=16 * profile.chassis_duration)
    car.backward_right(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)


def func_run_half_map(car, profile: HardwareProfile) -> None:
    """Drive the recorded half-map route and return path."""
    car.forward_left(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    time.sleep(1)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.left(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.backward(rpm=profile.chassis_rpm, duration=15 * profile.chassis_duration)
    car.left(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)

    car.right(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    time.sleep(0.1)
    car.backward(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
    time.sleep(0.1)
    car.left(rpm=profile.chassis_rpm, duration=7.5 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.right(rpm=profile.chassis_rpm, duration=16 * profile.chassis_duration)
    car.backward(rpm=profile.chassis_rpm, duration=8.5 * profile.chassis_duration)


def run_map_by_func(profile: HardwareProfile, func: Callable) -> None:
    """Confirm a route, then release the chassis even if it is interrupted."""
    with MecanumChassis._from_settings(profile=profile, port=_port()) as car:
        if _confirmed("运行地图"):
            func(car, profile)

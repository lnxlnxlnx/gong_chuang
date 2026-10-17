"""Short, operator-confirmed whole-chassis movement demo.

收尾统一交给上下文：``close`` 内部已经按"停 → 失能 → 释放驱动"的顺序做过，
脚本不再在它之前重复一遍 stop/disable。
"""

from __future__ import annotations

from modules.chassis.config import HardwareProfile
from modules.chassis.runtime import MecanumChassis


def _port() -> str:
    """步进总线的端口来自 motor 配置（与 arm 共用同一根线）。"""
    from modules.motor.config import load_bus_config

    return load_bus_config().port


def demo_chassis(profile: HardwareProfile) -> None:
    """Move the configured chassis forward and always release the motors."""
    with MecanumChassis._from_settings(profile=profile, port=_port()) as car:
        if input("准备底盘前进，回车执行，输入 q 跳过：").strip().lower() != "q":
            car.forward(rpm=profile.chassis_rpm, duration=10 * profile.chassis_duration)

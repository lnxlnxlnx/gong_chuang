"""Short, operator-confirmed whole-chassis movement demo."""

from __future__ import annotations

from modules.chassis.config import HardwareProfile, create_chassis


def demo_chassis(profile: HardwareProfile) -> None:
    """Move the configured chassis forward and always release the motors."""
    car = create_chassis(profile)
    try:
        car.initialize()
        car.enable_all()
        if input("准备底盘前进，回车执行，输入 q 跳过：").strip().lower() != "q":
            car.forward(rpm=profile.chassis_rpm, duration=10 * profile.chassis_duration)
    finally:
        car.stop()
        car.disable_all()
        car.close()

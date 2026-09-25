"""Real-hardware demo of MecanumDrive on the Jetson board.

在板子上运行（需先接好四台 Emm42 驱动器，共用一个 USB-TTL）：

    python test.py  # 选择 8

默认使用 `jetson` 配置；Windows 可传入 `windows`，串口和硬件地址都从配置文件读取。

注意：
  - 脚本全程 try/finally 保证结束时松轴并关闭串口；
  - 若某个方向反了，参考 README 用 pattern / wheel_signs / vx取反 修正。
"""

from __future__ import annotations

import time
import os

from mecanum_config import create_chassis, load_profile
from mecanum import WheelPosition


def main(profile_name: str | None = None) -> None:
    if profile_name is None:
        profile_name = "windows" if os.name == "nt" else "jetson"
    profile = load_profile(profile_name)
    car = create_chassis(profile).initialize()
    try:
        car.enable_all()
        time.sleep(0.2)

        # 依次演示：前进 → 后退 → 左平移 → 右平移 → 左旋 → 右旋
        car.forward(rpm=30, duration=2.0)
        car.backward(rpm=30, duration=2.0)
        car.strafe_left(rpm=20, duration=1.5)
        car.strafe_right(rpm=20, duration=1.5)
        car.rotate_ccw(rpm=20, duration=1.0)
        car.rotate_cw(rpm=20, duration=1.0)

        # 底层用法：单轮验证转向
        print("single wheel FL 5s ...")
        car.set_wheel_speed(WheelPosition.FL, signed_rpm=20, acceleration=5)
        time.sleep(5.0)
        car.stop()
    finally:
        car.disable_all()
        car.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)

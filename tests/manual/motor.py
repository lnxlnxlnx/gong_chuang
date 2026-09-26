"""Gentle end-to-end test for the Emm42 stepper driver on the configured port.

测试流程：

1. 使能扭矩，电机抱轴
2. 相对运动：正向 45°，60rpm，加速度 5 → 等待 1s
3. 相对运动：反向 -45°，60rpm，加速度 5 → 等待 1s
4. 连续速度模式：CW 30rpm 跑 1.5s → `motor.stop()` 停机
5. 连续速度模式：CCW 30rpm 跑 1.5s → `motor.stop()` 停机
6. 正常结束；异常时进入 except，finally 块保证：停止电机、关闭扭矩、关闭串口
"""

MENU_ENTRY = (5, "单电机基础测试")

import os
import sys
import time

from motor_drivers.emm42 import Direction, Emm42Driver, HomeMode
from mecanum_config import load_profile


def main() -> None:
    port = load_profile("windows" if os.name == "nt" else "jetson").port
    motor_id = 1
    print(f"[*] Motor test: {port}, id={motor_id}")
    motor = Emm42Driver(port, motor_id=motor_id).initialize()
    print("[*] serial opened OK")
    try:
        print("[*] enable torque (hold shaft)...")
        motor.enable(True)
        time.sleep(1.0)

        print("[*] short positive motion: move_relative(+45°, rpm=60, accel=5)")
        motor.move_relative(degrees=45.0, rpm=60, acceleration=5)
        time.sleep(1.0)

        print("[*] motion back: move_relative(-45°, rpm=60, accel=5)")
        motor.move_relative(degrees=-45.0, rpm=60, acceleration=5)
        time.sleep(1.0)

        print("[*] swap direction via set_speed: CW@30rpm 1.5s, then stop")
        motor.set_speed(Direction.CW, rpm=30, acceleration=5)
        time.sleep(1.5)
        motor.stop()
        time.sleep(0.5)

        motor.set_speed(Direction.CCW, rpm=30, acceleration=5)
        time.sleep(1.5)
        motor.stop()
        time.sleep(0.5)

        print("[*] motor test PASSED")
    except Exception as error:
        print(f"[!] motor test FAILED: {error!r}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            motor.stop()
        except Exception:
            pass
        motor.enable(False)
        motor.close()
        print("[*] torque released, port closed")


if __name__ == "__main__":
    main()

"""Manual direct bus-servo trial; separate from the controller path."""

from __future__ import annotations

MENU_ENTRY = (14, "夹爪直连调试")

import argparse
import time
from collections.abc import Callable

from modules.servo.drive.gripper import GripperConfig, load_gripper_config
from modules.servo.drive.bus_servo import BusServo


def run_direct_test(
    servo: BusServo,
    config: GripperConfig,
    *,
    prompt: Callable[[str], str] = input,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Check the reply, then confirm and verify each direct servo move."""
    print(f"直连舵机 ID {config.servo_id} 当前脉冲位置：{servo.read_position()}")
    for label, position in (("张开", config.open_position), ("闭合", config.closed_position)):
        answer = prompt(
            f"直连调试：将夹爪{label}到 {position}，耗时 {config.duration_ms} 毫秒；"
            "输入 YES 执行，其他输入跳过："
        )
        if answer.strip() != "YES":
            print(f"已跳过{label}")
            continue
        servo.set_position(position, config.duration_ms)
        sleep(config.duration_ms / 1000 + 0.2)
        print(f"{label}后读回：{servo.read_position()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="115200 bps 夹爪直连调试，非正式控制板路径")
    parser.add_argument("--port", default=None, help="覆盖 config/servo.json 中的平台串口")
    args = parser.parse_args()
    config = load_gripper_config()
    port = args.port if args.port is not None else config.port
    servo = BusServo(port, config.servo_id, timeout=0.5).initialize()
    try:
        run_direct_test(servo, config)
    finally:
        servo.close()


if __name__ == "__main__":
    main()

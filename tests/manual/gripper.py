"""Confirmed open/close test for the gripper servo."""

from __future__ import annotations

MENU_ENTRY = (13, "夹爪测试")

import time
from collections.abc import Callable

from gripper import GripperConfig, load_gripper_config
from motor_drivers.bus_servo import BusServo, load_bus_config


def run_manual(
    servo: BusServo,
    config: GripperConfig,
    *,
    prompt: Callable[[str], str] = input,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Read first; move to each configured position only after its own YES, then read back."""
    print(f"夹爪舵机 {config.servo_id} 当前脉冲位置：{servo.read_position()}")
    for label, position in (("张开", config.open_position), ("闭合", config.closed_position)):
        answer = prompt(
            f"夹爪将{label}到 {position} 脉冲，耗时 {config.duration_ms} 毫秒。"
            "确认夹爪有活动空间后输入 YES 执行，其他输入跳过："
        )
        if answer.strip() != "YES":
            print(f"已跳过{label}")
            continue
        servo.set_position(position, config.duration_ms)
        sleep(config.duration_ms / 1000 + 0.2)
        print(f"{label}后读回：{servo.read_position()}")


def main() -> None:
    """Open the configured bus servo, then close the UART on every exit path."""
    config = load_gripper_config()
    bus = load_bus_config()
    print(f"夹爪舵机 {config.servo_id} @ {bus.port} {bus.baudrate} bps")
    try:
        servo = BusServo(
            bus.port, config.servo_id, baudrate=bus.baudrate, timeout=0.5
        ).initialize()
    except Exception as error:
        # 端口被占用或设备名不对时直接说明，不再抛 traceback
        print(f"无法打开 {bus.port}：{error}")
        return
    try:
        try:
            run_manual(servo, config)
        except (RuntimeError, TimeoutError) as error:
            print(f"夹爪测试未通过：{error}")
    finally:
        servo.close()


if __name__ == "__main__":
    main()

"""Interactive position test for the Hiwonder 115200 bps bus servo."""

from __future__ import annotations

MENU_ENTRY = (12, "总线舵机测试")

import time
from collections.abc import Callable

from motor_drivers.bus_servo import BusServo, load_bus_config


def _number(text: str, name: str, minimum: int, maximum: int) -> int:
    """Validate one manually entered position or duration."""
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum}..{maximum} 之间")
    return value


def run_manual(servo: BusServo, servo_id: int, *, prompt: Callable[[str], str]) -> None:
    """Read first; send one position command only after an explicit YES."""
    current = servo.read_position()
    print(f"舵机 {servo_id} 当前脉冲位置：{current}")
    target_text = prompt("目标位置 0..1000（输入 q 只读不动）：").strip()
    if target_text.lower() == "q":
        return
    target = _number(target_text, "目标位置", 0, 1000)
    duration = _number(prompt("运动时长 1..30000 毫秒：").strip(), "运动时长", 1, 30000)
    if prompt(f"将舵机 {servo_id} 从 {current} 移到 {target}，输入 YES 确认：").strip() != "YES":
        print("已取消运动")
        return

    servo.set_position(target, duration)
    time.sleep(duration / 1000)
    print(f"运动后脉冲位置：{servo.read_position()}")


def main() -> None:
    """Read the port, baud rate and ID from config, then close the UART on every exit path."""
    config = load_bus_config()
    print(f"总线舵机 {config.port} @ {config.baudrate} bps，ID={config.servo_id}")
    try:
        servo = BusServo(
            config.port, config.servo_id, baudrate=config.baudrate, timeout=0.5
        ).initialize()
    except Exception as error:
        # 端口被占用或设备名不对时直接说明，不再抛 traceback
        print(f"无法打开 {config.port}：{error}")
        return
    try:
        try:
            run_manual(servo, config.servo_id, prompt=input)
        except (RuntimeError, TimeoutError) as error:
            print(f"舵机测试未通过：{error}")
    finally:
        servo.close()


if __name__ == "__main__":
    main()

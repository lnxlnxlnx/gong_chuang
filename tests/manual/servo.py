"""Interactive position test for the old project's 9600 baud servo controller."""

MENU_ENTRY = (12, "舵机控制板测试")

import json
import os
import time
from pathlib import Path
from typing import Callable

from motor_drivers.servo_controller import ServoController


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "servo.json"


def load_port(platform: str, config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    """Load the controller's serial port; reject an unset port."""
    try:
        settings = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取舵机配置：{config_path}") from error
    profile = settings.get(platform) if isinstance(settings, dict) else None
    port = profile.get("port") if isinstance(profile, dict) else None
    if not isinstance(port, str) or not port.strip():
        raise ValueError(f"请先填写 {config_path} 中的 {platform}.port")
    return port.strip()


def _number(text: str, name: str, minimum: int, maximum: int) -> int:
    """Validate one manually entered servo ID, position, or duration."""
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum}..{maximum} 之间")
    return value


def run_manual(controller: ServoController, servo_id: int, *, prompt: Callable[[str], str]) -> None:
    """Read first; send one position command only after an explicit YES."""
    current = controller.read_position(servo_id)
    print(f"舵机 {servo_id} 当前脉冲位置：{current}")
    target_text = prompt("目标位置 0..1000（输入 q 只读不动）：").strip()
    if target_text.lower() == "q":
        return
    target = _number(target_text, "目标位置", 0, 1000)
    duration = _number(prompt("运动时长 1..30000 毫秒：").strip(), "运动时长", 1, 30000)
    if prompt(f"将舵机 {servo_id} 从 {current} 移到 {target}，输入 YES 确认：").strip() != "YES":
        print("已取消运动")
        return

    controller.set_position(servo_id, position=target, duration_ms=duration)
    time.sleep(duration / 1000)
    print(f"运动后脉冲位置：{controller.read_position(servo_id)}")


def main() -> None:
    """Open only the configured controller UART, then close it on every exit path."""
    platform = "windows" if os.name == "nt" else "jetson"
    port = load_port(platform)
    id_text = input("舵机 ID 1..253（输入 q 退出）：").strip()
    if id_text.lower() == "q":
        return
    servo_id = _number(id_text, "舵机 ID", 1, 253)
    controller = ServoController(port).initialize()
    try:
        print(f"已连接舵机控制板串口 {port}，ID={servo_id}")
        try:
            run_manual(controller, servo_id, prompt=input)
        except (RuntimeError, TimeoutError) as error:
            print(f"舵机测试未通过：{error}")
    finally:
        controller.close()


if __name__ == "__main__":
    main()

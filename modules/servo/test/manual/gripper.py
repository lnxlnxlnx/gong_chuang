"""Confirmed open/close test for the gripper on the shared bus servo.

Menus 13 and 14 used to be two entries over the same 115200 bps direct path;
this is the single merged entry.  The bus port and baud rate come from the
servo bus configuration, while the positions come from the gripper section.
"""

from __future__ import annotations

MENU_ENTRY = (13, "夹爪测试")

import argparse
import time
from collections.abc import Callable

from modules.servo.config import GripperConfig, load_gripper_config
from modules.servo.drive.bus_servo import BusServo, load_bus_config


def run_manual(
    servo: BusServo,
    config: GripperConfig,
    *,
    prompt: Callable[[str], str] = input,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Read first; move to each configured position on its own confirmation, then read back.

    ``回车``就是确认——和转盘测试、机械臂测试用同一套手感；``q`` 结束剩下的动作，
    其他任何输入跳过这一项。

    ``sleep`` defaults to :func:`time.sleep` *at call time*: a default bound at
    definition time would ignore ``patch.object(time, "sleep")``.
    """
    sleep = sleep if sleep is not None else time.sleep
    print(f"夹爪舵机 {config.servo_id} 当前脉冲位置：{servo.read_position()}")
    for label, position in (("张开", config.open_position), ("闭合", config.closed_position)):
        answer = (
            prompt(
                f"夹爪将{label}到 {position} 脉冲，耗时 {config.duration_ms} 毫秒。"
                "确认夹爪有活动空间后按回车执行，输入 q 结束，其他输入跳过："
            )
            .strip()
            .lower()
        )
        if answer == "q":
            print("已结束后续夹爪动作")
            return
        if answer:
            print(f"已跳过{label}")
            continue
        servo.set_position(position, config.duration_ms)
        sleep(config.duration_ms / 1000 + 0.2)
        print(f"{label}后读回：{servo.read_position()}")


def parse_port(argv: list[str] | None = None) -> str | None:
    """Return the optional ``--port`` override for one-off wiring checks."""
    parser = argparse.ArgumentParser(description="夹爪测试（总线舵机直连）")
    parser.add_argument(
        "--port", default=None, help="覆盖 modules/servo/config.json 中的平台串口"
    )
    return parser.parse_args(argv).port


def main() -> None:
    """Open the bus at its configured baud rate, then close the UART on every exit path."""
    port_override = parse_port()
    config = load_gripper_config()
    bus = load_bus_config()
    port = port_override if port_override is not None else bus.port
    print(f"夹爪舵机 {config.servo_id} @ {port} {bus.baudrate} bps")
    try:
        servo = BusServo(
            port, config.servo_id, baudrate=bus.baudrate, timeout=0.5
        ).initialize()
    except Exception as error:
        # 端口被占用或设备名不对时直接说明，不再抛 traceback
        print(f"无法打开 {port}：{error}")
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

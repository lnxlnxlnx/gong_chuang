"""Confirmed open/close test for the Hiwonder gripper servo."""

from __future__ import annotations

MENU_ENTRY = (13, "夹爪控制板测试")

from collections.abc import Callable

from gripper import Gripper, load_gripper_config


def run_manual(gripper: Gripper, *, prompt: Callable[[str], str] = input) -> None:
    """Send each trial position only after a separate explicit confirmation."""
    config = gripper.config
    print(f"舵机 {config.servo_id} 当前脉冲位置：{gripper.read_position()}")
    for label, position, action in (
        ("张开", config.open_position, gripper.open_gripper),
        ("闭合", config.closed_position, gripper.close_gripper),
    ):
        answer = prompt(
            f"夹爪将{label}到 {position} 脉冲，耗时 {config.duration_ms} 毫秒。"
            "确认夹爪有活动空间后输入 YES 执行，其他输入跳过："
        )
        if answer.strip() == "YES":
            action()
        else:
            print(f"已跳过{label}")


def main() -> None:
    config = load_gripper_config()
    gripper = Gripper(config).initialize()
    try:
        run_manual(gripper)
    finally:
        gripper.close()


if __name__ == "__main__":
    main()

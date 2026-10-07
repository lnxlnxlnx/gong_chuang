"""Human-confirmed position-mode test for robot-arm motors 5, 6, and 7."""

from __future__ import annotations

MENU_ENTRY = (2, "机械臂三轴")

import argparse
import os
import sys
from typing import Callable, Optional, Tuple

from modules.arm.drive.robot_arm import RobotArm
from modules.arm.config import ArmConfig, load_arm_config
from modules.motor.config import load_bus_config

Action = Tuple[str, Optional[Callable[[int], object]]]


def run_sequence(
    arm: RobotArm,
    *,
    pulses: int,
    prompt: Callable[[str], str] = input,
) -> None:
    """Ask before each bounded relative move so the operator controls progression."""
    actions: Tuple[Action, ...] = (
        ("底座顺时针", arm.rotate_cw),
        ("底座逆时针", arm.rotate_ccw),
        ("机械臂向上", arm.move_up),
        ("机械臂向下", arm.move_down),
        ("机械臂前伸", arm.move_forward),
        ("机械臂后缩", arm.move_backward),
        ("确认机械臂后缩完成", None),
    )
    for label, action in actions:
        answer = (
            prompt(f"准备 {label} {pulses} 脉冲；回车执行，输入 q 结束：")
            .strip()
            .lower()
        )
        if answer == "q":
            return
        if action is None:
            continue
        report = action(pulses)
        print(
            f"{label}：电机 {report.motor_id}，{report.pulses} 脉冲，"
            f"电机轴 {report.motor_degrees:.4f}°"
            + (
                f"，底座输出轴 {report.output_degrees:.4f}°"
                if report.output_degrees is not None
                else ""
            )
            + f"（已按估算等待 {report.estimated_seconds:.2f}s 到位）"
        )


def run_hardware_test(
    arm: RobotArm,
    *,
    pulses: int,
    prompt: Callable[[str], str] = input,
) -> None:
    """Run the interactive sequence and always leave the arm stopped and unpowered."""
    try:
        arm.initialize()
        arm.enable_all()
        run_sequence(arm, pulses=pulses, prompt=prompt)
    finally:
        _cleanup(arm)


def _cleanup(arm: RobotArm) -> None:
    """Attempt every cleanup action without replacing the command that originally failed."""
    for name, action in (
        ("停止", arm.stop_all),
        ("失能", arm.disable_all),
        ("关闭串口", arm.close),
    ):
        try:
            action()
        except Exception as error:
            print(f"[!] 清理{name}失败：{error!r}", file=sys.stderr)


def parse_args(config: ArmConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZDT Emm42 机械臂三轴保守位置模式测试")
    default_profile = "windows" if os.name == "nt" else "jetson"
    parser.add_argument(
        "--port", default=load_bus_config().port,
        help="覆盖平台配置中的 Emm42 总线串口",
    )
    parser.add_argument(
        "--pulses",
        type=_pulse_count(config),
        default=config.default_pulses,
        metavar=f"1..{config.max_pulses}",
        help=f"每个相对位置动作的脉冲数，默认 {config.default_pulses}",
    )
    return parser.parse_args()


def _pulse_count(config: ArmConfig) -> Callable[[str], int]:
    """Build concise CLI validation from the shared configured pulse limit."""
    def parse(value: str) -> int:
        try:
            pulses = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not 1 <= pulses <= config.max_pulses:
            raise argparse.ArgumentTypeError(f"must be in 1..{config.max_pulses}")
        return pulses

    return parse


def main() -> None:
    config = load_arm_config()
    args = parse_args(config)
    arm = RobotArm(args.port, config=config)
    run_hardware_test(arm, pulses=args.pulses)


if __name__ == "__main__":
    main()

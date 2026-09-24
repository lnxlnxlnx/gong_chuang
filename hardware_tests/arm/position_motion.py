"""Human-confirmed position-mode test for robot-arm motors 5, 6, and 7."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from robot_arm import RobotArm

Action = tuple[str, Callable[[int], object]]


def run_sequence(
    arm: RobotArm,
    *,
    pulses: int,
    prompt: Callable[[str], str] = input,
) -> None:
    """Ask before each bounded relative move so the operator controls progression."""
    actions: tuple[Action, ...] = (
        ("底座顺时针", arm.rotate_cw),
        ("底座逆时针", arm.rotate_ccw),
        ("机械臂向上", arm.move_up),
        ("机械臂向下", arm.move_down),
        ("机械臂前伸", arm.move_forward),
        ("机械臂后缩", arm.move_backward),
    )
    for label, action in actions:
        answer = (
            prompt(f"准备 {label} {pulses} 脉冲；回车执行，输入 q 结束：")
            .strip()
            .lower()
        )
        if answer == "q":
            return
        report = action(pulses)
        print(
            f"{label}：电机 {report.motor_id}，{report.pulses} 脉冲，"
            f"电机轴 {report.motor_degrees:.4f}°"
            + (
                f"，底座输出轴 {report.output_degrees:.4f}°"
                if report.output_degrees is not None
                else ""
            )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZDT Emm42 机械臂三轴保守位置模式测试")
    parser.add_argument(
        "--port", required=True, help="Emm42 总线串口，例如 /dev/ttyUSB0 或 COM28"
    )
    parser.add_argument(
        "--pulses",
        type=int,
        default=600,
        choices=range(1, RobotArm.MAX_TEST_PULSES + 1),
        metavar="1..1000",
        help="每个相对位置动作的脉冲数，默认 600",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm = RobotArm(args.port)
    run_hardware_test(arm, pulses=args.pulses)


if __name__ == "__main__":
    main()

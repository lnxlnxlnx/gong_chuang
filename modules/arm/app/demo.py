"""Operator-confirmed arm demos; the port comes from the shared stepper bus.

收尾统一交给上下文/``close``：脚本不再在 ``close`` 之前额外 ``stop`` / ``disable``——
那两件事已经是 ``close`` 内部按顺序做的事，重复一遍只会让"停机到底发生了几次"说不清。
"""

from __future__ import annotations

from modules.arm.config import load_arm_config
from modules.arm.runtime import RobotArm


def _confirmed(label: str) -> bool:
    """Only execute a hardware action after the operator confirms it."""
    return input(f"准备{label}，回车执行，输入 q 跳过：").strip().lower() != "q"


def demo_arm(port: str) -> None:
    """Run one bounded position command in each arm direction."""
    with RobotArm._from_settings(config=load_arm_config(), port=port) as arm:
        motions = (
            ("底座顺时针", arm.rotate_cw),
            ("底座逆时针", arm.rotate_ccw),
            ("机械臂向上", arm.move_up),
            ("机械臂向下", arm.move_down),
            ("机械臂前伸", arm.move_forward),
            ("机械臂后缩", arm.move_backward),
        )
        for label, action in motions:
            if _confirmed(label):
                action(arm.config.default_pulses)


def arm_grab(port: str) -> None:
    """Run the shorter grab-position sequence used while tuning the gripper."""
    with RobotArm._from_settings(config=load_arm_config(), port=port) as arm:
        pulses = arm.config.default_pulses
        motions = (
            # ("底座顺时针", arm.rotate_cw, 5 * pulses),  # 差不多90度
            # ("底座逆时针", arm.rotate_ccw, 5 * pulses),
            # ("机械臂前伸", arm.move_forward, 3 * pulses),
            # ("机械臂后缩", arm.move_backward, 3 * pulses),
            # 默认差不多2cm->3个默认是5cm
            ("机械臂向上", arm.move_up, 3 * pulses),
            ("机械臂向下", arm.move_down, 3 * pulses),
            ("机械臂前伸", arm.move_forward, pulses),
        )
        for label, action, count in motions:
            if _confirmed(label):
                action(count)

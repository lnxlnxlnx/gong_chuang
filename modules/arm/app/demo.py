"""Operator-confirmed arm demos; the port comes from the shared stepper bus."""

from __future__ import annotations

from modules.arm.config import load_arm_config
from modules.arm.drive.robot_arm import RobotArm


def _confirmed(label: str) -> bool:
    """Only execute a hardware action after the operator confirms it."""
    return input(f"准备{label}，回车执行，输入 q 跳过：").strip().lower() != "q"


def demo_arm(port: str) -> None:
    """Run one bounded position command in each arm direction."""
    config = load_arm_config()
    arm = RobotArm(port, config=config)
    try:
        arm.initialize()
        arm.enable_all()
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
                action(config.default_pulses)
    finally:
        arm.stop_all()
        arm.disable_all()
        arm.close()


def arm_grab(port: str) -> None:
    """Run the shorter grab-position sequence used while tuning the gripper."""
    config = load_arm_config()
    arm = RobotArm(port, config=config)
    try:
        arm.initialize()
        arm.enable_all()
        motions = (
            # ("底座顺时针", arm.rotate_cw, 5 * config.default_pulses),  # 差不多90度
            # ("底座逆时针", arm.rotate_ccw, 5 * config.default_pulses),
            # ("机械臂前伸", arm.move_forward, 3 * config.default_pulses),
            # ("机械臂后缩", arm.move_backward, 3 * config.default_pulses),
            # 默认差不多2cm->3个默认是5cm
            ("机械臂向上", arm.move_up, 3 * config.default_pulses),
            ("机械臂向下", arm.move_down, 3 * config.default_pulses),
            ("机械臂前伸", arm.move_forward, config.default_pulses),
        )
        for label, action, pulses in motions:
            if _confirmed(label):
                action(pulses)
    finally:
        arm.stop_all()
        arm.disable_all()
        arm.close()

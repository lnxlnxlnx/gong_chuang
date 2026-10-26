'''
用于main中跑图、执行任务节点
'''

from __future__ import annotations

import time
from collections.abc import Callable

from gripper import place_box_from_repo_to_floor_v2

from modules.arm import MotionReport
from modules.chassis.config import HardwareProfile, load_profile
from modules.chassis.runtime import MecanumChassis
from modules.motion import MotionBatch, MotionBatchReport
from robot import Robot


def func_run_half_map(car, profile: HardwareProfile) -> None:
    """Drive the recorded half-map route and return path."""
    car.forward_left(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    time.sleep(1)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.left(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.backward(rpm=profile.chassis_rpm, duration=15 * profile.chassis_duration)
    car.left(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)

    car.right(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    time.sleep(0.1)
    car.backward(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
    time.sleep(0.1)
    car.left(rpm=profile.chassis_rpm, duration=7.5 * profile.chassis_duration)
    car.forward(rpm=profile.chassis_rpm, duration=8 * profile.chassis_duration)
    car.right(rpm=profile.chassis_rpm, duration=16 * profile.chassis_duration)
    car.backward(rpm=profile.chassis_rpm, duration=8.5 * profile.chassis_duration)
    
if __name__ == "__main__":
    with Robot.open(grasp=True, gripper=True, turntable=True, chassis = True) as robot:
        robot.turntable.go_home()
        robot.gripper.open_jaws()
        
        place_box_from_repo_to_floor_v2(robot)
        
        
        input("完成，回车继续")

'''
用于main中跑图、执行任务节点
'''

from __future__ import annotations

import time
from collections.abc import Callable

from gripper import (
    grab_box_from_floor_to_repo,
    grab_box_from_turntable_to_repo,
    place_box_from_repo_to_floor,
    place_box_from_repo_to_floor_v2,
    place_box_to_box,
)

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
        
        ###################
        place_box_from_repo_to_floor_v2(robot)
        ###################
        
        # robot.chassis.forward_left(rpm = 60, pulses=5400)      # 90
        # robot.chassis.forward(rpm=60, pulses=9600)     # 240

        # time.sleep(1)
        # input("开始抓取物料")
        # grab_box_from_turntable_to_repo(robot)
        # robot.chassis.forward(rpm = 60, duration = 8.5 * 0.5)
        
        # input("移动到拐角")
        # robot.chassis.forward(rpm = 60, pulses=12200)
        # robot.chassis.rotate_ccw(rpm = 60, pulses=4650)
        # robot.chassis.forward(rpm = 60, duration = 4)
        
        # 移动到圆环位置
        # robot.chassis.forward(rpm=60, pulses=9600)
        
        # 放置物料
        # place_box_from_repo_to_floor(robot)
        
        # 抓取物料(需要注意抓取时不能撞到物料盘)
        # grab_box_from_floor_to_repo(robot, order=(3, 2, 1))
        
        # 移动到拐角
        # robot.chassis.forward(rpm= 60, pulses=14300)     # 6000 + 9600 - 1300
        # robot.chassis.rotate_ccw(rpm = 60, pulses=4650)
        
        # 移动到细加工区
        # robot.chassis.forward(rpm=60, pulses=10600)
        # place_box_from_repo_to_floor(robot)
        
        # 回来转盘处
        # robot.chassis.backward(rpm=60, pulses=2700)
        # robot.chassis.left(rpm=60, pulses=12000)
        # robot.chassis.rotate_cw(rpm = 60, pulses=4650 * 2)
        # grab_box_from_turntable_to_repo(robot)
        
        # 再次回到拐角
        # robot.chassis.forward(rpm=60, pulses=11000)
        # robot.chassis.rotate_ccw(rpm= 60, pulses=4650)
        
        # 再次放置粗加工区
        # robot.chassis.forward(rpm = 60, pulses=10000)
        # place_box_from_repo_to_floor(robot)
        
        
        # 再次抓取
        # robot.chassis.backward(rpm=60, pulses=1000)
        # grab_box_from_floor_to_repo(robot)
        
        # 再次grab_box_from_floor_to_repo(robot, order=(3, 2, 1))
        # grab_box_from_floor_to_repo(robot, order=(3, 2, 1))
        
        # 移动拐角
        # robot.chassis.forward(rpm= 60, pulses=14300)     # 6000 + 9600 - 1300
        # robot.chassis.rotate_ccw(rpm = 60, pulses=4650)
        
        # 移动到细加工区
        # robot.chassis.forward(rpm=60, pulses=10600)
        # input("放置物块")
        # place_box_to_box(robot, _want="yellow")
        # robot.chassis.forward(rpm= 60, pulses=1950)
        # place_box_to_box(robot, _want="red")
        # robot.chassis.forward(rpm= 60, pulses=1950)
        # place_box_to_box(robot, _want="blue")
        
        input("完成，回车继续")

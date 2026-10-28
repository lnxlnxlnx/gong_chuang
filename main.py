"""项目应用入口。

只保留**业务动作**：装配交给 :class:`robot.Robot`，动作一行一个。
菜单选择式的人工测试仍在 :mod:`test` 的菜单里（``python test.py``），那里不经过本文件。
"""

from __future__ import annotations

import time

from app.gripper import (
    center_chassis_on_ring,  # 车横向对到圆环正前方
    grab_material_from_floor_to_repo,
    grab_material_from_floor_to_repo_by_xy,
    grab_material_from_turntable_to_repo,  # 圆盘->仓库
    grab_material_from_turntable_to_repo_follow,  # 圆盘->仓库（跟着料走）
    move_axes_to,
    place_material_from_repo_to_floor,  # 仓库->地面圆环
    place_material_from_repo_to_floor_by_pose,  # 仓库->地面圆环
    place_material_from_repo_to_floor_from_center_ring,  # 仓库->地面圆环
    place_material_on_material,  # 物料到物料
    place_material_on_material_by_order,  # 物料到物料（按 order 依次放三个位置）
    square_chassis_to_ring_row,  # 车纵向对到圆环正前方
    place_and_grab_material
)

SCENE_STABLE_TIMEOUT_SECONDS = 20.0

# 预览演示的**模块级**导入：``demo_material`` 只定义函数、不开设备，真正的装配发生在
# 它内部（读配置 + ``Vision.from_settings``）；保持模块级也让既有入口用例能直接替换它。
from modules.grasp.drive.visual_servo_model import VisualServoError
from modules.vision.app.demo import demo_material
from robot import Robot

ROTATE_PULSE = 4510

def main() -> None:
    """默认演示：物料定位预览。

    其余演示留在各自模块里：``modules.arm.app.demo``、``modules.servo.app.demo``、
    ``modules.chassis.app``（路线）。

    **上层调用的样子**见 :func:`locate_material` 与 :func:`pick_manual`：
    一条 ``with`` 完成装配，动作一行一个，收尾交给 ``Robot``。
    """
    with Robot.open(grasp=True, gripper=True, turntable=True, chassis=True) as robot:
        robot.gripper.set_position(350)
        robot.turntable.go_home()
        move_axes_to(robot, {"lift": 495, "extension": 580, "base": 0}).wait()
        move_axes_to(robot, {"lift": 950, "base": -80}).wait()  # 先让环进视野
        # square_chassis_to_ring_row(robot)
        # input("对齐角度")
        # center_chassis_on_ring(robot, require_digit=2)
        place_material_from_repo_to_floor_from_center_ring(robot)
        input("放置完成，按回车继续")
        # place_material_from_repo_to_floor_by_pose(robot)
        grab_material_from_floor_to_repo_by_xy(robot)
        input("环心对准，按回车继续")
        # grab_material_from_turntable_to_repo_follow(robot)
        # input("抓取完成，按回车继续")
        # place_material_from_repo_to_floor_by_pose(robot)
        input("放置完成，按回车继续")
        place_material_on_material_by_order(robot)
        input("结束!!!!!!!!!!!")


def locate_material(*, config_path=None) -> None:
    """**上层调用的样子**：一条 ``with`` 完成装配，动作一行。

    这是 ``docs/模块调用与统一初始化方案.md`` §2 的目标用法，也是
    ``tests/test_robot.py`` 覆盖的那条路：``Robot`` 负责开设备与收尾，
    业务代码只读 ``robot.vision`` 并调用动作。
    """
    with Robot.open(vision=True, vision_config_path=config_path) as robot:
        report = robot.vision.locate()
        if report.best is not None:
            material = report.best
            print(
                f"物料：{material.name or '未知'}，"
                f"相机系 xyz=({material.x_mm:.1f}, {material.y_mm:.1f}, "
                f"{material.z_mm:.1f}) mm"
            )
        else:
            print("画面里没有通过校验的物料")


def pick_manual() -> None:
    with Robot.open(grasp=True, gripper=True, turntable=True) as robot:
        robot.gripper.set_position(280)
        robot.turntable.go_home()
        # robot.arm.move_up(pulses=6000)          # 675度
        robot.arm.move_axis_to_motor_degrees("lift", 675)
        robot.arm.move_forward(3200)
        grab_material_from_turntable_to_repo(robot)
        

def _place_material(robot: Robot):
    try:
        report = robot.vision.locate()
    except Exception as e:
        print("视觉识别失败", e)
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return
        
    if report.best is None:
        print("画面里没有通过校验的物料")
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return
    material = report.best
    print(
        f"物料：{material.name or '未知'}，"
        f"相机系 xyz=({material.x_mm:.1f}, {material.y_mm:.1f}, "
        f"{material.z_mm:.1f}) mm"
    )

    try:
        robot.grasp.align_xy()
    except Exception as e:
        print("对准失败", e)
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return
    
    # 下降进行抓取
    robot.arm.move_axis_to_motor_degrees("lift", 563)
    robot.gripper.set_position(40)
    # 抬起物料，放进仓库
    robot.arm.move_axis_to_motor_degrees("lift", 900)
    robot.arm.move_axis_to_motor_degrees("extension", 10)
    robot.arm.move_axis_to_motor_degrees("base", 1048)
    robot.gripper.set_position(280)
    # 回到初始位置
    robot.arm.move_axis_to_motor_degrees("extension", 270.0)
    robot.arm.move_axis_to_motor_degrees("base", 0)
    robot.arm.move_axis_to_motor_degrees("lift", 675)
    robot.arm.move_forward(3200)

def run_map():
    with Robot.open(grasp=True, gripper=True, turntable=True, chassis = True) as robot:
        robot.turntable.go_home()
        robot.gripper.open_jaws()
        
        ###############
        # move_axes_to(robot, {"lift": 595, "extension": 350, "base": 0}).wait()
        # move_axes_to(robot, {"lift": 950, "base": 390}).wait()  # 先让环进视野
        # while True:
        #     try:
        #         robot.grasp.follow_material("red")   # YOLO 一帧 → 水平走一步
        #     except:
        #         continue                              # 跟丢一帧不算失败
        # move_axes_to(robot, {"lift": 495, "extension": 580, "base": 0}).wait()
        # move_axes_to(robot, {"lift": 950, "base": -80}).wait()  # 先让环进视野
        
        # robot.chassis.rotate_cw(pulses=4510*4)
        ###############
        place_and_grab_material(robot)
        # grab_material_from_turntable_to_repo(robot)
        input("结束!!!!!!!!!!!!!!!")
        
        ###################
        # 移动到物料盘
        # robot.chassis.forward_left(rpm = 60, pulses=4000)      # 90
        # robot.chassis.forward(rpm=60, pulses=11500)     # 240

        # 抓取物料
        grab_material_from_turntable_to_repo(robot)     # 完成后在extension=270的位置
        robot.chassis.rotate_ccw(rpm= 60, pulses=ROTATE_PULSE)
        
        # 移动到粗加工区
        # time.sleep(0.3)
        # batch = robot.motion.begin()
        # batch.add(robot.chassis.forward(rpm= 60, pulses=22500, wait=False))
        # batch.add(move_axes_to(robot, {"lift": 950, "extension": 580, "base": -80}))
        # batch.wait()
        
        # place_and_grab_material(robot)
        robot.chassis.forward(rpm= 60, pulses=22500)
        robot.chassis.left(rpm= 60, pulses=1600)
        move_axes_to(robot, {"lift": 495, "extension": 580, "base": 0}).wait()
        move_axes_to(robot, {"lift": 950, "base": -80}).wait()  # 先让环进视野
        square_chassis_to_ring_row(robot)
        center_chassis_on_ring(robot, require_digit=2)
        robot.chassis.left(pulses=500)
        place_material_from_repo_to_floor_by_pose(robot)
        grab_material_from_floor_to_repo_by_xy(robot)
        # 到细加工区
        # robot.chassis.right(rpm = 60, pulses=13200)
        # time.sleep(0.1)
        # robot.chassis.rotate_cw(rpm=60, pulses=ROTATE_PULSE)
        # robot.chassis.right(rpm = 60, pulses=10600)
        # move_axes_to(robot, {"lift": 495, "extension": 580, "base": 0}).wait()
        # move_axes_to(robot, {"lift": 950, "base": -80}).wait()  # 先让环进视野
        # square_chassis_to_ring_row(robot)
        # center_chassis_on_ring(robot, require_digit=2)
        # time.sleep(0.1)
        # robot.chassis.left(pulses=500)
        # place_material_from_repo_to_floor_by_pose(robot)
        
        
        # 第二次到圆盘
        # robot.chassis.right(pulses=14000)
        # robot.chassis.backward(pulses=10300)
        # grab_material_from_turntable_to_repo(robot)
        
        # 第二次到粗加工区
        # robot.chassis.left(pulses=23200)
        # robot.chassis.rotate_ccw(pulses=ROTATE_PULSE)
        # robot.chassis.left(pulses=1600)
        # move_axes_to(robot, {"lift": 495, "extension": 580, "base": 0}).wait()
        # move_axes_to(robot, {"lift": 950, "base": -80}).wait()  # 先让环进视野
        # square_chassis_to_ring_row(robot)
        # center_chassis_on_ring(robot, require_digit=2)
        # robot.chassis.left(pulses=500)
        # place_material_from_repo_to_floor_by_pose(robot)
        # grab_material_from_floor_to_repo_by_xy(robot)
        
        # 第二次到细加工区
        # robot.chassis.right(rpm = 60, pulses=14000)
        # time.sleep(0.1)
        # robot.chassis.rotate_cw(rpm=60, pulses=ROTATE_PULSE)
        # robot.chassis.right(rpm = 60, pulses=9600)
        # move_axes_to(robot, {"lift": 495, "extension": 580, "base": 0}).wait()
        # move_axes_to(robot, {"lift": 950, "base": -80}).wait()  # 先让环进视野
        # place_material_on_material_by_order(robot)
        
        # 回到起点
        # robot.chassis.right(pulses=1600)
        # robot.chassis.backward(pulses=23500)
        # robot.chassis.right(pulses=13200)
        # robot.chassis.backward(pulses=1500)
        
        input("调试!!!!!!!!")
        ###################
        
        # robot.chassis.forward_left(rpm = 60, pulses=4800)      # 90
        # robot.chassis.forward(rpm=60, pulses=11600)     # 240

        # # time.sleep(1)
        # # input("开始抓取物料")
        # grab_material_from_turntable_to_repo(robot)     # 已完成
        
        
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


if __name__ == "__main__":
    # main()
    run_map()

"""项目应用入口。

只保留**业务动作**：装配交给 :class:`robot.Robot`，动作一行一个。
菜单选择式的人工测试仍在 :mod:`test` 的菜单里（``python test.py``），那里不经过本文件。
"""

from __future__ import annotations

import time

from app.gripper import (
    _move_axes_to,
    center_chassis_on_ring,  # 车横向对到圆环正前方
    grab_box_from_floor_to_repo,
    grab_box_from_floor_to_repo_v2,
    grab_box_from_turntable_to_repo,  # 圆盘->仓库
    grab_box_from_turntable_to_repo_v2,  # 圆盘->仓库（跟着料走）
    place_box_from_repo_to_floor,  # 仓库->地面圆环
    place_box_from_repo_to_floor_v2,  # 仓库->地面圆环
    place_box_to_box,  # 物料到物料
    place_box_to_box_v2,  # 物料到物料（按 order 依次放三个位置）
)

SCENE_STABLE_TIMEOUT_SECONDS = 20.0

# 预览演示的**模块级**导入：``demo_material`` 只定义函数、不开设备，真正的装配发生在
# 它内部（读配置 + ``Vision._open_config``）；保持模块级也让既有入口用例能直接替换它。
from modules.vision.app.demo import demo_material
from robot import Robot


def main() -> None:
    """默认演示：物料定位预览。

    其余演示留在各自模块里：``modules.arm.app.demo``、``modules.servo.app.demo``、
    ``modules.chassis.app``（路线）。

    **上层调用的样子**见 :func:`locate_material`、:func:`pick_ready` 与
    :func:`pick_manual`：一条 ``with`` 完成装配，动作一行一个，收尾交给 ``Robot``。
    """
    # demo_material()
    with Robot.open(grasp=True, gripper=True, turntable=True, chassis=True) as robot:
        robot.gripper.set_position(280)
        robot.turntable.go_home()
        _move_axes_to(robot, {"lift": 495, "extension": 540, "base": 0}).wait()
        _move_axes_to(robot, {"lift": 800, "base": -32}).wait()  # 先让环进视野
        residual = center_chassis_on_ring(robot, require_digit=2) 
        input(f"环心对准，残差 {residual:+.1f} mm，按回车继续")
        grab_box_from_turntable_to_repo_v2(robot)
        input("抓取完成，按回车继续")
        grab_box_from_floor_to_repo_v2(robot)
        place_box_from_repo_to_floor_v2(robot)
        input("放置完成，按回车继续")
        place_box_to_box_v2(robot)
        # grab_box_from_floor_to_repo_v2(robot)
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
        grab_box_from_turntable_to_repo(robot)
        

def _place_box(robot: Robot):
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


if __name__ == "__main__":
    main()
    # pick_manual()
    # test_place_box()
    # test_vision()
    # adjust_arm_align()
    # locate_material()
    # pick_ready()
    # test_vision()
"""项目应用入口。

只保留**业务动作**：装配交给 :class:`robot.Robot`，动作一行一个。
菜单选择式的人工测试仍在 :mod:`test` 的菜单里（``python test.py``），那里不经过本文件。
"""

from __future__ import annotations

import time

from app.gripper import (
    _move_axes_to,
    align_box,
    grab_box_from_car,
    grab_box_from_floor,
    grab_box_from_turntable,
    place_box_to_repo,
)

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
    with Robot.open(grasp=True, gripper=True, turntable=True) as robot:
        robot.gripper.set_position(250)
        _move_axes_to(robot, {"lift": 675, "extension": 750.0, "base": 0}).wait()
        time.sleep(1.0)
        # grab_box_from_car(robot)
        # grab_box_from_car(robot)
        # grab_box_from_car(robot)
        try:
            for _ in range(3):
                result = robot.grasp.align_ring()      # 只算：环心 → 各轴绝对位置，不动电机
                base_deg     = robot.arm.turns_state.of("base").motor_degrees
                extension_deg = robot.arm.turns_state.of("extension").motor_degrees
                
                input("回车抓料，q 跳过：").strip().lower()
                grab_box_from_car(robot)
                # 到这里已经抓着料了：转到 plan 里算好的**绝对位置**（环的正上方）。
                # plan 是 robot.grasp.plan_ring() 的返回值；plan.degrees 就是 {"base": ...,
                # "extension": ...} 两个绝对电机角度，正好喂 _move_axes_to。
                if result.plan is not None:
                    new_dir = dict({"lift": 95}, **{"base": base_deg, "extension": extension_deg})
                    _move_axes_to(robot, new_dir).wait()
                    
                    robot.gripper.open_jaws()
                _move_axes_to(robot, {"lift": 675, "extension": 750.0, "base": 0}).wait()
        except Exception as e:
            print(f"Error occurred during pick and place operation: {e}")
            robot.arm.move_axis_to_motor_degrees("extension", 270.0)
            return
        
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)


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
        
        # grab_box(robot)
        try:
            for color in ("red", "green", "blue"):
                grab_box_from_turntable(robot, want_color=color)
                place_box_to_repo(robot, verbose=True)
        except Exception as e:
            print(f"Error occurred during pick and place operation: {e}")
            robot.arm.move_axis_to_motor_degrees("extension", 270.0)
            return

        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        robot.gripper.set_position(150)
        return 
        
        try:
            for _ in range(3):
                plan = robot.grasp.plan_ring()      # 只算：环心 → 各轴绝对位置，不动电机
                grab_box_from_car(robot)
                # 到这里已经抓着料了：转到 plan 里算好的**绝对位置**（环的正上方）。
                # plan 是 robot.grasp.plan_ring() 的返回值；plan.degrees 就是 {"base": ...,
                # "extension": ...} 两个绝对电机角度，正好喂 _move_axes_to。
                if plan is not None:
                    new_dir = dict({"lift": 95}, **plan.degrees)
                    _move_axes_to(robot, new_dir).wait()
                    
                    robot.gripper.open_jaws()
                _move_axes_to(robot, {"lift": 675, "extension": 270.0, "base": 0}).wait()
        except Exception as e:
            print(f"Error occurred during pick and place operation: {e}")
            robot.arm.move_axis_to_motor_degrees("extension", 270.0)
            return

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
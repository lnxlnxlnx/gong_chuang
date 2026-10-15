"""项目应用入口。

只保留**业务动作**：装配交给 :class:`robot.Robot`，动作一行一个。
菜单选择式的人工测试仍在 :mod:`test` 的菜单里（``python test.py``），那里不经过本文件。
"""

from __future__ import annotations

import time

from modules.arm.drive.kinematics import CylindricalPose

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
    demo_material()


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


def pick_ready(*, grasp_config_path=None, vision_config_path=None) -> None:
    """**抓取就位的例子**：抓取借用同一个 arm 与 vision，是否闭爪由业务显式决定。

    ``grasp=True`` 已经包含 arm 与 vision，不会再开第二份设备。
    """
    with Robot.open(
        grasp=True,
        gripper=True,
        grasp_config_path=grasp_config_path,
        vision_config_path=vision_config_path,
    ) as robot:
        result = robot.grasp.align_and_descend()
        print(result.summary())
        # 就位成功不等于夹到了物料：闭不闭爪是业务的显式动作。
        robot.gripper.close_jaws()

def pick_manual() -> None:
    with Robot.open(grasp=True, gripper=True) as robot:
        # 高度自己点动：相对脉冲，不需要位姿，也不会自己往上冲。
        # robot.arm.move_up(pulses=4000)
        # robot.arm.home_all()

            # 开机对位：把"臂现在停的地方"按你量的数声明成初始位姿
        robot.arm.record_measurement(
            CylindricalPose(base_degrees=0.0, radial_mm=180.0, height_mm=0.0)
        )
        robot.arm.move_by(x_mm=0, y_mm=0, z_mm=50)
        time.sleep(1)
        
        # robot.arm.
        report = robot.vision.locate()
        if report.best is None:
            print("画面里没有通过校验的物料")
            return
        material = report.best
        print(
            f"物料：{material.name or '未知'}，"
            f"相机系 xyz=({material.x_mm:.1f}, {material.y_mm:.1f}, "
            f"{material.z_mm:.1f}) mm"
        )

        # x/y 对准 → 下降 → 后缩；一次都不动高度（除了这一次开环下降）。
        # robot.grasp.align_xy()
        # robot.arm.move_by(x_mm=material.x_mm, y_mm=material.y_mm)
        # pulses = int((material.z_mm) * 40)
        # robot.arm.move_down(pulses = pulses)
        # # print(result.summary())
        # # 就位成功不等于夹到了物料：闭不闭爪是业务的显式动作。
        # robot.gripper.close_jaws()
        # time.sleep(1)
        # robot.gripper.open_jaws()

def test_vision():
    with Robot.open(vision=True) as robot:
        report = robot.vision.locate()
        if report.best is None:
            print("画面里没有通过校验的物料")
            return
        print("z =", report.best.z_mm)      # 必须在 55–160 之间，最好 80–120
        print("坐标:", report.best.x_mm, report.best.y_mm, report.best.z_mm)

if __name__ == "__main__":
    # main()
    pick_manual()
    # locate_material()
    # pick_ready()
    # test_vision()
"""停盘标定转盘抓取的最后可见高度；不合爪、不入库。"""

import argparse
import math

MENU_ENTRY = (28, "转盘抓取可见高度标定")


def main(argv=None):
    """只有人工确认转盘静止后才开设备；结果由调用者填入 v2 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", choices=("red", "green", "blue"), default="green")
    parser.add_argument("--step", type=float, default=10.0, help="每次下降的电机角度，默认 10°")
    args = parser.parse_args(argv)
    if not math.isfinite(args.step) or args.step <= 0:
        parser.error("--step 必须为正的有限角度")
    input("请让转盘停止、确认物料保持静止，再按回车开始标定：")

    from app.gripper import calibrate_turntable_grab_visibility, move_axes_to
    from robot import Robot

    with Robot.open(grasp=True, gripper=True, on_message=print) as robot:
        # 先单独抬高，再改变平面姿态，避免低位横扫物料。
        robot.arm.move_axis_to_motor_degrees("lift", 830)
        move_axes_to(robot, {"base": 420, "extension": 120}).wait()
        cutoff = calibrate_turntable_grab_visibility(robot, args.color, lift_step=args.step)
        print(f"标定结束：请给 v2 传 monitor_until_lift={cutoff:.2f}，本次未抓料。")


if __name__ == "__main__":
    main()

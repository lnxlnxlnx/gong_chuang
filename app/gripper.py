from robot import Robot


def grab_box_from_floor(robot: Robot):
    robot.gripper.set_position(280)
    try:
        # position = robot.vision.locate()
        robot.grasp.align_xy()
    except Exception as e:
        print(f"Error occurred while aligning the gripper: {e}")
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return

    robot.arm.move_axis_to_motor_degrees("lift", 75)
    robot.gripper.close_jaws()

def place_box_to_repo(robot: Robot):
    '''当爪子上有物体时，将物体放置在指定位置。'''
    robot.arm.move_axis_to_motor_degrees("lift", 900, wait=False)
    robot.arm.move_axis_to_motor_degrees("extension", 10, wait=False)
    robot.arm.move_axis_to_motor_degrees("base", 1048, wait=False)
    robot.gripper.set_position(280)
    robot.arm.move_axis_to_motor_degrees("extension", 270.0, wait=False)
    robot.arm.move_axis_to_motor_degrees("base", 0, wait=False)
    robot.arm.move_axis_to_motor_degrees("lift", 675, wait=False)
    robot.arm.move_forward(3200)

def align_box(robot: Robot):
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
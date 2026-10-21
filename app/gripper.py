from __future__ import annotations

import time

from modules.motion import MotionBatch, MotionBatchReport
from robot import Robot

TURN_TABLE_POSITIONS = [0, 500, 1000]  # 转盘的目标位置，单位为度
current_position_index = 0  # 当前转盘位置的索引


def _move_axes_to(robot: Robot, targets: dict[str, float]) -> MotionBatch:
    """让多根轴各自走到**绝对电机角度**，同时起步；返回动作组（还没等）。

    用的是 :meth:`RobotArm.move_axis_to_motor_degrees`：它就是绝对电机角度
    （3200 脉冲 = 360°，不取模），``wait=False`` 只把帧发出去并返回票据。帧一到
    驱动器轴就开始走，所以"同时"发生在硬件上，不需要线程；``MotionBatch.wait()``
    是唯一的 join 点。

    帧之间留 ``arm.inter_frame_seconds``：CH340 上背靠背两条位置帧会双双失效，
    ``RobotArm.move_axes`` 的多机同步路径也是这么处理的。

    已经停在目标上的轴不会发帧（协议不允许 0 位移）——所以"机械臂完全不动"时，
    先用 ``verbose=True`` 看每一项实际走了多少脉冲。
    """
    arm = robot.arm
    batch = robot.motion.begin()
    names = list(targets)
    for index, axis_name in enumerate(names):
        batch.add(
            arm.move_axis_to_motor_degrees(axis_name, targets[axis_name], wait=False)
        )
        if index + 1 < len(names) and arm.inter_frame_seconds > 0:
            time.sleep(arm.inter_frame_seconds)
    return batch


def _print_report(title: str, report: MotionBatchReport) -> None:
    """把一次动作组的结论打成几行：哪根轴走了多少、估时多少、实等多少。"""
    print(f"[{title}] {report.summary()}")
    for item in report.items:
        result = item.result
        if item.error is not None or result is None:
            continue
        print(
            f"    {item.subject}: {result.pulses} 脉冲 / "
            f"{result.motor_degrees:.1f}° / 估算 {result.estimated_seconds:.2f}s"
            f" / 完成于 {result.waited_seconds:.2f}s"
        )

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

def grab_box_from_turntable(robot: Robot, want_color = None):
    robot.gripper.set_position(280)
    try:
        # position = robot.vision.locate()
        robot.grasp.align_xy(want = want_color)
    except Exception as e:
        print(f"Error occurred while aligning the gripper: {e}")
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return

    robot.arm.move_axis_to_motor_degrees("lift", 530)
    robot.gripper.close_jaws()
    
def grab_box_from_car(robot: Robot):
    """从车上抓料，最后**转到 plan 里算好的绝对位置**（环正上方）。

    ``plan`` 是 :meth:`modules.grasp.runtime.Grasp.plan_ring` 的返回值——它只算不动，
    所以可以在抓料之前就算好（那时靶标在视野里），抓完再走。``None`` 表示这次不转
    到环上（只抓料）。
    """
    global current_position_index
    batch = robot.motion.begin()
    batch.add(robot.arm.move_axis_to_motor_degrees("lift", 675))
    batch.add(robot.gripper.set_position(150, wait=False))
    batch.wait()
    report = _move_axes_to(robot, {"lift": 980, "extension": 5, "base": 1048}).wait()
    robot.gripper.set_position(100)
    robot.arm.move_axis_to_motor_degrees("lift", 830)
    
    robot.gripper.close_jaws()
    robot.arm.move_axis_to_motor_degrees("lift", 980)
    _move_axes_to(robot, {"extension": 400.0, "base": 500}).wait()
    current_position_index = (current_position_index + 1) % len(TURN_TABLE_POSITIONS)
    batch = robot.motion.begin()
    batch.add(
        robot.turntable.go_to(TURN_TABLE_POSITIONS[current_position_index], wait=False)
    )
    batch.add(_move_axes_to(robot, {"lift": 500, "extension": 400.0, "base": 0}))
    batch.wait()
    

def place_box_to_repo(robot: Robot, *, verbose: bool = False):
    '''当爪子上有物体时，将物体放置在指定位置。'''
    global current_position_index
    arm = robot.arm

    # ① 抬升 + 回缩 + 转底座：三轴各自发**绝对角度**帧 → 一起动 → 一个 join 点。
    #    注意：底座这次要转约 9300 脉冲，超过"不许和伸缩并行"的 600 脉冲上限。
    #    那条限制只在 RobotArm.move_axes 里检查，单轴帧不拦，所以现场必须确认
    #    工作区是空的（伸出状态下横扫会把整条臂扫过去）。
    report = _move_axes_to(robot, {"lift": 900, "extension": 10, "base": 1048}).wait()
    if verbose:
        _print_report("放料前", report)

    # ② 三轴都到位之后才松爪。open_jaws 默认 wait=True，自己等到舵机走完并读回。
    robot.gripper.open_jaws()
    
    report = _move_axes_to(robot, {"lift": 980, "extension": 270.0}).wait()
    if verbose:
        _print_report("回位到安全位置", report)

    # ③ 回初始姿态：同样三轴一起走、一起等。
    report = _move_axes_to(robot, {"lift": 675, "base": 0}).wait()
    if verbose:
        _print_report("回位", report)

    # ④ 转盘（舵机总线）与机械臂回位（步进总线）分属两条连接，可以真并发：
    #    两边各自只发命令，一个 batch.wait() 收尾。
    current_position_index = (current_position_index + 1) % len(TURN_TABLE_POSITIONS)
    batch = robot.motion.begin()
    batch.add(
        robot.turntable.go_to(TURN_TABLE_POSITIONS[current_position_index], wait=False)
    )
    batch.add(arm.move_forward(3200, wait=False))
    batch.wait()

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
from __future__ import annotations

import time

from modules.motion import MotionBatch, MotionBatchReport
from robot import Robot

TURN_TABLE_POSITIONS = [0, 500, 1000]  # 转盘的目标位置，单位为度
current_position_index = 0  # 当前转盘位置的索引

#: 下降前等画面停稳的上限（秒）。转盘刚停下时画面还会抖一下；
#: 超过这个时间仍在动就不下降——物块会顺着转盘走掉。
SCENE_STABLE_TIMEOUT_SECONDS = 3.0


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

def grab_box_from_turntable(
    robot: Robot,
    want_colors=("red", "green", "blue")
):
    """把物料从转盘上抓下来，放到仓库里。不需要处理初始状态
    从转盘上抓料：对准 → 确认下一次画面停稳 → 下降 → 合爪。

    物块在转盘上，x/y 对准成功不代表它已经停住；下降前用
    ``vision.wait_scene_stable`` 再确认一次——它**不接受**"当前这一眼稳定"，
    要等下一次独立判断也稳定才放行。一直等不到就报错，不落下去扑空
    （``scene_stable_timeout`` 是这段等待的上限）。
    """
    robot.arm.move_axis_to_motor_degrees("lift", 600)
    _move_axes_to(robot, {"lift": 720, "extension": 620.0, "base": 0}).wait()
    try:
        for color in want_colors :
            # wait_scene_stable 返回 True = 等到了"下一次停稳"，可以抓；
            # False = 超时仍在动，这次不能下降。
            time.sleep(3.5)  # 等转盘停稳，避免刚停下时画面还在抖
            if not robot.vision.wait_scene_stable(timeout=SCENE_STABLE_TIMEOUT_SECONDS):
                raise RuntimeError("画面一直在动，无法进行下一步操作")
            _grab_box_from_turntable(
                robot,
                want_color=color
            )
            place_box_to_repo(robot, verbose=True)
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return

    robot.arm.move_axis_to_motor_degrees("extension", 270.0)
    robot.gripper.set_position(150)
    return 

def _grab_box_from_turntable(
    robot: Robot,
    want_color=None
):
    """内部函数
    从转盘上抓料：对准 → 确认下一次画面停稳 → 下降 → 合爪。

    物块在转盘上，x/y 对准成功不代表它已经停住；下降前用
    ``vision.wait_scene_stable`` 再确认一次——它**不接受**"当前这一眼稳定"，
    要等下一次独立判断也稳定才放行。一直等不到就报错，不落下去扑空
    （``scene_stable_timeout`` 是这段等待的上限）。
    """
    # robot.gripper.set_position(280)

    try:
        # position = robot.vision.locate()
        robot.grasp.align_xy(want = want_color)
    except Exception as e:
        print(f"Error occurred while aligning the gripper: {e}")
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return


    robot.arm.move_axis_to_motor_degrees("lift", 550)
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
    robot.gripper.set_position(100)
    
    robot.arm.move_axis_to_motor_degrees("lift", 980)
    _move_axes_to(robot, {"extension": 400.0, "base": 500}).wait()
    current_position_index = (current_position_index + 1) % len(TURN_TABLE_POSITIONS)
    batch = robot.motion.begin()
    batch.add(
        robot.turntable.go_to(TURN_TABLE_POSITIONS[current_position_index], wait=False)
    )
    batch.add(robot.gripper.open_jaws(wait=False))
    batch.add(_move_axes_to(robot, {"lift": 720, "extension": 620.0, "base": 0}))
    batch.wait()

def place_box_to_floor(robot: Robot, *, verbose: bool = False):
    '''把物料放到地面的圆环内。不需要考虑初始状态。'''
    robot.gripper.open_jaws()
    _move_axes_to(robot, {"lift": 375, "extension": 750.0, "base": 0}).wait()
    time.sleep(1.0)

    try:
        for _ in range(3):
            result = robot.grasp.align_ring()      # 只算：环心 → 各轴绝对位置，不动电机
            base_deg     = robot.arm.turns_state.of("base").motor_degrees
            extension_deg = robot.arm.turns_state.of("extension").motor_degrees
            
            grab_box_from_car(robot)
            # 到这里已经抓着料了：转到 plan 里算好的**绝对位置**（环的正上方）。
            # plan 是 robot.grasp.plan_ring() 的返回值；plan.degrees 就是 {"base": ...,
            # "extension": ...} 两个绝对电机角度，正好喂 _move_axes_to。
            if result.plan is not None:
                new_dir = dict({"lift": 95}, **{"base": base_deg, "extension": extension_deg})
                _move_axes_to(robot, new_dir).wait()
                
                robot.gripper.open_jaws()
            _move_axes_to(robot, {"lift": 375, "extension": 750.0, "base": 0}).wait()
            input("放料完成，回车继续，输入 q 跳过：").strip().lower()
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return
    
    robot.arm.move_axis_to_motor_degrees("extension", 270.0)

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
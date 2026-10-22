from __future__ import annotations

import time

from modules.motion import MotionBatch, MotionBatchReport
from robot import Robot

TURN_TABLE_POSITIONS = [0, 500, 1000]  # 转盘的目标位置，单位为度
current_position_index = 0  # 当前转盘位置的索引

#: 下降前等画面停稳的上限（秒）。转盘刚停下时画面还会抖一下；
#: 超过这个时间仍在动就不下降——物块会顺着转盘走掉。
#:
#: **不要太短。** 实测 3 s 会卡在边界上：大角度转底座（放料那一下 base 转 1000° 以上）
#: 之后桌面要好几秒才真正静下来，等待窗口刚好在"还在动"的最后一点用光，于是判失败，
#: 而紧跟其后的复测已经是 1.4 灰阶 / 0.001%（远低于上限）。8 s 给了这段余振衰减的时间；
#: 真正的持续运动照旧会被拒绝，只是多等几秒才下结论。
SCENE_STABLE_TIMEOUT_SECONDS = 20.0


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

def _grab_box_from_floor(robot: Robot):
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

def _scene_stability_detail(robot: Robot) -> str:
    """把"等待中最后一次判据"和"此刻复测"都带回报错里。

    两个数要一起看：等待中只超一点点、事后复测已经很小 → 是**等得太短**（把
    ``SCENE_STABLE_TIMEOUT_SECONDS`` 调大）；两次都超很多 → 才是判据太紧或画面真的在动。
    """
    parts = []
    last = getattr(robot.vision, "last_scene_stability", None)
    if last is not None:
        parts.append(f"等待中最后一次：{last.summary()}")
    try:
        parts.append(f"此刻复测：{robot.vision.scene_stability().summary()}")
    except Exception as problem:  # 相机掉线之类：别让报错信息变成另一条异常
        parts.append(f"此刻复测失败（{problem}）")
    return "；".join(parts)


def grab_box_from_turntable_to_repo(
    robot: Robot,
    want_colors=("red", "green", "blue")
):
    """把物料从转盘上抓下来，放到仓库里。不需要处理初始状态
    从转盘上抓料：等它看满最小等待还静止 → 对准 → 下降 → 合爪。

    物块在转盘上，x/y 对准成功不代表它已经停住；下降前用
    ``vision.wait_scene_stable`` 等一次——头 3.5 s 的"静止"一律不作数（物块正要被
    带走之前画面也是静的），看满这段窗口之后仍然静止才放行；等不到就报错、
    不落下去扑空（``SCENE_STABLE_TIMEOUT_SECONDS`` 是这段等待的上限，**不能设太短**：
    大动作之后桌面余振要好几秒才衰减完）。
    """
    robot.arm.move_axis_to_motor_degrees("lift", 600)
    _move_axes_to(robot, {"lift": 720, "extension": 620.0, "base": 0}).wait()
    try:
        for color in want_colors :
            # wait_scene_stable 返回 True = 看满最小等待（3.5 s）之后画面仍然静止，可以抓；
            # False = 等满上限一直在动，这次不能下降。
            time.sleep(1.5)  # 先让转盘停一下，别一进去就赶上它在转
            if not robot.vision.wait_scene_stable(timeout=SCENE_STABLE_TIMEOUT_SECONDS):
                raise RuntimeError(
                    "画面迟迟没有停稳，无法进行下一步操作；最近一次判据："
                    + _scene_stability_detail(robot)
                )
            _grab_box_from_turntable(
                robot,
                want_color=color
            )
            _place_box_to_repo(robot, verbose=True)
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        robot.arm.move_axis_to_motor_degrees("extension", 270.0)
        return

    robot.arm.move_axis_to_motor_degrees("extension", 270.0)
    robot.gripper.set_position(150)
    return 

def place_box_from_floor_to_repo(robot: Robot, want_colors: list[str]):
    """把物料从地面圆环放到仓库里。不需要考虑初始状态。"""
    robot.arm.move_axis_to_motor_degrees("lift", 375)
    robot.arm.move_axis_to_motor_degrees("extension", 750.0)
    time.sleep(1.0)
    try:
        for color in want_colors:
            result = robot.grasp.align_xy(want=color)      # 只算：环心 → 各轴绝对位置，不动电机
            base_deg     = robot.arm.turns_state.of("base").motor_degrees
            extension_deg = robot.arm.turns_state.of("extension").motor_degrees
            
            _grab_box_from_car(robot)
            # 到这里已经抓着料了：转到 plan 里算好的**绝对位置**（环的正上方）。
            # plan 是 robot.grasp.plan_ring() 的返回值；plan.degrees 就是 {"base": ...,
            # "extension": ...} 两个绝对电机角度，正好喂 _move_axes_to。
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

def _grab_box_from_turntable(
    robot: Robot,
    want_color=None
):
    """内部函数：对准 → 下降 → 合爪（**不再**查停稳）。

    停稳那一道闸在 :func:`grab_box_from_turntable` 的循环里、对准之前做；这里只把
    已经对准的爪子落到物块上再合爪。
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
    
def _grab_box_from_car(robot: Robot):
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

def _place_box_to_repo(robot: Robot, *, verbose: bool = False):
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

def place_box_from_repo_to_floor(robot: Robot, *, verbose: bool = False):
    '''把物料放到地面的圆环内。不需要考虑初始状态。'''
    batch = robot.motion.begin()
    batch.add(robot.gripper.open_jaws(wait=False))
    batch.add(robot.arm.move_axis_to_motor_degrees("lift", 375, wait=False))
    print("移动到安全位置")
    batch.wait()
    time.sleep(1.0)
    robot.arm.move_axis_to_motor_degrees("extension", 750.0)
    time.sleep(1.0)

    try:
        for _ in range(3):
            result = robot.grasp.align_ring()      # 只算：环心 → 各轴绝对位置，不动电机
            base_deg     = robot.arm.turns_state.of("base").motor_degrees
            extension_deg = robot.arm.turns_state.of("extension").motor_degrees
            
            _grab_box_from_car(robot)
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

def place_box_to_box(robot: Robot, *, _want: str):
    robot.grasp.align_xy(want=_want)
    extension_deg = robot.arm.turns_state.of("extension").motor_degrees
    base_deg = robot.arm.turns_state.of("base").motor_degrees
    _grab_box_from_car(robot)
    _move_axes_to(robot, {"lift": 520, "extension": extension_deg, "base": base_deg}).wait()
    robot.gripper.open_jaws()
    robot.arm.move_axis_to_motor_degrees("lift", 695)
    _move_axes_to(robot, {"lift": 720, "extension": 550, "base": 0}).wait()


    
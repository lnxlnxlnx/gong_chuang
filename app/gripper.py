from __future__ import annotations

import time

from modules.arm import MotionReport
from modules.chassis.drive.mecanum import WHEEL_ORDER
from modules.motion import MotionBatch, MotionBatchReport
from modules.motor.drive.emm42 import POSITION_UNITS_PER_REVOLUTION
from robot import Robot

TURN_TABLE_POSITIONS = [0, 500, 1000]  # 转盘的目标位置，单位为度
current_position_index = 0  # 当前转盘位置的索引

#: 地面 1/2/3 号圆环相邻两格之间的间距，单位是**位置命令脉冲**（车轮口径）。
#: 现场实测值，换场地/改摆放就改这一个数。
RING_STEP_PULSES = 1950
FLOOR_RINGS = (1, 2, 3)

SCENE_STABLE_TIMEOUT_SECONDS = 20.0

#: 三轴待命中转姿态（升降抬到工作高度、伸缩伸到够料位、底座回正）。
#: 只给一部分键也合法：``_move_axes_to`` 没给的轴一帧都不发。
READY_POSE = {"lift": 720, "extension": 370.0, "base": 350}

#: 出事时的安全回缩：只把伸缩轴收回**初始位**——``modules/arm/hardware.json`` 的
#: ``initial_motor_degrees.extension = 270.0``。和 ``READY_POSE`` 不是一回事：那个是
#: 伸手够料的中转位，这个是缩回来让出工作区，改一个不该顺手改另一个。
SAFE_RETRACT = {"extension": 270.0}


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


def _go_home(robot: Robot, home: dict[str, float] | None) -> None:
    """走到本次流程的入口姿态；``home=None`` 表示这次不走（跳过重复往返）。"""
    if home is not None:
        _move_axes_to(robot, home).wait()


def _print_report(title: str, report: MotionBatchReport) -> None:
    """把一次动作组的结论打成几行：哪根轴走了多少、估时多少、实等多少。"""
    print(f"[{title}] {report.summary()}")
    for item in report.items:
        result = item.result
        # 动作组允许混装：转盘/舵机的结论没有 pulses 这些字段，不是臂的报告就跳过。
        # 这里不用 assert——混装是合法用法，assert 会把"别的模块的正常结果"炸成异常。
        if item.error is not None or not isinstance(result, MotionReport):
            continue
        print(
            f"    {item.subject}: {result.pulses} 脉冲 / "
            f"{result.motor_degrees:.1f}° / 估算 {result.estimated_seconds:.2f}s"
            f" / 完成于 {result.waited_seconds:.2f}s"
        )

def _check_order(order) -> tuple:
    """环号必须都是 1/2/3：写错一格车会多走 1950 脉冲，而动作看起来完全正常。"""
    digits = tuple(int(digit) for digit in order)
    unknown = [digit for digit in digits if digit not in FLOOR_RINGS]
    if unknown:
        raise ValueError(f"order 里的环号只能是 {FLOOR_RINGS}，收到 {digits}")
    return digits


def _grid_counts(chassis, pulses: int) -> list:
    """前进 ``pulses`` 脉冲时四轮各自的带符号脉冲数。

    符号跟底盘自己的运动学走（``motion_transform="calibrated"`` 下前进是三负一正），
    这里不自己按"四轮同号"估——那样车会拧着走。
    """
    drive = chassis.drive
    raw = drive._kinematics(1.0, 0.0, 0.0)
    peak = max(abs(value) for value in raw)
    return [
        round(value / peak * pulses * drive.wheel_signs[wheel])
        for value, wheel in zip(raw, WHEEL_ORDER)
    ]


def _wheel_positions(chassis) -> list:
    """四轮的**实测**位置，换算成与位置命令同一把尺（命令脉冲）。

    ``0x36`` 给的是 1/65536 电机圈，位置命令用的是 ``pulses_per_revolution``，
    两者差一个常数；不换算会差 20 倍。
    """
    drive = chassis.drive
    to_pulses = drive.pulses_per_revolution / POSITION_UNITS_PER_REVOLUTION
    return [
        round(chassis.wheel(wheel).read_position() * to_pulses)
        for wheel in WHEEL_ORDER
    ]


def _ring_targets(chassis, *, start_digit: int) -> dict:
    """1/2/3 号位的四轮**绝对**脉冲位置；调用方保证车已停在 ``start_digit`` 号位前。

    编号沿前进方向递增，所以第 ``digit`` 号位 = ``start + (digit - start_digit) × 一格``。

    ponytail: 车没有场地里程计，起点靠人摆、轮子打滑看不见（编码器在电机侧，不是里程轮）。
    要更准就靠视觉编号兜底——``align_ring(require_digit=...)`` 已经会拒绝错环。
    """
    unit = _grid_counts(chassis, RING_STEP_PULSES)
    start = _wheel_positions(chassis)
    return {
        digit: [base + (digit - start_digit) * step for base, step in zip(start, unit)]
        for digit in FLOOR_RINGS
    }


def _drive_to_ring(chassis, targets, digit: int, *, rpm: int) -> None:
    """开到 ``digit`` 号位：读实测补差、走完等结算。已经在那一格就一步不走。

    发的是**相对**命令而不是绝对位置命令：``move_wheels_absolute_pulses`` 会把目标
    恰好为 0 的轮子整条跳过（那只是绝对零点，不是"不用动"），车会拧一下。
    目标本身是绝对的，所以中途丢一帧会在下一次补回来。
    """
    counts = [
        target - actual
        for target, actual in zip(targets[digit], _wheel_positions(chassis))
    ]
    if not any(counts):
        return
    chassis.move_wheels_relative_pulses(*counts, rpm=rpm)
    time.sleep(chassis.drive._travel_seconds(counts, rpm, None))


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
    want_colors=("red", "green", "blue"),
    *,
    home: dict[str, float] | None = READY_POSE,
):
    """把物料从转盘上抓下来，放到仓库里。不需要处理初始状态
    从转盘上抓料：等它看满最小等待还静止 → 对准 → 下降 → 合爪。

    物块在转盘上，x/y 对准成功不代表它已经停住；下降前用
    ``vision.wait_scene_stable`` 等一次——头 3.5 s 的"静止"一律不作数（物块正要被
    带走之前画面也是静的），看满这段窗口之后仍然静止才放行；等不到就报错、
    不落下去扑空（``SCENE_STABLE_TIMEOUT_SECONDS`` 是这段等待的上限，**不能设太短**：
    大动作之后桌面余振要好几秒才衰减完）。

    ``home`` 是进流程前先摆到的姿态，默认 :data:`READY_POSE`；传 ``None`` 表示这次不摆
    （调用方已经在这一姿态上时省一次往返；轴本来就停在目标上时也不会发帧）。
    """
    robot.arm.move_axis_to_motor_degrees("lift", 760)
    robot.arm.move_axis_to_motor_degrees("base", 450)
    robot.arm.move_axis_to_motor_degrees("extension", 10)
    
    # input("继续")
    try:
        for color in want_colors :
            # wait_scene_stable 返回 True = 看满最小等待（3.5 s）之后画面仍然静止，可以抓；
            # False = 等满上限一直在动，这次不能下降。
            time.sleep(1.5)  # 先让转盘停一下，别一进去就赶上它在转
            # interval=0.15：判一眼本身约 0.13–0.15 s（连拍 4 帧 + 差分），默认 0.3 s
            # 的间隔等于每次都白等一半时间；放行时刻由 min_seconds=3.5 定，不跟着变。
            if not robot.vision.wait_scene_stable(
                timeout=SCENE_STABLE_TIMEOUT_SECONDS, interval=0.15
            ):
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
        _move_axes_to(robot, SAFE_RETRACT).wait()
        return

    _move_axes_to(robot, SAFE_RETRACT).wait()
    robot.gripper.set_position(150)
    return 

def grab_box_from_floor_to_repo(
    robot: Robot,
    *,
    order=(3, 2, 1),
    rpm: int = 30,
    verbose: bool = False,
    home: dict[str, float] | None = {"lift": 760, "extension": 120.0, "base": 400},
):
    """把物料从地面圆环放到仓库里。不需要考虑初始状态。

    ``order`` 是**抓取顺序**（环号），位移由它派生：车约定停在 3 号环前，抓完
    ``order[i]`` 再开去 ``order[i + 1]``。默认 ``(3, 2, 1)`` = 老行为（一路后退；
    编号沿前进方向递增，所以从 3 号往回抓就是后退）。想跳着抓就传 ``(3, 1, 2)``——跨度
    由环号差算，不是"每抓一个固定后退一格"。

    抓哪个料不再按颜色选，而是按**环号**：``align_ring(require_digit=digit)`` 看到的
    不是目标环会直接报错。注意这条路径对准的是**环心**，``ring_offset_x/y_mm`` 的落料
    补偿照旧生效——料在环里偏得多时要按料心重新标一次补偿。

    ``home`` 默认是地面抓料位（伸缩比 :data:`READY_POSE` 短一点，够得到地面圆环）；
    传 ``None`` 表示这次不摆。
    """
    digits = _check_order(order)
    robot.arm.move_axis_to_motor_degrees("lift", 400)
    _go_home(robot, home)
    input("移动到抓料位置")
    # 圆环绝对位置在动手之前读一次：之后只补实测差值，中途丢帧不会累积。
    targets = _ring_targets(robot.chassis, start_digit=3)
    try:
        for digit in digits:
            _drive_to_ring(robot.chassis, targets, digit, rpm=rpm)
            time.sleep(1.5)  # 车停稳再对准
            robot.grasp.align_ring(require_digit=digit, tolerance_mm=2.0)
            robot.arm.move_axis_to_motor_degrees("lift", 90)
            robot.gripper.close_jaws()
            _place_box_to_repo(robot, verbose=verbose)
            robot.arm.move_axis_to_motor_degrees("extension", 120)
            input(f"已抓取 {digit} 号环的物料，回车继续，输入 q 跳过：").strip().lower()

    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        _move_axes_to(robot, SAFE_RETRACT).wait()
        return

    _go_home(robot, home)

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
        _move_axes_to(robot, SAFE_RETRACT).wait()
        return


    robot.arm.move_axis_to_motor_degrees("lift", 550, rpm=150)
    robot.gripper.close_jaws()
    
def _grab_box_from_car(robot: Robot):
    """从车上抓料，最后**转到 plan 里算好的绝对位置**（环正上方）。

    ``plan`` 是 :meth:`modules.grasp.runtime.Grasp.plan_ring` 的返回值——它只算不动，
    所以可以在抓料之前就算好（那时靶标在视野里），抓完再走。``None`` 表示这次不转
    到环上（只抓料）。
    """
    global current_position_index
    batch = robot.motion.begin()
    batch.add(robot.arm.move_axis_to_motor_degrees("lift", 675, wait=False))
    batch.add(robot.arm.move_axis_to_motor_degrees("base", 400, wait=False))
    batch.add(robot.gripper.set_position(150, wait=False))
    batch.wait()
    report = _move_axes_to(robot, {"lift": 980, "extension": 5, "base": 1045}).wait()
    robot.gripper.set_position(100)
    robot.arm.move_axis_to_motor_degrees("lift", 840)
    
    robot.gripper.close_jaws()
    
    robot.arm.move_axis_to_motor_degrees("lift", 980)
    robot.arm.move_axis_to_motor_degrees("base", 800)
    current_position_index = (current_position_index + 1) % len(TURN_TABLE_POSITIONS)
    # _move_axes_to(robot, {"lift": 760, "base": 450}).wait()
    batch.add(robot.arm.move_axis_to_motor_degrees("lift", 760, wait=False))
    batch.add(robot.arm.move_axis_to_motor_degrees("base", 400, wait=False))
    batch.add(robot.turntable.go_to(TURN_TABLE_POSITIONS[current_position_index], wait=False))
    batch.wait()

def _place_box_to_repo(robot: Robot, *, verbose: bool = False):
    '''当爪子上有物体时，将物体放置在指定位置。'''
    global current_position_index
    arm = robot.arm

    # ① 抬升 + 回缩 + 转底座：三轴各自发**绝对角度**帧 → 一起动 → 一个 join 点。
    #    注意：底座这次要转约 9300 脉冲，超过"不许和伸缩并行"的 600 脉冲上限。
    #    那条限制只在 RobotArm.move_axes 里检查，单轴帧不拦，所以现场必须确认
    #    工作区是空的（伸出状态下横扫会把整条臂扫过去）。
    robot.arm.move_axis_to_motor_degrees("lift", 500)
    _move_axes_to(robot, {"extension": 270.0, "base": 400, "lift": 900}).wait()
    report = _move_axes_to(robot, {"lift": 900, "extension": 10, "base": 1048}).wait()
    if verbose:
        _print_report("放料前", report)

    # ② 三轴都到位之后才松爪。open_jaws 默认 wait=True，自己等到舵机走完并读回。
    robot.gripper.set_position(100)
    
    robot.arm.move_axis_to_motor_degrees("lift", 980)
    
    robot.arm.move_axis_to_motor_degrees("base", 800)
    current_position_index = (current_position_index + 1) % len(TURN_TABLE_POSITIONS)
    batch = robot.motion.begin()
    batch.add(robot.arm.move_axis_to_motor_degrees("base", 200, wait=False))
    batch.add(robot.turntable.go_to(TURN_TABLE_POSITIONS[current_position_index], wait=False))
    batch.add(robot.gripper.open_jaws(wait=False))
    batch.wait()


def place_box_from_repo_to_floor(
    robot: Robot,
    *,
    order=(1, 2, 3),
    rpm: int | None = None,
    verbose: bool = False,
    home: dict[str, float] | None = {"lift": 760, "extension": 50.0, "base": 465},
):
    '''把物料放到地面的圆环内。不需要考虑初始状态。

    ``order`` 是**放置顺序**（环号），位移由它派生：车约定停在 1 号环前，放完
    ``order[i]`` 再开去 ``order[i + 1]``。``(1, 2, 3)`` = 老行为；``(2, 1, 3)`` =
    先放 2 号、退回 1 号、再开到 3 号。放完最后一格就停在那儿，不再多前进一格。

    入口那一段（开爪与抬升同组并行 → 伸缩伸出 → 两次停顿）走的是**放料位**，不是待命
    中转位，所以整段保持原样；``home`` 只用来在它**之前**额外摆一次姿态，
    ``None`` 表示不摆。
    '''
    digits = _check_order(order)
    batch = robot.motion.begin()
    batch.add(robot.gripper.open_jaws(wait=False))
    batch.add(robot.arm.move_axis_to_motor_degrees("lift", 375, wait=False))
    print("移动到安全位置")
    batch.wait()
    _go_home(robot, home)
    # 圆环绝对位置在动手之前读一次：之后每次只补实测差值，中途丢帧不会累积。
    targets = _ring_targets(robot.chassis, start_digit=1)
    # input("继续")

    try:
        for digit in digits:
            _drive_to_ring(robot.chassis, targets, digit, rpm=rpm)
            # 容差按 2 mm 收：全局 tolerance_mm 是 6 mm（还给抓料用），放料要的是"落在
            # 环心"，6 mm 的残差在 95 mm 的靶标上肉眼就看得出偏心。
            time.sleep(1.5)  # 先让车停稳
            # require_digit：看到的不是目标环就报错，不拿错环的角度去动机械臂。
            result = robot.grasp.align_ring(require_digit=digit, tolerance_mm=2.0)
            base_deg     = robot.arm.turns_state.of("base").motor_degrees
            extension_deg = robot.arm.turns_state.of("extension").motor_degrees
            
            _grab_box_from_car(robot)
            input("放料完成，回车继续，输入 q 跳过：").strip().lower()
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        _move_axes_to(robot, SAFE_RETRACT).wait()
        return
    
    _move_axes_to(robot, SAFE_RETRACT).wait()

def place_box_to_box(
    robot: Robot,
    *,
    _want: str,
    home: dict[str, float] | None = {"lift": 760, "extension": 50.0, "base": 475},
):
    # home 只用来在对准之前额外摆一次姿态；这个函数今天没有入口姿态动作，默认不摆。
    _go_home(robot, home)
    robot.grasp.align_xy(want=_want)
    extension_deg = robot.arm.turns_state.of("extension").motor_degrees
    base_deg = robot.arm.turns_state.of("base").motor_degrees
    _grab_box_from_car(robot)
    _move_axes_to(robot, {"lift": 480, "extension": extension_deg, "base": base_deg}).wait()
    robot.gripper.open_jaws()
    robot.arm.move_axis_to_motor_degrees("lift", 695)
    _go_home(robot, home)

def place_box_from_repo_to_floor_v2(
    robot: Robot,
    *,
    orders: tuple[int, int, int] = (1, 2, 3),
    home: dict[str, float] | None = {"lift": 760, "extension": 400, "base": -32},
):
    """把物料从转盘上抓下来，放到地面圆环里。不需要处理初始状态。

    ``home`` 是进流程前先摆到的姿态，默认 :data:`READY_POSE`；传 ``None`` 表示这次不摆
    （调用方已经在这一姿态上时省一次往返；轴本来就停在目标上时也不会发帧）。
    """
    robot.arm.move_axis_to_motor_degrees("lift", 400)
    _go_home(robot, home)
    RING_POSITION = [(-190, 800, 690), (-32, 800, 540), (100, 800, 690)]
    try:
        for order in orders:
            _move_axes_to(robot, {"lift": RING_POSITION[order - 1][1], "extension": RING_POSITION[order - 1][2], "base": RING_POSITION[order - 1][0]}).wait()
            # input("继续")       # 之后改为停顿1s
            time.sleep(1)  # 先让车停稳
            result = robot.grasp.align_ring(tolerance_mm=2, max_iterations = 5)
            base_deg = robot.arm.turns_state.of("base").motor_degrees
            extension_deg = robot.arm.turns_state.of("extension").motor_degrees
            
            _grab_box_from_car(robot)
            
            # 这趟的落点就是现在这个角度。抓料会把这几个轴都开走，所以先记下来：
            # new_dir = dict({"lift": 95}, **{"base": base_deg, "extension": extension_deg})
            _move_axes_to(robot, {"extension": extension_deg, "base": base_deg}).wait()
            robot.arm.move_axis_to_motor_degrees("lift", 95)
            
            robot.gripper.open_jaws()
            robot.arm.move_axis_to_motor_degrees("lift", 375)
            _go_home(robot, home)
            input("放料完成，回车继续，输入 q 跳过：").strip().lower()
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        _move_axes_to(robot, SAFE_RETRACT).wait()
        return

    _move_axes_to(robot, SAFE_RETRACT).wait()


def grab_box_from_floor_to_repo_v2(
    robot: Robot,
    *,
    orders: tuple[int, int, int] = (3, 2, 1),
    home: dict[str, float] | None = {"lift": 760, "extension": 400, "base": -32},
):
    """把物料从地面圆环放到仓库里。不需要考虑初始状态。

    ``home`` 是进流程前先摆到的姿态，默认 :data:`READY_POSE`；传 ``None`` 表示这次不摆
    （调用方已经在这一姿态上时省一次往返；轴本来就停在目标上时也不会发帧）。
    """
    robot.arm.move_axis_to_motor_degrees("lift", 400)
    _go_home(robot, home)
    RING_POSITION = [(-190, 580, 690), (-32, 580, 540), (120, 580, 690)]
    try:
        for order in orders:
            _move_axes_to(robot, {"lift": RING_POSITION[order - 1][1], "extension": RING_POSITION[order - 1][2], "base": RING_POSITION[order - 1][0]}).wait()
            time.sleep(1)  # 先让车停稳
            result = robot.grasp.align_xy(tolerance_mm=3.0)
            robot.arm.move_axis_to_motor_degrees("lift", 90)
            robot.gripper.close_jaws()
            _place_box_to_repo(robot, verbose=True)
            _go_home(robot, home)
            input("放料完成，回车继续，输入 q 跳过：").strip().lower()
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        _move_axes_to(robot, SAFE_RETRACT).wait()
        return
    _move_axes_to(robot, SAFE_RETRACT).wait()


#: 去台面三处**放置位**上方摆的姿态 ``(base, lift, extension)``，单位是电机角。
#: ``lift`` 就是放料高度（v1 的 480）：太低会压到已经在那里的料，太高夹爪打不开。
#: 换摆放/换台面就改这一张表。
_PLACE_POSITION = [(-190, 480, 690), (-32, 480, 540), (120, 480, 690)]


def place_box_to_box_v2(
    robot: Robot,
    *,
    orders: tuple[int, int, int] = (1, 2, 3),
    home: dict[str, float] | None = {"lift": 760, "extension": 270.0, "base": 0},
):
    """从仓库抓料，叠到台面上第 ``order`` 号位置的物料上；``orders`` 是**放置顺序**。

    每趟的顺序（与 :func:`place_box_to_box` 同一套动作，只是多了"先去哪个位置"）：

    1. 摆到 ``PLACE_POSITION[order - 1]``；
    2. ``align_xy`` 对准——没传 ``want``，按"离光轴最近的那个"对准；
    3. 记下对准后的底座/伸缩角；
    4. 回仓库抓一块料（:func:`_grab_box_from_car`，它会动三轴并转料盘）；
    5. 按记下的角度**回到刚才那个位置**、松爪、抬起。

    ``home`` 只在每趟开始前摆一次；``None`` = 不摆（调用方已经在这一姿态上）。
    """
    digits = _check_order(orders)
    try:
        for order in digits:
            base, lift, extension = _PLACE_POSITION[order - 1]
            _move_axes_to(robot, {"lift": lift, "extension": extension, "base": base}).wait()
            input("继续")       # 之后改为停顿1s
            robot.grasp.align_xy(tolerance_mm=2.0)
            # 抓料会把这几个轴都开走，所以先记下来：这趟的落点就是现在这个角度。
            base_deg = robot.arm.turns_state.of("base").motor_degrees
            extension_deg = robot.arm.turns_state.of("extension").motor_degrees
            _grab_box_from_car(robot)
            _move_axes_to(
                robot,
                {"extension": extension_deg, "base": base_deg},
            ).wait()
            robot.arm.move_axis_to_motor_degrees("lift", lift)
            robot.gripper.open_jaws()
            robot.arm.move_axis_to_motor_degrees("lift", 800)
            _go_home(robot, home)
            input(f"放置 {order} 号位完成，回车继续，输入 q 跳过：").strip().lower()
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        _move_axes_to(robot, SAFE_RETRACT).wait()
        return
    _move_axes_to(robot, SAFE_RETRACT).wait()
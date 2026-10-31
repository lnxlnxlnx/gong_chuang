from __future__ import annotations

import math
import time

from modules.arm import MotionReport
from modules.chassis.drive.mecanum import WHEEL_ORDER
from modules.motion import MotionBatch, MotionBatchReport
from modules.motor.drive.emm42 import POSITION_UNITS_PER_REVOLUTION
from modules.vision.drive.numbered_ring import derive_ring_row
from robot import Robot

TURNTABLE_POSITIONS = [0, 500, 1000]  # 转盘的目标位置，单位为度
_turntable_position_index = 0

#: 地面 1/2/3 号圆环相邻两格之间的间距，单位是**位置命令脉冲**（车轮口径）。
#: 现场实测值，换场地/改摆放就改这一个数。
RING_STEP_PULSES = 1950
FLOOR_RINGS = (1, 2, 3)

SCENE_STABLE_TIMEOUT_SECONDS = 20.0

#: 三轴待命中转姿态（升降抬到工作高度、伸缩伸到够料位、底座回正）。
#: 只给一部分键也合法：``move_axes_to`` 没给的轴一帧都不发。
READY_POSE = {"lift": 720, "extension": 370.0, "base": 350}

#: 出事时的安全回缩：只把伸缩轴收回**初始位**——``modules/arm/hardware.json`` 的
#: ``initial_motor_degrees.extension = 270.0``。和 ``READY_POSE`` 不是一回事：那个是
#: 伸手够料的中转位，这个是缩回来让出工作区，改一个不该顺手改另一个。
SAFE_RETRACT = {"extension": 270.0}


def move_axes_to(robot: Robot, targets: dict[str, float]) -> MotionBatch:
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
        move_axes_to(robot, home).wait()


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
        round(chassis.drive.get_wheel(wheel).read_position() * to_pulses)
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
    chassis.drive.move_wheels_relative_pulses(*counts, rpm=rpm)
    time.sleep(chassis.drive._travel_seconds(counts, rpm, None))


#: 车横向校准的门槛（毫米，相机系 x：右为正）。比臂的 2 mm 松一点：臂那边还会再对准一次。
RING_LATERAL_TOLERANCE_MM = 4.0
#: 还不知道"每脉冲挪多少毫米"时的第一步试走量（脉冲）。
LATERAL_TRIAL_PULSES = 300
#: 单步最多走多少脉冲：离得再远也别一脚油门扫过去。
LATERAL_MAX_PULSES = 3000
#: 读不到环时连读几帧再认输。一帧运动模糊 / 爪影扫过不该让整次校准失败。
RING_READ_ATTEMPTS = 5
#: 两次读环之间的间隔（秒）；一个来回约一次取帧，别把相机读空。
RING_READ_INTERVAL_SECONDS = 0.15
#: 车头与"三环圆心连线"的夹角门槛（度）。垂直就行了，所以目标倾角是 0。
RING_ROW_SQUARE_TOLERANCE_DEG = 1.0
#: 车头与"相机画面水平轴"的固定差角（度）。装出来的臂不是正对车头时补这个。
#: 它等于**车头摆正时**该量到的直线倾角：人眼把车头摆到与环排垂直，跑一次
#: :func:`square_chassis_to_ring_row` 读打印出来的倾角，把那个数填这里。
ROW_SQUARE_YAW_OFFSET_DEG = 4.01
#: 还不知道"每脉冲让倾角变多少度"时的第一步试转量（脉冲）。
ROW_ROTATE_TRIAL_PULSES = 150
#: 单步最多转多少脉冲：离得再远也别一把扫过去。
ROW_ROTATE_MAX_PULSES = 2000
#: 一帧看清三环、随后一次性放完三个环之前要摆的姿态（``lift``/``extension``/``base``，
#: 电机角度）：伸缩比逐环对准时短一点、镜头离地面远一些，整排 2×150 mm 才进得了画面。
#: 现场按取景改这一份；换场地只需要动它和 ``station.center_spacing_mm``。
RING_ROW_OBSERVE_POSE = {"lift": 960, "extension": 550, "base": -32}


def _read_ring(robot: Robot, *, digit: int | None = None, attempts: int = RING_READ_ATTEMPTS):
    """连读几帧找一个环；都找不到才返回 ``None``。"""
    for index in range(max(1, int(attempts))):
        if index:
            time.sleep(RING_READ_INTERVAL_SECONDS)
        target = robot.vision.locate_ring(digit=digit)
        if target is not None:
            return target
    return None


def _ring_diagnosis(robot: Robot) -> str:
    """把"这一帧为什么不能用"接到报错后面；没有（或换成假视觉）就什么也不加。"""
    detail = getattr(robot.vision, "last_ring_diagnosis", "")
    return f"；最近一次：{detail}" if isinstance(detail, str) and detail else ""


def center_chassis_on_ring(
    robot: Robot,
    *,
    require_digit: int | None = 2,
    tolerance_mm: float = RING_LATERAL_TOLERANCE_MM,
    max_iterations: int = 7,
    rpm: int = 30,
) -> float:
    """把**车**横向对到圆环正前方：读环心 x，按实测响应比解出该走多少轮脉冲。

    只对 x 闭环——前后（y）是臂的伸缩该管的，横向才是车的事。所以这个函数不碰机械臂，
    只动四个轮子；调用前先把臂摆到能看见环的姿态::

        move_axes_to(robot, {"lift": 580, "extension": 690, "base": 120}).wait()
        center_chassis_on_ring(robot, require_digit=2)

    第一步用固定试走量（``LATERAL_TRIAL_PULSES``）摸出"每脉冲让画面挪多少毫米"，之后
    按牛顿一步算到位：轮径、打滑、地面摩擦都不必预先标定，闭环自己学。方向反了（挪完
    反而更远）当场报错，不会越走越偏。

    :returns: 最后一次读到的横向残差（毫米，右为正）。走满 ``max_iterations`` 也没进
        容差时**不报错**：车停在最后一步，返回迭代过程中的**最小残差**（打印里两个数都在）。
    :raises RuntimeError: 画面里没有环 / 读数不可用 / 环号不符 / 方向反了。

    读环时**连读几帧才认输**（``RING_READ_ATTEMPTS``）：一帧运动模糊、或刚好被爪影
    扫过，不该让整次校准失败——这是"看一眼就抛"最容易误报的地方。

    环号是模型读的，代码不猜：同一帧里读出两个同号环（数字被料挡住、框贴边、离得
    太近时都会），这一帧**直接作废**、连读几帧仍如此就报错，而不是按分数挑一个。
    这就是"有时对到隔壁环"的来源：错的号会把车正对到 3 号环前面，而动作、残差、
    打印全都看起来完全正常；``square_chassis_to_ring_row`` 只要直线的倾角，所以编号
    错了它照样能用。

    **第一帧读到的那个环就是这次校准的环**：``require_digit`` 没给时按第一帧的号锁死，
    之后换号一律按"没看见这个环"报错。车一动，隔壁环进来是常事（只差一格），而
    "换了个环"在残差上就是跳 150 mm——拿它算方向会误报"横向运动学反了"，拿它算步长
    会把车开到隔壁那一格。
    """
    gain: float | None = None      # 每 +1 脉冲（左移）让环心 x 变化多少毫米
    previous: tuple[float, int] | None = None
    locked = require_digit          # 第一帧读到哪一号就锁定它（见下）
    best_residual: float | None = None   # 迭代到的最小残差，走不进容差时按它收工
    x_mm = 0.0
    for attempt in range(1, max(1, int(max_iterations)) + 1):
        target = _read_ring(robot, digit=locked)
        if target is None:
            raise RuntimeError(
                "画面里没有环：先把臂摆到能看见靶标的位置"
                + (f"（找的是 {locked} 号环）" if locked is not None else "")
                + _ring_diagnosis(robot)
            )
        if locked is not None and int(target.digit) != int(locked):
            raise RuntimeError(
                f"画面里的环是 {target.digit} 号，不是要求的 {locked} 号；"
                "拿另一个环横向对齐会把车开到错的位置"
            )
        if locked is None:
            locked = int(target.digit)
        reading = target.reading()
        if reading is None:
            raise RuntimeError("环心读数不可用（贴边 / 缺比例尺）：换个姿态再看")
        x_mm = float(reading[0])
        # 每一帧都报"这是几号环、模型多大把握"：认错环时残差照样收敛得很漂亮，
        # 唯一能当场看出来的就是这个数。
        print(f"  读到 {target.digit} 号：x={x_mm:+.1f} mm（模型 {target.score:.2f}）")

        # 先结算上一步：车真的动了多少，按脉冲折算；方向反了/没动就别再走。
        if previous is not None and previous[1]:
            observed = (x_mm - previous[0]) / previous[1]
            if observed <= 0.0:
                return 0.0
                raise RuntimeError(
                    f"车动了 {previous[1]:+d} 脉冲，画面里的环却没往反方向走"
                    f"（{previous[0]:+.1f} → {x_mm:+.1f} mm）：横向运动学反了、轮子没动，"
                    "或者打滑到量不出来"
                )
            gain = observed

        if abs(x_mm) <= tolerance_mm:
            print(f"  车已对正：横向残差 {x_mm:+.1f} mm（读了 {attempt} 帧）")
            return x_mm

        if best_residual is None or abs(x_mm) < abs(best_residual):
            best_residual = x_mm

        if gain is None:
            magnitude = LATERAL_TRIAL_PULSES
        else:
            magnitude = abs(int(round(-x_mm / gain)))
        magnitude = min(max(magnitude, 1), LATERAL_MAX_PULSES)
        # x>0 = 环在画面右边 = 车偏左 → 往右挪。底盘自己的方向方法会等到走完再返回。
        if x_mm > 0:
            robot.chassis.strafe_right(rpm=rpm, pulses=magnitude)
            step = -magnitude
        else:
            robot.chassis.strafe_left(rpm=rpm, pulses=magnitude)
            step = magnitude
        previous = (x_mm, step)

    # 走满步数不报错：车停在最后一步、把**最小残差**报出去。横移是一维的，没有"两个电机
    # 位置"可回退，而为了 0.3 mm 再挪一次不值当——真打滑时按最小残差继续还能干活。
    print(
        f"  车没进容差：{max_iterations} 步后残差 {x_mm:+.1f} mm，"
        f"最小残差 {best_residual if best_residual is not None else x_mm:+.1f} mm"
        f"（门槛 {tolerance_mm:.1f} mm）——检查轮子是不是打滑或撞到东西"
    )
    return float(best_residual if best_residual is not None else x_mm)


def _read_ring_row(robot: Robot, *, attempts: int = RING_READ_ATTEMPTS):
    """读**一帧**、把三环圆心那条直线解出来；凑不齐就重试几帧。

    凑齐 = 2 号环 + 至少一个相邻环都有可用读数。一帧运动模糊、或环刚好贴到画面边
    （贴边的框心不是环心，``reading()`` 会是 ``None``）都不该让整次解算失败。

    :returns: ``(row, middle_target)``。``middle_target`` 是 2 号环那一份
        ``RingTarget``：中间环的比例尺来源要从它身上取（同一帧只有这一把尺）。
    :raises RuntimeError: 连读 ``attempts`` 帧都凑不齐（带上最后看到的是哪些号）。
    """
    last_problem = "一帧都没读到"
    for index in range(max(1, int(attempts))):
        if index:
            time.sleep(RING_READ_INTERVAL_SECONDS)
        targets = robot.vision.locate_rings()
        readings = {
            digit: target.reading()
            for digit, target in targets.items()
            if target.reading() is not None
        }
        if 2 not in readings:
            last_problem = f"这一帧没有可用的 2 号环心（看到 {sorted(targets)}）"
            continue
        if len(readings) < 2:
            last_problem = "只看到 2 号环，定不出直线（1 号或 3 号要进画面且不贴边）"
            continue
        middle = targets[2]
        scale = middle.scale_mm_per_px
        if scale is None or not math.isfinite(scale) or scale <= 0:
            raise RuntimeError("三环直线解不出来：2 号环没有有效比例尺")
        middle_x, middle_y = readings[2]
        # 各框松紧不同，各自换算毫米会把水平的像素连线扭斜；同帧统一用中环那把尺。
        readings = {
            digit: (
                middle_x + (targets[digit].center_px[0] - middle.center_px[0]) * scale,
                middle_y + (targets[digit].center_px[1] - middle.center_px[1]) * scale,
            )
            for digit in readings
        }
        row = derive_ring_row(readings, spacing_mm=robot.vision.ring_center_spacing_mm)
        return row, targets[2]
    raise RuntimeError(f"三环直线解不出来：{last_problem}")


def square_chassis_to_ring_row(
    robot: Robot,
    *,
    yaw_offset_deg: float = ROW_SQUARE_YAW_OFFSET_DEG,
    tolerance_deg: float = RING_ROW_SQUARE_TOLERANCE_DEG,
    max_iterations: int = 6,
    rpm: int = 15,
) -> float:
    """转**车**让车头与三环圆心那条直线垂直：读一帧倾角，按实测响应比一步到位。

    倾角是 1→3 号环心连线的方向角（度，画面右为 0、下为正）。相机与车头若严格同向，
    车头正对时这条线在画面里是水平的（倾角 0）；实际装出来差的那几度就是
    ``yaw_offset_deg``——**车头摆正时应该量到的倾角**。第一步用固定试转量摸出"每脉冲
    转多少度"，之后按牛顿一步算到位：轮径、地面打滑、四轮方向都不必预先标定，闭环自己学。

    调 ``yaw_offset_deg``：人眼把车头摆到与环排垂直，跑一次本函数读打印出来的倾角，
    把那个数填进 :data:`ROW_SQUARE_YAW_OFFSET_DEG`（或直接按次传参）。之后它就停在
    那个角度上，而不是停在 0。它是**当前观察姿态下**的量：改了 ``RING_ROW_OBSERVE_POSE``
    里的 ``base``（底座转了几度，画面就跟着转几度）之后要重新调一次。

    与 :func:`center_chassis_on_ring` 只差在动哪个轴：那个用横移把环心推到光轴上，
    这个用自转把环排的倾角转到目标角。调用前先把臂摆到能同时看见 2 号与至少一个相邻环
    的姿态（见 :data:`RING_ROW_OBSERVE_POSE`）。只想读一个数、不想让车转（标定要的就是
    这个）就传 ``tolerance_deg=180``：取一帧就返回，倾角打在打印里。

    :returns: 最后量到的倾角**相对目标**的残差（度）；``0`` = 已经垂直。
    :raises RuntimeError: 凑不出三环直线、车转了却量不到角度变化、或转满步数仍不垂直。
    """
    gain: float | None = None  # 每 +1 脉冲（顺时针转）让直线倾角变化多少度
    previous: tuple[float, int] | None = None
    yaw = 0.0
    residual = 0.0
    for attempt in range(1, max(1, int(max_iterations)) + 1):
        row, _middle = _read_ring_row(robot)
        yaw = row.yaw_deg
        residual = (yaw - yaw_offset_deg + 90.0) % 180.0 - 90.0
        # 先结算上一步：车真的转了多少度；一步都没转就别再猜。
        if previous is not None and previous[1]:
            change = (residual - previous[0] + 90.0) % 180.0 - 90.0
            observed = change / previous[1]
            if abs(observed) < 1e-6:
                raise RuntimeError(
                    f"车转了 {previous[1]:+d} 脉冲，直线倾角却一点没变"
                    f"（残差 {previous[0]:+.2f}° → {residual:+.2f}°）："
                    "轮子没动、打滑到量不出来，或者相机根本没跟着车转"
                )
            gain = observed
        if abs(residual) <= tolerance_deg:
            print(
                f"  车头已与三环直线垂直：倾角 {yaw:+.2f}°"
                f"（目标 {yaw_offset_deg:+.2f}°、残差 {residual:+.2f}°；读了 {attempt} 帧）"
            )
            return residual
        correction = (
            math.copysign(ROW_ROTATE_TRIAL_PULSES, residual)
            if gain is None else -residual / gain
        )
        magnitude = abs(int(round(correction)))
        magnitude = min(max(magnitude, 1), ROW_ROTATE_MAX_PULSES)
        # 第一段试转后，方向必须服从实测响应，不能把算出的符号丢掉。
        if correction > 0:
            robot.chassis.rotate_cw(rpm=rpm, pulses=magnitude)
            step = magnitude
        else:
            robot.chassis.rotate_ccw(rpm=rpm, pulses=magnitude)
            step = -magnitude
        previous = (residual, step)

    raise RuntimeError(
        f"{max_iterations} 步还没把车头转到与三环直线垂直：最后倾角 {yaw:+.2f}°"
        f"（目标 {yaw_offset_deg:+.2f}°、残差 {residual:+.2f}°、门槛 {tolerance_deg:.2f}°）："
        "检查轮子是不是打滑或撞到东西"
    )


def _grab_material_from_floor(robot: Robot):
    robot.gripper.set_position(280)
    try:
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


def grab_material_from_turntable_to_repo(
    robot: Robot,
    want_colors=("red", "green", "blue"),
    *,
    home: dict[str, float] | None = {"lift": 860, "extension": 90.0, "base": 420}
):  # 420是中心位置
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
    if robot.arm.read_abs_state("lift").motor_degrees < 460:
        robot.arm.move_axis_to_motor_degrees("lift", 460)
    _go_home(robot, home)

    # input("继续")
    try:
        for color in want_colors :
            # wait_scene_stable 返回 True = 看满最小等待（3.5 s）之后画面仍然静止，可以抓；
            # False = 等满上限一直在动，这次不能下降。
            # time.sleep(1)  # 先让转盘停一下，别一进去就赶上它在转
            # interval=0.15：判一眼本身约 0.13–0.15 s（连拍 4 帧 + 差分），默认 0.3 s
            # 的间隔等于每次都白等一半时间；放行时刻由 min_seconds=3.5 定，不跟着变。
            if not robot.vision.wait_scene_stable(
                timeout=SCENE_STABLE_TIMEOUT_SECONDS, interval=0.15
            ):
                raise RuntimeError(
                    "画面迟迟没有停稳，无法进行下一步操作；最近一次判据："
                    + _scene_stability_detail(robot)
                )
            while True:
                res = _grab_material_from_turntable(
                    robot,
                    want_color=color
                )
                if res:
                    continue
                break
            # 将当前机械臂的状态保存下来
            extension_position = robot.arm.read_abs_state("extension").motor_degrees
            base_position = robot.arm.read_abs_state("base").motor_degrees
            position = {"extension": extension_position, "base": base_position}
            _stow_material_to_repo(robot, position=position)
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        return

    move_axes_to(robot, SAFE_RETRACT).wait()
    robot.gripper.set_position(150)
    return 


#: 跟随转盘上那块料最多跟多久（秒）。转盘一次换位是 1.5 s，跟 20 s 还没停稳就是出事了
#: （料被带出可达范围、或画面一直有东西在动），报错比一直转下去好。
FOLLOW_SECONDS = 20.0


def _follow_material_until_stable(
    robot: Robot, want_color, *, timeout: float = FOLLOW_SECONDS,
    min_follow_seconds: float = 4.0, stable_seconds: float = 0.5,
) -> None:
    """**跟着转盘上的料走**，直到料自己停住（停住那一刻才下降到抓）。

    跟踪在 :meth:`~modules.grasp.runtime.Grasp.follow_material` 里：
    短步运动期间取下一帧，
    首次识别后至少跟随四秒，短暂跟丢只重算停稳时间；连续停稳 stable_seconds 才返回。

    :raises RuntimeError: 跟满 ``timeout`` 秒料还在动（带上最后一次跟丢的原因）。
    """
    result = robot.grasp.follow_material(
        want_color, timeout=timeout,
        min_follow_seconds=min_follow_seconds, stable_seconds=stable_seconds,
    )
    print(
        f"  料停住了：读数 ({result.readings[-1][0]:+.2f}, {result.readings[-1][1]:+.2f}) mm，"
        f"误差 {result.error_mm:.2f} mm；跟了 {result.iterations} 帧"
    )


def grab_material_from_turntable_to_repo_follow(
    robot: Robot,
    want_colors=("red", "green", "blue"),
    *,
    home: dict[str, float] | None = {"lift": 830, "extension": 120.0, "base": 420},
    follow_seconds: float = FOLLOW_SECONDS,
    stable_seconds: float = 0.5,
):
    """**跟着料走**：不等它转到位，一路跟到它上方，停住了才抓。

    与 :func:`grab_material_from_turntable_to_repo` 只差中间那一段：

    - 那个入口：站在原地等 ``wait_scene_stable`` 看到画面停稳 → 对准 → 抓；
    - 这个入口：每块料先跟满四秒，连续停稳 stable_seconds（默认半秒）再精确对准；
      下降前重新检查画面，物料再次运动就重新跟随，但不重等四秒。

    换位/摆料这一段共用 :func:`_stow_material_to_repo`，两个入口不会各漂一份。
    """
    if robot.arm.read_abs_state("lift").motor_degrees < 460:
        robot.arm.move_axis_to_motor_degrees("lift", 460)
    _go_home(robot, home)

    input("继续")
    try:
        for color in want_colors:
            min_follow_seconds = 4.0
            while True:
                _follow_material_until_stable(
                    robot, color, timeout=follow_seconds,
                    min_follow_seconds=min_follow_seconds, stable_seconds=stable_seconds,
                )
                # 当前物料已跨过四秒门槛，抓取重试只需重新确认停稳。
                min_follow_seconds = 0.0
                res = _grab_material_from_turntable(
                    robot, want_color=color, require_stable=True,
                )
                if res:
                    continue
                break
            _stow_material_to_repo(robot)
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        return

    move_axes_to(robot, SAFE_RETRACT).wait()
    robot.gripper.set_position(150)

def grab_material_from_floor_to_repo(
    robot: Robot,
    *,
    order=(3, 2, 1),
    rpm: int = 30,
    verbose: bool = False,
    pre_locations: list[list[int]] | None = None, 
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
            time.sleep(1)  # 车停稳再对准
            if pre_locations is None:
                robot.grasp.align_xy(tolerance_mm=2.0)   # TODO:这里有问题，怎么会对准的是圆环
            else:
                move_axes_to(robot, {"base":pre_locations[digit][0], "lift":300, "extension": pre_locations[digit][1]})
            robot.arm.move_axis_to_motor_degrees("lift", 90)
            robot.gripper.close_jaws()
            _place_material_to_repo(robot, verbose=verbose)
            robot.arm.move_axis_to_motor_degrees("extension", 120)
            input(f"已抓取 {digit} 号环的物料，回车继续，输入 q 跳过：").strip().lower()

    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        return

    _go_home(robot, home)

def _grab_material_from_turntable(
    robot: Robot,
    want_color=None,
    *,
    require_stable: bool = False,
):
    """内部函数：对准 → 下降 → 合爪（**不再**查停稳）。

    跟随入口在精确对准之后、下降之前再确认画面停稳；若物料重新运动就返回重跟，
    避免把四秒跟随结束时的静止状态当成此刻也静止。

    对准走 :meth:`~modules.grasp.runtime.Grasp.align_xy_for_turntable` 的快速档：
    料已经停住（调用方先 ``wait_scene_stable``），所以每轮只取一帧、走一小步，
    **跳过颜色/霍夫的辅助定位**（约 0.6 s/帧），1 s 预算内修正几步；到点再取一帧确认。
    **没进容差就不下降**——"已知最好"不等于对准了，照着它下降就是抓空或抓偏。
    对准花了多久打出来：这是现场判断"哪一段慢"的唯一依据。
    """

    started = time.monotonic()
    try:
        result = robot.grasp.align_xy_for_turntable(want=want_color, fast=True, tolerance_mm = 13)
    except Exception as e:
        print(f"Error occurred while aligning the gripper: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        return 2

    print(f"  {result.summary()}（对准耗时 {time.monotonic() - started:.2f} s）")
    if not result.aligned:
        print(
            f"  1 s 预算内没进容差（误差 {result.error_mm:.1f} mm）："
            "停在已知最好的位置，这次不下降"
        )
        move_axes_to(robot, SAFE_RETRACT).wait()
        return 1

    if require_stable and not robot.vision.scene_stable():
        print("  精确对准后物料仍在运动：重新跟随，这次不下降")
        return 1
    robot.arm.move_axis_to_motor_degrees("lift", 550, rpm=120)
    robot.gripper.close_jaws()
    robot.arm.move_axis_to_motor_degrees("lift", 800, rpm=120)
    return 0


def _stow_material_to_repo(robot: Robot, position: dict[str, float] | None = None) -> None:
    """夹着料从转盘上方摆到仓库位、松爪、转盘进位。

    ``grab_material_from_turntable_to_repo`` 与 ``..._follow`` 共用这一段：换位/松爪的顺序
    只写一份，改一处两个入口一起变。
    """
    global _turntable_position_index
    move_axes_to(robot, {"lift": 900, "extension": 10, "base": 1048}).wait()
    robot.gripper.set_position(100)
    robot.arm.move_axis_to_motor_degrees("lift", 980)
    robot.arm.move_axis_to_motor_degrees("base", 800)
    _turntable_position_index = (_turntable_position_index + 1) % len(TURNTABLE_POSITIONS)
    batch = robot.motion.begin()
    if position is not None:
        batch.add(robot.arm.move_axis_to_motor_degrees("base", position["base"], wait=False))
        batch.add(robot.arm.move_axis_to_motor_degrees("extension", position["extension"], wait=False))
    else:
        batch.add(robot.arm.move_axis_to_motor_degrees("base", 420, wait=False))
        batch.add(robot.arm.move_axis_to_motor_degrees("extension", 90.0, wait=False))
    batch.add(robot.arm.move_axis_to_motor_degrees("lift", 830, wait=False))
    batch.add(
        robot.turntable.go_to(TURNTABLE_POSITIONS[_turntable_position_index], wait=False)
    )
    batch.add(robot.gripper.open_jaws(wait=False))
    batch.wait()


def _grab_material_from_car(robot: Robot):
    """从车上抓料，最后**转到 plan 里算好的绝对位置**（环正上方）。

    ``plan`` 是 :meth:`modules.grasp.runtime.Grasp.plan_ring` 的返回值——它只算不动，
    所以可以在抓料之前就算好（那时靶标在视野里），抓完再走。``None`` 表示这次不转
    到环上（只抓料）。
    """
    global _turntable_position_index
    batch = robot.motion.begin()
    batch.add(robot.arm.move_axis_to_motor_degrees("lift", 675, wait=False))
    batch.add(robot.arm.move_axis_to_motor_degrees("base", 400, wait=False))
    batch.add(robot.gripper.set_position(150, wait=False))
    batch.wait()
    report = move_axes_to(robot, {"lift": 980, "extension": 5, "base": 1045}).wait()
    robot.gripper.set_position(120)
    robot.arm.move_axis_to_motor_degrees("lift", 845)
    
    robot.gripper.close_jaws()
    
    robot.arm.move_axis_to_motor_degrees("lift", 980)
    robot.arm.move_axis_to_motor_degrees("base", 800)
    _turntable_position_index = (_turntable_position_index + 1) % len(TURNTABLE_POSITIONS)
    batch.add(robot.arm.move_axis_to_motor_degrees("lift", 760, wait=False))
    batch.add(robot.arm.move_axis_to_motor_degrees("base", 400, wait=False))
    batch.add(robot.turntable.go_to(TURNTABLE_POSITIONS[_turntable_position_index], wait=False))
    batch.wait()

def _place_material_to_repo(robot: Robot, *, verbose: bool = False):
    '''当爪子上有物体时，将物体放置在指定位置。'''
    global _turntable_position_index

    # ① 抬升 + 回缩 + 转底座：三轴各自发**绝对角度**帧 → 一起动 → 一个 join 点。
    #    注意：底座这次要转约 9300 脉冲，超过"不许和伸缩并行"的 600 脉冲上限。
    #    那条限制只在 RobotArm.move_axes 里检查，单轴帧不拦，所以现场必须确认
    #    工作区是空的（伸出状态下横扫会把整条臂扫过去）。
    robot.arm.move_axis_to_motor_degrees("lift", 500)
    move_axes_to(robot, {"extension": 270.0, "base": 400, "lift": 900}).wait()
    report = move_axes_to(robot, {"lift": 900, "extension": 10, "base": 1048}).wait()
    if verbose:
        _print_report("放料前", report)

    # ② 三轴都到位之后才松爪。open_jaws 默认 wait=True，自己等到舵机走完并读回。
    robot.gripper.set_position(100)
    
    robot.arm.move_axis_to_motor_degrees("lift", 980)
    
    robot.arm.move_axis_to_motor_degrees("base", 800)
    _turntable_position_index = (_turntable_position_index + 1) % len(TURNTABLE_POSITIONS)
    batch = robot.motion.begin()
    batch.add(robot.arm.move_axis_to_motor_degrees("base", 200, wait=False))
    batch.add(robot.turntable.go_to(TURNTABLE_POSITIONS[_turntable_position_index], wait=False))
    batch.add(robot.gripper.open_jaws(wait=False))
    batch.wait()


def place_material_from_repo_to_floor(
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
            
            _grab_material_from_car(robot)
            input("放料完成，回车继续，输入 q 跳过：").strip().lower()
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        return
    
    move_axes_to(robot, SAFE_RETRACT).wait()

def place_material_on_material(
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
    _grab_material_from_car(robot)
    move_axes_to(robot, {"lift": 480, "extension": extension_deg, "base": base_deg}).wait()
    robot.gripper.open_jaws()
    robot.arm.move_axis_to_motor_degrees("lift", 695)
    _go_home(robot, home)

def place_material_from_repo_to_floor_by_pose(
    robot: Robot,
    *,
    orders: tuple[int, int, int] = (1, 2, 3),
    home: dict[str, float] | None = {"lift": 760, "extension": 400, "base": -32},
)-> list[dict[str, float]] | None:
    """把物料从仓库放到地面圆环里：车停在原地，逐环摆姿态后用 ``align_ring`` 闭环。

    ``home`` 是进流程前先摆到的姿态；传 ``None`` 表示这次不摆（轴本来就停在目标上时
    也不会发帧）。
    """
    robot.arm.move_axis_to_motor_degrees("lift", 400)
    _go_home(robot, home)
    ring_position = [(-190, 800, 710), (-32, 800, 540), (130, 800, 740)]
    res = []
    try:
        for order in orders:
            move_axes_to(robot, {"lift": ring_position[order - 1][1], "extension": ring_position[order - 1][2], "base": ring_position[order - 1][0]}).wait()
            time.sleep(1)  # 先让车停稳
            # assume_digit：这一格按示教姿态就是 order 号环。模型对转过角度的靶标会读错号
            # （2026-10-29 现场：1 号环 5 帧全读成 2/3 号，环心一直是准的），用
            # require_digit 按号筛框只会让整段失败；环心照样由模型给，落料补偿按 order 取。
            try:
                result = robot.grasp.align_ring(
                    tolerance_mm=3, max_iterations=6, assume_digit=order
                )
            except Exception as e:
                print(f"Error occurred while aligning ring: {e}")
                robot.arm.move_axis_to_motor_degrees("extension", 580)
                result = robot.grasp.align_ring(
                    tolerance_mm=3, max_iterations=6, assume_digit=order
                )
                continue
            print("  " + result.summary())
            base_deg = robot.arm.turns_state.of("base").motor_degrees
            extension_deg = robot.arm.turns_state.of("extension").motor_degrees
            position = {"base": base_deg, "extension": extension_deg}
            res.append(position)
            
            _grab_material_from_car(robot)
            
            # 这趟的落点就是现在这个角度。抓料会把这几个轴都开走，所以先记下来：
            move_axes_to(robot, {"extension": extension_deg, "base": base_deg}).wait()
            robot.arm.move_axis_to_motor_degrees("lift", 95, rpm=100)
            
            robot.gripper.open_jaws()
            robot.arm.move_axis_to_motor_degrees("lift", 375, rpm = 100)
            _go_home(robot, home)
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        return None

    move_axes_to(robot, SAFE_RETRACT).wait()
    return res

#: 放料前先把臂摆到 2 号环前的姿态（``base``/``lift``/``extension``，电机角度）。
#: 取 :func:`place_material_from_repo_to_floor_by_pose` 里 2 号环那一份：相机在这个姿态下看得到 2 号环，
#: ``align_ring`` 从这里开始闭环；``ring_offsets`` 的那个补偿也是在这个高度上标出来的。
RING2_APPROACH_POSE = {"base": -32, "lift": 600, "extension": 540}

def _row_step_pulses(
    robot: Robot, *, spacing_mm: float, radius_mm: float, middle_yaw_deg: float,
    normal_offset_mm: float = 0.0,
) -> tuple[int, int]:
    """沿环排平移带符号的间距，返回底座和伸缩的带符号脉冲增量。

    ``middle_yaw_deg`` 是中环朝向相对环排法线的输出轴夹角。先把中环还原到以
    法线为 x 轴的平面坐标，再沿环排改变 y、沿法线补偿 x；不能把中环朝向直接
    当作法线，否则整排会倾斜。``normal_offset_mm`` 为正时向远离底座方向平移。
    """
    lead = robot.arm.config.axes["extension"].mm_per_motor_rev
    if lead is None:
        raise ValueError(
            "modules/arm/hardware.json 里缺 axes.extension.mm_per_motor_rev，"
            "算不出伸缩要伸多少"
        )
    if not radius_mm > 0:
        raise ValueError(f"radius_mm 必须是正数（毫米），收到 {radius_mm!r}")
    pulses_per_rev = robot.arm.pulses_per_revolution
    yaw = math.radians(middle_yaw_deg)
    normal_mm = radius_mm * math.cos(yaw)
    lateral_mm = radius_mm * math.sin(yaw)
    # 两向量的叉积/点积求转角增量，避免 atan2 跨过 ±180° 时多转一整圈。
    base_degrees = math.degrees(math.atan2(
        normal_mm * spacing_mm - lateral_mm * normal_offset_mm,
        radius_mm**2 + lateral_mm * spacing_mm + normal_mm * normal_offset_mm,
    ))
    base_pulses = round(
        base_degrees * robot.arm.config.base_reduction * pulses_per_rev / 360.0
    )
    radial_mm = math.hypot(
        normal_mm + normal_offset_mm, lateral_mm + spacing_mm
    ) - radius_mm
    extension_pulses = round(radial_mm * pulses_per_rev / lead)
    return base_pulses, extension_pulses


def _align_floor_ring(
    robot: Robot, *, digit: int, tolerance_mm: float
) -> tuple[int, int]:
    """保存中环对准后的关节位置，未对准时不能据此计算三环落点。"""
    aligned = robot.grasp.align_ring(
        require_digit=digit, tolerance_mm=tolerance_mm, max_iterations=6
    )
    for step in aligned.steps:
        print(f"  {step.line()}")
    if aligned.plan is not None:
        target_x, target_y = aligned.plan.target
        print(f"  {digit} 号环视觉目标读数：({target_x:+.2f}, {target_y:+.2f}) mm")
    if not aligned.aligned:
        residual = (
            "未知" if aligned.final_error_mm is None
            else f"{aligned.final_error_mm:.1f} mm"
        )
        raise RuntimeError(
            f"{digit} 号环未对准（残差 {residual}，门槛 {tolerance_mm:.1f} mm）：停止放料"
            + (f"；{aligned.note}" if aligned.note else "")
        )
    print(f"  {digit} 号环视觉已对准（视觉残差 {aligned.final_error_mm:.1f} mm）")
    return (
        robot.arm.turns_state.of("base").pulses,
        robot.arm.turns_state.of("extension").pulses,
    )


def place_material_from_repo_to_floor_from_center_ring(
    robot: Robot,
    *,
    orders: tuple[int, int, int] = (3, 2, 1),
    place_lift: float = 95.0,
    tolerance_mm: float = 2,
    need_calibrate: bool = False,
    row_yaw_correction_deg: float = 0,      # 往负数调，值越大1环离车越近
    row_normal_base_motor_degrees: float = -17.0,
    ring3_down_offset_mm: float = 7.6,
    ring3_right_offset_mm: float = 10.3,
):
    """只闭环对准中环一次，按可调补偿角计算三环落点，再从仓库放料。

    ``row_yaw_correction_deg`` 直接给环排的方向补偿，单位为输出轴角度（不再除减速比）。
    0° 对应原先以中环朝向为法线的对称算法；此前日志中自动补偿为 -13.21°，
    若该值过矫，可在 0° 与 -13.21° 之间调节。中环落点不受这个参数影响。

    ``ring3_right_offset_mm`` 单独补偿 3 号环的稳定左偏：沿环排 1→3 的方向为正，
    默认右移 15 mm；0 关闭，负数左移。先在平面坐标中平移，再同时反算底座和伸缩，
    因而不是只转底座走圆弧，也不会改变 1/2 号环的位置。

    ``ring3_down_offset_mm`` 单独补偿 3 号环的稳定上偏：正数沿环排法线向底座侧
    平移（观察画面中的下方），默认 10 mm；0 关闭，负数向上。它与向右补偿垂直，
    同样先平移平面落点，再反算关节位置，不改变放料高度。

    1. 摆到 :data:`RING2_APPROACH_POSE`，``align_ring(require_digit=2)`` 闭环对准
       —— 2 号环的位置是**量出来**的；
    2. 用中环的关节位置、环间距和方向补偿一次计算 1/3 号的底座与伸缩位置；
    3. 按 ``orders`` 去仓库抓料 → 回到已算好的绝对关节位置 → 降到
       ``place_lift`` 松爪 → 抬起。车不动，不再逐环对准。

    第 2 步必须**转+伸一起走**：只转底座的话末端是沿圆弧走的，三个料就会落在一条弧上
    （只调底座是修不好的）。``r`` 根据对准后的伸缩位置和硬件配置里的实测半径参考计算。
    """
    digits = _check_order(orders)
    if not math.isfinite(row_yaw_correction_deg):
        raise ValueError("row_yaw_correction_deg 必须是有限的输出轴角度")
    if not math.isfinite(ring3_right_offset_mm):
        raise ValueError("ring3_right_offset_mm 必须是有限的毫米数")
    if not math.isfinite(ring3_down_offset_mm):
        raise ValueError("ring3_down_offset_mm 必须是有限的毫米数")
    robot.arm.move_axis_to_motor_degrees("lift", 760)
    move_axes_to(robot, RING2_APPROACH_POSE).wait()
    if need_calibrate:
        robot.arm.move_axis_to_motor_degrees("lift", 960)
        square_chassis_to_ring_row(robot)
    time.sleep(1)  # 先让臂停稳再看
    try:
        base, extension = _align_floor_ring(
            robot, digit=2, tolerance_mm=tolerance_mm
        )
        arm = robot.arm
        spacing = float(robot.vision.ring_center_spacing_mm)
        radius = arm.config.extension_radius_mm(extension)
        degrees_per_pulse = 360.0 / arm.pulses_per_revolution
        poses = {2: (base, extension)}
        for digit in (1, 3):
            lateral_mm = (digit - 2) * spacing
            if digit == 3:
                lateral_mm += ring3_right_offset_mm
            step_base, step_extension = _row_step_pulses(
                robot, spacing_mm=lateral_mm,
                radius_mm=radius, middle_yaw_deg=row_yaw_correction_deg,
                normal_offset_mm=-ring3_down_offset_mm if digit == 3 else 0.0,
            )
            poses[digit] = (base + step_base, extension + step_extension)
        print(
            f"2 号环几何基准：当前半径 {radius:.2f} mm；"
            f"环排补偿角 {row_yaw_correction_deg:+.2f}°（输出轴）；"
            f"3 号环向右补偿 {ring3_right_offset_mm:+.1f} mm、"
            f"向下补偿 {ring3_down_offset_mm:+.1f} mm"
        )
        for digit in digits:
            base_pulses, extension_pulses = poses[digit]
            # 抓料会把这几个轴都开走；这里用的是**绝对**位置，中间怎么走都对得上。
            _grab_material_from_car(robot)
            move_axes_to(
                robot,
                {
                    "base": base_pulses * degrees_per_pulse,
                    "extension": extension_pulses * degrees_per_pulse,
                },
            ).wait()
            robot.arm.move_axis_to_motor_degrees("lift", place_lift)
            robot.gripper.open_jaws()
            robot.arm.move_axis_to_motor_degrees("lift", 375)
            print(f"  {digit} 号环：底座 {base_pulses} 脉冲、伸缩 {extension_pulses} 脉冲")
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        raise

    move_axes_to(robot, SAFE_RETRACT).wait()


def grab_material_from_floor_to_repo_by_xy(
    robot: Robot,
    *,
    orders: tuple[int, int, int] = (3, 2, 1),
    pre_locations: list[dict[str, float]] | None = None,
    home: dict[str, float] | None = {"lift": 760, "extension": 400, "base": -32},
):
    """把物料从地面圆环放到仓库里。不需要考虑初始状态。

    ``home`` 是进流程前先摆到的姿态，默认 :data:`READY_POSE`；传 ``None`` 表示这次不摆
    （调用方已经在这一姿态上时省一次往返；轴本来就停在目标上时也不会发帧）。
    """
    robot.arm.move_axis_to_motor_degrees("lift", 400)
    _go_home(robot, home)
    ring_position = [(-190, 580, 720), (-32, 580, 540), (110, 580, 720)]
    try:
        for order in orders:
            move_axes_to(robot, {"lift": ring_position[order - 1][1], "extension": ring_position[order - 1][2], "base": ring_position[order - 1][0]}).wait()
            if pre_locations is None:
                time.sleep(1)  # 先让车停稳
                result = robot.grasp.align_xy(tolerance_mm=3.0)
            else:
                position = pre_locations.pop(0)
                print(position)
                move_axes_to(robot, position).wait()
            robot.arm.move_axis_to_motor_degrees("lift", 90, rpm = 75)
            robot.gripper.close_jaws()
            _place_material_to_repo(robot, verbose=True)
            _go_home(robot, home)
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        return
    move_axes_to(robot, SAFE_RETRACT).wait()


#: 去台面三处**放置位**上方摆的姿态 ``(base, lift, extension)``，单位是电机角。
#: ``lift`` 就是放料高度（:func:`place_material_on_material` 用的 480）：太低会压到已经在那里
#: 的料，太高夹爪打不开。
#: 换摆放/换台面就改这一张表。
_PLACE_POSITION = [(-190, 480, 690), (-32, 480, 540), (110, 480, 690)]


def place_material_on_material_by_order(
    robot: Robot,
    *,
    orders: tuple[int, int, int] = (1, 2, 3),
    home: dict[str, float] | None = {"lift": 800, "extension": 270.0, "base": 0},
):
    """从仓库抓料，叠到台面上第 ``order`` 号位置的物料上；``orders`` 是**放置顺序**。

    每趟的顺序（与 :func:`place_material_on_material` 同一套动作，只是多了"先去哪个位置"）：

    1. 摆到 ``PLACE_POSITION[order - 1]``；
    2. ``align_xy`` 对准——没传 ``want``，按"离光轴最近的那个"对准；
    3. 记下对准后的底座/伸缩角；
    4. 回仓库抓一块料（:func:`_grab_material_from_car`，它会动三轴并转料盘）；
    5. 按记下的角度**回到刚才那个位置**、松爪、抬起。

    ``home`` 只在每趟开始前摆一次；``None`` = 不摆（调用方已经在这一姿态上）。
    """
    digits = _check_order(orders)
    try:
        for order in digits:
            base, lift, extension = _PLACE_POSITION[order - 1]
            move_axes_to(robot, {"lift": lift, "extension": extension, "base": base}).wait()
            time.sleep(1)  # 先让车停稳
            robot.grasp.align_xy(tolerance_mm=2.0)
            # 抓料会把这几个轴都开走，所以先记下来：这趟的落点就是现在这个角度。
            base_deg = robot.arm.turns_state.of("base").motor_degrees
            extension_deg = robot.arm.turns_state.of("extension").motor_degrees
            _grab_material_from_car(robot)
            move_axes_to(
                robot,
                {"extension": extension_deg, "base": base_deg},
            ).wait()
            robot.arm.move_axis_to_motor_degrees("lift", lift)
            robot.gripper.open_jaws()
            robot.arm.move_axis_to_motor_degrees("lift", 800)
            _go_home(robot, home)
    except Exception as e:
        print(f"Error occurred during pick and place operation: {e}")
        move_axes_to(robot, SAFE_RETRACT).wait()
        return
    move_axes_to(robot, SAFE_RETRACT).wait()
    
    
def place_and_grab_material(robot:Robot, place_orders = (1, 2, 3), grab_orders = None):
    if grab_orders is None:
        grab_orders = place_orders
    pre_locations = place_material_from_repo_to_floor_by_pose(robot, orders=place_orders)   # 这边应该返回三个位置
    grab_material_from_floor_to_repo_by_xy(robot, orders=grab_orders, pre_locations=pre_locations)   # 这边应该传入三个位置

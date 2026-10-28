"""eye-in-hand 视觉伺服：让爪子正下方对准物块（**只动平面**），再完成下降与后缩。

## 这套机器上相机是装在臂上的

相机随爪子一起动，所以世界坐标那条链（相机系 → 基座平面 → 圆柱关节量）**根本不适用**：
相机自己在动，"物块在相机系的坐标"里混着机械臂的运动，换算不出一个固定的世界位置。

但装在臂上有一个更省事的结果：

> 相机和爪子**刚性同体**，所以爪子抓取点在相机坐标系里的位置是**一个常数**，
> 记作 ``(tx, ty)``。

于是"爪子正下方对准物块" ⟺ "物块出现在画面里的 ``(tx, ty)``"。整件事变成
**把观测量伺服到定值**，一个世界坐标都不需要。

## 为什么用脉冲驱动

伺服要有"动多少"的量纲。用毫米就得先标定 ``mm_per_motor_rev``；用脉冲则**一个机器常数都不用**：
只要知道"每脉冲让画面里的物块挪多少毫米"，就能算出让误差归零的脉冲数。

那个量就是 2×2 雅可比 :class:`~modules.grasp.drive.visual_servo_model.VisualServoJacobian`：
它是**实测出来**的（:func:`probe_jacobian`），所以相机怎么装的、歪多少度、朝哪个方向，
全都不用假设。代价是标定一次。

## 为什么还要闭环

雅可比是线性的，而真实关系在大位移下会弯；背隙、丢步也会让"发了 800 脉冲"不等于
"走了 800 脉冲的量"。所以每次都用**新的一帧**重算。

**新的一帧**是硬要求：动作前记一个时刻，动作后只要晚于它的观测（``min_received_at``）。
用"多久算旧"挡不住"动作很快、旧帧也还新"的情况，那时读到的是动作**之前**的坐标。

## 本模块只留实际算法

配置与数值模型在 :mod:`modules.grasp.drive.visual_servo_model`，观测判据在
:mod:`modules.grasp.drive.visual_servo_observation`，脉冲原语在
:mod:`modules.grasp.drive.visual_servo_motion`。**主编排**（悬停 → 安全抬升 → x/y → 下降 →
后缩）已经移进 :class:`modules.grasp.runtime.Grasp` 的实例方法，这里不再保留同名的
自由函数转发壳——那种壳会让"到底谁在编排"重新变得说不清。

## 平面之外的两步是开环的

x/y 靠闭环；**平面一旦对准**，抓取前剩下的两步都是固定脉冲数：

```
descend_z          升降/6 号下降  z_descend_pulses
retract_extension  伸缩/7 号后缩  extension_retract_pulses
```

两个数都在 ``modules/grasp/eye_in_hand.json`` 里，换物块、换夹爪现场直接改，不用重标。
下降那一步会先确认平面已经对准（``z_center_tolerance_mm``）——它是这两步唯一的护栏。

## 自测

``python -m modules.grasp.drive.eye_in_hand`` 不需要硬件：验证 2×2 求逆、方向符号、
闭环收敛与三种失败路径。本模块仍可作为 ``python -m`` 入口，所以它**不能**在包导入时
被加载（见 ``modules/grasp/__init__.py``）。
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple

from modules.grasp.drive.visual_servo_motion import (
    _move_axes,
    _move_axis,
    _move_lift,
    _send_axis,
    _send_lift,
)
from modules.grasp.drive.visual_servo_model import (
    BASE_AXIS,
    DEFAULT_CONFIG_PATH as DEFAULT_CONFIG_PATH,
    LIFT_AXIS,
    MIN_JACOBIAN_DETERMINANT,
    MaterialMissingError,
    RADIAL_AXIS,
    GraspReadyResult,
    HoverResult,
    LiftScale,
    MaterialSelector,
    VisualServoConfig,
    VisualServoError,
    VisualServoJacobian,
    VisualServoResult,
    VisualServoTarget,
    ZDescentResult,
    ZResult,
    load_jacobian,
    load_lift_scale,
    load_material_selector,
    load_visual_servo_config,
    load_target,
    load_z_target,
    save_calibration,
)
from modules.grasp.drive.visual_servo_observation import (
    FrameSource,
    VISIBILITY_PROBE_FRAMES,
    _probe_visible,
    _read_material,
    _settled_sighting,
    measure_offset,
    measure_settled,
    select_material,
)
from modules.motor.drive.motion import estimate_move_seconds
from modules.vision.drive.sighting import MaterialSighting



def _apply_minimum(step: int, wanted: float, minimum: float) -> int:
    """指令小于背隙门槛时按原方向抬到门槛；本来就不动（取整成 0）就保持不动。

    太小的指令会被背隙整段吃掉、画面一动不动，闭环于是以为"没动过"、
    再发一次同样的指令——表现为原地打转。抬到门槛之上才能真正推动它。

    ``minimum`` 是**按该轴灵敏度换算出来的脉冲数**（见
    :meth:`VisualServoJacobian.minimum_pulses`），不是拍脑袋定的一串数。
    """
    if step == 0:
        return 0
    if abs(step) >= minimum:
        return step
    return int(math.copysign(max(1.0, minimum), wanted))


def _clamp_pulses(value: float, limit: int) -> int:
    """把浮点脉冲量取整并夹到 ``±limit``。"""
    rounded = int(round(value))
    return max(-limit, min(limit, rounded))


def _plan_planar_step(
    error_x: float,
    error_y: float,
    jacobian: VisualServoJacobian,
    config: VisualServoConfig,
    *,
    limit: int,
) -> Tuple[int, int]:
    """**平面误差 → 这一步该走多少脉冲**；闭环与单帧一步共用这一处。

    ``limit`` 是这一档的单步上限（粗调/细调的分档由调用方按
    :meth:`VisualServoConfig.step_limit_for` 定）。每个轴的门槛按**各自的灵敏度**换算
    （:meth:`VisualServoJacobian.minimum_pulses`），并且不超过这个上限——安全优先。

    这就是"逆运动解算"这一步的全部内容：``p = −J⁻¹·e``（:meth:`VisualServoJacobian.pulses_for`），
    再夹到脉冲预算里。它本身是一瞬间的事；慢的从来不是它。

    :raises VisualServoError: 两个轴都取整成 0，但误差还在（大不了调不动，不该静默收工）。
    """
    base_min, radial_min = jacobian.minimum_pulses(config.min_step_mm)
    base_wanted, radial_wanted = jacobian.pulses_for(error_x, error_y)
    base_step = _apply_minimum(
        _clamp_pulses(base_wanted, limit), base_wanted, min(base_min, limit)
    )
    radial_step = _apply_minimum(
        _clamp_pulses(radial_wanted, limit), radial_wanted, min(radial_min, limit)
    )
    if base_step == 0 and radial_step == 0:
        raise VisualServoError(
            f"这一轮算出来要走的脉冲都取整成 0（想要 底座 {base_wanted:.2f}、"
            f"伸缩 {radial_wanted:.2f}），但误差还有 {math.hypot(error_x, error_y):.2f} mm。"
            "把 max_pulses_per_step 调大，或者雅可比标定得太小需要重标"
        )
    return base_step, radial_step


def _given_offset(reading) -> Tuple[float, float]:
    """把"调用方已经量好的平面读数"校验成 ``(x_mm, y_mm)``。

    ``str`` / ``bytes`` 明确拒掉：``"12"`` 拆成两个字符也能转成 ``(1.0, 2.0)``——
    一个手滑写错的字符串会**静默变成**一个合法坐标，而机械臂会照着它动。
    这种错误必须在动任何东西之前响亮地报出来。
    """
    if isinstance(reading, (str, bytes)):
        raise ValueError(f"reading 要写成 (x_mm, y_mm) 两个数，收到字符串 {reading!r}")
    try:
        first, second = reading
        return (float(first), float(second))
    except (TypeError, ValueError) as problem:
        raise ValueError(f"reading 要写成 (x_mm, y_mm) 两个数，收到 {reading!r}") from problem


def _single_step(
    arm,
    target: VisualServoTarget,
    jacobian: VisualServoJacobian,
    config: VisualServoConfig,
    offset: Tuple[float, float],
    say: Callable[[str], None],
) -> VisualServoResult:
    """**单帧一步**：拿这一帧的读数算一次脉冲、走一步就返回；动作之后不再测量。

    为什么要它：跟随（物体一直在动）时，"等画面停稳"是在等一个**永远等不到**的东西——
    下一帧物体又动了。跟随循环每一轮都会立刻重新测一次，所以这里再测一遍只是白花两帧
    （约 250 ms，而且是在 30 fps 上）。

    代价写在 :attr:`~modules.grasp.drive.visual_servo_model.VisualServoResult.unverified` 上：
    结果里的读数与误差是**动作之前**那一次，不代表现在的位置。也正因为没有第二次测量，
    闭环里那两条"事后核对"的判据这里都没有：与雅可比预测方向相反、像素直径突变
    （爪子挡住物块）。单步幅度本来就被 ``step_limit_for`` 夹住，所以撞上去的风险不大，
    但**调用方必须自己看下一帧**——这本来就是它的循环在做的事。
    """
    error_x = offset[0] - target.x_mm
    error_y = offset[1] - target.y_mm
    error = math.hypot(error_x, error_y)
    say(f"单帧一步：物块 ({offset[0]:+.2f}, {offset[1]:+.2f}) mm，误差 {error:.2f} mm")

    def result(*, base_pulses: int, radial_pulses: int, resolution_limited: bool = False):
        return VisualServoResult(
            target=target,
            readings=(offset,),
            error_x=error_x,
            error_y=error_y,
            base_pulses=base_pulses,
            radial_pulses=radial_pulses,
            iterations=1,
            tolerance_mm=config.tolerance_mm,
            resolution_limited=resolution_limited,
            unverified=True,
        )

    if error <= config.tolerance_mm:
        say(f"  误差已在容差 {config.tolerance_mm:.1f} mm 内，一步都不动")
        return result(base_pulses=0, radial_pulses=0)
    if error < config.min_step_mm:
        say(
            f"  误差 {error:.2f} mm 已小于背隙地板 {config.min_step_mm:.2f} mm，"
            "再走只会跨过目标；按分辨率极限收工"
        )
        return result(base_pulses=0, radial_pulses=0, resolution_limited=True)

    limit = config.step_limit_for(error)
    base_step, radial_step = _plan_planar_step(
        error_x, error_y, jacobian, config, limit=limit
    )
    # 单帧一步没有"累计"可算，所以拿**整份预算**当这一步的上限：它拦不住慢慢磨，
    # 但拦得住"雅可比标反了于是想走几万脉冲"那种。
    if abs(base_step) > config.max_total_pulses or abs(radial_step) > config.max_total_pulses:
        raise VisualServoError(
            f"这一步就要 底座 {base_step:+d}、伸缩 {radial_step:+d} 脉冲，超过单次预算 "
            f"{config.max_total_pulses}。这通常说明雅可比标反了或符号错了，先跑 --calibrate 重标"
        )
    coarse = error > config.coarse_error_mm
    say(f"  走（{'粗调' if coarse else '细调'}，上限 {limit}）：底座 {base_step:+d}、伸缩 {radial_step:+d} 脉冲")
    move_seconds = _move_axes(arm, base_step, radial_step, config)
    say(f"  动作耗时 {move_seconds:.2f} s（这一步之后没有再验证）")
    return result(base_pulses=base_step, radial_pulses=radial_step)


def _result_when_occluded(
    readings: List[Tuple[float, float]],
    target: VisualServoTarget,
    config: VisualServoConfig,
    *,
    base_pulses: int,
    radial_pulses: int,
    iterations: int,
) -> Optional[VisualServoResult]:
    """物块看不见了，能不能按**已经测到**的那次读数收工；不能就返回 ``None``。

    这是"爪子已经悬到物块上方、自己把光挡住"的收工口。成立要同时满足两条：

    1. 这一次调用里**至少测到过一次**（一次都没有就无从谈起，只能报错问人）；
    2. 那一次的误差已经进入"可以下降"的平面精度（``z_center_tolerance_mm``，
       与 :func:`descend_z` 用的是同一个门槛）。

    两条一起的含义是：遮挡发生在"再走也只会过冲"的阶段，按那一次读数收工不会让机械臂落到
    一个没量过的地方。结果里的 ``occluded`` 明确表示**没有再验证过**——不是"已经对准"，
    调用方要据此决定后面还能不能做开环动作（例如下降）。

    还差得远就返回 ``None``（物块可能被人挡住、被挪走、或者被爪子推开了）——那种情况必须
    报错，不能猜。
    """
    if not readings:
        return None
    error_x = readings[-1][0] - target.x_mm
    error_y = readings[-1][1] - target.y_mm
    if math.hypot(error_x, error_y) > config.z_center_tolerance_mm:
        return None
    return VisualServoResult(
        target=target,
        readings=tuple(readings),
        error_x=error_x,
        error_y=error_y,
        base_pulses=base_pulses,
        radial_pulses=radial_pulses,
        iterations=iterations,
        tolerance_mm=config.tolerance_mm,
        occluded=True,
    )


def _step_is_reversed(
    jacobian: VisualServoJacobian,
    *,
    step: Tuple[int, int],
    measured: Tuple[float, float],
    config: VisualServoConfig,
) -> bool:
    """这一步的实测位移与雅可比预测是否**明显相反**。

    判据刻意做窄：点积为负，而且**两边都远高于停稳噪声**（``4 × settle_tolerance_mm``）。
    预测本来就很小的一步（细调阶段的 1 mm）不算——那种量级上噪声就能翻转符号。
    """
    predicted = jacobian.displacement_for(step[0], step[1])
    floor = 4.0 * config.settle_tolerance_mm
    if math.hypot(predicted[0], predicted[1]) <= floor:
        return False
    if math.hypot(measured[0], measured[1]) <= floor:
        return False
    return predicted[0] * measured[0] + predicted[1] * measured[1] < 0.0


def _reversed_step_error(
    jacobian: VisualServoJacobian,
    *,
    step: Tuple[int, int],
    measured: Tuple[float, float],
) -> VisualServoError:
    """连续两步都相反时的报错：把两个数都摆出来，并指向重探雅可比。"""
    predicted = jacobian.displacement_for(step[0], step[1])
    return VisualServoError(
        f"连着两步的实测画面位移都与雅可比预测**方向相反**：最后一次实测 "
        f"({measured[0]:+.2f}, {measured[1]:+.2f}) mm，预测 "
        f"({predicted[0]:+.2f}, {predicted[1]:+.2f}) mm"
        f"（底座 {step[0]:+d}、伸缩 {step[1]:+d} 脉冲）。\n"
        "  雅可比是“每脉冲让画面挪多少”的一次性标定，连续两步相反说明它和你现在的机器"
        "对不上：相机重新装过、皮带或联轴器滑过、轴的接线或方向改过。\n"
        "  重探一次（探针自带两段同向位移的一致性判据，符号错了它自己会报）：\n"
        "    python -m modules.grasp.test.manual.align_xy --calibrate"
    )


def probe_jacobian(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    config: VisualServoConfig,
    *,
    on_message: Optional[Callable[[str], None]] = None,
) -> VisualServoJacobian:
    """实测 2×2 雅可比：每个轴走两段**同向**位移，用第二段算斜率。

    为什么要走两段同向而不是"去一趟再回来"：反向的第一段位移会被**背隙**吃掉，
    算出来的斜率偏小。两段同向时第二段没有背隙损失，斜率是真的。

    测完把两个轴退回原位，所以这个过程不会把机械臂留在别处。

    :raises VisualServoError: 探针脉冲为 0、看不到物块、或探出来的雅可比奇异。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    pulses = config.probe_pulses

    def one_axis(axis: str, label: str) -> Tuple[float, float]:
        before = measure_settled(link, selector, config)
        checkpoint = time.monotonic()
        _move_axis(arm, axis, pulses)
        # 必须**等画面停稳**再读：按估算耗时读到的中点会混进第一段的残余位移，
        # 于是第二段的斜率被算大——实测这一处能把雅可比放大到 2.5 倍。
        mid = measure_settled(link, selector, config, after=checkpoint)
        checkpoint = time.monotonic()
        _move_axis(arm, axis, pulses)
        after = measure_settled(link, selector, config, after=checkpoint)
        # 退回原位：两段各退一次。
        _move_axis(arm, axis, -pulses)
        _move_axis(arm, axis, -pulses)
        # 斜率只用**第二段**：同向连续两段里，只有第二段没有背隙损失。
        # （用 "after − before" 除以 2·pulses 会把第一段的背隙摊进来，斜率偏小。）
        first_leg = (mid[0] - before[0], mid[1] - before[1])
        second_leg = (after[0] - mid[0], after[1] - mid[1])
        _require_consistent_legs(first_leg, second_leg, label=label, axis=axis, pulses=pulses)
        slope = (second_leg[0] / pulses, second_leg[1] / pulses)
        say(
            f"  探针 {label}：走 {2 * pulses} 脉冲，画面 "
            f"({before[0]:+.2f},{before[1]:+.2f}) -> ({mid[0]:+.2f},{mid[1]:+.2f}) "
            f"-> ({after[0]:+.2f},{after[1]:+.2f}) mm"
        )
        say(
            f"    第二段（无背隙）每 1000 脉冲 -> "
            f"({slope[0] * 1000:+.2f},{slope[1] * 1000:+.2f}) mm；"
            f"第一段位移 ({first_leg[0]:+.2f},{first_leg[1]:+.2f}) mm"
        )
        return slope

    base_slope = one_axis(BASE_AXIS, "底座")
    radial_slope = one_axis(RADIAL_AXIS, "伸缩")
    jacobian = VisualServoJacobian(
        base_dx=base_slope[0],
        base_dy=base_slope[1],
        radial_dx=radial_slope[0],
        radial_dy=radial_slope[1],
    )
    if jacobian.is_singular:
        raise VisualServoError(
            f"探出来的雅可比奇异（行列式 {jacobian.determinant:.3e}）："
            "底座与伸缩在画面里产生的位移方向几乎重合。"
            "请确认相机确实随伸缩轴一起移动，且两个轴的运动在画面里能分辨出方向差异"
        )
    return jacobian


def align_x_y(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    target: VisualServoTarget,
    jacobian: VisualServoJacobian,
    config: VisualServoConfig,
    *,
    ensure_safe_height: bool = True,
    quick_safe_lift: bool = False,
    lift_scale: Optional[LiftScale] = None,
    single_step: bool = False,
    reading: Optional[Sequence[float]] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> VisualServoResult:
    """闭环把物块伺服到 ``(tx, ty)``；只动 底座 与 伸缩，不碰高度。

    :param ensure_safe_height: 动手之前要不要先确认"已经抬到安全高度"（默认要）。
        连续用同一高度调用时（``--follow`` 的每一轮、或一次"对准+下降"里已经抬过）
        可以关掉：省下的是一次停稳测量（实测约 0.5 s），而高度在这些步骤里不会变。
        只有调用方**确实知道高度没变**才能关——低位转底盘会撞到物块。
    :param quick_safe_lift: 那次检查允许先用**单帧**快判（够高就放行，不确定退回停稳
        测量）。给"每一轮都要查"的 ``--follow`` 用：判据不放宽，但绝大多数轮次只花
        一帧（约 0.22 s）而不是一次停稳测量（约 0.54 s）。
    :param lift_scale: 已标定的升降灵敏度，交给 :func:`raise_before_rotating` 一步算到位。
    :param single_step: **单帧一步**：最多取一帧、算一次脉冲、走一步就返回，**动作之后不再
        测量**（见 :func:`_single_step`）。给跟随循环用——调用方本来就会立刻再测一次，
        而"等画面停稳"在物体一直在动时是在等一个永远等不到的东西。默认关：普通对准要的是
        "一直调到准为止"，那必须是闭环。
    :param reading: 调用方**已经量好**的平面读数 ``(x_mm, y_mm)``。给了它就连那一帧都不取
        （``single_step`` 自动为真）——这正是"``locate`` 一次 + 走一步"的组合::

            report = robot.vision.locate("green")        # 一帧：YOLO 像素 → x/y 毫米
            robot.grasp.align_xy(reading=(report.best.x_mm, report.best.y_mm))

        放进循环就是跟随：每轮只取**一帧**、只走**一步**，中间没有第二次测量。
    :raises VisualServoError: 看不到物块、雅可比奇异、脉冲预算用完，或迭代用尽仍未进容差。
    :raises ValueError: ``reading`` 不是两个数。

    物块**中途看不见**分两种，处理方式不同：

    - 已经测到过、而且那一次已经进入"可以下降"的平面精度（``z_center_tolerance_mm``，
      与 :func:`descend_z` 同一个门槛）→ 按最后一次读数收工，结果里 ``occluded=True``。
      爪子悬到物块正上方、自己把光挡住就是这个现象，收工口正是为它留的；
    - 一次都没测到、或者还差得远 → 原样报错（物块可能被挪走、被手挡住、被爪子推开）。

    画面里有多个物料时按**离光轴最近**的那个对准（见
    :func:`~modules.grasp.drive.visual_servo_observation.select_material`），不算遮挡、也不报错。

    ## 为什么不按"像素直径突变"判遮挡

    z 是按像素半径反推的（``z = f·D/2r``），所以直径一变，算出来的距离就跟着变。
    曾经在这里按"直径相对第一次测量变化超过 25%"判"爪子挡住了物块"并**直接报错**
    （``diameter_change_limit_ratio``），但真机上它总在误报：实测一次对准里直径从
    272 px 变成 397 px（1.46 倍，f=1012 px、顶面 30 mm，等价于 z 从 112 mm 报成 77 mm）。
    那次根本没有遮挡——报错文字说的"被遮住一部分 → 可见区域变小"是反方向。

    它也不是本函数的正确性问题：x/y 由像素**圆心**直接决定
    （``x = (u-cx)·z/f``），目标是画面中心时把 x 驱动到 0 就是 ``u→cx``，与 z 的尺度
    无关。尺度错了只影响**每一步走多远**、以及"进容差"实际有多严（差 1.46 倍就是 4 mm
    容差实际 5.8 mm），下一步重新测量就纠正回来，不会把爪子带到别处去。所以这条判据
    去掉了，闭环需要的反馈一点没少。

    "转底盘之前必须先够高"仍由 :func:`raise_before_rotating`（``ensure_safe_height``，
    按 **z** 判安全高度）负责——那是机械结构上的硬约束，与像素尺度无关。

    :func:`align_z` 里那条直径判据留着：那里升降**确实**按设计改变距离，而"检测换了
    目标"是另一回事（真物块被推出半径窗口后检测会锁到画面里别的东西上）。

    ## 为什么停稳判据不看 z

    每一次测量都要求"连续两帧读数一致"，而那个判据默认是**三维**的。本函数只用
    ``x_mm`` / ``y_mm``，z 一个数都不进解算——可 z 恰恰是
    最不稳的那个：它按像素半径反推（``z = f·D/2r``，本机约 0.42 mm/px），1 px 的半径
    抖动就顶掉整个细档 0.4 mm。

    实测（真机帧 + σ=1–2 灰阶噪声）：连续两帧的 Δxy 中位数是 0.02–0.26 mm（≤0.4 mm 的
    占 74–100%），Δz 中位数却是 1.7–2.7 mm（≤0.4 mm 的只占 13%）。于是细档判据**永远
    不满足**，每次测量都要等满 ``settle_timeout_seconds``（默认 4 s）才拿最后读数走人：
    对准本身很快，等的是这个。

    所以这里传 ``include_depth=False``，只比平面内的 x/y。测 z 的那几条路
    （:func:`raise_before_rotating` / :func:`hover_above_material` / :func:`align_z`）
    照旧比三维——那里的 z 就是被控量，抖动必须等它停下来。

    平面内的细档也不沿用 ``settle_tolerance_mm``（0.4 mm）：那个精度是给探针 / z 标定
    用的，对准的验收容差是 ``tolerance_mm``，读数噪声只要比它小一大截就够。这里取
    ``tolerance_mm / 4``（不低于 ``settle_tolerance_mm``、不高于粗档），于是"读数抖动
    1 mm 就等满 4 s"的那一段被去掉，每次测量回到"两帧收工"。

    ## 物块还没转进视野时（转盘）

    第一次测量把 ``config.target_wait_seconds`` 当作"等物块出现"的上限
    （``null`` = 一直等，``0`` = 保持旧语义），而不是沿用 ``settle_timeout_seconds`` 的
    4 s：转盘还没把物块转到镜头前时，本函数会一直看着画面等它出现，出现之后才按 4 s
    等它停稳。之后的每一次测量仍按 4 s 收工——"看见过又丢"要能按最后读数收工
    （``occluded=True``），不能被无限等待卡住。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    if reading is not None:
        # 参数错误要在**动任何东西之前**报出来；给了已经量好的读数就自然是单帧一步。
        given = _given_offset(reading)
        single_step = True
    # x/y 的每一步都要转底盘，所以先在**打开动作之前**抬到安全高度。这是机械结构上的
    # 硬约束：低位转底盘会撞到台面上的物块。刻意不自动降回去——降回去正是撞到它的动作。
    if ensure_safe_height:
        raise_before_rotating(
            arm,
            link,
            selector,
            config,
            scale=lift_scale,
            quick=quick_safe_lift,
            on_message=say,
        )
    if single_step:
        if reading is None:
            # 只取一帧，**不等停稳**：跟随的时候画面永远在变，等它停稳等于永远等下去。
            # 这一帧可能带运动模糊，但单步幅度本来就被 step_limit_for 夹住，
            # 而且下一步立刻会用新的一帧纠正回来。
            sighting = _read_material(link, selector, config)
            given = (sighting.x_mm, sighting.y_mm)
        return _single_step(arm, target, jacobian, config, given, say)
    readings: List[Tuple[float, float]] = []
    total_base = 0
    total_radial = 0
    #: 每次动完之后记一个时刻，下一次测量只接受晚于它的帧。
    #: **这是闭环正确性的关键**：动作很快时，队列里可能还压着动作**之前**的帧，
    #: 读到它就会以为"没动过"，于是再发一次同样的指令——表现为来回振荡。
    checkpoint: Optional[float] = None
    #: 上一次动作**之前**的读数，与那一次下发的脉冲。用来核对"实测位移 vs 雅可比预测"：
    #: 连续两步方向相反就是雅可比过时/装反的签名，必须停下而不是把脉冲预算烧完。
    previous_offset: Optional[Tuple[float, float]] = None
    previous_step: Optional[Tuple[int, int]] = None
    #: 连续几步方向相反。**一步不算**（物块被推动、检测跳目标都会这样）。
    contradictions = 0
    #: 细档判"停稳"用的一致性门槛。**不直接用** ``settle_tolerance_mm``（0.4 mm）：
    #: 那个数是给探针 / z 标定用的——那里的斜率就是被控量，0.4 mm 的读数噪声已经
    #: 足够把雅可比算歪。x/y 对准不需要那么严：它的验收容差是 ``tolerance_mm``，
    #: 读数噪声只要比它小一大截就够。而真机上连续两帧的 Δxy 尾巴会到 1 mm 以上
    #: （中位数 0.02–0.26 mm），按 0.4 mm 判就会每步都等满 ``settle_timeout_seconds``
    #: （默认 4 s）才拿最后读数走人——每多等一帧就是一次定位。
    #: 所以取"验收容差的四分之一"，下限仍是标定过的 ``settle_tolerance_mm``，
    #: 上限是粗档：细档不该比粗档更松。
    fine_settle_tolerance = min(
        config.coarse_settle_tolerance_mm,
        max(config.settle_tolerance_mm, 0.25 * config.tolerance_mm),
    )
    #: 这一轮判"停稳"用的一致性门槛。第一次用细档（要判"是不是已经对准了"），
    #: 之后误差还大就换粗档——粗档少等几帧，而每帧都要跑一次定位。
    settle_tolerance = fine_settle_tolerance

    for attempt in range(1, config.max_iterations + 1):
        started = time.monotonic()
        try:
            sighting = _settled_sighting(
                link,
                selector,
                config,
                after=checkpoint,
                tolerance=settle_tolerance,
                # 只比平面内的 x/y：本函数一个 z 都不参与解算（见 docstring 的
                # "为什么停稳判据不看 z"）。z 那一路的抖动会把每次测量拖到
                # settle_timeout_seconds 才收工，实测那是"对准要等好几秒"的唯一来源。
                include_depth=False,
                # 第一次测量：物块可能还在转盘上没转进视野，按 target_wait_seconds 等它出现
                # （null = 一直等）。测到过之后不再无限等——"看见过又被爪子挡住"的收工口
                # 要靠 settle_timeout_seconds 触发。
                wait_for_first_target=not readings,
            )
        except MaterialMissingError:
            # 只有"一个目标都没有"走这条路：可能只是爪子悬到物块上方把光挡住了。
            # "多个目标"与"等不到新观测"是别的类型，会照旧往上抛——那两种必须问人。
            fallback = _result_when_occluded(
                readings,
                target,
                config,
                base_pulses=total_base,
                radial_pulses=total_radial,
                iterations=attempt,
            )
            if fallback is None:
                raise
            say(
                f"  物块看不见了（第 {attempt} 次测量）：按第 {len(readings)} 次的读数 "
                f"({readings[-1][0]:+.2f}, {readings[-1][1]:+.2f}) mm 收工，误差 "
                f"{fallback.error_mm:.2f} mm，**未再验证**；"
                "多半是爪子已经悬到物块上方、自己把光挡住了"
            )
            return fallback
        measure_seconds = time.monotonic() - started
        offset = (sighting.x_mm, sighting.y_mm)
        readings.append(offset)

        # 遮挡不按"像素直径突变"判：真机上它总在误报，而 x/y 由像素圆心直接决定、
        # 与 z 的尺度无关（见 docstring 的"为什么不按像素直径突变判遮挡"）。
        error_x = offset[0] - target.x_mm
        error_y = offset[1] - target.y_mm
        error = math.hypot(error_x, error_y)
        # 下一轮判"停稳"用哪一档：还差得远就用粗档（少等几帧），近了回到细档。
        settle_tolerance = (
            config.coarse_settle_tolerance_mm
            if error > config.coarse_error_mm
            else fine_settle_tolerance
        )
        say(
            f"第 {attempt} 次测量：物块 ({offset[0]:+.2f}, {offset[1]:+.2f}) mm，"
            f"误差 {error:.2f} mm（测量 {measure_seconds * 1000:.0f} ms）"
        )
        if error <= config.tolerance_mm:
            return VisualServoResult(
                target=target,
                readings=tuple(readings),
                error_x=error_x,
                error_y=error_y,
                base_pulses=total_base,
                radial_pulses=total_radial,
                iterations=attempt,
                tolerance_mm=config.tolerance_mm,
            )

        # 上一次那一步到底把画面挪到哪了：**连续两步**都与雅可比预测方向相反才停下。
        # 一步不算：物块被爪子推动、检测跳到画面里别的圆上，都会让单独一步看起来相反。
        if previous_step is not None and previous_offset is not None:
            measured = (
                offset[0] - previous_offset[0],
                offset[1] - previous_offset[1],
            )
            predicted = jacobian.displacement_for(previous_step[0], previous_step[1])
            if _step_is_reversed(
                jacobian, step=previous_step, measured=measured, config=config
            ):
                contradictions += 1
                if contradictions >= 2:
                    raise _reversed_step_error(
                        jacobian, step=previous_step, measured=measured
                    )
                say(
                    f"  警告：这一步实测画面位移 ({measured[0]:+.2f}, {measured[1]:+.2f}) mm "
                    f"与雅可比预测 ({predicted[0]:+.2f}, {predicted[1]:+.2f}) mm 方向相反；"
                    "一步不算数（物块被推动、检测跳目标也会这样），再看下一步"
                )
            else:
                contradictions = 0
                say(
                    f"  这一步实测画面位移 ({measured[0]:+.2f}, {measured[1]:+.2f}) mm，"
                    f"雅可比预测 ({predicted[0]:+.2f}, {predicted[1]:+.2f}) mm"
                )

        # 粗调 / 细调分档：误差大时放开单步上限，一步把大部分误差消掉，
        # 免得拆成好几轮、每轮都重新取帧定位。
        coarse = error > config.coarse_error_mm
        limit = config.step_limit_for(error)

        # 剩下的误差比背隙能分辨的还小：**不能**再施加最小步长下限，
        # 那会一步跨过目标、下一轮又跨回来（表现为永远收敛不了）。
        # 如实收工，让调用方知道这是分辨率极限、不是失败。
        if error < config.min_step_mm:
            say(
                f"  误差 {error:.2f} mm 已小于背隙地板 {config.min_step_mm:.2f} mm，"
                "再走只会跨过目标；按分辨率极限收工"
            )
            return VisualServoResult(
                target=target,
                readings=tuple(readings),
                error_x=error_x,
                error_y=error_y,
                base_pulses=total_base,
                radial_pulses=total_radial,
                iterations=attempt,
                tolerance_mm=config.tolerance_mm,
                resolution_limited=True,
            )

        # 每个轴的门槛按各自灵敏度换算，并且不能超过这一档的单步上限（安全优先）。
        # 解算本身在 _plan_planar_step 里，和"单帧一步"共用同一处定义。
        base_step, radial_step = _plan_planar_step(
            error_x, error_y, jacobian, config, limit=limit
        )
        if abs(total_base + base_step) > config.max_total_pulses or abs(
            total_radial + radial_step
        ) > config.max_total_pulses:
            raise VisualServoError(
                f"脉冲预算用完了（底座已走 {total_base}、伸缩已走 {total_radial}，"
                f"上限 {config.max_total_pulses}），误差还有 {error:.2f} mm。"
                "这通常说明雅可比标反了或符号错了，先跑 --calibrate 重标并核对方向"
            )

        say(
            f"  走（{'粗调' if coarse else '细调'}，上限 {limit}）："
            f"底座 {base_step:+d}、伸缩 {radial_step:+d} 脉冲"
        )
        move_seconds = _move_axes(arm, base_step, radial_step, config)
        say(f"  动作耗时 {move_seconds:.2f} s")
        total_base += base_step
        total_radial += radial_step
        # 记下"走之前在哪、发了多少脉冲"，下一轮拿实测位移和雅可比预测对一次。
        previous_offset = offset
        previous_step = (base_step, radial_step)
        # 两个动作都已等完，从这一刻起才允许读新帧。
        checkpoint = time.monotonic() + config.frame_margin_seconds

    last = readings[-1]
    raise VisualServoError(
        f"x/y 伺服 {config.max_iterations} 次仍未进入容差 {config.tolerance_mm:.1f} mm；"
        f"最后读数 ({last[0]:+.2f}, {last[1]:+.2f}) mm、误差 "
        f"{math.hypot(last[0] - target.x_mm, last[1] - target.y_mm):.2f} mm。"
        "先跑 --calibrate 重标雅可比；若每次都在同一处停住，看是不是有线材或结构挡住了"
    )


#: 实时跟随**细调**一步的目标动作时长（秒）：单轴脉冲上限与等待时长都按它折算。
#:
#: 这个数就是跟随的"时间常数"：调大每步走得多、但一轮更慢，调小跟得勤、但每步
#: 只挪一点。现场按料的移动速度调（转盘快就调大）。
FOLLOW_STEP_SECONDS = 0.1
#: 误差还大（> ``coarse_error_mm``）时用这一档，一步把大部分误差吃掉。
#: 不然一个 100 mm 的起始误差要拆成四步，每一步都要付一次取帧的时间；实际动作
#: 时间本来就占大头，拆得越碎反而越慢。上限仍会被并行上限（600 脉冲）夹住。
FOLLOW_COARSE_STEP_SECONDS = 0.2
#: 动作估算里多留的停稳时间（秒）。估算公式只含加速 + 匀速，减速段没算进去；
#: 这一步之后马上要取下一帧，动作还没停完就读，画面是糊的。
FOLLOW_SETTLE_SECONDS = 0.02
#: 实时跟随的默认加减速档位。配置里 200 档在 80 rpm 下起转就要 0.22 s，
#: 加上 0.15 s 到位余量，一步至少 0.37 s——那正是"一顿一顿"的来源。
FOLLOW_ACCELERATION = 255
#: 静止料快速对准（:func:`_align_xy_fast`）的每步动作时长（秒）。比跟随稍大一点：
#: 料已经停住，不必每一步都只挪几毫米；在并行上限内一步走完更快。
TURNTABLE_FAST_STEP_SECONDS = 0.15
#: 跟随的**过冲系数**：解出来要走的脉冲再乘它。料在动，按"把当前误差走完"算永远慢
#: 半拍，多冲 40% 相当于把惯性算进去。离散闭环下残差按 ``|1-K| = 0.4`` 收敛，不会发散；
#: 到料停住那一刻最多来回两次就进容差。觉得追得太猛就调到 1.2，1.0 = 不过冲。
FOLLOW_GAIN = 1.4
#: 连续丢几帧才开始"盲找"。一帧丢是运动模糊/爪影，不是料真的没了。
FOLLOW_LOST_FRAMES = 3
#: 盲找时一次顺时针转多少脉冲（机械臂底座，CW=正）。转盘只会顺时针把料带走，
#: 所以丢料之后朝同一个方向扫，而不是原地等。20 脉冲 ≈ 11° 输出轴。
#: ponytail: 固定小步扫，扫多快没闭环；现场按画面丢失时长调，够不到就加大。
FOLLOW_LOST_BASE_PULSES = 20
#: 丢料后**一次性**把伸缩拉回多少脉冲（正数 = 往里缩）。料跑到后面时伸缩常常已经
#: 伸得太长，视野里根本没有它；120 脉冲（2.3 mm）不解套，按现场经验一次拉回
#: 2000 脉冲（≈38 mm）才够。每个"丢失片段"只拉一次，重新看见料后复位。
FOLLOW_LOST_RETRACT_PULSES = 2000
#: 一次跟随里最多拉回几次（安全阀）：3 × 2000 脉冲 ≈ 114 mm，再往下就快压到
#: 伸缩行程下限了。到顶之后只扫底座，不再继续收——收过头会顶在限位上。
FOLLOW_LOST_MAX_RETRACTS = 3


def follow_x_y(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    target: VisualServoTarget,
    jacobian: VisualServoJacobian,
    config: VisualServoConfig,
    *,
    step_seconds: float = FOLLOW_STEP_SECONDS,
    coarse_step_seconds: float = FOLLOW_COARSE_STEP_SECONDS,
    gain: float = FOLLOW_GAIN,
    acceleration: int = FOLLOW_ACCELERATION,
    after: Optional[float] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> VisualServoResult:
    """实时跟随的**一步**：一帧 → 一次短动作，**不等**那 0.15 s 到位余量。

    与 :func:`align_x_y` 的 ``single_step`` 分支算的是同一件事，差在**动作怎么等**：
    那边走 :func:`modules.grasp.drive.visual_servo_motion._move_axes`，按驱动器公式
    （加速段 + 匀速段 + 0.15 s 到位余量）睡完；配置里 200 档加速度在 80 rpm 下起转
    就要 0.22 s，所以**每一步至少 0.37 s**。跟一个正在动的料时，看起来就是"顿一下、
    走一步"，根本没有跟随的连续感。

    这里把一步压到 ``step_seconds`` 量级：

    - 单轴步长按 ``step_seconds`` 折算脉冲上限（``rpm * 脉冲每圈 / 60 * step_seconds``），
      误差再大也只走一小段，下一帧立刻纠正——误差大时不是走不动，是**分几步走**；
    - 加减速用满档（默认 255）：起转从 0.22 s 压到几毫秒；
    - 只等加速 + 匀速 + 少量停稳，不补 0.15 s 到位余量。

    代价写在结果里：``unverified=True``，这一步走完**不保证到位**，收敛靠外层循环::

        while True:
            robot.grasp.follow_material("red")      # 一轮 = 一帧 + 一小步

    **不动升降**：只发底座与伸缩两轴。高度由调用方负责。

    :param step_seconds: 这一帧的**细调**动作时长（秒）；按它换算单轴脉冲上限。
    :param coarse_step_seconds: 误差大于 ``config.coarse_error_mm`` 时用这一档，一步吃掉
        大部分误差；避免"一个 100 mm 的误差拆成四步、每步都付一次取帧"。
    :param gain: 过冲系数（默认 1.4）：料在动时按"走完当前误差"算会永远慢半拍，多冲
        一点相当于把惯性算进去。1.0 = 不过冲；离散闭环下 1.4 的残差按 0.4 收敛。
    :param acceleration: 加减速档位 0..255；越小越柔和，但起转越慢、跟随越顿。
    :param after: 只接受不早于这个时刻的观测；上一小步走完之后的时间戳能挡住
        "动作之前的旧帧"（走网络镜头 ``CameraLink`` 时那一层才真的按它过滤）。
    :raises VisualServoError: 这一帧看不到该物料、雅可比奇异。
    """
    if not step_seconds > 0:
        raise ValueError(f"step_seconds 必须是正数，收到 {step_seconds!r}")
    if not coarse_step_seconds > 0:
        raise ValueError(f"coarse_step_seconds 必须是正数，收到 {coarse_step_seconds!r}")
    if not gain > 0:
        raise ValueError(f"gain 必须是正数，收到 {gain!r}")
    say = on_message if on_message is not None else (lambda _message: None)
    # 跟随只认 YOLO 框：料离光轴远是正常状态，不该为它补跑颜色/霍夫（约 0.6 s/帧）。
    sighting = _read_material(link, selector, config, after=after, fast=True)
    offset = (sighting.x_mm, sighting.y_mm)
    error_x = offset[0] - target.x_mm
    error_y = offset[1] - target.y_mm
    error = math.hypot(error_x, error_y)

    def result(*, base_pulses: int, radial_pulses: int, resolution_limited: bool = False):
        return VisualServoResult(
            target=target,
            readings=(offset,),
            error_x=error_x,
            error_y=error_y,
            base_pulses=base_pulses,
            radial_pulses=radial_pulses,
            iterations=1,
            tolerance_mm=config.tolerance_mm,
            resolution_limited=resolution_limited,
            unverified=True,
        )

    if error <= config.tolerance_mm:
        say(f"跟随：误差 {error:.2f} mm 已在容差 {config.tolerance_mm:.1f} mm 内，这一步不动")
        return result(base_pulses=0, radial_pulses=0)
    if error < config.min_step_mm:
        say(
            f"跟随：误差 {error:.2f} mm 已小于背隙地板 {config.min_step_mm:.2f} mm，"
            "再走只会跨过目标；按分辨率极限收工"
        )
        return result(base_pulses=0, radial_pulses=0, resolution_limited=True)

    rpm = int(arm.config.rpm)
    pulses_per_revolution = int(arm.pulses_per_revolution)
    # 误差大时按粗调档走：实际动作时间本来就占大头，拆成很多小步只是多付几次取帧。
    # 上限取"时间预算折算的脉冲"与"允许并行的旋转上限"里小的那个：跟着料走时底座
    # 每步都转，超过并行上限就只能串行走，一轮立刻翻倍。
    step = coarse_step_seconds if error > config.coarse_error_mm else step_seconds
    limit = int(step * rpm * pulses_per_revolution / 60.0)
    limit = max(1, min(limit, config.parallel_rotation_limit_pulses))
    base_step, radial_step = _plan_planar_step(
        error_x, error_y, jacobian, config, limit=limit
    )
    if gain != 1.0:
        # 过冲：料在动，按"走完当前误差"算永远慢半拍。多冲的部分下一帧被新读数扣回来，
        # 离散闭环下的残差按 |1-gain| 收敛（1.4 -> 0.4），不会越追越远。
        base_step = _clamp_pulses(base_step * gain, limit)
        radial_step = _clamp_pulses(radial_step * gain, limit)
    say(
        f"跟随：误差 {error:.2f} mm → 底座 {base_step:+d}、伸缩 {radial_step:+d} 脉冲"
        f"（单步上限 {limit}，过冲 ×{gain:g}）"
    )
    # 两轴各发一条（先底座、隔一个帧间隔再伸缩），都不等动作走完；等待按**这次动作的
    # 实际时长**算（见模块常量）。注意轴名：``BASE_AXIS``/``RADIAL_AXIS`` 是伺服这边的
    # 抽象名（radial 在机械臂上叫 extension），所以这里走方向原语，不拼 ``move_axes``
    # 的轴名字典。
    _send_axis(arm, BASE_AXIS, base_step, rpm=rpm, acceleration=acceleration)
    if base_step and radial_step:
        time.sleep(config.inter_frame_seconds)
    _send_axis(arm, RADIAL_AXIS, radial_step, rpm=rpm, acceleration=acceleration)
    wait = max(
        estimate_move_seconds(
            pulses, rpm, pulses_per_revolution, acceleration, FOLLOW_SETTLE_SECONDS
        )
        for pulses in (abs(base_step), abs(radial_step))
        if pulses
    )
    time.sleep(wait)
    return result(base_pulses=base_step, radial_pulses=radial_step)


def _align_xy_fast(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    target: VisualServoTarget,
    jacobian: VisualServoJacobian,
    config: VisualServoConfig,
    *,
    seconds: float,
    tolerance: float,
    step_seconds: float,
    on_message: Callable[[str], None],
) -> VisualServoResult:
    """静止料的快速对准：每轮一帧、一小步，对齐或预算到点就停。

    与 :func:`align_xy_for_turntable` 的慢路径差在三处，而三处都在同一轮里叠加：

    - 帧走 ``frame_fast``：只跑 YOLO 框，不补颜色/霍夫（那一路约 0.6 s/帧）；
    - 动作走 :func:`follow_x_y` 的短步：加速度满档、只等实际动作那一小段，
      不补 0.15 s 到位余量；
    - 每轮只取一帧，修正靠下一步立刻重新测，不在动作之后等"连续两帧一致"。

    预算到点时如果最后一步动过，会再取一帧、**不移动**地确认一次：避免"其实已经到位，
    只是最后一步没被验证"导致调用方不敢下降。

    :param seconds: 从第一次拿到读数之后开始计的对准预算（秒）。
    :param tolerance: 这一次的到位门槛（毫米）。
    :param step_seconds: 每一步允许走多久（秒）；按它折算单轴脉冲上限。
    """
    say = on_message
    # ``follow_x_y`` 按 config 里的门槛判"这一步还要不要动"，所以这次调严/调松要带进去。
    fast_config = replace(config, tolerance_mm=tolerance)
    deadline: Optional[float] = None
    readings: List[Tuple[float, float]] = []
    total_base = 0
    total_radial = 0
    attempts = 0
    final_error_x = 0.0
    final_error_y = 0.0
    resolution_limited = False
    moved_last = False
    missing_last = False
    got_reading = False

    while True:
        attempts += 1
        try:
            result = follow_x_y(
                arm,
                link,
                selector,
                target,
                jacobian,
                fast_config,
                step_seconds=step_seconds,
                acceleration=FOLLOW_ACCELERATION,
                on_message=say,
            )
        except MaterialMissingError:
            missing_last = True
            if got_reading:
                say("  物块看不见了：按已知位置收工，不下降")
                break
            # YOLO 这一帧完全没认出料（可能是模型漏检、也可能还没转进视野）：
            # 退回带颜色/霍夫的慢路径。深蓝柱那种"只有辅助定位找得到"的现场就靠这一步，
            # 而它只在**第一帧**没框时付钱，正常有框的抓取不会变慢。
            say("  快速档第一帧没有 YOLO 框：退回带辅助定位的慢路径")
            return align_xy_for_turntable(
                arm,
                link,
                selector,
                target,
                jacobian,
                config,
                seconds=seconds,
                tolerance_mm=tolerance,
                fast=False,
                on_message=say,
            )
        missing_last = False
        got_reading = True
        offset = result.readings[-1]
        readings.append(offset)
        final_error_x = result.error_x
        final_error_y = result.error_y
        resolution_limited = result.resolution_limited
        total_base += result.base_pulses
        total_radial += result.radial_pulses
        moved_last = bool(result.base_pulses or result.radial_pulses)
        if deadline is None:
            # 预算从这里起算：等料转进视野的时间不算对准时间。
            deadline = time.monotonic() + max(0.0, seconds)
        if result.aligned or result.resolution_limited:
            break
        if time.monotonic() >= deadline:
            say(f"  {seconds:.1f} s 预算用完：停在已知最好位置，不下降")
            break

    if not got_reading:
        raise VisualServoError("一帧都没测到物块，无法对准")

    # 最后一步动过：它还没被验证，再取一帧（不移动）确认。
    verified = True
    if moved_last:
        try:
            sighting = _read_material(link, selector, fast_config, fast=True)
        except MaterialMissingError:
            verified = False
        else:
            offset = (sighting.x_mm, sighting.y_mm)
            readings.append(offset)
            final_error_x = offset[0] - target.x_mm
            final_error_y = offset[1] - target.y_mm
        if missing_last:
            verified = False

    return VisualServoResult(
        target=target,
        readings=tuple(readings),
        error_x=final_error_x,
        error_y=final_error_y,
        base_pulses=total_base,
        radial_pulses=total_radial,
        iterations=attempts,
        tolerance_mm=tolerance,
        resolution_limited=resolution_limited,
        unverified=not verified,
    )


def align_xy_for_turntable(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    target: VisualServoTarget,
    jacobian: VisualServoJacobian,
    config: VisualServoConfig,
    *,
    seconds: float = 1.0,
    max_iterations: int = 3,
    tolerance_mm: Optional[float] = None,
    step_limit: Optional[int] = None,
    fast: bool = False,
    step_seconds: float = TURNTABLE_FAST_STEP_SECONDS,
    on_message: Optional[Callable[[str], None]] = None,
) -> VisualServoResult:
    """转盘抓料用的 x/y 对准：**首帧只取一帧、一步按粗调上限走完、按秒收工**。

    与 :func:`align_x_y` 的差别，以及为什么这样更快：

    - **第一次测量只取一帧**：调用方（:func:`modules.grasp.runtime` 的转盘入口）已经用
      ``vision.wait_scene_stable`` 确认过整幅画面静止，再等"连续两帧一致"是白花一次定位
      （实测约 0.27 s）。物块还没转进视野时仍退回旧语义，按 ``target_wait_seconds`` 等它出现。
    - **动作之后**仍然等停稳：机械臂刚停的那一两帧读数还在漂，拿它去解算会再走一步反向的，
      表现为原地转——那是"更慢"，不是"更快"。
    - **一步按粗调上限走完**（``step_limit`` 默认 ``coarse_max_pulses_per_step``），
      不再拆成细调的小步。
    - **按墙上时间收工**：预算从**第一次拿到读数之后**才起算（前面可能在等物块转进视野，
      实测能等 6 s 以上，算进去就会"一次都没动就预算用完"）。第一次修正**一定**会走，
      之后每再走一步之前看预算；到点就退回到**测到过的误差最小位置**。返回值可能
      ``aligned=False``——那是"已知最好"，不是对准了，调用方**不能**拿它去下降。

    每一轮测量都带 ``after=checkpoint`` 过滤旧帧：动作很快时队列里还压着动作**之前**的画面，
    读到它就会以为没动过、再发一次同样的指令（表现为来回振荡）。

    :param seconds: 对准预算（秒），**从第一次拿到读数之后**起算。
    :param max_iterations: 最多测几次（首次 + 验证 + 补充步）；时间预算先到时以时间预算为准。
    :param tolerance_mm: 这一次的到位门槛；``None`` 用 ``config.tolerance_mm``。
    :param step_limit: 单轴单步脉冲上限；``None`` 用 ``config.coarse_max_pulses_per_step``。
    :param fast: 走 :func:`_align_xy_fast`：每轮一帧、一小步，跳过颜色/霍夫的辅助定位。
        物料**已经停住**时用这条（``wait_scene_stable`` 之后）；本来就要快速修正的
        场合，它比默认的"一大步 + 停稳验证"快得多。默认 ``False`` 保持原行为。
    :param step_seconds: ``fast=True`` 时每一步允许走多久（秒）。
    :raises VisualServoError: 一帧都测不到、脉冲预算用完；中途物块看不见时按已知最好位置收工。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    tolerance = config.tolerance_mm if tolerance_mm is None else float(tolerance_mm)
    if tolerance <= 0:
        raise ValueError("tolerance_mm 必须是正数")
    if fast:
        return _align_xy_fast(
            arm,
            link,
            selector,
            target,
            jacobian,
            config,
            seconds=seconds,
            tolerance=tolerance,
            step_seconds=step_seconds,
            on_message=say,
        )
    limit = config.coarse_max_pulses_per_step if step_limit is None else int(step_limit)
    # ponytail: 预算内只保证"一次测量 + 一步 + 一次验证"。初始误差大于一步的修正量时，
    # 到点就停在已知最好位置、不下降——要"抓到"而不是"快点抓空"，把 seconds 调到 2–3
    # 或者先让转盘停位对齐，比在这里多塞小步划算。
    budget = max(0.0, float(seconds))
    started = time.monotonic()
    #: 预算截止时刻；**第一次拿到读数之后**才定。等物块转进视野（``target_wait_seconds``
    #: 可能是 ``null``，一直等）不是对准时间，算进去就会出现"一次都没动就预算用完"。
    deadline: Optional[float] = None
    #: 验证那几轮判"停稳"用的一致性门槛，与 ``align_x_y`` 同一个取法。
    fine_settle = min(
        config.coarse_settle_tolerance_mm,
        max(config.settle_tolerance_mm, 0.25 * tolerance),
    )

    #: ``(误差, 偏移, 累计底座脉冲, 累计伸缩脉冲)``，按测量顺序。
    history: List[Tuple[float, Tuple[float, float], int, int]] = []
    total_base = 0
    total_radial = 0
    checkpoint: Optional[float] = None
    attempts = 0

    while True:
        attempts += 1
        if not history:
            try:
                # 场景刚被判过静止：这一帧就是可信的读数，不必再等第二帧。
                sighting = _read_material(link, selector, config)
            except MaterialMissingError:
                # 物块还没转进视野。这不是失败，退回旧语义等它出现。
                sighting = _settled_sighting(
                    link,
                    selector,
                    config,
                    tolerance=fine_settle,
                    include_depth=False,
                    wait_for_first_target=True,
                    on_message=say,
                )
        else:
            try:
                # 动作之后必须等停稳：刚停那一两帧读数还在漂，拿它解算会再走反方向。
                sighting = _settled_sighting(
                    link,
                    selector,
                    config,
                    after=checkpoint,
                    tolerance=fine_settle,
                    include_depth=False,
                    on_message=say,
                )
            except MaterialMissingError:
                say("  物块看不见了：按已知误差最小的位置收工")
                break
        offset = (sighting.x_mm, sighting.y_mm)
        error_x = offset[0] - target.x_mm
        error_y = offset[1] - target.y_mm
        error = math.hypot(error_x, error_y)
        history.append((error, offset, total_base, total_radial))
        if deadline is None:
            # 预算从这里起算，所以第一次修正**一定**会走：没动就收工等于没对准。
            deadline = time.monotonic() + budget
        say(
            f"第 {attempts} 次（{time.monotonic() - started:.2f} s）："
            f"物块 ({offset[0]:+.2f}, {offset[1]:+.2f}) mm，误差 {error:.2f} mm"
        )
        if error <= tolerance:
            return VisualServoResult(
                target=target,
                readings=tuple(item[1] for item in history),
                error_x=error_x,
                error_y=error_y,
                base_pulses=total_base,
                radial_pulses=total_radial,
                iterations=attempts,
                tolerance_mm=tolerance,
            )
        if error < config.min_step_mm:
            say(
                f"  误差 {error:.2f} mm 已小于背隙地板 {config.min_step_mm:.2f} mm，"
                "再走只会跨过目标；按分辨率极限收工"
            )
            return VisualServoResult(
                target=target,
                readings=tuple(item[1] for item in history),
                error_x=error_x,
                error_y=error_y,
                base_pulses=total_base,
                radial_pulses=total_radial,
                iterations=attempts,
                tolerance_mm=tolerance,
                resolution_limited=True,
            )
        if attempts >= max_iterations:
            say(f"  测量次数用完了（{max_iterations} 次）：按已知误差最小的位置收工")
            break
        if attempts > 1 and deadline is not None and time.monotonic() >= deadline:
            say(f"  {budget:.1f} s 预算用完（首测之后开始计）：按已知误差最小的位置收工")
            break
        base_step, radial_step = _plan_planar_step(
            error_x, error_y, jacobian, config, limit=limit
        )
        if abs(total_base + base_step) > config.max_total_pulses or abs(
            total_radial + radial_step
        ) > config.max_total_pulses:
            raise VisualServoError(
                f"脉冲预算用完了（底座已走 {total_base}、伸缩已走 {total_radial}，"
                f"上限 {config.max_total_pulses}），误差还有 {error:.2f} mm。"
                "先跑 --calibrate 重标雅可比并核对方向"
            )
        say(f"  走（大步，单轴上限 {limit}）：底座 {base_step:+d}、伸缩 {radial_step:+d} 脉冲")
        _move_axes(arm, base_step, radial_step, config)
        total_base += base_step
        total_radial += radial_step
        checkpoint = time.monotonic() + config.frame_margin_seconds

    last = history[-1]
    best = min(history, key=lambda item: item[0])
    final = last
    retreat_base = best[2] - total_base
    retreat_radial = best[3] - total_radial
    # 只在"最好那次明显比现在好"时退：差不到一个背隙地板，退回去只是白花一次动作。
    if best[0] < last[0] - config.min_step_mm and (retreat_base or retreat_radial):
        say(f"  退回误差最小那一次：底座 {retreat_base:+d}、伸缩 {retreat_radial:+d} 脉冲")
        _move_axes(arm, retreat_base, retreat_radial, config)
        final = best
    readings = [item[1] for item in history if item is not final]
    readings.append(final[1])
    return VisualServoResult(
        target=target,
        readings=tuple(readings),
        error_x=final[1][0] - target.x_mm,
        error_y=final[1][1] - target.y_mm,
        base_pulses=final[2],
        radial_pulses=final[3],
        iterations=len(history),
        tolerance_mm=tolerance,
    )


def _blind_search_step(
    arm,
    *,
    base_pulses: int,
    radial_pulses: int,
    rpm: int,
    pulses_per_revolution: int,
    acceleration: int,
    inter_frame_seconds: float,
) -> float:
    """丢料后的**盲扫一步**：按已知方向转底座、同时回收伸缩；返回等待秒数。

    ``base_pulses`` 的符号就是搜索方向（正 = 机械臂 CW）。转盘只会顺时针把料带走，
    所以这里默认也朝 CW 扫；伸缩往回缩是为了把因为伸得太长而丢掉的视野收回来。
    两条都只发一小段、按实际动作时长等，和 :func:`follow_x_y` 同一个节奏。
    """
    _send_axis(arm, BASE_AXIS, base_pulses, rpm=rpm, acceleration=acceleration)
    if base_pulses and radial_pulses:
        time.sleep(inter_frame_seconds)
    _send_axis(arm, RADIAL_AXIS, radial_pulses, rpm=rpm, acceleration=acceleration)
    wait = max(
        estimate_move_seconds(
            pulses, rpm, pulses_per_revolution, acceleration, FOLLOW_SETTLE_SECONDS
        )
        for pulses in (abs(base_pulses), abs(radial_pulses))
        if pulses
    )
    time.sleep(wait)
    return wait


def follow_material_xy(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    target: VisualServoTarget,
    jacobian: VisualServoJacobian,
    config: VisualServoConfig,
    *,
    timeout: float = 20.0,
    tolerance_mm: Optional[float] = None,
    step_seconds: float = FOLLOW_STEP_SECONDS,
    coarse_step_seconds: float = FOLLOW_COARSE_STEP_SECONDS,
    gain: float = FOLLOW_GAIN,
    acceleration: int = FOLLOW_ACCELERATION,
    lost_frames: int = FOLLOW_LOST_FRAMES,
    lost_base_pulses: int = FOLLOW_LOST_BASE_PULSES,
    lost_retract_pulses: int = FOLLOW_LOST_RETRACT_PULSES,
    on_message: Optional[Callable[[str], None]] = None,
) -> VisualServoResult:
    """**跟着走起来的物块走**：每轮取一帧、走一小步，料停住才收工。

    每一步交给 :func:`follow_x_y`：误差大时按 ``coarse_step_seconds`` 一步吃掉大部分
    （底座最多约 49 mm），误差小了再按 ``step_seconds`` 细跟。加速度满档、只等实际
    动作那一小段——不是"走一大步再按估算等 0.37 s"那个一顿一顿的节奏。

    **料跑到后面去、看不见了**：连续丢 ``lost_frames`` 帧后开始盲扫——底座朝上次
    跟随的方向（默认 CW，转盘就是顺时针把料带走的）转 ``lost_base_pulses`` 脉冲；
    每个丢失片段**先一次性**把伸缩拉回 ``lost_retract_pulses`` 脉冲（默认 2000 ≈ 38 mm），
    把"伸得太长、视野已经不在料上"这件事解套；之后每丢一帧再扫一步底座，
    直到重新看见或超时。方向用**上一次成功的底座步向**（还没有就用
    ``lost_base_pulses`` 的符号，默认 CW），所以底座装反了不用改代码。

    "料停住了"要**同时**满足两条，缺一不可：

    1. 这一步不需要发动作（``base_pulses == radial_pulses == 0``），且与上一帧读数相差
       不超过 ``still_tolerance``；
    2. 观测源支持 ``scene_stable()`` 时（本机 :class:`~modules.vision.runtime.Vision` 有），
       再确认**整幅画面**静止。

    只比 1 会在"跟得很准"时误判：相机装在爪上，料还在走但始终停在画面同一处，读数本来
    就不变；整幅画面才看得见背景还在动。连续两次满足才返回。

    跟丢（转出去／被爪影挡住）的那几帧不算失败，接着跟；跟满 ``timeout`` 才报错。

    **一次升降帧都不发**：只发底座与伸缩两轴。

    返回时位置可信度有限（``unverified``）：``aligned`` 说的是料停住时读数在不在容差内；
    精确对准由 :func:`align_xy_for_turntable` 接手。

    :param timeout: 最多跟多久（秒）。
    :param tolerance_mm: 到位门槛；``None`` 用 ``config.tolerance_mm``。
    :param step_seconds: 细跟一步允许走多久（秒），折算成单轴脉冲上限。
    :param coarse_step_seconds: 误差大于 ``config.coarse_error_mm`` 时用的粗调档。
    :param gain: 过冲系数（默认 1.4）：料在动时多冲一点，少追几步。
    :param acceleration: 加减速档位 0..255；越小越柔和，但起转越慢、跟得越顿。
    :param lost_frames: 连续丢几帧才开始盲扫。
    :param lost_base_pulses: 盲扫时底座每次转多少脉冲（正 = CW）。
    :param lost_retract_pulses: 每个丢失片段**一次性**回收多少伸缩脉冲（正数 = 往里缩）。
        这两个都给 0 就关掉盲扫（只等转盘把料带回来）。
    :raises VisualServoError: 跟满 ``timeout`` 料还没停住（或一直看不见）。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    tolerance = config.tolerance_mm if tolerance_mm is None else float(tolerance_mm)
    if tolerance <= 0:
        raise ValueError("tolerance_mm 必须是正数")
    #: "料停住了"的读数判据：连续两帧读数之差不超过它。比细档稍松，免得读数抖动当成在动。
    still_tolerance = min(
        config.coarse_settle_tolerance_mm,
        max(config.settle_tolerance_mm, 0.25 * tolerance),
    )
    #: 整幅画面静止的判据只在"料看起来停住"时才问，避免每轮多付一次连拍（约 0.13 s）。
    scene_stable = getattr(link, "scene_stable", None)
    seconds = max(0.0, float(timeout))
    started = time.monotonic()
    deadline = started + seconds

    reads = 0
    total_base = 0
    total_radial = 0
    misses = 0
    #: 这一轮"丢失片段"里已经把伸缩拉回来了吗：一次丢失只拉一大把，不每帧都拉。
    lost_retracted = False
    #: 本次跟随一共拉回了几次（安全阀见 ``FOLLOW_LOST_MAX_RETRACTS``）。
    lost_retracts = 0
    #: 上一次成功的底座步向（+1 = CW）；丢料后的盲扫沿这个方向继续。
    #: 还没跟出过一步时为 ``None``：那时按 ``lost_base_pulses`` 的符号定方向（默认 CW）。
    last_base_sign: Optional[int] = None
    previous: Optional[Tuple[float, float]] = None
    still = 0
    last_error: Optional[Exception] = None
    #: 上一步走完的时刻；下一次取帧只接受晚于它的观测（网络镜头那条路真的按它过滤）。
    checkpoint: Optional[float] = None
    rpm = int(arm.config.rpm)
    pulses_per_revolution = int(arm.pulses_per_revolution)

    while time.monotonic() < deadline:
        try:
            result = follow_x_y(
                arm,
                link,
                selector,
                target,
                jacobian,
                config,
                step_seconds=step_seconds,
                coarse_step_seconds=coarse_step_seconds,
                gain=gain,
                acceleration=acceleration,
                after=checkpoint,
                on_message=say,
            )
        except MaterialMissingError as error:
            # 跟着走的时候"这一帧没看见"是常态：爪影、转出去、运动模糊都算，接着跟。
            last_error = error
            previous = None
            still = 0
            misses += 1
            # 连续丢料：转盘只会顺时针把料带走，就沿同一个方向扫；伸缩先一次性拉回一大把
            # （伸得太长时视野根本不在料上，小步回收不解套）。
            if (
                misses >= max(1, int(lost_frames))
                and (lost_base_pulses or lost_retract_pulses)
            ):
                direction = (
                    last_base_sign
                    if last_base_sign is not None
                    else (1 if lost_base_pulses >= 0 else -1)
                )
                base_search = direction * abs(int(lost_base_pulses))
                radial_search = 0
                if (
                    lost_retract_pulses
                    and not lost_retracted
                    and lost_retracts < FOLLOW_LOST_MAX_RETRACTS
                ):
                    radial_search = -abs(int(lost_retract_pulses))
                    lost_retracted = True
                    lost_retracts += 1
                if base_search or radial_search:
                    wait = _blind_search_step(
                        arm,
                        base_pulses=base_search,
                        radial_pulses=radial_search,
                        rpm=rpm,
                        pulses_per_revolution=pulses_per_revolution,
                        acceleration=acceleration,
                        inter_frame_seconds=config.inter_frame_seconds,
                    )
                    total_base += base_search
                    total_radial += radial_search
                    checkpoint = time.monotonic()
                    say(
                        f"  丢料 {misses} 帧：盲扫（底座 {base_search:+d}、"
                        f"伸缩 {radial_search:+d} 脉冲，等 {wait:.2f} s）"
                    )
            continue
        checkpoint = time.monotonic()
        misses = 0
        lost_retracted = False
        offset = result.readings[-1]
        reads += 1
        total_base += result.base_pulses
        total_radial += result.radial_pulses
        if result.base_pulses:
            last_base_sign = 1 if result.base_pulses > 0 else -1
        moved = (
            math.inf
            if previous is None
            else math.hypot(offset[0] - previous[0], offset[1] - previous[1])
        )
        previous = offset
        # 这一帧不需要动、读数也没变，才可能是"料停住"；还要整幅画面也静止（有那条判据时）。
        if (
            moved <= still_tolerance
            and result.base_pulses == 0
            and result.radial_pulses == 0
        ):
            still += 1
        else:
            still = 0
        if still >= 2:
            if scene_stable is None or scene_stable():
                say(f"  料停住了（误差 {result.error_mm:.2f} mm）：交给精确对准")
                return VisualServoResult(
                    target=target,
                    readings=(offset,),
                    error_x=result.error_x,
                    error_y=result.error_y,
                    base_pulses=total_base,
                    radial_pulses=total_radial,
                    iterations=reads,
                    tolerance_mm=tolerance,
                    unverified=True,
                )
            # 读数没变、背景还在动：那是"跟得很准但料还在走"，不能当成停住。
            still = 0
            say(f"  读数没变但画面还在动：继续跟（误差 {result.error_mm:.2f} mm）")

    raise VisualServoError(
        f"跟了 {seconds:.0f} s 料还没停住（或一直看不见）；最后一次跟丢的原因："
        f"{last_error if last_error is not None else '没有（读数一直在变）'}"
    )


def _require_consistent_legs(
    first_leg: Tuple[float, float],
    second_leg: Tuple[float, float],
    *,
    label: str,
    axis: str,
    pulses: int,
) -> None:
    """两段同向位移必须自洽；不自洽就报错，而不是默默采信第二段。

    同向连走两段时，物理上只有一种可能：**两段朝向相同，而且第一段不比第二段大**
    （第一段要吃掉背隙，只会更小）。不满足就说明测的不是机械臂的运动：

    - 物块被爪子推走了；
    - 物块移出画面、检测跳到了别的圆上；
    - 画面还没停稳就开始读了。

    这三种情况下算出来的"斜率"毫无意义，而它又会被当成雅可比去驱动机械臂——
    实测这一处曾把底座那一列算大 2.5 倍，表现就是闭环每步过冲、来回振荡永不收敛。
    """
    first_len = math.hypot(*first_leg)
    second_len = math.hypot(*second_leg)
    if second_len <= 1e-9 and first_len <= 1e-9:
        raise VisualServoError(
            f"探针 {label} 走了 {2 * pulses} 脉冲，画面却一点没动。"
            "这个轴没有驱动到画面（确认它真的接了、且方向没接反），或者物块没在视野里"
        )
    dot = first_leg[0] * second_leg[0] + first_leg[1] * second_leg[1]
    if second_len > 1e-9 and dot < 0:
        raise VisualServoError(
            f"探针 {label} 两段同向位移却朝相反方向（第一段 {first_leg}、第二段 {second_leg}）："
            "说明测到的不是这个轴的运动。常见原因：物块被爪子推走了、物块移出画面后"
            "检测跳到了别的圆上。请把物块挪到画面中央再重试"
        )
    if first_len > second_len * 1.3 + 0.5:
        raise VisualServoError(
            f"探针 {label} 第一段位移 ({first_leg}) 比第二段 ({second_leg}) 还大："
            "同向连走两段时第一段要吃掉背隙，只会更小。多半是画面还没停稳就开始读了，"
            "或者物块被推动了。"
            f"可以把 settle_timeout_seconds（现在决定等多久）调大，或把 probe_pulses 调小"
        )


def hover_above_material(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    config: VisualServoConfig,
    *,
    scale: Optional[LiftScale] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> HoverResult:
    """**就位的第一步**：把镜头到物块的距离伺服到 ``hover_z_mm``（高了降、低了抬）。

    ## 为什么需要它

    x/y 对准**不动高度**，所以"对准时爪子停在哪一层"原本完全取决于机械臂当时在哪——
    只由 ``safe_lift_z_mm`` 兜一个下限（低了才抬，高了不管）。而 :func:`descend_z` 的下降
    是**开环的固定脉冲数**：起点高度不固定，落点就不固定，抓取高度就会一轮一个样。

    配上 ``hover_z_mm`` 之后，每轮开始都先回到同一层，下降的落点才有重复性。
    ``null``（默认）= 不主动调整，行为与以前完全一致。

    ## 两条硬约束

    1. **往下降只降到 ``safe_lift_z_mm`` 为止**。悬停再低也不能低过"转底盘前的安全高度"——
       那正是低位转底盘的成因（配置层面也拦：``hover_z_mm < safe_lift_z_mm`` 直接拒绝加载）。
    2. **往下走之前必须先知道"一步走多远"**。有标定的 ``lift_scale`` 就直接用；没有就先抬
       一段**量出真实灵敏度**（升和降共用同一个 |Δz/脉冲|），量出来之前绝不往下走——
       往下是在朝物块靠近，盲走没有第二次机会。

    到位判据用 ``z_tolerance_mm``（和视觉版 z 同一个"到位"容差）。每段之后都重新测，
    并用这一段实测的 Δz/脉冲修正下一段——和 :func:`raise_before_rotating` 同一套做法。

    :param scale: 已标定的升降灵敏度；``None`` 表示没标定（那就只抬不降）。
    :raises VisualServoError: 看不到物块（起手就要能看见它，否则悬停无从谈起）。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    target = config.hover_z_mm
    if target is None:
        say("  没配 hover_z_mm：不主动调整起始高度（只由 safe_lift_z_mm 兜下限）")
        return HoverResult(configured=False)

    slope: Optional[float] = None
    if scale is not None and abs(scale.dz_per_pulse) > 1e-9:
        slope = abs(scale.dz_per_pulse)
    moved = 0
    start_z: Optional[float] = None
    previous: Optional[Tuple[float, int]] = None
    dead_segments = 0
    for attempt in range(1, config.safe_lift_max_steps + 1):
        # 起手这一测要等停稳：它同时决定"要不要动"和"下一步走多远"。
        sighting = _settled_sighting(link, selector, config)
        z_mm = sighting.z_mm
        if start_z is None:
            start_z = z_mm
        if previous is not None:
            before_z, pulses = previous
            change_mm = z_mm - before_z
            if slope is not None and abs(slope * pulses) > 4 * config.settle_tolerance_mm:
                if abs(change_mm) < config.settle_tolerance_mm:
                    say(
                        f"  警告：走了 {pulses:+d} 脉冲，z 只变了 {change_mm:+.2f} mm"
                        f"（{before_z:.1f} -> {z_mm:.1f} mm）：多半已经到行程端点。"
                        "停在这里，请手动确认高度再继续。"
                    )
                    return HoverResult(
                        configured=True, pulses=moved, start_z_mm=start_z, end_z_mm=z_mm
                    )
            if change_mm != 0 and pulses != 0:
                measured = abs(change_mm / pulses)
                if measured > 0:
                    dead_segments = 0
                    slope = measured
            else:
                dead_segments += 1
                if dead_segments >= 2:
                    say("  警告：连着两段 z 都没变，停在这里；请手动确认高度")
                    return HoverResult(
                        configured=True, pulses=moved, start_z_mm=start_z, end_z_mm=z_mm
                    )

        error = z_mm - target
        if abs(error) <= config.z_tolerance_mm:
            say(
                f"  已在悬停高度：z={z_mm:.1f} mm（目标 {target:.1f} ± "
                f"{config.z_tolerance_mm:.1f}）"
                + (f"，本轮走了 {moved:+d} 脉冲" if moved else "")
            )
            return HoverResult(
                configured=True, pulses=moved, start_z_mm=start_z, end_z_mm=z_mm
            )

        if slope is None:
            if error < 0:
                step_pulses = config.safe_lift_step_pulses
                reason = f"没标定升降灵敏度：只抬不降，按固定 {step_pulses} 脉冲一段抬"
            else:
                say(
                    f"  现在 z={z_mm:.1f} mm 比悬停高度 {target:.1f} mm 远，需要下降，"
                    "但没标定升降灵敏度（lift_scale），不敢盲降。维持现状——"
                    "要不先把升降灵敏度标出来（--align-z-visual 会顺手标）"
                )
                return HoverResult(
                    configured=True,
                    pulses=moved,
                    start_z_mm=start_z,
                    end_z_mm=z_mm,
                    stopped_without_scale=True,
                )
        else:
            # 这里用**当前的** slope：可能是标定值，也可能是上一段实测出来的。
            # 一条约定与 LiftScale.pulses_for 相同：误差为正（z 偏大）就要往下走。
            wanted = int(round(-error / slope))
            step_pulses = max(
                -config.safe_lift_max_pulses_per_step,
                min(config.safe_lift_max_pulses_per_step, wanted),
            )
            if step_pulses < 0:
                # 下降方向的护栏：最多降到 safe_lift_z_mm 那一层。
                allowed = int((z_mm - config.safe_lift_z_mm) / slope)
                if allowed <= 0:
                    say(
                        f"  已经到安全高度 {config.safe_lift_z_mm:.1f} mm，不能再往下悬停；"
                        "要么把 hover_z_mm 调高，要么把 safe_lift_z_mm 调低（后者要谨慎）"
                    )
                    return HoverResult(
                        configured=True,
                        pulses=moved,
                        start_z_mm=start_z,
                        end_z_mm=z_mm,
                        stopped_at_safe_floor=True,
                    )
                step_pulses = -min(abs(step_pulses), allowed)
            reason = (
                f"按 {slope * 1000:+.1f} mm/1000 脉冲算，误差 {error:+.1f} mm -> "
                f"{step_pulses:+d} 脉冲"
            )
        if step_pulses == 0:
            return HoverResult(
                configured=True, pulses=moved, start_z_mm=start_z, end_z_mm=z_mm
            )
        say(
            f"  悬停调整：z={z_mm:.1f} mm（目标 {target:.1f}），{reason}"
            f"（第 {attempt}/{config.safe_lift_max_steps} 次）"
        )
        _move_lift(arm, step_pulses)
        moved += step_pulses
        previous = (z_mm, step_pulses)

    say(
        f"  警告：调了 {config.safe_lift_max_steps} 次仍没到悬停高度 "
        f"{target:.1f} mm（现在 {previous[0] if previous else 0:.1f} mm）。"
        "请手动确认高度再继续。"
    )
    return HoverResult(
        configured=True, pulses=moved, start_z_mm=start_z, end_z_mm=None
    )


def raise_before_rotating(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    config: VisualServoConfig,
    *,
    scale: Optional[LiftScale] = None,
    quick: bool = False,
    on_message: Optional[Callable[[str], None]] = None,
) -> int:
    """**转底盘之前先抬到安全高度**；返回抬升用的脉冲数。

    这一条是机械结构决定的硬约束，不是保守起见：这台机器上**移动 x/y 就是旋转底盘**，
    而爪子和相机在低位时转过去会直接撞到台面上的物块——现场撞倒过一次。

    判据优先用 **z**（镜头到物块顶面的距离）而不是脉冲数：z 由摄像头直接测出来，
    与传动比无关，所以在 ``mm_per_motor_rev`` 还没标定的时候就能生效。

    ## 能算就算，算不了才一段段试

    标了 ``lift_scale``（且 ``safe_lift_use_scale`` 为真）时，按"还差多少毫米"**算一步**
    再走。为什么这条值钱：一次停稳测量实测约 537 ms，而"固定 500 脉冲一段、每段测一次"
    最坏要 20 段——省下来的是**十几段测量**，不是动作时间。

    算出来的那一步夹在 ``safe_lift_max_pulses_per_step`` 以内，而且每段之后都重新测一次：
    标定值不准时最多多走一段，不会一口气冲进行程端点。每段实测到的 Δz/脉冲还会**反过来
    修正下一段的灵敏度**——它比配置里那个数更接近这台机器此刻的状态。

    没标定时保持原样："抬一段、测一次"，直到 ``z >= safe_lift_z_mm``，不依赖任何标定值。

    看不到物块时退回 ``safe_lift_pulses``；那个也是 0 就**明确警告**而不是偷偷放过。

    刻意**不**在事后自动降回去：降回去正是撞到物块的那个动作。什么时候下降由
    :func:`descend_z`（脉冲版）或 :func:`align_z`（视觉版）决定。
    """
    say = on_message if on_message is not None else (lambda _message: None)

    if config.safe_lift_z_mm > 0:
        if quick:
            # 单帧快判：**只有"比安全高度还高出一截"才直接放行**，其余一律退回停稳测量。
            # 为什么要它：每一轮都要查一次的调用方（--follow）原本每轮付一次停稳测量
            # （实测约 0.54 s），而"还在不在安全高度"这件事用一帧就够——前提是留余量：
            # 运动模糊的一帧可能把 z **读大**，而"以为够高"正是低位转底盘的成因。
            try:
                probe = _read_material(link, selector, config)
            except VisualServoError:
                probe = None
            if (
                probe is not None
                and probe.z_mm
                >= config.safe_lift_z_mm + config.safe_lift_quick_margin_mm
            ):
                say(
                    f"  已在安全高度：z={probe.z_mm:.1f} mm ≥ {config.safe_lift_z_mm:.1f} mm"
                    f"（单帧快判，余量 {config.safe_lift_quick_margin_mm:.1f} mm）"
                )
                return 0
            say("  单帧快判不足以放行（读数偏低或余量不够），退回停稳测量")
        slope: Optional[float] = None
        if scale is not None and config.safe_lift_use_scale:
            slope = abs(scale.dz_per_pulse) if abs(scale.dz_per_pulse) > 1e-9 else None
        #: 判断"这一段是不是几乎没动"的参照灵敏度：算得出一步时就是那个值，
        #: 固定步长那条路则用第一段实测出来的值——没有它就没法区分"灵敏度小"和"根本没动"。
        reference = slope
        #: 固定步长那条路上"连着几段几乎没动"。没有灵敏度可参照时，靠它收手。
        dead_segments = 0
        raised = 0
        #: 上一段：抬起**之前**的 z 与那一段走的脉冲。
        previous: Optional[Tuple[float, int]] = None
        for step in range(1, config.safe_lift_max_steps + 1):
            try:
                sighting = _settled_sighting(link, selector, config)
            except VisualServoError:
                say("  看不到物块，没法用 z 判安全高度；改用脉冲数抬升")
                break
            z_mm = sighting.z_mm
            if previous is not None:
                before_z, pulses = previous
                change_mm = z_mm - before_z
                # "走了脉冲但 z 几乎没动"只可能是到行程端点或升降轴没接上。
                # 判据拿**这一段本该动的量**做参照：够长的一段只动了不到一个停稳容差，
                # 就不是"灵敏度偏小"，而是根本没动——再按 2000 脉冲顶下去会顶坏东西。
                if (
                    reference is not None
                    and reference * pulses > 4 * config.settle_tolerance_mm
                    and abs(change_mm) < config.settle_tolerance_mm
                ):
                    say(
                        f"  警告：抬了 {pulses} 脉冲，z 只变了 {change_mm:+.2f} mm"
                        f"（{before_z:.1f} -> {z_mm:.1f} mm）：多半已经到行程端点，"
                        "或者升降轴没接上。停在这里，请手动确认高度再继续。"
                    )
                    return raised
                if change_mm > 0:
                    reference = change_mm / pulses
                    dead_segments = 0
                    if slope is not None:
                        slope = reference
                elif reference is None:
                    # 固定步长那条路（没有灵敏度可参照）：连着两段"几乎没动"就停。
                    # 一段不动可能是背隙或噪声，连着两段就不是了。
                    dead_segments += 1
                    if dead_segments >= 2:
                        say(
                            f"  警告：连着 {dead_segments} 段抬升，z 几乎没变"
                            f"（现在 {z_mm:.1f} mm）：多半已经到行程端点，或者升降轴没接上。"
                            "停在这里，请手动确认高度再继续。"
                        )
                        return raised
            if z_mm >= config.safe_lift_z_mm:
                say(
                    f"  已在安全高度：z={z_mm:.1f} mm ≥ {config.safe_lift_z_mm:.1f} mm"
                    + (f"（本轮抬了 {raised} 脉冲）" if raised else "")
                )
                return raised
            if slope is not None:
                need_mm = config.safe_lift_z_mm - z_mm
                wanted = max(1, int(math.ceil(need_mm / slope * 1.02)))
                step_pulses = min(wanted, config.safe_lift_max_pulses_per_step)
                reason = (
                    f"按 {slope * 1000:+.1f} mm/1000 脉冲算，还差 {need_mm:.1f} mm -> "
                    f"{step_pulses} 脉冲" + ("（夹到单段上限）" if wanted > step_pulses else "")
                )
            else:
                step_pulses = config.safe_lift_step_pulses
                reason = f"没标定升降灵敏度，按固定 {step_pulses} 脉冲一段试"
            say(
                f"  转底盘前抬升：z={z_mm:.1f} mm < {config.safe_lift_z_mm:.1f} mm，{reason}"
                f"（第 {step}/{config.safe_lift_max_steps} 次）"
            )
            _move_lift(arm, step_pulses)
            raised += step_pulses
            previous = (z_mm, step_pulses)
        else:
            say(
                f"  警告：抬了 {config.safe_lift_max_steps} 次仍没到 "
                f"{config.safe_lift_z_mm:.1f} mm；可能是物块已经不在画面里、"
                "或者升降到了行程端点。请手动确认高度再继续。"
            )
            return raised

    if config.safe_lift_pulses > 0:
        say(f"  先抬升 {config.safe_lift_pulses} 脉冲（转底盘前必须先抬起来）")
        seconds = _move_lift(arm, config.safe_lift_pulses)
        say(f"  抬升动作耗时 {seconds:.2f} s")
        return config.safe_lift_pulses

    say(
        "  警告：safe_lift_z_mm 和 safe_lift_pulses 都是 0，转底盘前不会抬升；"
        "低位转动会撞到台面上的物块。确认已经手动抬起来再继续。"
    )
    return 0


def find_material(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    config: VisualServoConfig,
    *,
    max_steps: Optional[int] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> bool:
    """逐步**抬升**直到画面里看得见物块；找到返回 ``True``。

    为什么这是标准起手式：物块太近时既模糊、又掉出单目的工作距离窗口，
    检测会直接失效——看起来像"什么都没有"，其实只是贴太近了。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    steps = config.find_max_steps if max_steps is None else max_steps
    for step in range(1, steps + 1):
        if not _probe_visible(link, selector, config):
            say(
                f"  画面里看不到物块：抬升 {config.find_lift_step_pulses} 脉冲再找"
                f"（第 {step}/{steps} 次）"
            )
            _move_lift(arm, config.find_lift_step_pulses)
            continue
        if step > 1:
            say(f"  抬升 {step - 1} 次后看到了物块")
        return True
    return False


def _try_lift_direction(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    config: VisualServoConfig,
    direction: int,
    say: Callable[[str], None],
) -> Optional[LiftScale]:
    """朝一个方向走两段同向位移，量出 z 灵敏度；测不出来返回 ``None``。

    ``direction`` 取 ``+1``（抬升）或 ``-1``（下降）。测出什么就照实返回，
    连同符号：``dz_per_pulse`` 始终以"抬升为正"表达。
    """
    pulses = config.lift_probe_pulses
    before = _settled_sighting(link, selector, config)
    checkpoint = time.monotonic()
    _move_lift(arm, direction * pulses)
    mid = _settled_sighting(link, selector, config, after=checkpoint)
    checkpoint = time.monotonic()
    _move_lift(arm, direction * pulses)
    after = _settled_sighting(link, selector, config, after=checkpoint)
    _move_lift(arm, -direction * pulses)
    _move_lift(arm, -direction * pulses)

    first_leg = mid.z_mm - before.z_mm
    second_leg = after.z_mm - mid.z_mm
    say(
        f"  升降探针（{'抬升' if direction > 0 else '下降'}）："
        f"z {before.z_mm:.1f} -> {mid.z_mm:.1f} -> {after.z_mm:.1f} mm"
    )
    # 自洽性：z 是按直径反推的（z ∝ 1/直径），所以直径必须跟着一起变。
    # 对不上就说明检测换了目标——升降把真物块推出半径窗口之后，
    # 检测会锁到画面里别的东西上，报出一个看起来很正常、其实毫无关系的 z。
    for tag, first, second in (("第一段", before, mid), ("第二段", mid, after)):
        if not (first.diameter_px and second.diameter_px) or second.z_mm <= 0:
            continue
        expected = first.diameter_px * first.z_mm / second.z_mm
        ratio = second.diameter_px / expected
        if ratio > 1.25 or ratio < 0.8:
            say(
                f"    {tag}的直径与 z 对不上（直径 {first.diameter_px:.0f}->"
                f"{second.diameter_px:.0f} px，按 z 应该到 {expected:.0f} px）："
                "检测换了目标，这一趟不算"
            )
            return None
    # 第二段必须给出足够大的、方向正确的变化才算数。
    # 太小说明这一趟根本没动（撞到行程端点、脉冲被背隙吃掉、或者轴没接上）。
    if direction * second_leg < config.settle_tolerance_mm:
        say(f"    第二段只变了 {second_leg:+.2f} mm，测不出来")
        return None
    scale = LiftScale(dz_per_pulse=direction * second_leg / pulses)
    say(f"    {scale.summary()}")
    return scale


def probe_lift_scale(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    config: VisualServoConfig,
    *,
    on_message: Optional[Callable[[str], None]] = None,
) -> LiftScale:
    """实测升降的 z 灵敏度：走两段**同向**位移，用第二段算斜率。

    和 x/y 的探针同一条道理：第二段没有背隙损失。测完退回原位。

    **两个方向都试**：只朝一个方向探的话，如果那个方向已经到行程端点（例如刚用
    "抬起来找物块"把升降抬到了顶），量到的变化是零，看起来像"这个轴坏了"。
    先试抬升、不成再试下降，两边都测不出来才报错。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    # 探针要转底盘（底座那一列），所以先抬到安全高度。
    raise_before_rotating(arm, link, selector, config, on_message=say)
    upward = _try_lift_direction(arm, link, selector, config, +1, say)
    if upward is not None:
        return upward
    say("  抬升方向测不出来（多半已经到顶），改测下降方向")
    downward = _try_lift_direction(arm, link, selector, config, -1, say)
    if downward is not None:
        return downward
    raise VisualServoError(
        f"抬升和下降两个方向都测不出 z 的变化（每段 {config.lift_probe_pulses} 脉冲）。"
        "常见原因：升降已经贴在行程端点上、脉冲被背隙吃掉（把 lift_probe_pulses 调大）、"
        "或者这个轴没接上。"
    )


def _lift_step_within_guard(
    *, z_mm: float, step: int, scale: LiftScale, config: VisualServoConfig
) -> Tuple[int, bool]:
    """把这一步夹到"预测 z 不低于 ``z_min_mm``"；返回 ``(步长, 是否被夹)``。

    下降护栏用的是**测到的 z**，不是脉冲数——因为没标定 ``mm_per_motor_rev`` 时，
    脉冲数换算不出绝对高度，而 z 本身就是"离镜头多远"，正是撞上去之前会先变小的量。
    """
    predicted = z_mm + step * scale.dz_per_pulse
    if predicted >= config.z_min_mm:
        return step, False
    allowed = (config.z_min_mm - z_mm) / scale.dz_per_pulse
    clamped = int(math.copysign(min(abs(step), abs(allowed)), step))
    return clamped, True


def align_z(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    target_mm: float,
    scale: LiftScale,
    config: VisualServoConfig,
    *,
    search: bool = True,
    on_message: Optional[Callable[[str], None]] = None,
) -> ZResult:
    """驱动升降，让物块深度伺服到 ``target_mm``（毫米）。

    :raises VisualServoError: 看不到物块（且抬升也没找到）、撞上下降护栏、预算用完、迭代用尽。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    if target_mm < config.z_min_mm:
        raise VisualServoError(
            f"目标 z={target_mm:.1f} mm 低于下降护栏 z_min_mm={config.z_min_mm:.1f} mm；"
            "那样永远到不了。要么放宽 z_min_mm，要么把目标改大"
        )
    readings: List[float] = []
    total_pulses = 0
    checkpoint: Optional[float] = None
    #: 第一次测到的像素直径。升降改变的是距离，所以直径**本来就该**随之变化——
    #: 但变化必须是**连续、单调**的；一步之内翻倍或减半只可能是换了目标。
    baseline_diameter: Optional[float] = None

    for attempt in range(1, config.max_iterations + 1):
        try:
            sighting = _settled_sighting(link, selector, config, after=checkpoint)
        except VisualServoError:
            if not search or not find_material(arm, link, selector, config, on_message=say):
                raise
            checkpoint = time.monotonic()
            continue

        # 检测换目标时必须停下来：升降把真物块推出**半径窗口**（窗口按工作距离算）之后，
        # 检测会转而锁到画面里别的东西上，报出一个看起来很正常、其实毫无关系的读数。
        # z 是按直径反推的，所以"直径一步之内翻倍/减半"就是这个事故的签名。
        if sighting.diameter_px:
            if baseline_diameter is None:
                baseline_diameter = sighting.diameter_px
            else:
                ratio = sighting.diameter_px / baseline_diameter
                if ratio > 1.6 or ratio < 0.625:
                    raise VisualServoError(
                        f"画面里物块的像素直径从 {baseline_diameter:.0f} px 变成 "
                        f"{sighting.diameter_px:.0f} px（{ratio:.2f} 倍），一步之内变了这么多"
                        "只可能是**检测换了目标**——多半是真物块被推出了半径窗口"
                        "（窗口按工作距离 z_max_mm 算：镜头抬得太高时物块变小、\n"
                        "  半径掉到窗口下限以下就被丢弃），检测于是锁到画面里别的东西上。\n"
                        "  对策：别把它抬那么高（把 z_max_mm 调大或把目标 z 设小一点），"
                        "或者把视野里其余的东西挪开。"
                    )

        # **下降之前必须确认平面已经对准。** z 是按直径反推的，而直径与物块在画面
        # 哪个位置无关——z 伺服看不见爪子有没有对准，不检查就会让爪子斜着压下去。
        planar = math.hypot(sighting.x_mm, sighting.y_mm)
        if planar > config.z_center_tolerance_mm:
            raise VisualServoError(
                f"物块离光轴 {planar:.1f} mm，超过下降前允许的 "
                f"{config.z_center_tolerance_mm:.1f} mm。平面还没对准就下降会让爪子斜着"
                "压到物块上。先做 x/y 对准（`--follow`，或直接跑一次无参数的菜单 21），"
                "再回来做 z。"
            )

        z_mm = sighting.z_mm
        readings.append(z_mm)
        error = z_mm - target_mm
        say(
            f"第 {attempt} 次测量：z={z_mm:.1f} mm（直径 {sighting.diameter_px or 0:.0f} px），"
            f"误差 {error:+.2f} mm"
        )
        if abs(error) <= config.z_tolerance_mm:
            return ZResult(
                target_mm=target_mm, readings=tuple(readings), error_mm=error,
                pulses=total_pulses, iterations=attempt, tolerance_mm=config.z_tolerance_mm,
            )

        wanted = scale.pulses_for(error)
        step = _clamp_pulses(wanted, config.z_max_pulses_per_step)
        if step == 0:
            return ZResult(
                target_mm=target_mm, readings=tuple(readings), error_mm=error,
                pulses=total_pulses, iterations=attempt, tolerance_mm=config.z_tolerance_mm,
                stopped_by_guard=True,
            )
        step, clamped = _lift_step_within_guard(
            z_mm=z_mm, step=step, scale=scale, config=config
        )
        if step == 0:
            say(f"  下降护栏：z 已经到 {z_mm:.1f} mm（下限 {config.z_min_mm:.1f} mm），不再往下")
            return ZResult(
                target_mm=target_mm, readings=tuple(readings), error_mm=error,
                pulses=total_pulses, iterations=attempt, tolerance_mm=config.z_tolerance_mm,
                stopped_by_guard=True,
            )
        if abs(total_pulses + step) > config.z_max_total_pulses:
            raise VisualServoError(
                f"升降脉冲预算用完了（已走 {total_pulses}，上限 {config.z_max_total_pulses}），"
                f"z 还有 {error:+.2f} mm。多半是灵敏度标反了，重跑 z 标定"
            )

        say(
            f"  {'抬升' if step > 0 else '下降'} {abs(step)} 脉冲"
            + ("（已按下降护栏夹小）" if clamped else "")
        )
        seconds = _move_lift(arm, step)
        say(f"  动作耗时 {seconds:.2f} s")
        total_pulses += step
        checkpoint = time.monotonic() + config.frame_margin_seconds

    last = readings[-1]
    raise VisualServoError(
        f"z 伺服 {config.max_iterations} 次仍未进入容差 {config.z_tolerance_mm:.1f} mm；"
        f"最后 z={last:.1f} mm、误差 {last - target_mm:+.2f} mm。"
        "如果每次都在同一处停住，先看是不是已经贴到物块了（护栏会拦住）"
    )


def descend_z(
    arm,
    link: FrameSource,
    selector: MaterialSelector,
    config: VisualServoConfig,
    *,
    pulses: Optional[int] = None,
    scale: Optional[LiftScale] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> ZDescentResult:
    """z 对准（脉冲版）：**确认平面已对准之后，按固定脉冲数直接下降**。

    和 :func:`align_z` 的区别只有一句：这里不闭环。爪子压下去之后画面里就没有物块了
    （爪瓣挡住、也掉出工作距离窗口），z 这一段本来就拿不到反馈；现场真正好用的量是
    "从平面对准的高度到夹持高度走了多少脉冲"——那是机械安装常数，量一次就固定下来，
    放在 :attr:`VisualServoConfig.z_descend_pulses` 里随时能改。换物块高度、换夹爪都只改这个数，
    不用重标任何东西。

    ## 唯一的护栏：平面已经对准

    脉冲数只保证"往下走了这么多"，**不保证下面正好是物块**。所以走之前先测一次**停稳**
    的读数：物块离光轴超过 ``z_center_tolerance_mm`` 就拒绝下降（一个动作帧都不发），
    并让你先去做 x/y 对准。顺序反了就是让爪子斜着压下去——现场因此差点压坏爪子。

    :param pulses: 覆盖配置里的脉冲数（现场试数用）；``None`` 表示用配置值。
    :param scale: 已标定的升降灵敏度，只用来说一句"预计降到多深"，**不参与决策**。
    :raises VisualServoError: 看不到物块、平面还没对准、或覆盖的脉冲数不是正整数。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    count = config.z_descend_pulses if pulses is None else pulses
    if type(count) is not int or count < 1:
        raise VisualServoError(f"下降脉冲数必须是正整数，拿到 {count!r}")

    sighting = _settled_sighting(link, selector, config)
    planar = math.hypot(sighting.x_mm, sighting.y_mm)
    if planar > config.z_center_tolerance_mm:
        raise VisualServoError(
            f"物块离光轴 {planar:.1f} mm，超过下降前允许的 "
            f"{config.z_center_tolerance_mm:.1f} mm。平面还没对准就下降会让爪子斜着压到"
            "物块上。先做 x/y 对准（`--follow`，或直接跑一次无参数的菜单 21），再回来下降。"
        )
    say(
        f"  下降前：z={sighting.z_mm:.1f} mm，物块离光轴 {planar:.2f} mm"
        f"（要求 ≤ {config.z_center_tolerance_mm:.1f} mm）"
    )
    if scale is not None:
        # 估算一下"这一步大概会降到多深"。只是给人核对的：真正决定走多少的是脉冲数，
        # 不是这个估算——所以它不参与任何判断（灵敏度没标定时整段也不出现）。
        predicted = sighting.z_mm - count * scale.dz_per_pulse
        say(
            f"  按已标定的 {scale.summary()}，这 {count} 脉冲预计把 z 从 "
            f"{sighting.z_mm:.1f} mm 带到约 {predicted:.1f} mm"
            "（只是估算，用来核对脉冲数改对没有）"
        )
    say(f"  下降 {count} 脉冲（升降/6 号；盲降，走完之前画面里就没有物块了）")
    seconds = _move_lift(arm, -count)
    say(f"  下降完成，耗时约 {seconds:.2f} s")
    return ZDescentResult(
        pulses=count, planar_mm=planar, start_z_mm=sighting.z_mm, seconds=seconds
    )


def retract_extension(
    arm,
    config: VisualServoConfig,
    *,
    pulses: Optional[int] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> int:
    """伸缩轴（7 号电机）**后缩**固定脉冲数；返回实际走的脉冲数。

    它是 :func:`descend_z` 之后的第二步（顺序就是"先下降、再后缩"）。单独调它之前先想清楚
    爪子下面是什么：它**不检查**平面有没有对准（那是 :func:`descend_z` 的闸），发出去的就是
    一次水平位移。

    这个数同样来自 :attr:`VisualServoConfig.extension_retract_pulses`：开环、可改、不用标定。

    :param pulses: 覆盖配置里的脉冲数（现场试数用）；``None`` 表示用配置值。
    :raises VisualServoError: 覆盖的脉冲数不是正整数。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    count = config.extension_retract_pulses if pulses is None else pulses
    if type(count) is not int or count < 1:
        raise VisualServoError(f"后缩脉冲数必须是正整数，拿到 {count!r}")
    say(f"  伸缩/7 号后缩 {count} 脉冲")
    seconds = _move_axis(arm, RADIAL_AXIS, -count)
    say(f"  后缩完成，耗时约 {seconds:.2f} s")
    return count


def _self_test() -> int:
    """不需要硬件：验证 2×2 求逆、方向符号、闭环收敛与三种失败路径。"""
    class FakeArm:
        """按真实雅可比把脉冲换算成画面位移；可选每次欠一个固定比例。"""

        def __init__(self, jacobian: VisualServoJacobian, *, gain: float = 1.0) -> None:
            self.jacobian = jacobian
            self.gain = gain
            self.offset = [-8.3, 1.8]
            self.calls: List[Tuple[str, int]] = []
            self.fail_on_radial = False

        def _apply(self, axis: str, pulses: int) -> None:
            dx, dy = self.jacobian.displacement_for(
                pulses if axis == BASE_AXIS else 0,
                pulses if axis == RADIAL_AXIS else 0,
            )
            self.offset[0] += dx * self.gain
            self.offset[1] += dy * self.gain

        def _axis(self, axis: str, action: str, pulses: int) -> None:
            if axis == RADIAL_AXIS and self.fail_on_radial:
                raise AssertionError("这一步不该动伸缩")
            signed = pulses if action in ("cw", "forward") else -pulses
            self.calls.append((f"{axis}:{action}", pulses))
            self._apply(axis, signed)

        def rotate_cw(self, pulses, **_kwargs): self._axis(BASE_AXIS, "cw", pulses)
        def rotate_ccw(self, pulses, **_kwargs): self._axis(BASE_AXIS, "ccw", pulses)
        def move_forward(self, pulses, **_kwargs): self._axis(RADIAL_AXIS, "forward", pulses)
        def move_backward(self, pulses, **_kwargs): self._axis(RADIAL_AXIS, "backward", pulses)

    class FakeLink:
        """模拟一台**带缓冲**的相机：只有明确要求"不早于某时刻"时才给新帧。

        没带过滤的读会拿到上一次发布的快照——真机上就是这样：动作刚做完，
        队列里可能还压着动作**之前**的帧。闭环要是忘了过滤，就会读到旧坐标、
        以为没动过，然后重复发同一条指令（表现为来回振荡）。
        所以这个假件刻意**不**自动变新，它逼调用方把时间戳过滤用对。
        """

        def __init__(self, arm: FakeArm) -> None:
            self.arm = arm
            self.error = None
            self.guarded_reads = 0
            self.unguarded_reads = 0

        def _publish(self, stamp: float) -> Tuple[float, float, float]:
            return (self.arm.offset[0], self.arm.offset[1], stamp)

        def frame(self, *, timeout=None, max_age_seconds=None, min_received_at=None):
            from modules.vision.drive.sighting import parse_sighting

            if min_received_at is None:
                self.unguarded_reads += 1
            else:
                self.guarded_reads += 1
            # 每次都发布当前真实位姿（真摄像头每次读都是一帧新画面）。
            x, y, stamp = self._publish(max(time.monotonic(), min_received_at or 0.0))
            item = parse_sighting(
                f"物料 4: @(566,376) 绿色物料（green），xyz=({x:.4f},{y:.4f},113.2) mm",
                received_at=stamp,
            )
            return (item,)

    # 一个物理上说得通的雅可比：底座让物块在画面里沿一个方向跑，伸缩沿另一个方向跑。
    truth = VisualServoJacobian(
        base_dx=0.0120, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091
    )
    assert not truth.is_singular, truth

    # --- 求逆的正确性：走算出来的脉冲，必须把误差消掉 ---
    for error in ((3.0, -2.0), (-5.5, 1.25), (0.5, 0.5)):
        base_pulses, radial_pulses = truth.pulses_for(*error)
        moved = truth.displacement_for(base_pulses, radial_pulses)
        assert abs(moved[0] + error[0]) < 1e-9, (error, moved)
        assert abs(moved[1] + error[1]) < 1e-9, (error, moved)

    # 误差为 0 时不该要求任何动作。
    assert truth.pulses_for(0.0, 0.0) == (0.0, 0.0)

    # --- 奇异必须拒绝 ---
    singular = VisualServoJacobian(base_dx=0.01, base_dy=0.01, radial_dx=0.02, radial_dy=0.02)
    assert singular.is_singular
    try:
        singular.pulses_for(1.0, 1.0)
    except VisualServoError as error:
        assert "奇异" in str(error), error
    else:
        raise AssertionError("奇异雅可比必须报 VisualServoError")

    # --- 闭环收敛：目标 (0,0) 就是把物块调到相机轴上 ---
    config = VisualServoConfig(
        tolerance_mm=0.5, max_iterations=12, max_pulses_per_step=2000, min_step_mm=0.1
    )
    arm = FakeArm(truth)
    link = FakeLink(arm)
    target = VisualServoTarget(0.0, 0.0)
    result = align_x_y(arm, link, MaterialSelector(material_id=4), target, truth, config)
    assert result.aligned, result
    assert result.iterations <= 4, result.iterations
    assert abs(result.readings[-1][0]) <= 0.5 and abs(result.readings[-1][1]) <= 0.5, result
    assert arm.calls, "应该至少发过一次动作"
    # 关键：动作之后的每一次读都必须带时间戳过滤。假件只在被要求时才发布新帧，
    # 所以少了这个过滤闭环根本收敛不了——这条断言把"确实做了过滤"钉下来。
    assert link.guarded_reads >= result.iterations - 1, (
        link.guarded_reads,
        result.iterations,
    )

    # --- 闭环能修掉系统偏差（每次只走了 70%）---
    biased_arm = FakeArm(truth, gain=0.7)
    biased_link = FakeLink(biased_arm)
    biased = align_x_y(biased_arm, biased_link, MaterialSelector(material_id=4), target, truth, config)
    assert biased.aligned, biased
    assert biased.iterations > result.iterations, (biased.iterations, result.iterations)

    # --- 不收敛要报错，而不是一直走下去 ---
    stuck_arm = FakeArm(truth, gain=0.02)
    stuck_link = FakeLink(stuck_arm)
    try:
        align_x_y(stuck_arm, stuck_link, MaterialSelector(material_id=4), target, truth,
                  VisualServoConfig(tolerance_mm=0.1, max_iterations=3, max_pulses_per_step=2000))
    except VisualServoError as error:
        assert "3 次" in str(error), error
    else:
        raise AssertionError("不收敛必须报 VisualServoError")

    # --- 脉冲预算用完要报错 ---
    budget_arm = FakeArm(truth, gain=0.05)
    budget_link = FakeLink(budget_arm)
    try:
        align_x_y(budget_arm, budget_link, MaterialSelector(material_id=4), target, truth,
                  VisualServoConfig(tolerance_mm=0.01, max_iterations=50, max_pulses_per_step=50,
                              max_total_pulses=100))
    except VisualServoError as error:
        assert "预算" in str(error), error
    else:
        raise AssertionError("预算用完必须报 VisualServoError")

    # --- 看不到物块要报错 ---
    class EmptyLink:
        error = None

        def frame(self, **_kwargs):
            return None

    try:
        align_x_y(
            FakeArm(truth), EmptyLink(), MaterialSelector(material_id=4), target, truth,
            # 停稳超时调小：这里测的是"看不到物块要报错"，不该真等 4 秒。
            VisualServoConfig(tolerance_mm=0.5, settle_timeout_seconds=0.2, settle_poll_seconds=0.02),
        )
    except VisualServoError as error:
        assert "稳定看到物块" in str(error), error
    else:
        raise AssertionError("看不到物块必须报 VisualServoError")

    # --- 等停稳：读数还在变的时候不能拿它当观测 ---
    class DriftingLink:
        """前几次读数在变，之后稳定——机械臂还没停稳就是这个样子。"""

        error = None

        def __init__(self) -> None:
            self._series = [(0.0, 0.0), (30.0, 5.0), (42.0, 8.0)]
            self.reads = 0

        def frame(self, **_kwargs):
            from modules.vision.drive.sighting import parse_sighting

            self.reads += 1
            x, y = self._series.pop(0) if self._series else (42.0, 8.0)
            return (
                parse_sighting(
                    f"物料 4: 绿色物料（green），xyz=({x:.4f},{y:.4f},113.2) mm",
                    received_at=time.monotonic(),
                ),
            )

    drifting = DriftingLink()
    settled = measure_settled(
        drifting, MaterialSelector(material_id=4),
        VisualServoConfig(settle_timeout_seconds=2.0, settle_poll_seconds=0.01),
    )
    assert settled == (42.0, 8.0), settled
    assert drifting.reads == 4, drifting.reads

    # --- 目标判别 ---
    from modules.vision.drive.sighting import parse_sighting

    one = (parse_sighting("物料 4: 绿色物料（green），xyz=(1,2,100) mm"),)
    assert select_material(one, MaterialSelector()).material_id == 4
    assert select_material(one, MaterialSelector(color="green")).material_id == 4
    assert select_material(one, MaterialSelector(material_id=4)).material_id == 4
    two = one + (parse_sighting("物料 5: 蓝色物料（blue），xyz=(3,4,100) mm"),)
    try:
        select_material(two, MaterialSelector())
    except VisualServoError as error:
        assert "多个目标" in str(error), error
    else:
        raise AssertionError("没配判别条件且有多个目标时必须报 VisualServoError")
    # 配了就选得出来。
    assert select_material(two, MaterialSelector(material_id=5)).material_id == 5
    assert select_material(two, MaterialSelector(color="green")).material_id == 4
    try:
        select_material(two, MaterialSelector(material_id=99))
    except VisualServoError as error:
        assert "没有物块" in str(error), error
    else:
        raise AssertionError("配了编号但画面里没有它时必须报 VisualServoError")

    # --- 探针：两个轴各走两段同向位移，最后必须回到原位 ---
    probe_arm = FakeArm(truth)
    probe_link = FakeLink(probe_arm)
    start = list(probe_arm.offset)
    probed = probe_jacobian(
        probe_arm, probe_link, MaterialSelector(material_id=4),
        VisualServoConfig(probe_pulses=800),
    )
    for field in ("base_dx", "base_dy", "radial_dx", "radial_dy"):
        assert abs(getattr(probed, field) - getattr(truth, field)) < 1e-12, (field, probed, truth)
    assert abs(probe_arm.offset[0] - start[0]) < 1e-9, (probe_arm.offset, start)
    assert abs(probe_arm.offset[1] - start[1]) < 1e-9, (probe_arm.offset, start)

    # 探出来奇异的机器要报错（两个轴在画面里同向）。
    # **必须**用同一个假臂喂给探针和链路：两个不同的 FakeArm 会让链路读到一个永远
    # 不动的假臂，"画面没动"会被零位移检查抓出来。
    flat_truth = VisualServoJacobian(base_dx=0.01, base_dy=0.01, radial_dx=0.02, radial_dy=0.02)
    flat_arm = FakeArm(flat_truth)
    try:
        probe_jacobian(
            flat_arm, FakeLink(flat_arm),
            MaterialSelector(material_id=4), VisualServoConfig(probe_pulses=800),
        )
    except VisualServoError as error:
        assert "奇异" in str(error), error
    else:
        raise AssertionError("探出奇异雅可比必须报 VisualServoError")

    print(f"eye-in-hand 伺服自测通过：{result.summary()}")
    print(f"  {truth.summary()}")
    return 0

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

那个量就是 2×2 雅可比 :class:`~modules.grasp.drive.servo_model.ServoJacobian`：
它是**实测出来**的（:func:`probe_jacobian`），所以相机怎么装的、歪多少度、朝哪个方向，
全都不用假设。代价是标定一次。

## 为什么还要闭环

雅可比是线性的，而真实关系在大位移下会弯；背隙、丢步也会让"发了 800 脉冲"不等于
"走了 800 脉冲的量"。所以每次都用**新的一帧**重算。

**新的一帧**是硬要求：动作前记一个时刻，动作后只要晚于它的观测（``min_received_at``）。
用"多久算旧"挡不住"动作很快、旧帧也还新"的情况，那时读到的是动作**之前**的坐标。

## 本模块只留实际算法

配置与数值模型在 :mod:`modules.grasp.drive.servo_model`，观测判据在
:mod:`modules.grasp.drive.servo_observation`，脉冲原语在
:mod:`modules.grasp.drive.servo_motion`。**主编排**（悬停 → 安全抬升 → x/y → 下降 →
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
from typing import Callable, List, Optional, Sequence, Tuple

from modules.grasp.drive.servo_motion import (
    _move_axes,
    _move_axis,
    _move_lift,
    _send_lift,
)
from modules.grasp.drive.servo_model import (
    BASE_AXIS,
    DEFAULT_CONFIG_PATH as DEFAULT_CONFIG_PATH,
    LIFT_AXIS,
    MIN_JACOBIAN_DETERMINANT,
    RADIAL_AXIS,
    GraspReadyResult,
    HoverResult,
    LiftScale,
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoResult,
    ServoTarget,
    ZDescentResult,
    ZResult,
    load_jacobian,
    load_lift_scale,
    load_object_selector,
    load_servo_config,
    load_target,
    load_z_target,
    save_calibration,
)
from modules.grasp.drive.servo_observation import (
    FrameSource,
    VISIBILITY_PROBE_FRAMES,
    _probe_visible,
    _read_object,
    _settled_sighting,
    measure_offset,
    measure_settled,
    select_object,
)
from modules.vision.drive.sighting import MaterialSighting



def _apply_minimum(step: int, wanted: float, minimum: float) -> int:
    """指令小于背隙门槛时按原方向抬到门槛；本来就不动（取整成 0）就保持不动。

    太小的指令会被背隙整段吃掉、画面一动不动，闭环于是以为"没动过"、
    再发一次同样的指令——表现为原地打转。抬到门槛之上才能真正推动它。

    ``minimum`` 是**按该轴灵敏度换算出来的脉冲数**（见
    :meth:`ServoJacobian.minimum_pulses`），不是拍脑袋定的一串数。
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


def probe_jacobian(
    arm,
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    on_message: Optional[Callable[[str], None]] = None,
) -> ServoJacobian:
    """实测 2×2 雅可比：每个轴走两段**同向**位移，用第二段算斜率。

    为什么要走两段同向而不是"去一趟再回来"：反向的第一段位移会被**背隙**吃掉，
    算出来的斜率偏小。两段同向时第二段没有背隙损失，斜率是真的。

    测完把两个轴退回原位，所以这个过程不会把机械臂留在别处。

    :raises ServoError: 探针脉冲为 0、看不到物块、或探出来的雅可比奇异。
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
    jacobian = ServoJacobian(
        base_dx=base_slope[0],
        base_dy=base_slope[1],
        radial_dx=radial_slope[0],
        radial_dy=radial_slope[1],
    )
    if jacobian.is_singular:
        raise ServoError(
            f"探出来的雅可比奇异（行列式 {jacobian.determinant:.3e}）："
            "底座与伸缩在画面里产生的位移方向几乎重合。"
            "请确认相机确实随伸缩轴一起移动，且两个轴的运动在画面里能分辨出方向差异"
        )
    return jacobian


def align_x_y(
    arm,
    link: FrameSource,
    selector: ObjectSelector,
    target: ServoTarget,
    jacobian: ServoJacobian,
    config: ServoConfig,
    *,
    ensure_safe_height: bool = True,
    quick_safe_lift: bool = False,
    lift_scale: Optional[LiftScale] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> ServoResult:
    """闭环把物块伺服到 ``(tx, ty)``；只动 底座 与 伸缩，不碰高度。

    :param ensure_safe_height: 动手之前要不要先确认"已经抬到安全高度"（默认要）。
        连续用同一高度调用时（``--follow`` 的每一轮、或一次"对准+下降"里已经抬过）
        可以关掉：省下的是一次停稳测量（实测约 0.5 s），而高度在这些步骤里不会变。
        只有调用方**确实知道高度没变**才能关——低位转底盘会撞到物块。
    :param quick_safe_lift: 那次检查允许先用**单帧**快判（够高就放行，不确定退回停稳
        测量）。给"每一轮都要查"的 ``--follow`` 用：判据不放宽，但绝大多数轮次只花
        一帧（约 0.22 s）而不是一次停稳测量（约 0.54 s）。
    :param lift_scale: 已标定的升降灵敏度，交给 :func:`raise_before_rotating` 一步算到位。
    :raises ServoError: 看不到物块、雅可比奇异、脉冲预算用完，或迭代用尽仍未进容差。
    """
    say = on_message if on_message is not None else (lambda _message: None)
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
    readings: List[Tuple[float, float]] = []
    total_base = 0
    total_radial = 0
    #: 每次动完之后记一个时刻，下一次测量只接受晚于它的帧。
    #: **这是闭环正确性的关键**：动作很快时，队列里可能还压着动作**之前**的帧，
    #: 读到它就会以为"没动过"，于是再发一次同样的指令——表现为来回振荡。
    checkpoint: Optional[float] = None
    #: 第一次测到的像素直径，用来发现"物块被挡住了"。
    baseline_diameter: Optional[float] = None
    #: 这一轮判"停稳"用的一致性门槛。第一次用细档（要判"是不是已经对准了"），
    #: 之后误差还大就换粗档——粗档少等几帧，而每帧都要跑一次 178 ms 的定位。
    settle_tolerance = config.settle_tolerance_mm

    for attempt in range(1, config.max_iterations + 1):
        started = time.monotonic()
        sighting = _settled_sighting(
            link, selector, config, after=checkpoint, tolerance=settle_tolerance
        )
        measure_seconds = time.monotonic() - started
        offset = (sighting.x_mm, sighting.y_mm)
        readings.append(offset)

        # 物块被挡住时，可见区域从整圆变成月牙，等效半径变小——测距公式按直径反推距离，
        # 于是会报成"物块突然远了几十毫米"。那看起来完全像一次正常观测，其实与机械臂
        # 位置毫无关系；照着它解算只会让闭环空转。这里把它变成一句说人话的报错。
        if sighting.diameter_px:
            if baseline_diameter is None:
                baseline_diameter = sighting.diameter_px
            else:
                ratio = sighting.diameter_px / baseline_diameter
                if abs(ratio - 1.0) > config.diameter_change_limit_ratio:
                    raise ServoError(
                        f"画面里物块的像素直径从 {baseline_diameter:.0f} px 变成 "
                        f"{sighting.diameter_px:.0f} px（{ratio:.2f} 倍），"
                        f"距镜头 {sighting.z_mm:.0f} mm。多半是**爪子或别的结构挡住了物块**"
                        "（被遮住一部分时可见区域变小，测距就会把它算成变远）。\n"
                        "  x/y 对准必须让爪子**高于**物块，否则转动底座时爪瓣会从物块上扫过去。"
                        "先把升降抬起来（或用菜单 2 的「机械臂向上」点动几下）再试。"
                    )
        error_x = offset[0] - target.x_mm
        error_y = offset[1] - target.y_mm
        error = math.hypot(error_x, error_y)
        # 下一轮判"停稳"用哪一档：还差得远就用粗档（少等几帧），近了回到细档。
        settle_tolerance = (
            config.coarse_settle_tolerance_mm
            if error > config.coarse_error_mm
            else config.settle_tolerance_mm
        )
        say(
            f"第 {attempt} 次测量：物块 ({offset[0]:+.2f}, {offset[1]:+.2f}) mm，"
            f"误差 {error:.2f} mm（测量 {measure_seconds * 1000:.0f} ms）"
        )
        if error <= config.tolerance_mm:
            return ServoResult(
                target=target,
                readings=tuple(readings),
                error_x=error_x,
                error_y=error_y,
                base_pulses=total_base,
                radial_pulses=total_radial,
                iterations=attempt,
                tolerance_mm=config.tolerance_mm,
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
            return ServoResult(
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
        base_min, radial_min = jacobian.minimum_pulses(config.min_step_mm)
        base_wanted, radial_wanted = jacobian.pulses_for(error_x, error_y)
        base_step = _apply_minimum(
            _clamp_pulses(base_wanted, limit), base_wanted, min(base_min, limit)
        )
        radial_step = _apply_minimum(
            _clamp_pulses(radial_wanted, limit), radial_wanted, min(radial_min, limit)
        )
        if base_step == 0 and radial_step == 0:
            raise ServoError(
                f"这一轮算出来要走的脉冲都取整成 0（想要 底座 {base_wanted:.2f}、"
                f"伸缩 {radial_wanted:.2f}），但误差还有 {error:.2f} mm。"
                "把 max_pulses_per_step 调大，或者雅可比标定得太小需要重标"
            )
        if abs(total_base + base_step) > config.max_total_pulses or abs(
            total_radial + radial_step
        ) > config.max_total_pulses:
            raise ServoError(
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
        # 两个动作都已等完，从这一刻起才允许读新帧。
        checkpoint = time.monotonic() + config.frame_margin_seconds

    last = readings[-1]
    raise ServoError(
        f"x/y 伺服 {config.max_iterations} 次仍未进入容差 {config.tolerance_mm:.1f} mm；"
        f"最后读数 ({last[0]:+.2f}, {last[1]:+.2f}) mm、误差 "
        f"{math.hypot(last[0] - target.x_mm, last[1] - target.y_mm):.2f} mm。"
        "先跑 --calibrate 重标雅可比；若每次都在同一处停住，看是不是有线材或结构挡住了"
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
        raise ServoError(
            f"探针 {label} 走了 {2 * pulses} 脉冲，画面却一点没动。"
            "这个轴没有驱动到画面（确认它真的接了、且方向没接反），或者物块没在视野里"
        )
    dot = first_leg[0] * second_leg[0] + first_leg[1] * second_leg[1]
    if second_len > 1e-9 and dot < 0:
        raise ServoError(
            f"探针 {label} 两段同向位移却朝相反方向（第一段 {first_leg}、第二段 {second_leg}）："
            "说明测到的不是这个轴的运动。常见原因：物块被爪子推走了、物块移出画面后"
            "检测跳到了别的圆上。请把物块挪到画面中央再重试"
        )
    if first_len > second_len * 1.3 + 0.5:
        raise ServoError(
            f"探针 {label} 第一段位移 ({first_leg}) 比第二段 ({second_leg}) 还大："
            "同向连走两段时第一段要吃掉背隙，只会更小。多半是画面还没停稳就开始读了，"
            "或者物块被推动了。"
            f"可以把 settle_timeout_seconds（现在决定等多久）调大，或把 probe_pulses 调小"
        )


def hover_above_object(
    arm,
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
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
    :raises ServoError: 看不到物块（起手就要能看见它，否则悬停无从谈起）。
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
    selector: ObjectSelector,
    config: ServoConfig,
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
                probe = _read_object(link, selector, config)
            except ServoError:
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
            except ServoError:
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


def find_object(
    arm,
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
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
    selector: ObjectSelector,
    config: ServoConfig,
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
    selector: ObjectSelector,
    config: ServoConfig,
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
    raise ServoError(
        f"抬升和下降两个方向都测不出 z 的变化（每段 {config.lift_probe_pulses} 脉冲）。"
        "常见原因：升降已经贴在行程端点上、脉冲被背隙吃掉（把 lift_probe_pulses 调大）、"
        "或者这个轴没接上。"
    )


def _lift_step_within_guard(
    *, z_mm: float, step: int, scale: LiftScale, config: ServoConfig
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
    selector: ObjectSelector,
    target_mm: float,
    scale: LiftScale,
    config: ServoConfig,
    *,
    search: bool = True,
    on_message: Optional[Callable[[str], None]] = None,
) -> ZResult:
    """驱动升降，让物块深度伺服到 ``target_mm``（毫米）。

    :raises ServoError: 看不到物块（且抬升也没找到）、撞上下降护栏、预算用完、迭代用尽。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    if target_mm < config.z_min_mm:
        raise ServoError(
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
        except ServoError:
            if not search or not find_object(arm, link, selector, config, on_message=say):
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
                    raise ServoError(
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
            raise ServoError(
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
            raise ServoError(
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
    raise ServoError(
        f"z 伺服 {config.max_iterations} 次仍未进入容差 {config.z_tolerance_mm:.1f} mm；"
        f"最后 z={last:.1f} mm、误差 {last - target_mm:+.2f} mm。"
        "如果每次都在同一处停住，先看是不是已经贴到物块了（护栏会拦住）"
    )


def descend_z(
    arm,
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    pulses: Optional[int] = None,
    scale: Optional[LiftScale] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> ZDescentResult:
    """z 对准（脉冲版）：**确认平面已对准之后，按固定脉冲数直接下降**。

    和 :func:`align_z` 的区别只有一句：这里不闭环。爪子压下去之后画面里就没有物块了
    （爪瓣挡住、也掉出工作距离窗口），z 这一段本来就拿不到反馈；现场真正好用的量是
    "从平面对准的高度到夹持高度走了多少脉冲"——那是机械安装常数，量一次就固定下来，
    放在 :attr:`ServoConfig.z_descend_pulses` 里随时能改。换物块高度、换夹爪都只改这个数，
    不用重标任何东西。

    ## 唯一的护栏：平面已经对准

    脉冲数只保证"往下走了这么多"，**不保证下面正好是物块**。所以走之前先测一次**停稳**
    的读数：物块离光轴超过 ``z_center_tolerance_mm`` 就拒绝下降（一个动作帧都不发），
    并让你先去做 x/y 对准。顺序反了就是让爪子斜着压下去——现场因此差点压坏爪子。

    :param pulses: 覆盖配置里的脉冲数（现场试数用）；``None`` 表示用配置值。
    :param scale: 已标定的升降灵敏度，只用来说一句"预计降到多深"，**不参与决策**。
    :raises ServoError: 看不到物块、平面还没对准、或覆盖的脉冲数不是正整数。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    count = config.z_descend_pulses if pulses is None else pulses
    if type(count) is not int or count < 1:
        raise ServoError(f"下降脉冲数必须是正整数，拿到 {count!r}")

    sighting = _settled_sighting(link, selector, config)
    planar = math.hypot(sighting.x_mm, sighting.y_mm)
    if planar > config.z_center_tolerance_mm:
        raise ServoError(
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
    config: ServoConfig,
    *,
    pulses: Optional[int] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> int:
    """伸缩轴（7 号电机）**后缩**固定脉冲数；返回实际走的脉冲数。

    它是 :func:`descend_z` 之后的第二步（顺序就是"先下降、再后缩"）。单独调它之前先想清楚
    爪子下面是什么：它**不检查**平面有没有对准（那是 :func:`descend_z` 的闸），发出去的就是
    一次水平位移。

    这个数同样来自 :attr:`ServoConfig.extension_retract_pulses`：开环、可改、不用标定。

    :param pulses: 覆盖配置里的脉冲数（现场试数用）；``None`` 表示用配置值。
    :raises ServoError: 覆盖的脉冲数不是正整数。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    count = config.extension_retract_pulses if pulses is None else pulses
    if type(count) is not int or count < 1:
        raise ServoError(f"后缩脉冲数必须是正整数，拿到 {count!r}")
    say(f"  伸缩/7 号后缩 {count} 脉冲")
    seconds = _move_axis(arm, RADIAL_AXIS, -count)
    say(f"  后缩完成，耗时约 {seconds:.2f} s")
    return count


def _self_test() -> int:
    """不需要硬件：验证 2×2 求逆、方向符号、闭环收敛与三种失败路径。"""
    class FakeArm:
        """按真实雅可比把脉冲换算成画面位移；可选每次欠一个固定比例。"""

        def __init__(self, jacobian: ServoJacobian, *, gain: float = 1.0) -> None:
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
            self._published: Optional[Tuple[float, float, float]] = None
            self.guarded_reads = 0
            self.unguarded_reads = 0

        def _publish(self, stamp: float) -> None:
            self._published = (self.arm.offset[0], self.arm.offset[1], stamp)

        def frame(self, *, timeout=None, max_age_seconds=None, min_received_at=None):
            from modules.vision.drive.sighting import parse_sighting

            if min_received_at is None:
                self.unguarded_reads += 1
            else:
                self.guarded_reads += 1
            # 每次都发布当前真实位姿（真摄像头每次读都是一帧新画面）。
            self._publish(max(time.monotonic(), min_received_at or 0.0))
            x, y, stamp = self._published
            item = parse_sighting(
                f"物料 4: @(566,376) 绿色物料（green），xyz=({x:.4f},{y:.4f},113.2) mm",
                received_at=stamp,
            )
            return (item,)

    # 一个物理上说得通的雅可比：底座让物块在画面里沿一个方向跑，伸缩沿另一个方向跑。
    truth = ServoJacobian(
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
    singular = ServoJacobian(base_dx=0.01, base_dy=0.01, radial_dx=0.02, radial_dy=0.02)
    assert singular.is_singular
    try:
        singular.pulses_for(1.0, 1.0)
    except ServoError as error:
        assert "奇异" in str(error), error
    else:
        raise AssertionError("奇异雅可比必须报 ServoError")

    # --- 闭环收敛：目标 (0,0) 就是把物块调到相机轴上 ---
    config = ServoConfig(
        tolerance_mm=0.5, max_iterations=12, max_pulses_per_step=2000, min_step_mm=0.1
    )
    arm = FakeArm(truth)
    link = FakeLink(arm)
    target = ServoTarget(0.0, 0.0)
    result = align_x_y(arm, link, ObjectSelector(material_id=4), target, truth, config)
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
    biased = align_x_y(biased_arm, biased_link, ObjectSelector(material_id=4), target, truth, config)
    assert biased.aligned, biased
    assert biased.iterations > result.iterations, (biased.iterations, result.iterations)

    # --- 不收敛要报错，而不是一直走下去 ---
    stuck_arm = FakeArm(truth, gain=0.02)
    stuck_link = FakeLink(stuck_arm)
    try:
        align_x_y(stuck_arm, stuck_link, ObjectSelector(material_id=4), target, truth,
                  ServoConfig(tolerance_mm=0.1, max_iterations=3, max_pulses_per_step=2000))
    except ServoError as error:
        assert "3 次" in str(error), error
    else:
        raise AssertionError("不收敛必须报 ServoError")

    # --- 脉冲预算用完要报错 ---
    budget_arm = FakeArm(truth, gain=0.05)
    budget_link = FakeLink(budget_arm)
    try:
        align_x_y(budget_arm, budget_link, ObjectSelector(material_id=4), target, truth,
                  ServoConfig(tolerance_mm=0.01, max_iterations=50, max_pulses_per_step=50,
                              max_total_pulses=100))
    except ServoError as error:
        assert "预算" in str(error), error
    else:
        raise AssertionError("预算用完必须报 ServoError")

    # --- 看不到物块要报错 ---
    class EmptyLink:
        error = None

        def frame(self, **_kwargs):
            return None

    try:
        align_x_y(
            FakeArm(truth), EmptyLink(), ObjectSelector(material_id=4), target, truth,
            # 停稳超时调小：这里测的是"看不到物块要报错"，不该真等 4 秒。
            ServoConfig(tolerance_mm=0.5, settle_timeout_seconds=0.2, settle_poll_seconds=0.02),
        )
    except ServoError as error:
        assert "稳定看到物块" in str(error), error
    else:
        raise AssertionError("看不到物块必须报 ServoError")

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
        drifting, ObjectSelector(material_id=4),
        ServoConfig(settle_timeout_seconds=2.0, settle_poll_seconds=0.01),
    )
    assert settled == (42.0, 8.0), settled
    assert drifting.reads == 4, drifting.reads

    # --- 目标判别 ---
    from modules.vision.drive.sighting import parse_sighting

    one = (parse_sighting("物料 4: 绿色物料（green），xyz=(1,2,100) mm"),)
    assert select_object(one, ObjectSelector()).material_id == 4
    assert select_object(one, ObjectSelector(color="green")).material_id == 4
    assert select_object(one, ObjectSelector(material_id=4)).material_id == 4
    two = one + (parse_sighting("物料 5: 蓝色物料（blue），xyz=(3,4,100) mm"),)
    try:
        select_object(two, ObjectSelector())
    except ServoError as error:
        assert "多个目标" in str(error), error
    else:
        raise AssertionError("没配判别条件且有多个目标时必须报 ServoError")
    # 配了就选得出来。
    assert select_object(two, ObjectSelector(material_id=5)).material_id == 5
    assert select_object(two, ObjectSelector(color="green")).material_id == 4
    try:
        select_object(two, ObjectSelector(material_id=99))
    except ServoError as error:
        assert "没有物块" in str(error), error
    else:
        raise AssertionError("配了编号但画面里没有它时必须报 ServoError")

    # --- 探针：两个轴各走两段同向位移，最后必须回到原位 ---
    probe_arm = FakeArm(truth)
    probe_link = FakeLink(probe_arm)
    start = list(probe_arm.offset)
    probed = probe_jacobian(
        probe_arm, probe_link, ObjectSelector(material_id=4),
        ServoConfig(probe_pulses=800),
    )
    for field in ("base_dx", "base_dy", "radial_dx", "radial_dy"):
        assert abs(getattr(probed, field) - getattr(truth, field)) < 1e-12, (field, probed, truth)
    assert abs(probe_arm.offset[0] - start[0]) < 1e-9, (probe_arm.offset, start)
    assert abs(probe_arm.offset[1] - start[1]) < 1e-9, (probe_arm.offset, start)

    # 探出来奇异的机器要报错（两个轴在画面里同向）。
    # 注意**必须**用同一个假臂喂给探针和链路：之前这里写成了两个不同的 FakeArm，
    # 链路读的是一个永远不动的假臂，于是"画面没动"——被上面那条零位移检查抓了出来。
    flat_truth = ServoJacobian(base_dx=0.01, base_dy=0.01, radial_dx=0.02, radial_dy=0.02)
    flat_arm = FakeArm(flat_truth)
    try:
        probe_jacobian(
            flat_arm, FakeLink(flat_arm),
            ObjectSelector(material_id=4), ServoConfig(probe_pulses=800),
        )
    except ServoError as error:
        assert "奇异" in str(error), error
    else:
        raise AssertionError("探出奇异雅可比必须报 ServoError")

    print(f"eye-in-hand 伺服自测通过：{result.summary()}")
    print(f"  {truth.summary()}")
    return 0

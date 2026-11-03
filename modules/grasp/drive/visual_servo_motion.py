"""动作侧：把脉冲真的发给机械臂，并等它走完。

这里只有"发指令 + 等"这一件事，**不读视觉、不判断该走多少、不做策略选择**：

- :func:`_move_axis` / :func:`_send_axis`：单轴；
- :func:`_move_axes`：x/y 两个关节按配置估算耗时后一起发；
- :func:`_send_lift` / :func:`_move_lift`：升降轴（6 号）。

估算的等待时间是"按脉冲数折算"，只用来决定等一下多久，不参与任何判据。
依赖方向见 :mod:`modules.grasp.drive.visual_servo_model` 的模块说明：本模块只依赖
``visual_servo_model``，不导入观测与控制器。
"""

from __future__ import annotations

import time
from typing import Optional

from modules.grasp.drive.visual_servo_model import BASE_AXIS, RADIAL_AXIS, VisualServoConfig


#: 伸缩轴在机械臂上的真实轴名（``RADIAL_AXIS`` 是伺服这边的抽象名）。
EXTENSION_AXIS = "extension"

#: 伸缩轴的**安全下限**（电机口径：3200 脉冲 = 360°，与 ``read_abs_state`` 同一把尺）。
#:
#: 0° ≈ 完全缩回的机械端位，再往里发位置帧只会顶在端位上堵转——现场看到的就是
#: "跟随退到 extension=0，电机一直响"。跟随的丢料盲扫一次发 −2000 脉冲（225°），
#: 从 90° 起步算出 −135°，所以必须在**发帧之前**按当前读数夹一次。留 30°（≈5 mm）
#: 余量：换机构/换伸缩节就改这一个数，别按加工图纸改——真机的端位只有撞到才知道。
RADIAL_MIN_MOTOR_DEGREES = 30.0


def clamp_radial_retract(arm, signed_pulses: int) -> int:
    """把一次**回收**的伸缩脉冲夹到行程下限之上；伸出方向原样返回。

    依据是机械臂的**命令里程表**（``read_abs_state``，只读内存、不发串口），不是编码器：
    真堵过之后里程表会比实际更靠里，那只让后续回收更保守，不会反向往外顶。
    """
    if signed_pulses >= 0:
        return signed_pulses
    read_state = getattr(arm, "read_abs_state", None)
    if read_state is None:  # 测试替身没有里程表：没有位置可夹，按原值发
        return signed_pulses
    current = getattr(read_state(EXTENSION_AXIS), "motor_degrees", None)
    if not isinstance(current, (int, float)) or isinstance(current, bool):
        return signed_pulses  # 同上：Mock 替身读到的是 Mock，不是位置
    room = int((current - RADIAL_MIN_MOTOR_DEGREES) * arm.pulses_per_revolution / 360.0)
    if room <= 0:
        return 0
    return max(signed_pulses, -room)


def _move_axis(arm, axis: str, signed_pulses: int) -> float:
    """驱动一个轴并等它走完（按估算耗时）；返回等了多久。"""
    seconds = _send_axis(arm, axis, signed_pulses)
    time.sleep(seconds)
    return seconds


def _move_ring_axis(arm, axis: str, signed_pulses: int) -> None:
    """圆环对准逐段等实测停稳，再由视觉判断误差；小步先越过目标再返回。"""
    if signed_pulses == 0:
        return
    segments = (signed_pulses,)
    if abs(signed_pulses) < 50:
        extra = 300 if signed_pulses > 0 else -300
        segments = (signed_pulses + extra, -extra)
    for index, pulses in enumerate(segments, start=1):
        # 监控接口在返回时按编码器增量修正里程表，下一段必须从实测位置继续。
        target = arm.turns_state.of(axis).pulses + pulses
        context = (
            f"圆环对准 {axis}：净修正 {signed_pulses:+d} 脉冲，"
            f"第 {index}/{len(segments)} 段 {pulses:+d} 脉冲；"
        )
        try:
            arm.move_axis_to_motor_degrees_monitored(
                axis, target * 360.0 / arm.pulses_per_revolution,
                keep_moving=lambda: True, stop_when_still=True,
            )
        except TimeoutError as error:
            raise TimeoutError(context + str(error)) from error
        except RuntimeError as error:
            raise RuntimeError(context + str(error)) from error


def _send_axis(
    arm,
    axis: str,
    signed_pulses: int,
    *,
    rpm: Optional[int] = None,
    acceleration: Optional[int] = None,
) -> float:
    """只**发出**一个轴的动作、不等；返回估算耗时（秒）。

    Emm42 是"发完就返回、不回 ACK"的，所以 ``wait=False`` 立刻返回，
    耗时靠 :func:`estimate_move_seconds` 估算——这正是能并行两个轴的前提。

    ``rpm`` / ``acceleration`` 原样交给机械臂的动作方法：跟随要压短每一步的起转时间，
    就会传更高的档位；不传则用 ``modules/arm/config.json`` 里的默认值。

    伸缩在这里**统一夹一次行程下限**（:func:`clamp_radial_retract`）：跟随、盲扫、
    对准所有伸缩动作都从这一处出去，所以"退到 0° 堵转"只需要在这一个地方拦。
    """
    if signed_pulses == 0:
        return 0.0
    if axis == RADIAL_AXIS:
        signed_pulses = clamp_radial_retract(arm, signed_pulses)
        if signed_pulses == 0:
            return 0.0
    if axis == BASE_AXIS:
        action = arm.rotate_cw if signed_pulses > 0 else arm.rotate_ccw
    else:
        action = arm.move_forward if signed_pulses > 0 else arm.move_backward
    report = action(
        abs(signed_pulses), wait=False, rpm=rpm, acceleration=acceleration
    )
    return float(getattr(report, "estimated_seconds", 0.0) or 0.0)


def _move_axes(arm, base_pulses: int, radial_pulses: int, config: "VisualServoConfig") -> float:
    """走一步平面修正，返回估算耗时（秒）。

    **两个轴同时发**：驱动器不回 ACK，串行 ``wait`` 等于把两段时间**相加**，
    而实际可以并行、只等较长的那个。跟踪速度有一半就丢在这里。

    大角度旋转时**不**并行：伸出状态下横扫底座会把整条臂扫过工作区，
    所以改成"先伸缩、再旋转"，和 ``pipeline`` 是同一条规矩。
    """
    if base_pulses == 0 and radial_pulses == 0:
        return 0.0

    both = base_pulses != 0 and radial_pulses != 0
    if both and abs(base_pulses) > config.parallel_rotation_limit_pulses:
        # 大角度：先缩回/伸出到位，再转。
        seconds = _send_axis(arm, RADIAL_AXIS, radial_pulses)
        time.sleep(seconds)
        seconds += _send_axis(arm, BASE_AXIS, base_pulses)
    else:
        first = _send_axis(arm, BASE_AXIS, base_pulses)
        if both:
            # 两帧之间必须留间隔，否则 CH340 上两帧都失效（见 inter_frame_seconds）。
            time.sleep(config.inter_frame_seconds)
        second = _send_axis(arm, RADIAL_AXIS, radial_pulses)
        seconds = max(first, second)
    time.sleep(seconds)
    return seconds


def _send_lift(arm, signed_pulses: int) -> float:
    """只**发出**升降动作、不等；返回估算耗时（秒）。正数 = 抬升。"""
    if signed_pulses == 0:
        return 0.0
    action = arm.move_up if signed_pulses > 0 else arm.move_down
    report = action(abs(signed_pulses), wait=False)
    return float(getattr(report, "estimated_seconds", 0.0) or 0.0)


def _move_lift(arm, signed_pulses: int) -> float:
    seconds = _send_lift(arm, signed_pulses)
    time.sleep(seconds)
    return seconds

"""动作侧：把脉冲真的发给机械臂，并等它走完。

这里只有"发指令 + 等"这一件事，**不读视觉、不判断该走多少、不做策略选择**：

- :func:`_move_axis` / :func:`_send_axis`：单轴；
- :func:`_move_axes`：x/y 两个关节按配置估算耗时后一起发；
- :func:`_send_lift` / :func:`_move_lift`：升降轴（6 号）。

估算的等待时间是"按脉冲数折算"，只用来决定等一下多久，不参与任何判据。
依赖方向见 :mod:`modules.grasp.drive.servo_model` 的模块说明：本模块只依赖
``servo_model``，不导入观测与控制器。
"""

from __future__ import annotations

from modules.grasp.drive.servo_model import ServoConfig



def _move_axis(arm, axis: str, signed_pulses: int) -> float:
    """驱动一个轴并等它走完（按估算耗时）；返回等了多久。"""
    seconds = _send_axis(arm, axis, signed_pulses)
    time.sleep(seconds)
    return seconds


def _send_axis(arm, axis: str, signed_pulses: int) -> float:
    """只**发出**一个轴的动作、不等；返回估算耗时（秒）。

    Emm42 是"发完就返回、不回 ACK"的，所以 ``wait=False`` 立刻返回，
    耗时靠 :func:`estimate_move_seconds` 估算——这正是能并行两个轴的前提。
    """
    if signed_pulses == 0:
        return 0.0
    if axis == BASE_AXIS:
        action = arm.rotate_cw if signed_pulses > 0 else arm.rotate_ccw
    else:
        action = arm.move_forward if signed_pulses > 0 else arm.move_backward
    report = action(abs(signed_pulses), wait=False)
    return float(getattr(report, "estimated_seconds", 0.0) or 0.0)


def _move_axes(arm, base_pulses: int, radial_pulses: int, config: "ServoConfig") -> float:
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

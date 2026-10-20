"""圆环中心 → 底座**绝对角度**：一次线性化解算，不动任何设备。

## 为什么是"里程表增量"而不是世界坐标

相机装在臂上（eye-in-hand），画面里的环心是**相机系**读数，单张图里没有"环在世界哪"
这件事——相机自己在动。所以能算的只有"该动多少"；而 :class:`modules.arm.runtime.RobotArm`
本来就有**命令脉冲里程表**（``turns_state``，一个标定都不需要）：

```
绝对脉冲 = 当前里程表读数 + Δ脉冲
```

所以这里返回的角度与 :meth:`modules.arm.runtime.RobotArm.move_axis_to_motor_degrees`
是同一个口径（电机轴，3200 脉冲 = 360°，不取模）：

```python
plan = robot.grasp.plan_ring()                       # 一帧定位 + 解算
robot.arm.move_axis_to_motor_degrees("base", plan.base_motor_degrees)
```

## 一次线性化：够用，但要复核

``jacobian`` 是实测的"每脉冲让画面挪多少毫米"，所以 ``p = −J⁻¹·e`` 是把误差 ``e``
一次消掉的两个轴步长。真实关系在大角度下会弯（转 30° 之后雅可比已经变了），所以
这是**一次**解算，不是"走到就一定对"：转完再看一帧，偏差大就再算一次——
和 :meth:`modules.grasp.runtime.Grasp.align_xy` 的闭环是同一个道理，只是这里把
"要转到哪个绝对角度"先算出来交给你。

## 单位必须与雅可比一致

``reading`` 与 ``jacobian`` 必须是**同一把尺**（同一套毫米/像素）。物料那条路线的读数
按针孔标定 ``z/f`` 换算；环心读数如果退回 ``station.mm_per_px``（没有 ``table_depth_mm``
与 ``focal_px`` 时就是这样），两者可以差好几倍——角度也会跟着整体差好几倍，而且看起来
完全正常。所以 :meth:`modules.grasp.runtime.Grasp.plan_ring` 会先核对比例尺来源，
对不上就报错，而不是给一个像样的错数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from modules.grasp.drive.servo_model import ServoError, ServoJacobian

#: 底座在画面里"几乎推不动"的判据（毫米/脉冲）。低于它就解不出角度。
MIN_BASE_SENSITIVITY_MM_PER_PULSE = 1e-6


@dataclass(frozen=True)
class RingBasePlan:
    """从**一帧环心读数**解算出来的"转到哪"。

    角度有两个口径，别混用：

    - ``base_motor_degrees``：**电机轴**绝对角度（3200 脉冲 = 360°，不取模），
      与 :meth:`modules.arm.runtime.RobotArm.move_axis_to_motor_degrees` 一致；
    - ``base_output_degrees``：**输出轴**角度（底座减速比之后，物理上的底盘转角）。

    绝对值的原点由里程表定义：回零 / 绝对归位之后才是"驱动器零点"，否则是
    "程序启动那一刻"（见 :mod:`modules.arm.arm_state`）。
    """

    #: 这一帧用的环心读数与目标读数（相机系毫米）。
    reading: Tuple[float, float]
    target: Tuple[float, float]
    #: ``reading - target``：要消掉的误差。
    error_mm: Tuple[float, float]
    #: 环号（模型给的），没定位到环时不会有这个计划。
    digit: Optional[int] = None
    #: 一帧的解算置信度（模型分数），方便现场判断这一步该不该信。
    score: Optional[float] = None

    #: 底座的增量（带符号，整数脉冲）。
    base_pulses: int = 0
    #: 底座的绝对目标（里程表口径）。
    base_motor_pulses: int = 0
    base_motor_degrees: float = 0.0
    base_output_degrees: float = 0.0
    #: 底座每脉冲多少度（输出轴）与多少电机度，打印/核对用。
    base_output_degrees_per_pulse: float = 0.0
    base_motor_degrees_per_pulse: float = 0.0

    #: 伸缩的增量与绝对目标；只转底座时用不到，但"爪子正对环心"需要它。
    radial_pulses: int = 0
    radial_motor_pulses: int = 0
    radial_motor_degrees: float = 0.0

    #: 两轴都走之后的预测残差（毫米）与是否进容差。
    residual_mm: float = 0.0
    tolerance_mm: float = 0.0
    aligned: bool = False

    #: 人话说明：这一步是什么、还剩多少、下一步该做什么。
    note: str = ""

    def base_command(self) -> Tuple[str, float]:
        """可以直接交给 ``move_axis_to_motor_degrees`` 的一对 ``("base", 角度)``。"""
        return ("base", self.base_motor_degrees)

    def summary(self) -> str:
        where = (
            f"环 {self.digit}（{self.score:.2f}）" if self.digit is not None else "环"
        )
        return (
            f"{where} 读数 ({self.reading[0]:+.1f}, {self.reading[1]:+.1f}) mm，"
            f"误差 {math.hypot(*self.error_mm):.1f} mm → "
            f"底座绝对 {self.base_motor_degrees:+.1f}°（电机轴，"
            f"{self.base_pulses:+d} 脉冲；输出轴 {self.base_output_degrees:+.1f}°）"
            f"，伸缩 {self.radial_pulses:+d} 脉冲"
            f"，预测残差 {self.residual_mm:.1f} mm"
            f"{'（进容差）' if self.aligned else '（还要再迭代一次）'}。"
        )


def plan_ring_base(
    *,
    reading: Sequence[float],
    target: Sequence[float],
    jacobian: ServoJacobian,
    base_pulses: int,
    radial_pulses: int,
    pulses_per_revolution: int,
    reduction_ratio: float,
    tolerance_mm: float = 4.0,
) -> RingBasePlan:
    """解算"把环心读数推到 ``target`` 需要的底座绝对角度"。

    Parameters:
        reading: 环心在**相机系**的毫米坐标（与 ``jacobian`` 同一把尺）。
        target: 爪口在相机系里的目标读数（``eye_in_hand.json`` 的 ``target``）。
        jacobian: 实测的两轴灵敏度（毫米/脉冲）。
        base_pulses / radial_pulses: 两根轴**当前**的里程表读数（可以是负的）。
        pulses_per_revolution: 电机一圈的脉冲数（``3200``）。
        reduction_ratio: 底座减速比（电机转几圈输出轴转一圈）。
        tolerance_mm: 判"已经够准"的门槛；只影响 :attr:`RingBasePlan.aligned`。

    :raises ServoError: 雅可比奇异（两轴在画面里的位移方向重合），或底座那一列
        几乎为零（转底座画面不动，解不出角度）。
    """
    first, second = _pair(reading, "reading")
    target_x, target_y = _pair(target, "target")
    _require_int(base_pulses, "base_pulses")
    _require_int(radial_pulses, "radial_pulses")
    _require_positive_int(pulses_per_revolution, "pulses_per_revolution")
    if not reduction_ratio > 0:
        raise ValueError("reduction_ratio 必须是正数（电机转几圈输出轴转一圈）")
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm 不能是负数")

    error = (first - target_x, second - target_y)
    # 一次线性化：走 p 脉冲让读数变化 J·p，要 e + J·p = 0，即 p = −J⁻¹·e。
    base_step_float, radial_step_float = jacobian.pulses_for(*error)
    if jacobian.base_norm < MIN_BASE_SENSITIVITY_MM_PER_PULSE:  # pragma: no cover - 与奇异判据重叠
        raise ServoError(
            f"底座在画面里几乎推不动（{jacobian.base_norm:.2e} mm/脉冲），解不出角度；"
            "检查相机是不是离转轴太近、或雅可比标定把两个轴记反了"
        )
    base_step = _round(base_step_float)
    radial_step = _round(radial_step_float)

    residual = _residual(error, jacobian, base_step, radial_step)
    base_motor_pulses = base_pulses + base_step
    radial_motor_pulses = radial_pulses + radial_step
    motor_degrees_per_pulse = 360.0 / float(pulses_per_revolution)
    base_motor_degrees = base_motor_pulses * motor_degrees_per_pulse
    radial_motor_degrees = radial_motor_pulses * motor_degrees_per_pulse
    base_output_degrees = base_motor_degrees / float(reduction_ratio)
    aligned = residual <= tolerance_mm

    if aligned:
        note = (
            f"一次解算就把残差降到 {residual:.1f} mm（容差 {tolerance_mm:.1f} mm）。"
        )
    else:
        note = (
            f"一次线性化解算后预测残差 {residual:.1f} mm（容差 {tolerance_mm:.1f} mm）："
            "大角度下雅可比会变，走完再看一帧重算一次就能收敛"
        )
    return RingBasePlan(
        reading=(first, second),
        target=(target_x, target_y),
        error_mm=error,
        base_pulses=base_step,
        base_motor_pulses=base_motor_pulses,
        base_motor_degrees=base_motor_degrees,
        base_output_degrees=base_output_degrees,
        base_output_degrees_per_pulse=motor_degrees_per_pulse / float(reduction_ratio),
        base_motor_degrees_per_pulse=motor_degrees_per_pulse,
        radial_pulses=radial_step,
        radial_motor_pulses=radial_motor_pulses,
        radial_motor_degrees=radial_motor_degrees,
        residual_mm=residual,
        tolerance_mm=float(tolerance_mm),
        aligned=aligned,
        note=note,
    )


# ---------------------------------------------------------------------- 内部
def _residual(
    error: Tuple[float, float],
    jacobian: ServoJacobian,
    base_step: int,
    radial_step: int,
) -> float:
    """按**整数**脉冲推算走完之后的残差长度（四舍五入的余量也要算进去）。"""
    moved_x, moved_y = jacobian.displacement_for(base_step, radial_step)
    return math.hypot(error[0] + moved_x, error[1] + moved_y)


def _pair(value: Sequence[float], name: str) -> Tuple[float, float]:
    """把 ``(x, y)`` 校验成两个有限数。"""
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} 要写成 (x, y) 两个数，收到 {value!r}")
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} 里的每一项都必须是数字，收到 {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} 里有非有限数：{item!r}")
        numbers.append(number)
    return (numbers[0], numbers[1])


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数脉冲，收到 {value!r}")
    return int(value)


def _require_positive_int(value: object, name: str) -> int:
    number = _require_int(value, name)
    if number <= 0:
        raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
    return number


def _round(value: float) -> int:
    """四舍五入到整数脉冲（``.5`` 一律远离零，避免银行家舍入把 2.5 变成 2）。"""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要机械臂、不需要相机的自测：验算一遍角度与绝对脉冲。"""
    #: 实测值（modules/grasp/eye_in_hand.json）：底座每脉冲让画面挪 (−0.0817, −0.0076) mm。
    jacobian = ServoJacobian(
        base_dx=-0.08174204217107724,
        base_dy=-0.007583697824866524,
        radial_dx=-0.0002735146342596506,
        radial_dy=0.01812172800295296,
    )
    pulses_per_revolution = 3200
    reduction_ratio = 5

    problems = []

    # 1) 已经在目标上：一帧都不该发
    plan = plan_ring_base(
        reading=(0.0, 0.0), target=(0.0, 0.0), jacobian=jacobian,
        base_pulses=1000, radial_pulses=-500,
        pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
    )
    if plan.base_pulses != 0 or plan.radial_pulses != 0:
        problems.append(f"已在目标上却给了步长：{plan.base_pulses}/{plan.radial_pulses}")
    if plan.base_motor_pulses != 1000 or not plan.aligned:
        problems.append(f"绝对值应当是当前脉冲，实际 {plan.base_motor_pulses}")

    # 2) 读数偏 +10 mm（沿底座那一列的反方向）：应当往正方向转
    #    Δ = −(e·j)/|j|² ；e=(10,0)、j=(−0.0817,−0.0076) ⇒ Δ ≈ +121.5 脉冲
    plan = plan_ring_base(
        reading=(10.0, 0.0), target=(0.0, 0.0), jacobian=jacobian,
        base_pulses=0, radial_pulses=0,
        pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
    )
    if not 118 <= plan.base_pulses <= 125:
        problems.append(f"底座步长应当在 +121 附近，实际 {plan.base_pulses}")
    print(plan.summary())

    # 3) 绝对角度 = (当前 + 增量) × 360/PPR
    expected_degrees = plan.base_pulses * 360.0 / pulses_per_revolution  # 当前脉冲是 0
    if abs(plan.base_motor_degrees - expected_degrees) > 1e-9:
        problems.append(
            f"绝对角度不对：{plan.base_motor_degrees} != {expected_degrees}"
        )
    if abs(plan.base_output_degrees - expected_degrees / reduction_ratio) > 1e-9:
        problems.append(f"输出轴角度没除减速比：{plan.base_output_degrees}")

    # 4) 两轴一起走：残差应当接近 0
    plan = plan_ring_base(
        reading=(3.0, -4.0), target=(0.0, 0.0), jacobian=jacobian,
        base_pulses=0, radial_pulses=0,
        pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
        tolerance_mm=1.0,
    )
    if plan.residual_mm > 1.0:
        problems.append(f"两轴都走之后残差应当很小，实际 {plan.residual_mm:.2f} mm")

    # 5) 只差目标读数里的那一点：target 本身就是爪口位置
    plan = plan_ring_base(
        reading=(2.0, 0.0), target=(2.0, 0.0), jacobian=jacobian,
        base_pulses=0, radial_pulses=0,
        pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
    )
    if plan.base_pulses != 0:
        problems.append(f"读数等于目标时不该动，实际 {plan.base_pulses}")

    # 6) 奇异雅可比：直接拒绝，不能编一个角度出来
    singular = ServoJacobian(base_dx=0.01, base_dy=0.02, radial_dx=0.02, radial_dy=0.04)
    try:
        plan_ring_base(
            reading=(1.0, 1.0), target=(0.0, 0.0), jacobian=singular,
            base_pulses=0, radial_pulses=0,
            pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
        )
    except ServoError as error:
        if "奇异" not in str(error):
            problems.append(f"奇异时要说清楚，实际 {error}")
    else:
        problems.append("奇异雅可比应当报错")

    # 7) 参数写错：在算之前就拒绝
    for bad in ((1.0,), ("a", "b"), (1.0, float("nan"))):
        try:
            plan_ring_base(
                reading=bad, target=(0.0, 0.0), jacobian=jacobian,
                base_pulses=0, radial_pulses=0,
                pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
            )
        except ValueError:
            pass
        else:
            problems.append(f"非法读数 {bad!r} 应当被拒绝")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：绝对角度、两轴步长、奇异与参数校验都按约定处理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

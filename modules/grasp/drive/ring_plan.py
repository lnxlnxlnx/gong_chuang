"""圆环中心 → **各电机的绝对位置**：一次线性化解算，不动任何设备。

## 用法：先算、后走

```python
plan = robot.grasp.plan_ring()          # 只算：一帧定位 + 一次解算，电机一帧都不动
...                                     # 中间去抓料、摆料，随便走多少步

robot.arm.move_axis_to_motor_degrees("base", plan.base)            # 底座绝对角度
robot.arm.move_axis_to_motor_degrees("extension", plan.extension)   # 伸缩绝对角度
# 或者一步走完：plan.apply(robot.arm)；只转底座：plan.apply(robot.arm, axes=("base",))
```

``plan.degrees`` / ``plan.pulses`` 是同一位置的字典，键就是轴名::

    app.gripper._move_axes_to(robot, plan.degrees)      # {"base": 222.5, "extension": 5.7}

## 为什么是"里程表增量"而不是世界坐标

相机装在臂上（eye-in-hand），画面里的环心是**相机系**读数，单张图里没有"环在世界哪"
这件事——相机自己在动。所以能算的只有"该动多少"；而 :class:`modules.arm.runtime.RobotArm`
本来就有**命令脉冲里程表**（``turns_state``，一个标定都不需要）：

```
绝对脉冲 = 当前里程表读数 + Δ脉冲
```

所以这里返回的角度与 :meth:`modules.arm.runtime.RobotArm.move_axis_to_motor_degrees`
是同一个口径（电机轴，3200 脉冲 = 360°，不取模）。因为是绝对位置，算完隔多久再走都
对得上——只要中间**没清零里程表**（现在没有代码会自动清零，只有手动
``turns_state.mark_origin()`` 或改 ``initial_motor_degrees`` 才会重定义原点）。

## 一次线性化：够用，但要复核

``jacobian`` 是实测的"每脉冲让画面挪多少毫米"，所以 ``p = −J⁻¹·e`` 是把误差 ``e``
一次消掉的两个轴步长。真实关系在大角度下会弯（转 30° 之后雅可比已经变了），所以
这是**一次**解算，不是"走到就一定对"：转完再看一帧，偏差大就再算一次——
和 :meth:`modules.grasp.runtime.Grasp.align_xy` 的闭环是同一个道理，只是这里把
"要转到哪个绝对位置"先算出来交给你。

## 单位必须与雅可比一致

``reading`` 与 ``jacobian`` 必须是**同一把尺**（物理毫米）。物料那条路线的读数按针孔
标定 ``z/f`` 换算；环心读数正常也走物理毫米（``depth/focal`` 或"已知外径 / 模型框"）。
一旦退回 ``station.mm_per_px``（给固定相机标的值），两者可以差好几倍——角度也会跟着
整体差好几倍，而且看起来完全正常。所以 :meth:`modules.grasp.runtime.Grasp.plan_ring`
会先核对比例尺来源，对不上就报错，而不是给一个像样的错数。
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
    extension_pulses: int = 0
    extension_motor_pulses: int = 0
    extension_motor_degrees: float = 0.0

    #: 两轴都走之后的预测残差（毫米）与是否进容差。
    residual_mm: float = 0.0
    tolerance_mm: float = 0.0
    aligned: bool = False

    #: 人话说明：这一步是什么、还剩多少、下一步该做什么。
    note: str = ""

    # ---------------------------------------------------- 各电机的绝对位置
    @property
    def base(self) -> float:
        """底座应当走到的**绝对角度**（电机轴口径，度）。

        直接可喂 :meth:`modules.arm.runtime.RobotArm.move_axis_to_motor_degrees`::

            robot.arm.move_axis_to_motor_degrees("base", plan.base)
        """
        return self.base_motor_degrees

    @property
    def extension(self) -> float:
        """伸缩应当走到的**绝对角度**（电机轴口径，度），喂 ``"extension"`` 轴。"""
        return self.extension_motor_degrees

    @property
    def degrees(self) -> "dict[str, float]":
        """各轴绝对角度，键就是轴名（``base`` / ``extension``），方便循环发命令::

            for axis, degrees in plan.degrees.items():
                robot.arm.move_axis_to_motor_degrees(axis, degrees)
        """
        return {"base": self.base, "extension": self.extension}

    @property
    def pulses(self) -> "dict[str, int]":
        """各轴应当走到的**绝对脉冲**（里程表口径），与 :attr:`degrees` 同一位置。"""
        return {"base": self.base_motor_pulses, "extension": self.extension_motor_pulses}

    def apply(
        self,
        arm,
        *,
        axes: Optional[Sequence[str]] = None,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
    ):
        """把机械臂走到这份计划的绝对位置；**不做安全高度检查**，由调用方负责。

        逐轴发绝对帧（``move_axis_to_motor_pulses``：增量按当时的里程表算，所以中间
        干了多少别的动作都对得上）。``axes`` 默认两根都走；只想转底座时传
        ``axes=("base",)``。

        计划记录的是**算的时候**那个姿态下的绝对角度，所以执行前不要回零
        （清零里程表之后，之前算出来的绝对值就不再指向同一个位置）。

        :returns: 每根轴一份 ``MotionReport``（按 ``axes`` 顺序）。
        """
        wanted = self.pulses
        names = tuple(wanted) if axes is None else tuple(axes)
        unknown = [name for name in names if name not in wanted]
        if unknown:
            raise ValueError(
                f"这份计划里没有 {unknown} 轴；可用的只有 {sorted(wanted)}"
            )
        return tuple(
            arm.move_axis_to_motor_pulses(
                name, wanted[name], rpm=rpm, acceleration=acceleration
            )
            for name in names
        )

    def summary(self) -> str:
        where = f"环 {self.digit}" if self.digit is not None else "环"
        if self.digit is not None and self.score is not None:
            where += f"（{self.score:.2f}）"
        return (
            f"{where} 读数 ({self.reading[0]:+.1f}, {self.reading[1]:+.1f}) mm，"
            f"误差 {math.hypot(*self.error_mm):.1f} mm → "
            f"底座绝对 {self.base:+.1f}°（电机轴，{self.base_pulses:+d} 脉冲；"
            f"输出轴 {self.base_output_degrees:+.1f}°），"
            f"伸缩绝对 {self.extension:+.1f}°（{self.extension_pulses:+d} 脉冲），"
            f"预测残差 {self.residual_mm:.1f} mm"
            f"{'（进容差）' if self.aligned else '（还要再迭代一次）'}。"
        )


def plan_ring_base(
    *,
    reading: Sequence[float],
    target: Sequence[float],
    jacobian: ServoJacobian,
    base_pulses: int,
    extension_pulses: int,
    pulses_per_revolution: int,
    reduction_ratio: float,
    tolerance_mm: float = 4.0,
) -> RingBasePlan:
    """解算"把环心读数推到 ``target`` 需要的底座绝对角度"。

    Parameters:
        reading: 环心在**相机系**的毫米坐标（与 ``jacobian`` 同一把尺）。
        target: 爪口在相机系里的目标读数（``eye_in_hand.json`` 的 ``target``）。
        jacobian: 实测的两轴灵敏度（毫米/脉冲）。
        base_pulses / extension_pulses: 两根轴**当前**的里程表读数（可以是负的）。
        pulses_per_revolution: 电机一圈的脉冲数（``3200``）。
        reduction_ratio: 底座减速比（电机转几圈输出轴转一圈）。
        tolerance_mm: 判"已经够准"的门槛；只影响 :attr:`RingBasePlan.aligned`。

    :raises ServoError: 雅可比奇异（两轴在画面里的位移方向重合），或底座那一列
        几乎为零（转底座画面不动，解不出角度）。
    """
    first, second = _pair(reading, "reading")
    target_x, target_y = _pair(target, "target")
    _require_int(base_pulses, "base_pulses")
    _require_int(extension_pulses, "extension_pulses")
    _require_positive_int(pulses_per_revolution, "pulses_per_revolution")
    if not reduction_ratio > 0:
        raise ValueError("reduction_ratio 必须是正数（电机转几圈输出轴转一圈）")
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm 不能是负数")

    error = (first - target_x, second - target_y)
    # 一次线性化：走 p 脉冲让读数变化 J·p，要 e + J·p = 0，即 p = −J⁻¹·e。
    base_step_float, extension_step_float = jacobian.pulses_for(*error)
    if jacobian.base_norm < MIN_BASE_SENSITIVITY_MM_PER_PULSE:  # pragma: no cover - 与奇异判据重叠
        raise ServoError(
            f"底座在画面里几乎推不动（{jacobian.base_norm:.2e} mm/脉冲），解不出角度；"
            "检查相机是不是离转轴太近、或雅可比标定把两个轴记反了"
        )
    base_step = _round(base_step_float)
    extension_step = _round(extension_step_float)

    residual = _residual(error, jacobian, base_step, extension_step)
    base_motor_pulses = base_pulses + base_step
    extension_motor_pulses = extension_pulses + extension_step
    motor_degrees_per_pulse = 360.0 / float(pulses_per_revolution)
    base_motor_degrees = base_motor_pulses * motor_degrees_per_pulse
    extension_motor_degrees = extension_motor_pulses * motor_degrees_per_pulse
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
        extension_pulses=extension_step,
        extension_motor_pulses=extension_motor_pulses,
        extension_motor_degrees=extension_motor_degrees,
        residual_mm=residual,
        tolerance_mm=float(tolerance_mm),
        aligned=aligned,
        note=note,
    )


# ---------------------------------------------------------------------- 闭环
@dataclass(frozen=True)
class RingAlignStep:
    """闭环里的一步：看到了什么、走了多少、走完实测挪了多少。"""

    attempt: int
    reading: Tuple[float, float]
    error_mm: float
    base_pulses: int
    extension_pulses: int
    #: 这一段走完之后**实测**挪了多少（下一步的读数减这一步的读数）。
    measured_mm: Optional[Tuple[float, float]] = None
    #: 按配置的雅可比**预测**该挪多少。
    predicted_mm: Optional[Tuple[float, float]] = None
    #: 实测 / 预测 的投影比：``1`` 说明尺是对的，``0.5`` 说明实际灵敏度只有一半。
    response: Optional[float] = None

    def line(self) -> str:
        moved = ""
        if self.measured_mm is not None and self.predicted_mm is not None:
            moved = (
                f"，实测 ({self.measured_mm[0]:+.2f},{self.measured_mm[1]:+.2f}) mm / "
                f"预测 ({self.predicted_mm[0]:+.2f},{self.predicted_mm[1]:+.2f}) mm"
            )
            if self.response is not None:
                moved += f"，响应比 {self.response:.2f}"
        return (
            f"第 {self.attempt} 步：读数 ({self.reading[0]:+.2f},{self.reading[1]:+.2f}) mm，"
            f"误差 {self.error_mm:.2f} mm → 底座 {self.base_pulses:+d}、"
            f"伸缩 {self.extension_pulses:+d} 脉冲{moved}"
        )


@dataclass(frozen=True)
class RingAlignResult:
    """一次环心闭环的结论：每一步步长与实测响应，**以及最后那一份 plan**。

    ``plan`` 是收敛时算出来的那份 :class:`RingBasePlan`——它和直接调
    ``plan_ring()`` 得到的**是同一个东西**（同一套字段、同一个绝对位置口径），
    所以"先闭环对准、再拿绝对位置去干别的"不需要再调一次 ``plan_ring``：
    直接存 ``result.plan`` 就行。
    """

    steps: Tuple[RingAlignStep, ...] = ()
    aligned: bool = False
    final_reading: Optional[Tuple[float, float]] = None
    final_error_mm: Optional[float] = None
    tolerance_mm: float = 0.0
    digit: Optional[int] = None
    #: 最后一次那一帧环心读数的比例尺来源（``depth/focal`` / ``ring box`` 等）。
    #: 之后要拿"这个读数 ± 偏移"去解算别的点时，比例尺得跟它同一把尺。
    scale_source: str = ""
    #: 最后一次估计出来的"实测/预测"比（``None`` = 一步都没走成）。
    response: Optional[float] = None
    #: 收敛时那一份计划：各轴绝对位置、残差、备注都在这上面。
    plan: Optional[RingBasePlan] = None
    note: str = ""

    # ------------------------------------------------ 与 plan_ring 同一套读法
    @property
    def base(self) -> Optional[float]:
        """底座应当停下的**绝对角度**（电机轴），与 ``plan_ring().base`` 同义。"""
        return None if self.plan is None else self.plan.base

    @property
    def extension(self) -> Optional[float]:
        """伸缩应当停下的**绝对角度**（电机轴）。"""
        return None if self.plan is None else self.plan.extension

    @property
    def degrees(self) -> "dict[str, float]":
        """``{"base": 角度, "extension": 角度}``，键就是轴名。"""
        return {} if self.plan is None else self.plan.degrees

    @property
    def pulses(self) -> "dict[str, int]":
        """``{"base": 脉冲, "extension": 脉冲}``（绝对，里程表口径）。"""
        return {} if self.plan is None else self.plan.pulses

    def apply(self, arm, **kwargs):
        """把机械臂走到**收敛时那个绝对位置**（已经在那儿了，一般不用调）。"""
        if self.plan is None:
            raise ValueError("这次闭环没有算出计划（一步都没走成），没有位置可应用")
        return self.plan.apply(arm, **kwargs)

    def summary(self) -> str:
        head = f"环 {self.digit}：" if self.digit is not None else "环："
        if self.final_reading is None:
            return head + "一步也没走成（" + self.note + "）"
        state = "已到位" if self.aligned else "**没到位**"
        text = (
            f"{head}{state}，最终读数 ({self.final_reading[0]:+.2f},"
            f"{self.final_reading[1]:+.2f}) mm，误差 "
            f"{self.final_error_mm:.2f} mm（容差 {self.tolerance_mm:.2f} mm），"
            f"用了 {len(self.steps)} 步"
        )
        if self.response is not None:
            text += f"，实测/预测响应比 {self.response:.2f}"
        if self.plan is not None:
            text += (
                f"；最终绝对位置：底座 {self.plan.base:+.1f}°"
                f"（{self.plan.base_motor_pulses:+d} 脉冲）、"
                f"伸缩 {self.plan.extension:+.1f}°"
                f"（{self.plan.extension_motor_pulses:+d} 脉冲）"
            )
        if self.note:
            text += "；" + self.note
        return text


# ---------------------------------------------------------------------- 内部
def _residual(
    error: Tuple[float, float],
    jacobian: ServoJacobian,
    base_step: int,
    extension_step: int,
) -> float:
    """按**整数**脉冲推算走完之后的残差长度（四舍五入的余量也要算进去）。"""
    moved_x, moved_y = jacobian.displacement_for(base_step, extension_step)
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
        base_pulses=1000, extension_pulses=-500,
        pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
    )
    if plan.base_pulses != 0 or plan.extension_pulses != 0:
        problems.append(f"已在目标上却给了步长：{plan.base_pulses}/{plan.extension_pulses}")
    if plan.base_motor_pulses != 1000 or not plan.aligned:
        problems.append(f"绝对值应当是当前脉冲，实际 {plan.base_motor_pulses}")

    # 2) 读数偏 +10 mm（沿底座那一列的反方向）：应当往正方向转
    #    Δ = −(e·j)/|j|² ；e=(10,0)、j=(−0.0817,−0.0076) ⇒ Δ ≈ +121.5 脉冲
    plan = plan_ring_base(
        reading=(10.0, 0.0), target=(0.0, 0.0), jacobian=jacobian,
        base_pulses=0, extension_pulses=0,
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
        base_pulses=0, extension_pulses=0,
        pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
        tolerance_mm=1.0,
    )
    if plan.residual_mm > 1.0:
        problems.append(f"两轴都走之后残差应当很小，实际 {plan.residual_mm:.2f} mm")

    # 5) 只差目标读数里的那一点：target 本身就是爪口位置
    plan = plan_ring_base(
        reading=(2.0, 0.0), target=(2.0, 0.0), jacobian=jacobian,
        base_pulses=0, extension_pulses=0,
        pulses_per_revolution=pulses_per_revolution, reduction_ratio=reduction_ratio,
    )
    if plan.base_pulses != 0:
        problems.append(f"读数等于目标时不该动，实际 {plan.base_pulses}")

    # 6) 奇异雅可比：直接拒绝，不能编一个角度出来
    singular = ServoJacobian(base_dx=0.01, base_dy=0.02, radial_dx=0.02, radial_dy=0.04)
    try:
        plan_ring_base(
            reading=(1.0, 1.0), target=(0.0, 0.0), jacobian=singular,
            base_pulses=0, extension_pulses=0,
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
                base_pulses=0, extension_pulses=0,
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

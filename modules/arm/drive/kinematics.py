"""机械臂圆柱运动学与单位换算（纯数学，不打开串口、不读配置文件）。

机械臂是**圆柱坐标**结构，不是笛卡尔龙门：

| 轴 | 电机地址 | 物理动作 | 圆柱量 |
|---|---|---|---|
| ``base`` | 5 | 底座旋转 | ``base_degrees`` |
| ``lift`` | 6 | 升降 | ``height_mm`` |
| ``extension`` | 7 | 前后伸缩 | ``radial_mm`` |

所以「末端到某个坐标」的逆解只有两行：

```
base_degrees = atan2(Y, X)        radial_mm = hypot(X, Y)
```

高度 ``height_mm`` 是**离台面**的高度，向上为正。它由相机深度换算而来
（``table_depth − point_depth``），不是机械臂自己的绝对坐标——本机械臂只按脉冲走。

## 单位与脉冲

只用两个机器常数，且都来自实测标定：

- 底座：``reduction_ratio`` = 电机转几圈输出轴转一圈，于是
  ``角度 = 脉冲 × 360 / (PPR × reduction_ratio)``；
- 升降 / 伸缩：``mm_per_motor_rev`` = **电机转一圈走多少毫米**，于是
  ``位移 = 脉冲 × mm_per_motor_rev / PPR``。

``mm_per_motor_rev`` 就是丝杠导程 ``L`` 与减速比 ``g`` 的商。文献里要求分别给出
``L`` 和 ``g``，但公式里永远只出现 ``L / g``，所以标定这个商即可，不必拆机量导程。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

#: 与 ``modules.arm.config.AXIS_NAMES`` 对应。
BASE_AXIS = "base"
LIFT_AXIS = "lift"
EXTENSION_AXIS = "extension"


@dataclass(frozen=True)
class CylindricalPose:
    """末端在基座圆柱坐标系里的位置。

    - ``base_degrees``：底座转角，绕基座旋转轴；符号与 ``RobotArm.rotate_cw`` 一致（CW 为正）；
    - ``radial_mm``：末端到基座旋转轴的水平距离，总是非负；
    - ``height_mm``：末端离台面的高度，向上为正。
    """

    base_degrees: float
    radial_mm: float
    height_mm: float

    def summary(self) -> str:
        return (
            f"θ={self.base_degrees:.1f}°，r={self.radial_mm:.1f} mm，"
            f"h={self.height_mm:.1f} mm"
        )


@dataclass(frozen=True)
class TravelLimit:
    """一个轴的软行程范围；``None`` 表示尚未标定。"""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not self.minimum <= self.maximum:
            raise ValueError("travel limit minimum must not exceed maximum")

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum

    def describe(self) -> str:
        return f"{self.minimum:.1f}..{self.maximum:.1f}"


# ---------------------------------------------------------------- 单位换算
def motor_pulses_for_degrees(
    degrees: float,
    *,
    pulses_per_revolution: int,
) -> int:
    """**电机轴**转角（度）换算成脉冲数（四舍五入，带符号）。

    注意和 :func:`base_pulses_for_degrees` 的区别：那个给的是**输出轴**角度、
    要乘减速比；这个是电机自己转过的角度，``3200 脉冲 = 360°``，与机构无关，
    三个轴用同一把尺。

    这台机器的三个轴行程都超过一个电机圈（底座 1800°、升降 1024°、伸缩 1057°），
    所以这里**不做取模、不做归一**——转了几圈就是几圈。累加与原点见
    :class:`modules.arm.arm_state.MotorTurnsState`。
    """
    _require_positive_integer("pulses_per_revolution", pulses_per_revolution)
    _require_finite("degrees", degrees)
    return _round_half_away(degrees * pulses_per_revolution / 360.0)


def motor_degrees_for_pulses(
    pulses: int,
    *,
    pulses_per_revolution: int,
) -> float:
    """**电机轴**脉冲数换算成转角（度）；与 :func:`motor_pulses_for_degrees` 互逆。"""
    _require_positive_integer("pulses_per_revolution", pulses_per_revolution)
    return pulses * 360.0 / pulses_per_revolution


def base_pulses_for_degrees(
    degrees: float,
    *,
    pulses_per_revolution: int,
    reduction_ratio: float,
) -> int:
    """底座转角（度）换算成脉冲数（四舍五入，带符号）。"""
    _require_positive_integer("pulses_per_revolution", pulses_per_revolution)
    _require_positive_number("reduction_ratio", reduction_ratio)
    _require_finite("degrees", degrees)
    return _round_half_away(degrees * pulses_per_revolution * reduction_ratio / 360.0)


def base_degrees_for_pulses(
    pulses: int,
    *,
    pulses_per_revolution: int,
    reduction_ratio: float,
) -> float:
    """底座脉冲数换算成转角（度）。"""
    _require_positive_integer("pulses_per_revolution", pulses_per_revolution)
    _require_positive_number("reduction_ratio", reduction_ratio)
    return pulses * 360.0 / (pulses_per_revolution * reduction_ratio)


def linear_pulses_for_mm(
    millimeters: float,
    *,
    pulses_per_revolution: int,
    mm_per_motor_rev: Optional[float],
) -> int:
    """直线轴位移（毫米）换算成脉冲数（四舍五入，带符号）。

    ``mm_per_motor_rev`` 为 ``None`` 时抛 ``ValueError``：没标定就不许下发毫米指令，
    这是 ``modules/arm/docs/机械臂Emm42三轴控制说明.md`` 第 7 节定下的规矩。

    返回值可能是 0（位移小于半个脉冲），调用方应把它当作"已经到位"而不是发一条空帧。
    """
    _require_positive_integer("pulses_per_revolution", pulses_per_revolution)
    _require_calibrated(mm_per_motor_rev)
    _require_finite("millimeters", millimeters)
    return _round_half_away(millimeters * pulses_per_revolution / mm_per_motor_rev)


def mm_for_linear_pulses(
    pulses: int,
    *,
    pulses_per_revolution: int,
    mm_per_motor_rev: Optional[float],
) -> float:
    """直线轴脉冲数换算成位移（毫米）。"""
    _require_positive_integer("pulses_per_revolution", pulses_per_revolution)
    _require_calibrated(mm_per_motor_rev)
    return pulses * mm_per_motor_rev / pulses_per_revolution


# ---------------------------------------------------------------- 圆柱运动学
def inverse_cylindrical(plane_x_mm: float, plane_y_mm: float, height_mm: float) -> CylindricalPose:
    """机械臂平面坐标 + 高度 → 圆柱关节量。"""
    _require_finite("plane_x_mm", plane_x_mm)
    _require_finite("plane_y_mm", plane_y_mm)
    _require_finite("height_mm", height_mm)
    return CylindricalPose(
        base_degrees=math.degrees(math.atan2(plane_y_mm, plane_x_mm)),
        radial_mm=math.hypot(plane_x_mm, plane_y_mm),
        height_mm=float(height_mm),
    )


def normalize_degrees(degrees: float) -> float:
    """把绝对角度归一到 ``(-180, 180]``，与 ``atan2`` 的输出同域。

    位姿预测必须做和测量同样的归一。否则「从 170° 再转 20°」会被记成 190°——物理上就是
    -170°，却通不过 ±180 的行程检查，抓取会在半路莫名其妙地失败。

    注意 ``-180`` 会归成 ``180``：两者指同一个方向，但统一成后者比较结果才唯一。
    """
    _require_finite("degrees", degrees)
    wrapped = float(degrees) % 360.0
    return wrapped - 360.0 if wrapped > 180.0 else wrapped


def _round_half_away(value: float) -> int:
    """四舍五入（.5 远离零），避免 Python 的银行家舍入让对称位移不对称。"""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _require_calibrated(mm_per_motor_rev: Optional[float]) -> None:
    if mm_per_motor_rev is None:
        raise ValueError(
            "该直线轴还没有 mm_per_motor_rev 标定值；先用实验 B/C 标定后再下发毫米指令"
        )
    _require_positive_number("mm_per_motor_rev", mm_per_motor_rev)


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive_number(name: str, value: Optional[float]) -> None:
    _require_finite(name, value)
    if not float(value) > 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要硬件：验证换算、逆解、最短转角和可达性。"""
    ppr, ratio = 3200, 5

    # 16 细分、g=5：输出轴一圈 16000 脉冲，1000 脉冲 = 22.5°。
    assert base_pulses_for_degrees(360.0, pulses_per_revolution=ppr, reduction_ratio=ratio) == 16000
    assert abs(base_degrees_for_pulses(1000, pulses_per_revolution=ppr, reduction_ratio=ratio) - 22.5) < 1e-9
    assert base_pulses_for_degrees(-22.5, pulses_per_revolution=ppr, reduction_ratio=ratio) == -1000

    # mm/电机圈 = 40 mm 时，3200 脉冲正好 40 mm。
    assert linear_pulses_for_mm(40.0, pulses_per_revolution=ppr, mm_per_motor_rev=40.0) == 3200
    assert abs(mm_for_linear_pulses(1600, pulses_per_revolution=ppr, mm_per_motor_rev=40.0) - 20.0) < 1e-9
    # 半个脉冲以下的位移取整成 0，调用方当作"已到位"。
    assert linear_pulses_for_mm(0.001, pulses_per_revolution=ppr, mm_per_motor_rev=40.0) == 0

    # 逆解（基座 XYZ -> 圆柱）落在文档写的位置上。
    pose = inverse_cylindrical(0.0, 100.0, 30.0)
    assert abs(pose.base_degrees - 90.0) < 1e-9 and abs(pose.radial_mm - 100.0) < 1e-9
    # 没标定的直线轴必须拒绝毫米指令。
    try:
        linear_pulses_for_mm(10.0, pulses_per_revolution=ppr, mm_per_motor_rev=None)
    except ValueError as error:
        assert "标定" in str(error), error
    else:
        raise AssertionError("未标定的直线轴必须报错")

    print("圆柱运动学自测通过：换算、逆解、未标定保护都正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

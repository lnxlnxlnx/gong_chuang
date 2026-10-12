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
（见 :func:`height_above_table`），不是机械臂自己的绝对坐标——本机械臂只有相对
位置模式，绝对高度一律以视觉测量为准。

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
    这是 ``docs/机械臂Emm42三轴控制说明.md`` 第 7 节定下的规矩。

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


def forward_cylindrical(pose: CylindricalPose) -> Tuple[float, float, float]:
    """圆柱关节量 → 机械臂平面坐标 ``(X, Y, Z)``；用来验证逆解。"""
    if pose.radial_mm < 0:
        raise ValueError("radial_mm must not be negative")
    radians = math.radians(pose.base_degrees)
    return (
        pose.radial_mm * math.cos(radians),
        pose.radial_mm * math.sin(radians),
        pose.height_mm,
    )


def shortest_angle_delta_degrees(start_degrees: float, target_degrees: float) -> float:
    """从 ``start`` 转到 ``target`` 的最短转角，结果落在 ``(-180, 180]``。

    底座能转几百度的臂上这一条很要紧：不做最短路径，θ 从 170° 到 -170° 会走 340° 的
    长路，把整条臂横扫过工作区。
    """
    _require_finite("start_degrees", start_degrees)
    _require_finite("target_degrees", target_degrees)
    delta = (target_degrees - start_degrees) % 360.0
    return delta - 360.0 if delta > 180.0 else delta


def normalize_degrees(degrees: float) -> float:
    """把绝对角度归一到 ``(-180, 180]``，与 ``atan2`` 的输出同域。

    位姿预测必须做和测量同样的归一。否则「从 170° 再转 20°」会被记成 190°——物理上就是
    -170°，却通不过 ±180 的行程检查，抓取会在半路莫名其妙地失败。

    注意 ``-180`` 会归成 ``180``：两者指同一个方向，但统一成后者比较结果才唯一。
    """
    _require_finite("degrees", degrees)
    wrapped = float(degrees) % 360.0
    return wrapped - 360.0 if wrapped > 180.0 else wrapped


def height_above_table(*, table_depth_mm: float, point_depth_mm: float) -> float:
    """相机深度换算成离台面高度。

    镜头俯视时深度沿光轴向下递增，所以台面深度减去物点深度就是高度。
    ``table_depth_mm`` 是一次性标定的常数（见 ``docs/抓取坐标链路.md``）。
    """
    _require_finite("table_depth_mm", table_depth_mm)
    _require_finite("point_depth_mm", point_depth_mm)
    return float(table_depth_mm) - float(point_depth_mm)


def marker_height_for_grasp(
    *,
    material_height_mm: float,
    grip_ratio: float,
    marker_to_grasp_mm: float,
) -> float:
    """侧向夹取时，爪上靶标需要到达的高度。

    - ``grip_ratio``：夹在物料的哪个高度，``0`` 贴台面、``0.5`` 半高、``1`` 顶面；
    - ``marker_to_grasp_mm``：靶标相对实际夹持点的竖直偏移，靶标在夹持点**上方**时取正值。

    侧向夹取的夹持点高度依赖物料高度，所以单目测出的物料高度在这里真正被用到；
    这也是 ``MaterialLocator`` 的深度不只为 XY 服务的地方。
    """
    _require_finite("material_height_mm", material_height_mm)
    _require_finite("marker_to_grasp_mm", marker_to_grasp_mm)
    if not 0.0 <= grip_ratio <= 1.0:
        raise ValueError("grip_ratio must be in 0..1")
    if material_height_mm < 0:
        raise ValueError("material_height_mm must not be negative")
    return material_height_mm * grip_ratio + marker_to_grasp_mm


# ---------------------------------------------------------------- 可达性
def check_reachable(
    pose: CylindricalPose,
    *,
    base_limit: Optional[TravelLimit] = None,
    radial_limit: Optional[TravelLimit] = None,
    height_limit: Optional[TravelLimit] = None,
) -> Tuple[str, ...]:
    """返回所有越界说明；全部在范围内时返回空元组。

    传 ``None`` 的轴**不检查**。没标定行程就等于没有软件保护，这一点调用方必须清楚，
    所以消息里会写明是哪个轴、越了多少。
    """
    problems = []
    for value, limit, name, unit in (
        (pose.base_degrees, base_limit, BASE_AXIS, "°"),
        (pose.radial_mm, radial_limit, EXTENSION_AXIS, " mm"),
        (pose.height_mm, height_limit, LIFT_AXIS, " mm"),
    ):
        if limit is not None and not limit.contains(value):
            problems.append(
                f"{name} 目标 {value:.1f}{unit} 超出行程 {limit.describe()}{unit}"
            )
    return tuple(problems)


def require_reachable(
    pose: CylindricalPose,
    *,
    base_limit: Optional[TravelLimit] = None,
    radial_limit: Optional[TravelLimit] = None,
    height_limit: Optional[TravelLimit] = None,
) -> None:
    """越界时抛 ``ValueError``；必须在发出任何运动帧之前调用。"""
    problems = check_reachable(
        pose,
        base_limit=base_limit,
        radial_limit=radial_limit,
        height_limit=height_limit,
    )
    if problems:
        raise ValueError("；".join(problems))


# ---------------------------------------------------------------- 内部校验
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

    # 逆解与正解互逆。
    pose = inverse_cylindrical(0.0, 100.0, 30.0)
    assert abs(pose.base_degrees - 90.0) < 1e-9 and abs(pose.radial_mm - 100.0) < 1e-9
    back = forward_cylindrical(pose)
    assert abs(back[0]) < 1e-9 and abs(back[1] - 100.0) < 1e-9 and abs(back[2] - 30.0) < 1e-9

    # 最短转角走短边。
    assert abs(shortest_angle_delta_degrees(170.0, -170.0) - 20.0) < 1e-9
    assert abs(shortest_angle_delta_degrees(-170.0, 170.0) + 20.0) < 1e-9
    assert abs(shortest_angle_delta_degrees(0.0, 180.0) - 180.0) < 1e-9

    # 深度换算成高度：台面 300 mm，物点 285 mm -> 高 15 mm。
    assert abs(height_above_table(table_depth_mm=300.0, point_depth_mm=285.0) - 15.0) < 1e-9

    # 侧向夹取：物料高 20 mm，夹半高，靶标在夹持点上方 12 mm -> 需要到 22 mm。
    assert abs(
        marker_height_for_grasp(material_height_mm=20.0, grip_ratio=0.5, marker_to_grasp_mm=12.0) - 22.0
    ) < 1e-9

    # 可达性。
    limit = TravelLimit(-110.0, 110.0)
    assert check_reachable(CylindricalPose(90.0, 80.0, 10.0), base_limit=limit) == ()
    problems = check_reachable(
        CylindricalPose(150.0, 999.0, 10.0),
        base_limit=limit,
        radial_limit=TravelLimit(40.0, 160.0),
    )
    assert len(problems) == 2, problems

    # 没标定的直线轴必须拒绝毫米指令。
    try:
        linear_pulses_for_mm(10.0, pulses_per_revolution=ppr, mm_per_motor_rev=None)
    except ValueError as error:
        assert "标定" in str(error), error
    else:
        raise AssertionError("未标定的直线轴必须报错")

    print("圆柱运动学自测通过：换算、逆解、最短转角、可达性、未标定保护都正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

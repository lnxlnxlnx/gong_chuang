"""让爪子正下方对准物块：平面（θ, r）的精确解。

用户要的第一步就是这件事——**只动平面，不动高度**：

```
当前:  爪子 在 (θ, r)，镜头测出物料在同一平面里的 (θ_t, r_t)
目标:  转 Δθ、伸 Δr，使爪子的平面位置正好落在物料正上方
```

## 解是什么

就两行：

```
Δθ = θ_t − θ        （取最短转角，见 kinematics.shortest_angle_delta_degrees）
Δr = r_t − r
```

## 为什么是这个，而不是别的

底座在圆柱坐标里是**纯旋转**：它把爪子沿半径 ``r`` 的圆弧搬运，不改变 ``r``。
伸缩轴在旋转之后沿**当前朝向**推 ``Δr``。所以一步"转 + 伸"之后爪子的位置是：

```
(r + Δr) · û(θ + Δθ)          其中 û(α) = (cos α, sin α)
```

要它等于物料的 ``T = r_t · û(θ_t)``，就要求方向对齐（``θ + Δθ = θ_t``）且半径对齐
（``r + Δr = r_t``）。两个条件互相独立，各给一个未知数，所以**平面内永远有精确解**，
不需要迭代，也不会有第二个解。

> 这里最容易写错的一步：把 ``Δr`` 也乘一次 ``cos Δθ``。``Δr`` 本身就是沿
> ``û(θ + Δθ)`` 的位移，再旋转一次就变成另一个量，径向会解错几十毫米——
> 那正是本模块第一版犯的错，现场表现为"转了也对、伸了也对，就是没对上"。

## 失败只有两种

1. **径向解成负数**——物料在旋转轴的另一侧。物理上那是把伸缩轴顶死，直接拒绝；
2. **行程不够**——解出来的 ``(θ_t, r_t)`` 落在 ``limits`` 之外。这一条**不在这里判**：
   行程是绝对位姿的限制，由调用方用 ``kinematics.require_reachable`` 在发帧之前检查
   （和 ``pipeline`` 的约定一致）。

## 求出来之后谁来发指令

这一层只算，不发帧。原因和二里那条一样：**"伸出状态下不许大角度旋转"属于动作编排**，
不属于运动学。真正发指令的是 :class:`~modules.grasp.drive.pipeline.GraspPipeline`
或调用方自己——``RobotArm`` 只有相对位置模式，也没有任何绝对行程保护。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from modules.arm.drive.kinematics import (
    CylindricalPose,
    forward_cylindrical,
    normalize_degrees,
    shortest_angle_delta_degrees,
)

#: 小于这个角度就当作"角度不用动"。
ROTATION_EPSILON_DEGREES = 1e-6

#: 小于这个长度就当作"这一维不用动"。
LENGTH_EPSILON_MM = 1e-9


class AlignmentError(ValueError):
    """这个目标在平面上解不出来（爪子和物料都在旋转轴上，或径向会变成负的）。"""


@dataclass(frozen=True)
class PlanarAlignment:
    """从当前位姿到"爪子正对物料上方"需要走的两个关节量。"""

    #: 底座要转的角度（带符号，CW 为正）。
    delta_base_degrees: float
    #: 伸缩轴要走的距离（带符号，伸出为正）。
    delta_radial_mm: float
    #: 走完之后的平面径向距离 ``r + Δr``，总是非负。
    radial_after_mm: float
    #: 走完之后爪子中心到物料的平面距离（毫米）。闭式解下这是舍入误差级别。
    residual_mm: float
    #: 当前平面距离；用它和 ``residual_mm`` 对比可以看出这一步省了多少。
    error_before_mm: float

    @property
    def rotation_only(self) -> bool:
        """只需要转、不需要伸出缩回。"""
        return abs(self.delta_radial_mm) <= LENGTH_EPSILON_MM

    @property
    def rotation_free(self) -> bool:
        """只需要伸出缩回、不需要转。"""
        return abs(self.delta_base_degrees) <= ROTATION_EPSILON_DEGREES

    def summary(self) -> str:
        return (
            f"转 {self.delta_base_degrees:+.3f}°，伸缩 {self.delta_radial_mm:+.2f} mm"
            f"（径向 {self.radial_after_mm:.1f} mm）；"
            f"平面误差 {self.error_before_mm:.1f} → {self.residual_mm:.3f} mm"
        )


def planar_error_mm(first: CylindricalPose, second: CylindricalPose) -> float:
    """两个位姿在水平面上的距离（毫米），**忽略高度差**。"""
    first_x, first_y, _ = forward_cylindrical(first)
    second_x, second_y, _ = forward_cylindrical(second)
    return math.hypot(second_x - first_x, second_y - first_y)


def solve_planar_alignment(
    current: CylindricalPose,
    target: CylindricalPose,
) -> PlanarAlignment:
    """解出"爪子平面位置落到物料正上方"所需的 ``(Δθ, Δr)``。

    高度不参与：调用方自己决定要不要同步动升降。``target.height_mm`` 会被忽略。

    :raises AlignmentError: 目标径向为负，或者解出来的两个量没有把爪子送到物料正上方
        （后者只可能是解算器自身的 bug，会带着数字报出来）。
    """
    r = float(current.radial_mm)
    r_target = float(target.radial_mm)
    if r < 0 or r_target < 0:
        raise AlignmentError(
            f"径向距离必须非负，收到 current={r}、target={r_target}"
        )

    error_before = planar_error_mm(current, target)

    # 约束：转完再伸出之后，爪子中心必须正好落在物料上。
    #
    #   r·û(θ)  --转 Δθ-->  r·û(θ+Δθ)  --伸 Δr-->  (r + Δr)·û(θ+Δθ)  =  T
    #
    # 关键一步：Δr 本身就是**沿 û(θ+Δθ) 方向**的位移，所以它不该再被旋转一次
    # （写成 (r + Δr·cos Δθ)·û 是最容易犯的错，那会让径向解出错误的量）。
    #
    # 于是这一个向量方程直接给出两个独立的条件：
    #
    #   θ + Δθ = θ_t          （方向对齐）
    #   r + Δr = r_t          （半径对齐）
    #
    # 也就是"底座转到物料的角度、伸缩伸到物料的半径"。平面内这是个精确解，
    # 不是近似，也不需要迭代——只有在行程或可达性上才可能失败。
    delta_theta = math.radians(
        shortest_angle_delta_degrees(current.base_degrees, target.base_degrees)
    )
    radial_after = r_target
    delta_radial = radial_after - r
    delta_degrees = math.degrees(delta_theta)

    if radial_after < -LENGTH_EPSILON_MM:
        raise AlignmentError(
            f"解出来的径向距离是负的（{radial_after:.2f} mm）："
            f"物料在径向 {r_target:.1f} mm、角度 {target.base_degrees:.1f}°。"
            "请确认基座中心标定是否正确"
        )

    # 自己验一遍：按这两个量走完，爪子中心必须落在物料上。
    residual = planar_error_mm(pose_after(current, delta_degrees, delta_radial), target)
    if residual > 1e-6:
        raise AlignmentError(
            f"解出来的两个量没有把爪子送到物料正上方（残差 {residual:.6f} mm）："
            f"转 {delta_degrees:+.3f}°、伸缩 {delta_radial:+.3f} mm。"
            "这是解算器自身的错误，不是现场问题，请连同这条消息一起反馈"
        )
    return PlanarAlignment(
        delta_base_degrees=delta_degrees,
        delta_radial_mm=delta_radial,
        radial_after_mm=max(0.0, radial_after),
        residual_mm=residual,
        error_before_mm=error_before,
    )


def pose_after(
    current: CylindricalPose,
    delta_base_degrees: float,
    delta_radial_mm: float,
) -> CylindricalPose:
    """按给定的两个关节量推算走完之后的位姿（高度不变）。

    调用方发完指令要用它推进 :class:`~modules.grasp.drive.arm_state.ArmState`；
    解算器用它验证闭式解自己。
    """
    return CylindricalPose(
        base_degrees=normalize_degrees(current.base_degrees + delta_base_degrees),
        radial_mm=current.radial_mm + delta_radial_mm,
        height_mm=current.height_mm,
    )


def alignment_to_pose(
    current: CylindricalPose,
    target: CylindricalPose,
) -> CylindricalPose:
    """解出"对准之后爪子所在的位姿"（沿用 ``current`` 的高度）。"""
    alignment = solve_planar_alignment(current, target)
    return pose_after(current, alignment.delta_base_degrees, alignment.delta_radial_mm)


def is_aligned(
    current: CylindricalPose,
    target: CylindricalPose,
    *,
    tolerance_mm: float,
) -> bool:
    """爪子是否已经落在物料正上方（平面距离在容差内）。"""
    if not tolerance_mm > 0:
        raise ValueError("tolerance_mm 必须是正数")
    return planar_error_mm(current, target) <= tolerance_mm


def describe_planar_step(
    current: CylindricalPose,
    target: CylindricalPose,
    *,
    tolerance_mm: float = 1.0,
) -> str:
    """一句话说明"现在差多少、要动什么"，给真机调试打印用。"""
    error = planar_error_mm(current, target)
    if error <= tolerance_mm:
        return f"已经对准：平面误差 {error:.2f} mm（容差 {tolerance_mm:.1f} mm）"
    try:
        alignment = solve_planar_alignment(current, target)
    except AlignmentError as error_message:
        return f"平面误差 {error:.2f} mm，但解不出来：{error_message}"
    return alignment.summary()


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要硬件：验证闭式解真的把爪子送到物料正上方。"""
    # 解出来的位姿，平面距离必须落到舍入误差量级——这是这一层唯一的正确性判据。
    cases = [
        # (当前 θ, 当前 r, 物料 θ, 物料 r) —— 覆盖各种相对位置
        (0.0, 80.0, 30.0, 90.0),      # 常规：要转也要伸
        (0.0, 80.0, 0.0, 90.0),       # 同角度，只需伸出
        (30.0, 90.0, 30.0, 90.0),     # 已经对准
        (30.0, 90.0, 40.0, 60.0),     # 要转要缩
        (-170.0, 100.0, 170.0, 100.0),  # 跨 ±180：必须走 20° 短边，不是 340°
        (0.0, 100.0, 180.0, 100.0),   # 正对面 180°
        (10.0, 100.0, -170.0, 40.0),  # 大角度 + 大幅缩回
        (90.0, 5.0, -90.0, 120.0),    # 爪子几乎贴在轴上（行列式小但非零）
    ]
    for theta, radius, theta_t, radius_t in cases:
        current = CylindricalPose(theta, radius, 50.0)
        target = CylindricalPose(theta_t, radius_t, 20.0)
        alignment = solve_planar_alignment(current, target)
        after = CylindricalPose(
            normalize_degrees(theta + alignment.delta_base_degrees),
            alignment.radial_after_mm,
            50.0,
        )
        remaining = planar_error_mm(after, target)
        assert remaining < 1e-6, (current, target, alignment, remaining)
        assert alignment.radial_after_mm >= 0.0, alignment
        # 解出来的累计角必须落在 (-180, 180]，否则行程检查和最短转角都会算错。
        assert -180.0 < after.base_degrees <= 180.0, after

    # 跨 ±180 必须走短边：170° → -170° 是 20°，不是 340°。
    crossing = solve_planar_alignment(
        CylindricalPose(-170.0, 100.0, 50.0), CylindricalPose(170.0, 100.0, 20.0)
    )
    assert abs(crossing.delta_base_degrees) < 90.0, crossing

    # 已经对准时两个量都应该是 0，不能"抖"出一个来回。
    settled = solve_planar_alignment(
        CylindricalPose(30.0, 90.0, 50.0), CylindricalPose(30.0, 90.0, 20.0)
    )
    assert abs(settled.delta_base_degrees) < 1e-9, settled
    assert abs(settled.delta_radial_mm) < 1e-9, settled
    assert settled.rotation_only and settled.rotation_free, settled
    assert settled.error_before_mm < 1e-9, settled

    # 只伸出（角度相同）时不能顺手转一下。
    extend_only = solve_planar_alignment(
        CylindricalPose(0.0, 80.0, 50.0), CylindricalPose(0.0, 90.0, 20.0)
    )
    assert abs(extend_only.delta_base_degrees) < 1e-9, extend_only
    assert abs(extend_only.delta_radial_mm - 10.0) < 1e-9, extend_only
    assert extend_only.rotation_free, extend_only

    # 径向为负的目标是物理上不存在的（伸缩轴不可能伸到轴的另一侧）：拒绝。
    try:
        solve_planar_alignment(
            CylindricalPose(0.0, 100.0, 50.0), CylindricalPose(180.0, -5.0, 20.0)
        )
    except AlignmentError as error:
        assert "非负" in str(error), error
    else:
        raise AssertionError("径向为负的目标必须报 AlignmentError")

    # 目标正好落在旋转轴上（径向 0）是合法的：那不是错误，只是机械上要够得着。
    on_axis = solve_planar_alignment(
        CylindricalPose(0.0, 100.0, 50.0), CylindricalPose(180.0, 0.0, 20.0)
    )
    assert abs(on_axis.radial_after_mm) < 1e-9, on_axis
    assert abs(on_axis.delta_radial_mm + 100.0) < 1e-9, on_axis

    # 行程检查**不在这一层**：解算器只管平面几何，越界由调用方用 require_reachable 判。
    # 这里明确把它记下来，免得以后有人以为解算器会拦住越界目标。
    from modules.arm.drive.kinematics import TravelLimit, require_reachable

    far = CylindricalPose(0.0, 200.0, 20.0)
    solved = alignment_to_pose(CylindricalPose(0.0, 80.0, 50.0), far)
    try:
        require_reachable(solved, radial_limit=TravelLimit(40.0, 160.0))
    except ValueError as error:
        assert "行程" in str(error), error
    else:
        raise AssertionError("越界目标必须被 require_reachable 拦住")

    # alignment_to_pose 与 solve 必须自洽，并且高度沿用当前值。
    current = CylindricalPose(12.0, 70.0, 45.0)
    target = CylindricalPose(-140.0, 130.0, 20.0)
    pose = alignment_to_pose(current, target)
    assert abs(pose.height_mm - 45.0) < 1e-9, pose
    assert planar_error_mm(pose, target) < 1e-6, (pose, target)

    # 解出来的就是"转到物料的角度、伸到物料的半径"——两条都要单独钉住，
    # 否则以后有人改了其中一条，另一条可能悄悄跟着变。
    assert abs(solved.base_degrees - far.base_degrees) < 1e-9, solved
    assert abs(solved.radial_mm - far.radial_mm) < 1e-9, solved

    # is_aligned 的容差语义。
    assert is_aligned(CylindricalPose(30.0, 90.0, 50.0), CylindricalPose(30.0, 90.5, 20.0), tolerance_mm=1.0)
    assert not is_aligned(CylindricalPose(30.0, 90.0, 50.0), CylindricalPose(30.0, 95.0, 20.0), tolerance_mm=1.0)

    print(f"平面精确对准自测通过：{len(cases)} 组位姿残差 < 1e-6 mm；"
          f"跨 ±180 走短边（{crossing.delta_base_degrees:+.2f}°）；"
          f"越界目标由 require_reachable 拦住。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

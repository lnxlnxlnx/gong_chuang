"""绝对运动的纯计算规划：把"目标位姿"变成一串**已量化、已预检**的单轴动作帧。

## 这一层不碰设备

纯数学：输入当前夹持点位姿、目标位姿、行程、安全位与传动常数，输出一串
:class:`PlannedSegment`。:class:`modules.arm.runtime.RobotArm` 负责执行。
于是"目标够不够得着""中间会不会撞到""哪一段会超单帧脉冲上限"可以在**发第一条运动帧
之前**全部算完——后段非法时前段也不会先动。

## 安全廊道（沿用既有约定）

要转角度时，顺序固定为：

```
先到安全半径 → 再到安全高度 → 再旋转 → 再到目标半径 → 最后到目标高度
```

不转角度时只走"先改半径、再改高度"。只用现有相对脉冲串行执行，**不增加多轴插补**，
也不宣称末端沿 XYZ 直线运动。软件行程检查不等于环境避障。

## 量化与实际执行用同一份结果

每个轴只能走**整数脉冲**，所以"目标 mm"与"实际走到的 mm"通常差不到半个脉冲。
规划时逐段按量化脉冲推算下一段的起点，**与实际执行使用同一结果**。

超过单帧上限（``ArmConfig.max_pulses``）时，**在规划阶段拆成多帧**——这与"预测完整
路径"并不矛盾：拆段是规划的一部分，拆完之后每一帧的起点、落点都已经被检查过，
而不是边走边算。规划成功后执行阶段不再做任何决策。

## 角度：不做无条件归一化

底座角的合法性是"**实际受限角位置**"，所以要检查的是**整段连线**，不是只有两个端点：
行程 ``[-170°, 170°]`` 上的 170° 与 -170° 两个端点都合法，但两条连线（±180° 与 340°）
都必须经过区间之外，因此**拒绝**，而不是悄悄走穿过限位的那一条。

可用路线有两条：实际上最短的那条，以及补角那条（|Δ|=360-|Δ短|）。两条分别按"整段
连续角位置都在行程内、且每一帧脉冲都不超上限"检查，都合法时取转角小的那条。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from modules.arm.drive.kinematics import (
    BASE_AXIS,
    EXTENSION_AXIS,
    LIFT_AXIS,
    CylindricalPose,
    TravelLimit,
    base_degrees_for_pulses,
    base_pulses_for_degrees,
    linear_pulses_for_mm,
    mm_for_linear_pulses,
    normalize_degrees,
    shortest_angle_delta_degrees,
)

#: 小于这个量就当作"没动"，不发空帧。
ANGLE_EPSILON_DEGREES = 1e-9
LENGTH_EPSILON_MM = 1e-9

#: 浮点比较用的松弛量：行程检查不该因为 1e-13 的舍入而失败。
_EPSILON = 1e-9


class PlanError(ValueError):
    """目标无法安全到达（缺标定、越界、角度无合法路线、脉冲不可分解）。

    统一在**第一条运动帧之前**抛出，所以看到它就说明一个动作帧都没发。
    """


@dataclass(frozen=True)
class PlannedSegment:
    """一条**单轴单帧**动作：走多少整数脉冲，以及按量化脉冲算出的落点。"""

    #: ``"base"`` / ``"lift"`` / ``"extension"``。
    axis: str
    #: 本帧要走的带符号整数脉冲。
    pulses: int
    #: 走完本帧之后的夹持点位姿（按 ``pulses`` 反算，不是按请求值）。
    resulting_pose: CylindricalPose

    @property
    def is_empty(self) -> bool:
        """量化为 0 脉冲：调用方不应为它发帧，也不应改变位置。"""
        return self.pulses == 0


@dataclass(frozen=True)
class MotionPlan:
    """一次绝对定位的完整计划。"""

    segments: Tuple[PlannedSegment, ...]
    #: 调用方请求的目标位姿（可能不是整数脉冲能精确到达的那个）。
    requested: CylindricalPose
    #: 按量化脉冲推算的**实际**落点位姿。执行完就是它。
    final_pose: CylindricalPose

    @property
    def moving_segments(self) -> Tuple[PlannedSegment, ...]:
        """真正会发帧的那些帧（跳过量化为 0 的）。"""
        return tuple(segment for segment in self.segments if not segment.is_empty)

    def summary(self) -> str:
        moving = self.moving_segments
        if not moving:
            return "无需动作（量化后没有要走的脉冲）"
        parts = [f"{segment.axis} {segment.pulses:+d}" for segment in moving]
        return f"{len(moving)} 帧：" + " → ".join(parts)


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(f"{name} 必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise PlanError(f"{name} 必须是有限值（不能是 NaN 或无穷）")
    return number


def _require_inside(name: str, value: float, limit: Optional[TravelLimit], unit: str) -> None:
    if limit is None:
        raise PlanError(f"{name} 无法检查：对应的 limits 还没标定")
    if not limit.contains(value):
        raise PlanError(f"{name} {value:.3f}{unit} 超出行程 {limit.describe()}{unit}")


def _require_calibrated_axes(config) -> None:
    """两个绝对入口的前置条件：毫米标定、底座减速比、三轴行程、安全位都要齐。"""
    missing = []
    if config.radial_limit is None:
        missing.append("limits.radial_mm")
    if config.height_limit is None:
        missing.append("limits.height_mm")
    if config.base_limit is None:
        missing.append("limits.base_degrees")
    if config.safe_pose is None:
        missing.append("safe_pose")
    if config.mm_per_motor_rev("lift") is None:
        missing.append("axes.lift.mm_per_motor_rev")
    if config.mm_per_motor_rev("extension") is None:
        missing.append("axes.extension.mm_per_motor_rev")
    try:
        config.base_reduction
    except ValueError:
        missing.append("axes.base.reduction_ratio")
    if missing:
        raise PlanError(
            "绝对定位还缺这些标定/行程，无法在发帧前完成预检："
            + "、".join(missing)
            + "（现场标定见 modules/grasp/docs/抓取坐标链路.md）"
        )


# ---------------------------------------------------------------- 角度路线
@dataclass(frozen=True)
class AngleRoute:
    """一条可走的角度路线。"""

    #: 终点连续角位置（可能不在 ``(-180, 180]`` 内，因为行程可能更宽）。
    endpoint_degrees: float
    #: 相对于起点要转的角度（带符号，CW 为正）。
    delta_degrees: float
    #: 本路线需要的（未拆帧的）总脉冲。
    pulses: int

    @property
    def travel(self) -> float:
        return abs(self.delta_degrees)


def _within_limit(angle: float, limit: TravelLimit) -> bool:
    """连续角位置（可能超出 ``(-180, 180]``）是否落在行程区间内。"""
    return limit.minimum - _EPSILON <= angle <= limit.maximum + _EPSILON


def angle_routes(
    *,
    config,
    start_degrees: float,
    target_degrees: float,
) -> Tuple[AngleRoute, ...]:
    """列出所有**整段合法**的角度路线，按转角从小到大排序。

    对目标方向角的每一个等价角 ``θ+360k``，只考虑两条连线：实际最短的那条，以及
    补角那条。整段连续角位置都必须落在行程内——只验端点是不够的。
    """
    limit = config.base_limit
    start = _require_finite("start_degrees", start_degrees)
    target = _require_finite("target_degrees", target_degrees)
    if not limit.contains(start):
        raise PlanError(
            f"当前底座角 {start:.3f}° 已经在行程 {limit.describe()}° 之外，无法规划"
        )
    ppr = config.pulses_per_revolution
    reduction = config.base_reduction

    routes = []
    seen = set()
    base = normalize_degrees(target)
    lowest = math.floor((limit.minimum - start) / 360.0) - 1
    highest = math.ceil((limit.maximum - start) / 360.0) + 1
    for k in range(lowest, highest + 1):
        equivalence = base + 360.0 * k
        short = shortest_angle_delta_degrees(start, equivalence)
        # 两条候选连线：实际最短的那条，以及补角那条（另一个转向）。
        candidates = [short]
        complement = math.copysign(360.0, short) - short if short else 0.0
        if abs(complement) > ANGLE_EPSILON_DEGREES:
            candidates.append(complement)
        for delta in candidates:
            if not _path_is_inside(start, delta, limit):
                continue
            endpoint = start + delta
            key = round(endpoint, 9)
            if key in seen:
                continue
            seen.add(key)
            pulses = base_pulses_for_degrees(
                delta, pulses_per_revolution=ppr, reduction_ratio=reduction
            )
            routes.append(
                AngleRoute(endpoint_degrees=endpoint, delta_degrees=delta, pulses=pulses)
            )
    routes.sort(key=lambda route: (route.travel, -route.delta_degrees))
    return tuple(routes)


def _path_is_inside(start: float, delta: float, limit: TravelLimit) -> bool:
    """从 ``start`` 转 ``delta`` 度的整条连线是否都在行程内。

    只验两个端点是不够的：行程 ``[-170, 170]`` 上的 170 与 -170 都合法，
    但两条连线都会经过区间之外。
    """
    endpoint = start + delta
    low, high = (start, endpoint) if delta >= 0 else (endpoint, start)
    return _within_limit(low, limit) and _within_limit(high, limit)


# ---------------------------------------------------------------- 执行器（唯一换算处）
class _Translation:
    """按整数脉冲逐帧推进的"当前位姿"，规划与实际执行共用同一套换算。"""

    def __init__(self, config, pose: CylindricalPose) -> None:
        self._config = config
        self.pose = pose

    # -- 单帧构造 --------------------------------------------------------
    def frame_for(self, axis: str, target: float) -> PlannedSegment:
        """构造**一帧**：朝 ``target`` 走，但不超过单帧脉冲上限。"""
        limit = self._config.max_pulses
        if axis == BASE_AXIS:
            delta = target - self.pose.base_degrees
            pulses = base_pulses_for_degrees(
                delta,
                pulses_per_revolution=self._config.pulses_per_revolution,
                reduction_ratio=self._config.base_reduction,
            )
            pulses = _clamp(pulses, limit)
            landed = self.pose.base_degrees + base_degrees_for_pulses(
                pulses,
                pulses_per_revolution=self._config.pulses_per_revolution,
                reduction_ratio=self._config.base_reduction,
            )
            _require_inside("旋转帧的底座角", landed, self._config.base_limit, "°")
            self.pose = CylindricalPose(
                normalize_degrees(landed), self.pose.radial_mm, self.pose.height_mm
            )
        else:
            current = (
                self.pose.radial_mm if axis == EXTENSION_AXIS else self.pose.height_mm
            )
            mm_per_rev = self._config.mm_per_motor_rev(axis)
            if mm_per_rev is None:
                raise PlanError(f"{axis} 轴缺少 mm_per_motor_rev，无法下发毫米动作")
            pulses = linear_pulses_for_mm(
                target - current,
                pulses_per_revolution=self._config.pulses_per_revolution,
                mm_per_motor_rev=mm_per_rev,
            )
            pulses = _clamp(pulses, limit)
            landed = current + mm_for_linear_pulses(
                pulses,
                pulses_per_revolution=self._config.pulses_per_revolution,
                mm_per_motor_rev=mm_per_rev,
            )
            if axis == EXTENSION_AXIS:
                _require_inside("伸缩帧的半径", landed, self._config.radial_limit, " mm")
                self.pose = CylindricalPose(self.pose.base_degrees, landed, self.pose.height_mm)
            else:
                _require_inside("升降帧的高度", landed, self._config.height_limit, " mm")
                self.pose = CylindricalPose(self.pose.base_degrees, self.pose.radial_mm, landed)
        return PlannedSegment(axis=axis, pulses=pulses, resulting_pose=self.pose)

    # -- 走到目标（自动拆帧） --------------------------------------------
    def move_to(self, axis: str, target: float, label: str) -> list:
        frames = []
        for _ in range(_MAX_FRAMES_PER_MOVE):
            current = self._axis_value(axis)
            if abs(target - current) <= self._epsilon(axis):
                break
            frame = self.frame_for(axis, target)
            if frame.pulses == 0:
                # 误差已经小于半个脉冲：物理上到不了更近了，接受这个落点。
                break
            frames.append(frame)
        else:
            raise PlanError(
                f"{label} 需要超过 {_MAX_FRAMES_PER_MOVE} 帧才能走完，"
                "请检查行程与 mm_per_motor_rev 是否合理"
            )
        return frames

    def _axis_value(self, axis: str) -> float:
        if axis == BASE_AXIS:
            return self.pose.base_degrees
        if axis == EXTENSION_AXIS:
            return self.pose.radial_mm
        return self.pose.height_mm

    def _epsilon(self, axis: str) -> float:
        return ANGLE_EPSILON_DEGREES if axis == BASE_AXIS else LENGTH_EPSILON_MM

    # -- 角度路线（整条路线一次定好，不按最短路径逐帧逼近） --------------
    def turn_along(self, route: AngleRoute) -> list:
        limit = self._config.max_pulses
        frames = []
        total = route.pulses
        if total == 0:
            return frames
        sign = 1 if total > 0 else -1
        remaining = abs(total)
        while remaining > 0:
            step = min(remaining, limit)
            remaining -= step
            pulses = sign * step
            landed = self.pose.base_degrees + base_degrees_for_pulses(
                pulses,
                pulses_per_revolution=self._config.pulses_per_revolution,
                reduction_ratio=self._config.base_reduction,
            )
            _require_inside("旋转帧的底座角", landed, self._config.base_limit, "°")
            self.pose = CylindricalPose(
                normalize_degrees(landed), self.pose.radial_mm, self.pose.height_mm
            )
            frames.append(
                PlannedSegment(axis=BASE_AXIS, pulses=pulses, resulting_pose=self.pose)
            )
        return frames


#: 一段动作最多拆多少帧。超了说明标定或行程有问题，宁可报错也不无限循环。
_MAX_FRAMES_PER_MOVE = 64


def _clamp(pulses: int, limit: int) -> int:
    if limit <= 0:
        raise PlanError(f"max_pulses 必须是正数，实际 {limit}")
    return max(-limit, min(limit, pulses))


# ---------------------------------------------------------------- 入口
def plan_absolute(
    *,
    config,
    current: CylindricalPose,
    target: CylindricalPose,
) -> MotionPlan:
    """把一次绝对定位规划成已量化、已预检的单轴动作帧序列。

    :param config: :class:`modules.arm.config.ArmConfig`（行程、安全位与传动常数）。
    :param current: 当前**夹持点**位姿；调用方负责确定位姿已知。
    :param target: 目标夹持点位姿；``base_degrees`` 是实际受限角位置。
    :raises PlanError: 缺标定、越界、角度没有合法路线。
    """
    _require_calibrated_axes(config)
    current_radial = _require_finite("current.radial_mm", current.radial_mm)
    current_height = _require_finite("current.height_mm", current.height_mm)
    if current_radial < 0:
        raise PlanError("当前半径不能是负数")
    _require_inside("当前位姿的底座角", current.base_degrees, config.base_limit, "°")
    _require_inside("当前位姿的半径", current_radial, config.radial_limit, " mm")
    _require_inside("当前位姿的高度", current_height, config.height_limit, " mm")

    target_radial = _require_finite("target.radial_mm", target.radial_mm)
    target_height = _require_finite("target.height_mm", target.height_mm)
    if target_radial < 0:
        raise PlanError("目标半径不能是负数（r 是水平距离，总是非负）")
    _require_inside("目标半径", target_radial, config.radial_limit, " mm")
    _require_inside("目标高度", target_height, config.height_limit, " mm")

    safe = config.safe_pose
    _require_inside("安全半径", safe.radial_mm, config.radial_limit, " mm")
    _require_inside("安全高度", safe.height_mm, config.height_limit, " mm")

    routes = angle_routes(
        config=config,
        start_degrees=current.base_degrees,
        target_degrees=target.base_degrees,
    )
    route = routes[0] if routes else None
    if route is None:
        if abs(shortest_angle_delta_degrees(current.base_degrees, target.base_degrees)) > ANGLE_EPSILON_DEGREES:
            raise PlanError(
                f"在行程 {config.base_limit.describe()}° 内找不到从 "
                f"{current.base_degrees:.1f}° 到 {target.base_degrees:.1f}° 的合法路线："
                "两条方向都会穿过限位。不要走穿过限位的路线，也不要自动分段绕路"
            )
        # 角度没变但也没有合法路线：说明当前位置本身有问题，继续按"不转角"处理。
        route = AngleRoute(
            endpoint_degrees=current.base_degrees, delta_degrees=0.0, pulses=0
        )

    motion = _Translation(
        config, CylindricalPose(current.base_degrees, current_radial, current_height)
    )
    frames = []

    if route.pulses != 0:
        # 先缩到安全半径，再抬到安全高度，然后**一次定好整条角度路线**，最后去目标。
        if motion.pose.radial_mm > safe.radial_mm + LENGTH_EPSILON_MM:
            frames.extend(motion.move_to(EXTENSION_AXIS, safe.radial_mm, "缩回到安全半径"))
        if motion.pose.height_mm < safe.height_mm - LENGTH_EPSILON_MM:
            frames.extend(motion.move_to(LIFT_AXIS, safe.height_mm, "抬到安全高度"))
        frames.extend(motion.turn_along(route))
        frames.extend(motion.move_to(EXTENSION_AXIS, target_radial, "伸到目标半径"))
        frames.extend(motion.move_to(LIFT_AXIS, target_height, "到目标高度"))
    else:
        # 不转角度：先改半径、再改高度（下降落到物料上是这条路径）。
        frames.extend(motion.move_to(EXTENSION_AXIS, target_radial, "伸到目标半径"))
        frames.extend(motion.move_to(LIFT_AXIS, target_height, "到目标高度"))

    requested = CylindricalPose(
        normalize_degrees(route.endpoint_degrees), target_radial, target_height
    )
    return MotionPlan(
        segments=tuple(frames), requested=requested, final_pose=motion.pose
    )


def plan_xyz(
    *,
    config,
    current: CylindricalPose,
    x_mm: float,
    y_mm: float,
    z_mm: float,
) -> Tuple[MotionPlan, CylindricalPose]:
    """绝对基座 XYZ → 圆柱 → 同一套规划；返回 ``(计划, 逆解出的目标位姿)``。

    ``x=y=0`` 时 XYZ 没有确定底座角度：**保留当前角度**，再检查 r=0 是否在行程内。
    不得由 ``atan2(0,0)`` 隐式产生一次转到 0° 的动作。
    """
    x = _require_finite("x_mm", x_mm)
    y = _require_finite("y_mm", y_mm)
    z = _require_finite("z_mm", z_mm)
    radial = math.hypot(x, y)
    if radial == 0.0:
        # 轴心目标：角度无定义，沿用当前位置；圆柱入口显式给角度时不受这条影响。
        base_degrees = current.base_degrees
    else:
        base_degrees = math.degrees(math.atan2(y, x))
    target = CylindricalPose(base_degrees, radial, z)
    return plan_absolute(config=config, current=current, target=target), target

"""eye-in-hand 视觉伺服：让爪子正下方对准物块（**只动平面**）。

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

那个量就是 2×2 雅可比 :class:`ServoJacobian`：

| | 底座每脉冲 | 伸缩每脉冲 |
|---|---|---|
| 画面 x 位移 | ``base_dx`` | ``radial_dx`` |
| 画面 y 位移 | ``base_dy`` | ``radial_dy`` |

它是**实测出来**的（:func:`probe_jacobian`），所以相机怎么装的、歪多少度、朝哪个方向，
全都不用假设。代价是标定一次。

## 为什么还要闭环

雅可比是线性的，而真实关系在大位移下会弯；背隙、丢步也会让"发了 800 脉冲"不等于
"走了 800 脉冲的量"。所以每次都用**新的一帧**重算，而不是相信"发了这么多就应该到了"。

**新的一帧**是硬要求：动作前记一个时刻，动作后只要晚于它的观测（``min_received_at``）。
用"多久算旧"挡不住"动作很快、旧帧也还新"的情况，那时读到的是动作**之前**的坐标，
闭环会以为没动过、然后再发一次——看起来像在振荡。

## 失败长什么样

三种都**响亮地失败**，不猜：

1. **画面里没有物块**（或不止一个而没配判别条件）——不猜哪一个是目标；
2. **雅可比奇异**——两个关节在画面里产生的位移方向几乎重合，2×2 反不过来。
   物理含义是"这两个轴动起来，物块在画面里往同一个方向跑"，靠它们凑不出任意方向；
3. **迭代用尽**——报出最后误差与总脉冲，让你去看是标定不准还是被挡住了。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Protocol, Sequence, Tuple

from modules.vision.drive.sighting import MaterialSighting

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eye_in_hand.json"


class FrameSource(Protocol):
    """观测源需要提供的那一件事：取一组观测。

    故意只要这一个方法，于是两种来源都能插进来，而这里不需要知道是哪种：

    - :class:`modules.camera_link.drive.link.CameraLink`（远端镜头走 TCP）；
    - :class:`modules.grasp.drive.camera_source.CameraFrameSource`（本机 USB 摄像头，
      默认走这条）。

    ``min_received_at`` 是"只要不早于这个时刻的观测"——机械臂动完之后
    **必须**带上它，否则可能读到动作**之前**的画面（详见模块开头）。
    """

    def frame(
        self,
        *,
        timeout: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
        min_received_at: Optional[float] = None,
    ) -> Optional[Sequence[MaterialSighting]]: ...

#: 判定雅可比奇异的门槛（行列式绝对值）。行列式是"两个关节在画面里位移方向"张成的
#: 面积；小到接近零时 2×2 求逆会把噪声放大成巨大的脉冲指令。
MIN_JACOBIAN_DETERMINANT = 1e-6

BASE_AXIS = "base"
RADIAL_AXIS = "radial"


class ServoError(RuntimeError):
    """伺服无法继续（看不到物块、雅可比奇异、不收敛、超出行程预算）。"""


@dataclass(frozen=True)
class ObjectSelector:
    """怎么从一堆观测里认出"哪条是物块"。

    - ``material_id``：按编号认（最稳）；
    - ``color``：按颜色认。

    两个都不给时要求**画面里只有一个目标**：多于一个就直接报错，
    因为猜错目标会让机械臂朝错误的方向动，而动作本身看起来完全正常。
    """

    material_id: Optional[int] = None
    color: Optional[str] = None

    def __post_init__(self) -> None:
        if self.color is not None:
            object.__setattr__(self, "color", self.color.strip().lower())

    def matches(self, sighting: MaterialSighting) -> bool:
        if self.material_id is not None and sighting.material_id != self.material_id:
            return False
        if self.color is not None and sighting.color != self.color:
            return False
        return True

    @property
    def is_open(self) -> bool:
        """没配判别条件时为真，此时要求画面里只有一个目标。"""
        return self.material_id is None and self.color is None

    def describe(self) -> str:
        if self.is_open:
            return "物块 = 画面里唯一的目标"
        parts = []
        if self.material_id is not None:
            parts.append(f"编号 {self.material_id}")
        if self.color is not None:
            parts.append(f"颜色 {self.color}")
        return "物块 = " + "、".join(parts)


def select_object(
    sightings: Sequence[MaterialSighting], selector: ObjectSelector
) -> MaterialSighting:
    """从一帧里挑出物块。

    :raises ServoError: 一个都没有，或者没有一个明确的。
    """
    matched = [item for item in sightings if selector.matches(item)]
    if not matched:
        listing = "、".join(
            f"物料 {item.material_id}({item.color})" for item in sightings
        ) or "（空）"
        raise ServoError(
            f"这一帧里没有物块（{selector.describe()}）；收到 {len(sightings)} 条：{listing}"
        )
    if len(matched) > 1 and selector.is_open:
        listing = "、".join(f"物料 {item.material_id}({item.color})" for item in matched)
        raise ServoError(
            f"画面里有多个目标（{listing}），不知道该对准哪一个；"
            "请在 modules/grasp/eye_in_hand.json 的 object 里填 material_id 或 color"
        )
    return matched[-1]


@dataclass(frozen=True)
class ServoTarget:
    """``(tx, ty)``：爪子抓取点在相机坐标系里的位置（毫米）。"""

    x_mm: float
    y_mm: float

    def summary(self) -> str:
        return f"目标读数 ({self.x_mm:+.2f}, {self.y_mm:+.2f}) mm"


@dataclass(frozen=True)
class ServoJacobian:
    """每脉冲引起的画面位移（毫米/脉冲）。

    正方向与 ``RobotArm`` 的动作方法一致：底座正脉冲 = CW，伸缩正脉冲 = 前伸。
    """

    base_dx: float
    base_dy: float
    radial_dx: float
    radial_dy: float

    @property
    def determinant(self) -> float:
        return self.base_dx * self.radial_dy - self.radial_dx * self.base_dy

    @property
    def is_singular(self) -> bool:
        return abs(self.determinant) < MIN_JACOBIAN_DETERMINANT

    def pulses_for(self, error_x: float, error_y: float) -> Tuple[float, float]:
        """要让画面位移改变 ``(−error_x, −error_y)``，两个轴各走多少脉冲。

        推导：线性化下走 ``p`` 脉冲会让观测量变化 ``J·p``；当前误差是 ``e``，
        要让它归零就是 ``e + J·p = 0``，即 ``p = −J⁻¹·e``。

        :raises ServoError: 雅可比奇异。
        """
        if self.is_singular:
            raise ServoError(
                f"雅可比奇异（行列式 {self.determinant:.3e}）：底座与伸缩在画面里"
                "产生的位移方向几乎重合，两个轴凑不出任意方向。"
                "请检查相机是不是装得让两个轴的运动在画面里同向，或重新标定 --calibrate"
            )
        determinant = self.determinant
        return (
            (self.radial_dx * error_y - self.radial_dy * error_x) / determinant,
            (self.base_dy * error_x - self.base_dx * error_y) / determinant,
        )

    def displacement_for(self, base_pulses: float, radial_pulses: float) -> Tuple[float, float]:
        """走这两个脉冲量会让画面里的物块挪多少（线性化预测）。"""
        return (
            self.base_dx * base_pulses + self.radial_dx * radial_pulses,
            self.base_dy * base_pulses + self.radial_dy * radial_pulses,
        )

    @property
    def base_norm(self) -> float:
        """底座轴的灵敏度（毫米/脉冲，不分方向）。"""
        return math.hypot(self.base_dx, self.base_dy)

    @property
    def radial_norm(self) -> float:
        """伸缩轴的灵敏度（毫米/脉冲，不分方向）。"""
        return math.hypot(self.radial_dx, self.radial_dy)

    def minimum_pulses(self, min_step_mm: float) -> Tuple[float, float]:
        """让每个轴至少推动画面 ``min_step_mm`` 毫米所需的脉冲数。

        用毫米表达门槛、再按**各自的灵敏度**换算成脉冲，是因为两个轴的灵敏度可以差
        十几倍（实测这台机器底座 0.164 mm/脉冲、伸缩 0.0139 mm/脉冲）。同一个"最小脉冲数"
        落在底座上等于二十多毫米的走动，会直接把闭环推到振荡。

        某一列几乎为零（那个轴动不了画面）时返回 0——那种情况本来就该由奇异判据拦下，
        不该在这里编一个巨大的脉冲数出来。
        """
        if min_step_mm <= 0:
            return (0.0, 0.0)
        base = min_step_mm / self.base_norm if self.base_norm > 1e-12 else 0.0
        radial = min_step_mm / self.radial_norm if self.radial_norm > 1e-12 else 0.0
        return (base, radial)

    def summary(self) -> str:
        return (
            f"雅可比：底座每 1000 脉冲 -> ({self.base_dx * 1000:+.2f}, {self.base_dy * 1000:+.2f}) mm；"
            f"伸缩每 1000 脉冲 -> ({self.radial_dx * 1000:+.2f}, {self.radial_dy * 1000:+.2f}) mm；"
            f"行列式 {self.determinant:.3e}"
        )


@dataclass(frozen=True)
class ServoConfig:
    """伺服的步长、预算与容差。

    ## 粗调 / 细调两档

    误差大于 ``coarse_error_mm`` 时按**粗调**走：单步上限放到
    ``coarse_max_pulses_per_step``，一步把大部分误差消掉。误差小了再按**细调**走，
    单步上限收紧到 ``max_pulses_per_step``。

    为什么要分档：单步上限卡得太死时，大误差要拆成好几轮，**每一轮都要重新取帧+定位**
    （定位是整个循环里最慢的一段之一），于是"能一步到位"变成"走三步"。
    分档之后大误差一轮收敛、小误差不会因为步子太大而反复过冲。
    """

    #: 平面误差小于这个值就认为对准了（毫米）。
    tolerance_mm: float = 2.0
    #: 最多重新测量几次。
    max_iterations: int = 12
    #: **细调**档的单轴脉冲上限。
    max_pulses_per_step: int = 1200
    #: **粗调**档的单轴脉冲上限；误差大于 ``coarse_error_mm`` 时用它。
    coarse_max_pulses_per_step: int = 4000
    #: 误差大于这个值就进粗调档（毫米）。
    coarse_error_mm: float = 10.0
    #: 单轴指令要让画面至少挪这么多毫米，否则按原方向抬到这么多。
    #: 抬的原因是**背隙**：指令太小时会被背隙整段吃掉，画面一动不动，
    #: 闭环于是以为"没动过"、再发一次同样的指令——看起来就是原地打转。
    #:
    #: **刻意用毫米而不是脉冲**：两个轴的灵敏度可以差十几倍（实测这台机器底座
    #: 0.164 mm/脉冲、伸缩 0.0139 mm/脉冲），同一个"最小脉冲数"落在底座上等于
    #: 二十多毫米的走动，会直接把闭环推到振荡。
    min_step_mm: float = 1.0
    #: 转角不超过这个脉冲数时，两个轴**同时**发；超过就改成"先伸缩、再旋转"。
    #: 伸出状态下横扫底座会把整条臂扫过工作区，大角度旋转不能和伸缩并行。
    parallel_rotation_limit_pulses: int = 600
    #: 每个轴在一次对准里允许走的总脉冲；用完了就报错。
    #: 这是**没有标定行程时唯一的兜底**，别设得太大。
    max_total_pulses: int = 10000
    #: 标定雅可比时，每个轴探两段同向位移；这是每段的脉冲数。
    probe_pulses: int = 200
    #: 等新观测的超时（秒）。
    frame_timeout_seconds: float = 2.0
    #: 只接受"接收时刻晚于动作完成时刻"的观测；这里额外留一点余量（秒）。
    frame_margin_seconds: float = 0.0
    #: 连续两次读数相差不超过这个值（毫米）才算"画面停稳了"。
    settle_tolerance_mm: float = 0.4
    #: 等画面停稳的最长时间（秒）；超时就拿最后一次读到的值，并让调用方知道。
    settle_timeout_seconds: float = 4.0
    #: 两条位置帧之间的最小间隔（秒）。
    #:
    #: **不能省。** 实测这台机器的 CH340 上，两条 ``FD`` 位置帧背靠背连发（间隔 0 ms）
    #: 会让**两帧都失效**——不是丢一帧，是两个轴都不动，而且驱动器 ``Response=None``
    #: 不会报任何错。留 20 ms 就正常。省掉它的代价是"所有两轴同时要动的步骤变成空操作"，
    #: 表现为闭环怎么走都不动、迭代用尽。
    inter_frame_seconds: float = 0.03
    #: 等停稳时的轮询间隔（秒）。
    settle_poll_seconds: float = 0.05
    #: 画面里物块的**像素直径**相对第一次测量变化超过这个比例就判为"被遮挡了"。
    #:
    #: 为什么用直径：相机在臂上时，爪瓣扫过物块会把它挡住一部分，可见区域从整圆变成
    #: 月牙，等效半径随之变小——而测距公式按直径反推距离，于是会被算成"物块突然远了
    #: 几十毫米"。那正是"爪子挡住了物块"的签名，而且它看起来完全像一次正常的观测。
    diameter_change_limit_ratio: float = 0.25

    def __post_init__(self) -> None:
        # 与 load_servo_config 的判据保持一致：容差、超时、粗调门槛为 0 都没有意义
        # （容差 0 等于要求"分毫不差"，超时 0 等于"根本不等帧"，
        # 粗调门槛 0 等于"永远走粗调"），而 frame_margin_seconds 取 0 是合法的。
        for name in (
            "tolerance_mm",
            "frame_timeout_seconds",
            "coarse_error_mm",
            "min_step_mm",
            "settle_timeout_seconds",
            "settle_poll_seconds",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} 必须是正数")
        if self.frame_margin_seconds < 0:
            raise ValueError("frame_margin_seconds 不能是负数")
        if self.settle_tolerance_mm < 0:
            raise ValueError("settle_tolerance_mm 不能是负数")
        if self.diameter_change_limit_ratio <= 0:
            raise ValueError("diameter_change_limit_ratio 必须是正数")
        if self.inter_frame_seconds < 0:
            raise ValueError("inter_frame_seconds 不能是负数")
        for name in (
            "max_iterations",
            "max_pulses_per_step",
            "coarse_max_pulses_per_step",
            "parallel_rotation_limit_pulses",
            "max_total_pulses",
            "probe_pulses",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须是正整数")

    def step_limit_for(self, error_mm: float) -> int:
        """当前误差该用哪一档的单步上限。"""
        return (
            self.coarse_max_pulses_per_step
            if error_mm > self.coarse_error_mm
            else self.max_pulses_per_step
        )


def load_servo_config(config_path: Path = DEFAULT_CONFIG_PATH) -> ServoConfig:
    """读取 ``modules/grasp/eye_in_hand.json`` 的伺服参数部分。"""
    document = _load_document(config_path)
    defaults = ServoConfig()
    return ServoConfig(
        tolerance_mm=_positive(document.get("tolerance_mm", defaults.tolerance_mm), "tolerance_mm"),
        max_iterations=_integer(document.get("max_iterations", defaults.max_iterations), "max_iterations"),
        max_pulses_per_step=_integer(
            document.get("max_pulses_per_step", defaults.max_pulses_per_step), "max_pulses_per_step"
        ),
        coarse_max_pulses_per_step=_integer(
            document.get("coarse_max_pulses_per_step", defaults.coarse_max_pulses_per_step),
            "coarse_max_pulses_per_step",
        ),
        coarse_error_mm=_positive(
            document.get("coarse_error_mm", defaults.coarse_error_mm), "coarse_error_mm"
        ),
        min_step_mm=_positive(
            document.get("min_step_mm", defaults.min_step_mm), "min_step_mm"
        ),
        parallel_rotation_limit_pulses=_integer(
            document.get("parallel_rotation_limit_pulses", defaults.parallel_rotation_limit_pulses),
            "parallel_rotation_limit_pulses",
        ),
        max_total_pulses=_integer(
            document.get("max_total_pulses", defaults.max_total_pulses), "max_total_pulses"
        ),
        probe_pulses=_integer(document.get("probe_pulses", defaults.probe_pulses), "probe_pulses"),
        frame_timeout_seconds=_positive(
            document.get("frame_timeout_seconds", defaults.frame_timeout_seconds),
            "frame_timeout_seconds",
        ),
        frame_margin_seconds=_non_negative(
            document.get("frame_margin_seconds", defaults.frame_margin_seconds),
            "frame_margin_seconds",
        ),
        settle_tolerance_mm=_non_negative(
            document.get("settle_tolerance_mm", defaults.settle_tolerance_mm),
            "settle_tolerance_mm",
        ),
        settle_timeout_seconds=_positive(
            document.get("settle_timeout_seconds", defaults.settle_timeout_seconds),
            "settle_timeout_seconds",
        ),
        settle_poll_seconds=_positive(
            document.get("settle_poll_seconds", defaults.settle_poll_seconds),
            "settle_poll_seconds",
        ),
        diameter_change_limit_ratio=_positive(
            document.get("diameter_change_limit_ratio", defaults.diameter_change_limit_ratio),
            "diameter_change_limit_ratio",
        ),
        inter_frame_seconds=_non_negative(
            document.get("inter_frame_seconds", defaults.inter_frame_seconds),
            "inter_frame_seconds",
        ),
    )


def load_object_selector(config_path: Path = DEFAULT_CONFIG_PATH) -> ObjectSelector:
    """读取 ``modules/grasp/eye_in_hand.json`` 的 ``object`` 段。"""
    document = _load_document(config_path)
    block = document.get("object")
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ServoError("object 必须是 JSON 对象")

    material_id = block.get("material_id")
    if material_id is not None and (type(material_id) is not int or material_id < 0):
        raise ServoError("object.material_id 必须是非负整数或 null")
    color = block.get("color")
    if color is not None and (not isinstance(color, str) or not color.strip()):
        raise ServoError("object.color 必须是非空字符串或 null")
    return ObjectSelector(material_id=material_id, color=color)


def load_target(config_path: Path = DEFAULT_CONFIG_PATH) -> Optional[ServoTarget]:
    """读取已标定的 ``(tx, ty)``；没标定过返回 ``None``。"""
    document = _load_document(config_path)
    block = document.get("target")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ServoError("target 必须是 JSON 对象")
    if block.get("x_mm") is None or block.get("y_mm") is None:
        return None
    return ServoTarget(
        x_mm=_number(block["x_mm"], "target.x_mm"),
        y_mm=_number(block["y_mm"], "target.y_mm"),
    )


def load_jacobian(config_path: Path = DEFAULT_CONFIG_PATH) -> Optional[ServoJacobian]:
    """读取已标定的雅可比；没标定过返回 ``None``。"""
    document = _load_document(config_path)
    block = document.get("jacobian")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ServoError("jacobian 必须是 JSON 对象")
    fields = ("base_dx", "base_dy", "radial_dx", "radial_dy")
    if any(block.get(field) is None for field in fields):
        return None
    return ServoJacobian(**{field: _number(block[field], f"jacobian.{field}") for field in fields})


def save_calibration(
    *,
    target: Optional[ServoTarget] = None,
    jacobian: Optional[ServoJacobian] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """把标定结果写回配置文件（只改这两个块，别的键原样保留）。"""
    document = _load_document(config_path)
    if target is not None:
        document["target"] = {"x_mm": target.x_mm, "y_mm": target.y_mm}
    if jacobian is not None:
        document["jacobian"] = {
            "base_dx": jacobian.base_dx,
            "base_dy": jacobian.base_dy,
            "radial_dx": jacobian.radial_dx,
            "radial_dy": jacobian.radial_dy,
        }
    config_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config_path


@dataclass(frozen=True)
class ServoResult:
    """一次 x/y 对准的结果。"""

    target: ServoTarget
    #: 每次测量到的画面偏移，按顺序；最后一个是最终位置。
    readings: Tuple[Tuple[float, float], ...]
    #: 最终误差分量与模长（毫米）。
    error_x: float
    error_y: float
    #: 用掉的底座 / 伸缩脉冲（带符号累计）。
    base_pulses: int
    radial_pulses: int
    iterations: int
    tolerance_mm: float
    #: 是不是"卡在分辨率极限上"停下的：误差已经比背隙能分辨的还小，
    #: 再走一步只会跨过目标。这时 :attr:`aligned` 可能是假，但那不是失败。
    resolution_limited: bool = False

    @property
    def error_mm(self) -> float:
        return math.hypot(self.error_x, self.error_y)

    @property
    def aligned(self) -> bool:
        return self.error_mm <= self.tolerance_mm

    def summary(self) -> str:
        if self.aligned:
            verdict = "完成"
        elif self.resolution_limited:
            verdict = "已到分辨率极限（背隙决定，再走只会跨过目标）"
        else:
            verdict = "未达标"
        return (
            f"x/y 对准{verdict}："
            f"最终读数 ({self.readings[-1][0]:+.2f}, {self.readings[-1][1]:+.2f}) mm，"
            f"误差 {self.error_mm:.2f} mm（容差 {self.tolerance_mm:.1f} mm）；"
            f"底座 {self.base_pulses:+d} 脉冲、伸缩 {self.radial_pulses:+d} 脉冲；"
            f"测量 {self.iterations} 次"
        )


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


def _read_object(link: FrameSource, selector: ObjectSelector, config: ServoConfig, *, after: Optional[float] = None) -> MaterialSighting:
    """等一帧，并返回物块那一观测。``after`` 给出"只要不早于这个时刻"的过滤。"""
    frame = link.frame(
        timeout=config.frame_timeout_seconds,
        min_received_at=after,
    )
    if frame is None:
        error = getattr(link, "error", None)
        raise ServoError(
            "等不到新的观测"
            + (f"；接收链路报错：{error}" if error else "；确认镜头端已连上并正在发数据")
        )
    return select_object(frame, selector)


def measure_offset(
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    after: Optional[float] = None,
) -> Tuple[float, float]:
    """读一帧，返回物块在相机系的水平偏移 ``(x, y)``（毫米）。"""
    sighting = _read_object(link, selector, config, after=after)
    return (sighting.x_mm, sighting.y_mm)


def _settled_sighting(
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    after: Optional[float] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> MaterialSighting:
    """等到画面不再变，返回那一刻的整条观测（不只是坐标）。

    返回观测而不只是 ``(x, y)``，是因为**像素直径**也要用：爪瓣扫过物块会把它挡住一部分，
    那是闭环唯一能自己发现的"机械干涉"信号。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    deadline = time.monotonic() + config.settle_timeout_seconds
    previous: Optional[Tuple[float, float]] = None
    last: Optional[MaterialSighting] = None
    while True:
        try:
            current = _read_object(link, selector, config, after=after)
        except ServoError:
            # 运动模糊时检不到物块是常态，不该当失败——接着等。
            previous = None
        else:
            last = current
            point = (current.x_mm, current.y_mm)
            if previous is not None and math.hypot(
                point[0] - previous[0], point[1] - previous[1]
            ) <= config.settle_tolerance_mm:
                return current
            previous = point
        if time.monotonic() >= deadline:
            if last is None:
                raise ServoError(
                    f"等了 {config.settle_timeout_seconds:.1f} s 也没能稳定看到物块；"
                    "确认物块在画面里、没有被人或线材挡住"
                )
            say(
                f"  等停稳超时（{config.settle_timeout_seconds:.1f} s）："
                f"画面仍在变，先按最后读数 ({last.x_mm:+.2f}, {last.y_mm:+.2f}) mm 继续"
            )
            return last
        time.sleep(config.settle_poll_seconds)


def measure_settled(
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    after: Optional[float] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> Tuple[float, float]:
    """测到画面**不再变**为止，返回那个读数。

    为什么不能"按估算耗时等一等就读"：实测 ``wait=True`` 在 0.652 s 就返回了，
    可读数在那之后还在大幅变化（``-23.65,+39.77`` → ``-33.38,+23.03``），
    中间还反复因为运动模糊整个检不到物块。拿那种数去解算，等于用噪声当观测——
    闭环会振荡，探针会把雅可比算大好几倍。

    判据是**连续两次读数足够接近**：把"等多久"交给观测本身，而不是交给一个估算公式。
    """
    sighting = _settled_sighting(link, selector, config, after=after, on_message=on_message)
    return (sighting.x_mm, sighting.y_mm)


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
    on_message: Optional[Callable[[str], None]] = None,
) -> ServoResult:
    """闭环把物块伺服到 ``(tx, ty)``；只动 底座 与 伸缩，不碰高度。

    :raises ServoError: 看不到物块、雅可比奇异、脉冲预算用完，或迭代用尽仍未进容差。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    readings: List[Tuple[float, float]] = []
    total_base = 0
    total_radial = 0
    #: 每次动完之后记一个时刻，下一次测量只接受晚于它的帧。
    #: **这是闭环正确性的关键**：动作很快时，队列里可能还压着动作**之前**的帧，
    #: 读到它就会以为"没动过"，于是再发一次同样的指令——表现为来回振荡。
    checkpoint: Optional[float] = None
    #: 第一次测到的像素直径，用来发现"物块被挡住了"。
    baseline_diameter: Optional[float] = None

    for attempt in range(1, config.max_iterations + 1):
        started = time.monotonic()
        sighting = _settled_sighting(link, selector, config, after=checkpoint)
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


def _clamp_pulses(value: float, limit: int) -> int:
    """把浮点脉冲量取整并夹到 ``±limit``。"""
    rounded = int(round(value))
    return max(-limit, min(limit, rounded))


def _load_document(config_path: Path) -> Mapping[str, object]:
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ServoError(f"无法读取 eye-in-hand 配置：{config_path}") from error
    except json.JSONDecodeError as error:
        raise ServoError(f"eye-in-hand 配置不是合法 JSON：{config_path}") from error
    if not isinstance(document, dict):
        raise ServoError("eye-in-hand 配置必须是 JSON 对象")
    return document


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServoError(f"{name} 必须是数字")
    if not math.isfinite(float(value)):
        raise ServoError(f"{name} 必须是有限数")
    return float(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ServoError(f"{name} 必须是正整数")
    return value


def _positive(value: object, name: str) -> float:
    number = _number(value, name)
    if not number > 0:
        raise ServoError(f"{name} 必须是正数")
    return number


def _non_negative(value: object, name: str) -> float:
    number = _number(value, name)
    if number < 0:
        raise ServoError(f"{name} 不能是负数")
    return number


# ---------------------------------------------------------------------- 自测
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


if __name__ == "__main__":
    raise SystemExit(_self_test())

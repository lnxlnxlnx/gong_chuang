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

## 平面之外的两步是开环的

x/y 靠闭环（就是上面这一套）；**平面一旦对准**，抓取前剩下的两步都是固定脉冲数：

```
descend_z          升降/6 号下降  z_descend_pulses
retract_extension  伸缩/7 号后缩  extension_retract_pulses
```

两个数都在 ``modules/grasp/eye_in_hand.json`` 里，换物块、换夹爪现场直接改，不用重标。
下降那一步会先确认平面已经对准（``z_center_tolerance_mm``）——它是这两步唯一的护栏。
想做"拿物块深度当反馈"的版本才用 :func:`align_z`（视觉伺服，要先标定升降灵敏度）。

## 一次调用

:func:`align_and_descend` 把"抬到安全高度 → x/y 对准 → z 下降 → 后缩"串成一次调用，
并把每一段的耗时记进 :class:`GraspReadyResult`。现场实测：一次定位 178 ms、一次停稳测量
约 537 ms，所以"段数"比动作本身更容易吃掉时间——这个结果对象就是用来盯这件事的。
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

    # ---------------------------------------------------------------- z 轴
    #
    # z 是"物块离镜头多远"，伺服目标就是"让它在 70 mm 左右"。
    # 它和 x/y 是同一套机制，但多一条**下降护栏**：没有 mm_per_motor_rev 就不知道
    # 升降的绝对高度，唯一能兜底的就是"测到的 z 不许低于 z_min_mm"。

    #: **下降前要求平面已对准**：物块的 ``hypot(x, y)`` 必须小于这个值（毫米）。
    #:
    #: 为什么必须有：``z`` 是按**直径**反推的，而直径与"物块在画面哪个位置"无关。
    #: 也就是说 z 伺服**看不见爪子有没有对准**——它只知道自己离物块多远。
    #: 不先对准就下降，等于让爪子斜着压下去。现场因此差点压坏爪子。
    z_center_tolerance_mm: float = 6.0
    #: z 误差小于这个值就算到位（毫米）。
    z_tolerance_mm: float = 3.0
    #: **下降护栏**：测到的 z 低于它就拒绝再往下走（毫米）。
    #:
    #: 这个值必须按**爪子相对镜头的下探量**来定：z 是"镜头到物块顶面"的距离，
    #: 而抓取点比镜头低，所以"爪子碰到物块"发生在 z 远大于 0 的地方。
    #: 定得太小等于没有护栏——现场就是这样差点压坏爪子的。
    #: 经验做法：先量出"爪尖刚好碰到物块顶面"时的 z，再把护栏设得比它大 5–10 mm。
    z_min_mm: float = 25.0
    #: z 的上限（毫米）。默认取视觉模块工作距离窗口的上端：出了窗口测距本身就不可信。
    z_max_mm: float = 160.0
    #: 升降单步脉冲上限与总预算。
    z_max_pulses_per_step: int = 600
    z_max_total_pulses: int = 8000

    # ---------------------------------------------- 脉冲版 z（开环，抓取前两步）
    #
    # 上面那几条是**视觉伺服**版 z（:func:`align_z`）用的：先标定升降灵敏度，再拿
    # "物块离镜头多远"做反馈。下面两条是它的**开环替代**：平面一旦对准，就按固定
    # 脉冲数把该做的两步走完。现场用的是这一套——数小、改起来快、不用标定任何东西。

    #: 平面对准之后，升降（6 号电机）**直接下降**多少脉冲到夹持高度。
    #:
    #: 为什么可以是固定脉冲数：爪子压下去之后画面里就没有物块了（爪瓣挡住、也掉出
    #: 工作距离窗口），z 这一段**本来就拿不到反馈**；而"从平面对准的高度到夹持高度"
    #: 是一段机械安装常数，实测一次就固定下来。换物块、换夹爪改这个数就行。
    #:
    #: 执行它的是 :func:`descend_z`——那一步会先确认平面已经对准，再走这么多脉冲。
    z_descend_pulses: int = 1700
    #: 下降之后，伸缩（7 号电机）**后缩**多少脉冲。
    #:
    #: 和 :attr:`z_descend_pulses` 一样是开环的固定值（现场定出来的那个数）。
    #: 正数 = 后缩（``RobotArm.move_backward``），执行它的是 :func:`retract_extension`。
    extension_retract_pulses: int = 150

    #: 标定升降灵敏度时，每段走多少脉冲。
    lift_probe_pulses: int = 120
    #: **找不到物块时**每次抬升多少脉冲、最多抬几次（"先抬起来再找"）。
    #: 物块太近时既模糊又出工作距离窗口，检测会直接失效；抬起来往往就看见了。
    find_lift_step_pulses: int = 500
    find_max_steps: int = 10
    #: 等新观测的超时（秒）。
    frame_timeout_seconds: float = 2.0
    #: 只接受"接收时刻晚于动作完成时刻"的观测；这里额外留一点余量（秒）。
    frame_margin_seconds: float = 0.0
    #: 连续两次读数相差不超过这个值（毫米）才算"画面停稳了"。
    settle_tolerance_mm: float = 0.4
    #: **粗调阶段**接受的读数一致度（毫米）。
    #:
    #: 粗调时误差还有十几毫米，没必要为了 0.4 mm 的一致性多等好几帧——而每一帧都要跑
    #: 一次定位（本机实测 178 ms）。远的时候只求"往哪边走、大概多远"，近了再按
    #: ``settle_tolerance_mm`` 严格判。
    coarse_settle_tolerance_mm: float = 3.0
    #: 等画面停稳的最长时间（秒）；超时就拿最后一次读到的值，并让调用方知道。
    settle_timeout_seconds: float = 4.0
    #: **转底盘前要抬到多高**：测到的 ``z``（镜头到物块顶面的距离）至少要有这么多毫米。
    #:
    #: 为什么用 z 而不是脉冲数：这台机器上**移动 x/y 就是旋转底盘**，而低位转过去会撞到
    #: 物块（现场撞倒过）。"抬够高度"的物理本质就是"镜头/爪子离物块顶面足够远"，
    #: 而 ``z`` 正是那个距离——它由摄像头直接测出来，**与传动比无关**，
    #: 所以不需要先标定 ``mm_per_motor_rev`` 或 ``dz_per_pulse`` 就能生效。
    #:
    #: ``0`` 表示不用这个判据，退回 ``safe_lift_pulses``。
    safe_lift_z_mm: float = 100.0
    #: 用 z 判据时**每一步最多**抬多少脉冲。
    #:
    #: 这是抬升的"步子大小"上限：标过 ``lift_scale`` 时按实测灵敏度算一步（算出来超过
    #: 它就夹到它），没标过时就直接用它当固定步长。夹一个上限是为了**标定值不准时也不
    #: 会一口气冲进行程端点**——每一步之后都会重新测一次 z 再决定下一步。
    safe_lift_step_pulses: int = 500
    #: 标过升降灵敏度时，用它**一步算到位**（而不是固定 500 脉冲一段地试）。
    #:
    #: 为什么快这么多：一次"停稳测量"实测约 537 ms（两次定位，每次 178 ms），而按
    #: 灵敏度算一步通常 1–2 步就到位——省下的是十几段测量，不是动作时间。
    safe_lift_use_scale: bool = True
    #: 抬到安全高度时，**按灵敏度算出来的那一步**最多走多少脉冲。
    #: 与 ``safe_lift_step_pulses`` 的区别：那个是"没标定时的固定步长"，这个是"算出来
    #: 的一步最多能让它走多远"。标定值偏小时它会兜住，免得一步冲进行程端点。
    safe_lift_max_pulses_per_step: int = 2000
    #: 用 z 判据时最多抬几次，免得对着一个永远够不到的目标空转。
    safe_lift_max_steps: int = 20
    #: 兜底：**看不到物块时**按这个脉冲数抬升（``0`` 表示不抬，只警告）。
    safe_lift_pulses: int = 0
    #: 转底盘的角度超过这么多度时，就必须先抬升。
    safe_lift_angle_degrees: float = 1.0
    #: 两条位置帧之间的最小间隔（秒）。
    #: **不能省。** 实测这台机器的 CH340 上，两条 ``FD`` 位置帧背靠背连发（间隔 0 ms）
    #: 会让**两帧都失效**——不是丢一帧，是两个轴都不动，而且驱动器 ``Response=None``
    #: 不会报任何错。留 20 ms 就正常。省掉它的代价是"所有两轴同时要动的步骤变成空操作"，
    #: 表现为闭环怎么走都不动、迭代用尽。
    inter_frame_seconds: float = 0.03
    #: 等停稳时的轮询间隔（秒）。
    #:
    #: 摄像头本身约 32 ms 一帧，所以这个值只该是"补一帧"的零头：50 ms 等于每次轮询
    #: 白等 50 ms，而一次停稳测量至少要轮询两次。别调成 0——那样会在没新帧时打转空转
    #: CPU，也失去了"等一帧"的语义。
    settle_poll_seconds: float = 0.01
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
        if self.coarse_settle_tolerance_mm < 0:
            raise ValueError("coarse_settle_tolerance_mm 不能是负数")
        if self.coarse_settle_tolerance_mm < self.settle_tolerance_mm:
            raise ValueError(
                "coarse_settle_tolerance_mm 不能小于 settle_tolerance_mm："
                "粗调档只该更宽松，不该更严"
            )
        if self.diameter_change_limit_ratio <= 0:
            raise ValueError("diameter_change_limit_ratio 必须是正数")
        if self.inter_frame_seconds < 0:
            raise ValueError("inter_frame_seconds 不能是负数")
        if self.safe_lift_pulses < 0:
            raise ValueError("safe_lift_pulses 不能是负数")
        if self.safe_lift_z_mm < 0:
            raise ValueError("safe_lift_z_mm 不能是负数")
        if not self.safe_lift_angle_degrees > 0:
            raise ValueError("safe_lift_angle_degrees 必须是正数")
        for name in (
            "max_iterations",
            "max_pulses_per_step",
            "coarse_max_pulses_per_step",
            "parallel_rotation_limit_pulses",
            "max_total_pulses",
            "probe_pulses",
            "z_max_pulses_per_step",
            "z_max_total_pulses",
            "z_descend_pulses",
            "extension_retract_pulses",
            "safe_lift_step_pulses",
            "safe_lift_max_pulses_per_step",
            "safe_lift_max_steps",
            "lift_probe_pulses",
            "find_lift_step_pulses",
            "find_max_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("z_tolerance_mm", "z_min_mm", "z_max_mm"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} 必须是正数")
        if self.z_min_mm >= self.z_max_mm:
            raise ValueError("z_min_mm 必须小于 z_max_mm")

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
        coarse_settle_tolerance_mm=_non_negative(
            document.get("coarse_settle_tolerance_mm", defaults.coarse_settle_tolerance_mm),
            "coarse_settle_tolerance_mm",
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
        safe_lift_z_mm=_non_negative(
            document.get("safe_lift_z_mm", defaults.safe_lift_z_mm), "safe_lift_z_mm"
        ),
        safe_lift_step_pulses=_integer(
            document.get("safe_lift_step_pulses", defaults.safe_lift_step_pulses),
            "safe_lift_step_pulses",
        ),
        safe_lift_use_scale=_boolean(
            document.get("safe_lift_use_scale", defaults.safe_lift_use_scale),
            "safe_lift_use_scale",
        ),
        safe_lift_max_pulses_per_step=_integer(
            document.get(
                "safe_lift_max_pulses_per_step", defaults.safe_lift_max_pulses_per_step
            ),
            "safe_lift_max_pulses_per_step",
        ),
        safe_lift_max_steps=_integer(
            document.get("safe_lift_max_steps", defaults.safe_lift_max_steps),
            "safe_lift_max_steps",
        ),
        safe_lift_pulses=_non_negative_integer(
            document.get("safe_lift_pulses", defaults.safe_lift_pulses), "safe_lift_pulses"
        ),
        safe_lift_angle_degrees=_positive(
            document.get("safe_lift_angle_degrees", defaults.safe_lift_angle_degrees),
            "safe_lift_angle_degrees",
        ),
        z_tolerance_mm=_positive(
            document.get("z_tolerance_mm", defaults.z_tolerance_mm), "z_tolerance_mm"
        ),
        z_center_tolerance_mm=_positive(
            document.get("z_center_tolerance_mm", defaults.z_center_tolerance_mm),
            "z_center_tolerance_mm",
        ),
        z_min_mm=_positive(document.get("z_min_mm", defaults.z_min_mm), "z_min_mm"),
        z_max_mm=_positive(document.get("z_max_mm", defaults.z_max_mm), "z_max_mm"),
        z_max_pulses_per_step=_integer(
            document.get("z_max_pulses_per_step", defaults.z_max_pulses_per_step),
            "z_max_pulses_per_step",
        ),
        z_max_total_pulses=_integer(
            document.get("z_max_total_pulses", defaults.z_max_total_pulses),
            "z_max_total_pulses",
        ),
        z_descend_pulses=_integer(
            document.get("z_descend_pulses", defaults.z_descend_pulses), "z_descend_pulses"
        ),
        extension_retract_pulses=_integer(
            document.get("extension_retract_pulses", defaults.extension_retract_pulses),
            "extension_retract_pulses",
        ),
        lift_probe_pulses=_integer(
            document.get("lift_probe_pulses", defaults.lift_probe_pulses), "lift_probe_pulses"
        ),
        find_lift_step_pulses=_integer(
            document.get("find_lift_step_pulses", defaults.find_lift_step_pulses),
            "find_lift_step_pulses",
        ),
        find_max_steps=_integer(
            document.get("find_max_steps", defaults.find_max_steps), "find_max_steps"
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


def load_z_target(config_path: Path = DEFAULT_CONFIG_PATH) -> Optional[float]:
    """读取已标定的 z 目标（毫米）；没标定过返回 ``None``。"""
    document = _load_document(config_path)
    value = document.get("z_target_mm")
    return None if value is None else _number(value, "z_target_mm")


def load_lift_scale(config_path: Path = DEFAULT_CONFIG_PATH) -> Optional[LiftScale]:
    """读取已标定的升降灵敏度；没标定过返回 ``None``。"""
    document = _load_document(config_path)
    block = document.get("lift_scale")
    if not isinstance(block, dict) or block.get("dz_per_pulse") is None:
        return None
    return LiftScale(dz_per_pulse=_number(block["dz_per_pulse"], "lift_scale.dz_per_pulse"))


def save_calibration(
    *,
    target: Optional[ServoTarget] = None,
    jacobian: Optional[ServoJacobian] = None,
    z_target_mm: Optional[float] = None,
    lift_scale: Optional[LiftScale] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """把标定结果写回配置文件（只改这几个块，别的键原样保留）。"""
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
    if z_target_mm is not None:
        document["z_target_mm"] = z_target_mm
    if lift_scale is not None:
        document["lift_scale"] = {"dz_per_pulse": lift_scale.dz_per_pulse}
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
    tolerance: Optional[float] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> MaterialSighting:
    """等到画面不再变，返回那一刻的整条观测（不只是坐标）。

    返回观测而不只是 ``(x, y)``，是因为**像素直径**也要用：爪瓣扫过物块会把它挡住一部分，
    那是闭环唯一能自己发现的"机械干涉"信号。

    ``tolerance`` 覆盖一致性判据：粗调阶段（离目标还有十几毫米）用
    ``coarse_settle_tolerance_mm`` 就够，不必为了 0.4 mm 多等好几帧——每一帧都要跑一次
    定位，本机实测 178 ms。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    allowed = config.settle_tolerance_mm if tolerance is None else tolerance
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
            point = (current.x_mm, current.y_mm, current.z_mm)
            if previous is not None and math.sqrt(
                (point[0] - previous[0]) ** 2
                + (point[1] - previous[1]) ** 2
                + (point[2] - previous[2]) ** 2
            ) <= allowed:
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
    ensure_safe_height: bool = True,
    lift_scale: Optional[LiftScale] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> ServoResult:
    """闭环把物块伺服到 ``(tx, ty)``；只动 底座 与 伸缩，不碰高度。

    :param ensure_safe_height: 动手之前要不要先确认"已经抬到安全高度"（默认要）。
        连续用同一高度调用时（``--follow`` 的每一轮、或一次"对准+下降"里已经抬过）
        可以关掉：省下的是一次停稳测量（实测约 0.5 s），而高度在这些步骤里不会变。
        只有调用方**确实知道高度没变**才能关——低位转底盘会撞到物块。
    :param lift_scale: 已标定的升降灵敏度，交给 :func:`raise_before_rotating` 一步算到位。
    :raises ServoError: 看不到物块、雅可比奇异、脉冲预算用完，或迭代用尽仍未进容差。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    # x/y 的每一步都要转底盘，所以先在**打开动作之前**抬到安全高度。这是机械结构上的
    # 硬约束：低位转底盘会撞到台面上的物块。刻意不自动降回去——降回去正是撞到它的动作。
    if ensure_safe_height:
        raise_before_rotating(arm, link, selector, config, scale=lift_scale, on_message=say)
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


# ------------------------------------------------------------------ z 轴伺服
#
# z 就是 :attr:`MaterialSighting.z_mm`——**物块沿光轴离镜头多远**。抓取时希望它落在
# 7 cm 左右，所以这一步的活是"驱动升降，让测到的 z 等于目标"。
#
# 和 x/y 同一套机制（测 → 算 → 走 → 再测），但多两件事：
#
# 1. **下降护栏**：``mm_per_motor_rev`` 没标定时不知道升降的绝对高度，唯一能兜底的
#    就是"测到的 z 不许低于 z_min_mm"。相机撞到物块之前，z 一定先变小。
# 2. **找不到就先抬起来**：物块太近时既模糊又掉出工作距离窗口，检测会直接失效。
#    抬起来往往就看见了，所以 :func:`find_object` 是这一步的标准起手式。

LIFT_AXIS = "lift"


@dataclass(frozen=True)
class LiftScale:
    """升降的 z 灵敏度：每脉冲让物块深度变化多少毫米（**抬升为正**）。"""

    dz_per_pulse: float

    def pulses_for(self, error_mm: float) -> float:
        """把误差 ``error_mm`` **消掉**所需的升降脉冲（带符号）。

        ``error_mm`` 是"当前 z − 目标 z"。要让 z 变化 ``−error_mm``，所以这里取负——
        和 :meth:`ServoJacobian.pulses_for` 同一个约定（那条内部也带负号）。
        **漏掉这个负号会让升降朝反方向走**：z 高于目标时它去抬升，越走越远。
        """
        if abs(self.dz_per_pulse) < 1e-9:
            raise ServoError(
                "升降灵敏度是 0：这个轴没有改变物块深度。确认升降真的接上了、"
                "且探针时画面里有完整的物块"
            )
        return -error_mm / self.dz_per_pulse

    def summary(self) -> str:
        return f"升降每 1000 脉冲 -> z {self.dz_per_pulse * 1000:+.1f} mm"


@dataclass(frozen=True)
class ZResult:
    """一次 z 对准的结果。"""

    target_mm: float
    #: 每次测到的 z，按顺序；最后一个是最终值。
    readings: Tuple[float, ...]
    error_mm: float
    pulses: int
    iterations: int
    tolerance_mm: float
    #: 因为下降护栏拦住而停在半路（不是失败，是"再往下会撞"）。
    stopped_by_guard: bool = False

    @property
    def aligned(self) -> bool:
        return abs(self.error_mm) <= self.tolerance_mm

    def summary(self) -> str:
        if self.aligned:
            verdict = "完成"
        elif self.stopped_by_guard:
            verdict = "被下降护栏拦住（再往下会撞上物块）"
        else:
            verdict = "未达标"
        return (
            f"z 对准{verdict}：最终 z={self.readings[-1]:.1f} mm，"
            f"误差 {self.error_mm:+.2f} mm（容差 {self.tolerance_mm:.1f} mm）；"
            f"升降 {self.pulses:+d} 脉冲；测量 {self.iterations} 次"
        )


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


def raise_before_rotating(
    arm,
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    scale: Optional[LiftScale] = None,
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


#: 快速探一次"物块在不在画面里"时最多读几帧。每帧约 221 ms（含 178 ms 定位）。
VISIBILITY_PROBE_FRAMES = 2


def _probe_visible(link: FrameSource, selector: ObjectSelector, config: ServoConfig) -> bool:
    """快速回答"物块在不在画面里"，**不等停稳**。

    为什么可以不等：这一步只问"在不在"，不问"在哪"——差几毫米无所谓。而等停稳
    （连续两帧一致）至少要 537 ms；看不到时还要一直等到 ``settle_timeout_seconds``
    （默认 4 s）才认输，乘上"抬起来找"的步数就是几十秒。

    代价是可能被一帧运动模糊骗到：所以连读 :data:`VISIBILITY_PROBE_FRAMES` 帧，
    任何一帧认出来就算"在"。
    """
    for _ in range(VISIBILITY_PROBE_FRAMES):
        try:
            _read_object(link, selector, config)
        except ServoError:
            continue
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


# ------------------------------------------- 脉冲版 z：下降与后缩（开环，现场在用）
#
# 平面一旦对准，剩下的抓取前两步就是两个固定脉冲数：升降下降 ``z_descend_pulses``、
# 伸缩后缩 ``extension_retract_pulses``。
#
# 为什么这里可以开环：爪子压下去之后画面里就没有物块了（爪瓣挡住、也掉出工作距离
# 窗口），**这一段本来就拿不到反馈**；而"从平面对准的高度到夹持高度"是机械安装常数，
# 量一次就固定。真正要保证的不是"下降了多少"，而是**下降之前平面已经对准**——脉冲数
# 只保证往下走了那么多，不保证下面正好是物块。


@dataclass(frozen=True)
class ZDescentResult:
    """一次"按脉冲下降"的结果。"""

    #: 实际发出的下降脉冲数（正数）。
    pulses: int
    #: 下降前测到的平面误差（毫米）——这一步的前置条件。
    planar_mm: float
    #: 下降前测到的物块深度（毫米），事后用来核对"当时停在哪一层"。
    start_z_mm: float
    #: 下降动作的估算耗时（秒）。
    seconds: float

    def summary(self) -> str:
        return (
            f"下降 {self.pulses} 脉冲完成（下降前 z={self.start_z_mm:.1f} mm、"
            f"平面误差 {self.planar_mm:.2f} mm）；耗时约 {self.seconds:.2f} s"
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


@dataclass(frozen=True)
class GraspReadyResult:
    """一次"抬升 + x/y 对准 + z 下降 + 后缩"的结果。"""

    #: 转底盘前抬到安全高度用了多少脉冲（本来就在安全高度就是 0）。
    raised_pulses: int
    #: x/y 对准的结果：误差、迭代次数、两个轴各走了多少脉冲。
    aligned: ServoResult
    #: z 下降的结果：走了多少脉冲、下降前多深、下降前的平面误差。
    descent: ZDescentResult
    #: 伸缩后缩的脉冲数。
    retracted_pulses: int
    #: 各段的墙钟耗时（秒）。记下来是为了**一眼看出时间花在哪一段**——
    #: 定位一次 178 ms、停稳测量一次约 537 ms，段数比动作本身更容易吃掉时间。
    lift_seconds: float
    align_seconds: float
    descend_seconds: float
    retract_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.lift_seconds
            + self.align_seconds
            + self.descend_seconds
            + self.retract_seconds
        )

    def summary(self) -> str:
        return (
            f"就位完成：抬升 {self.raised_pulses} 脉冲、x/y 误差 {self.aligned.error_mm:.2f} mm"
            f"（{self.aligned.iterations} 次测量）、下降 {self.descent.pulses} 脉冲、后缩 "
            f"{self.retracted_pulses} 脉冲；共 {self.total_seconds:.1f} s"
            f"（抬升 {self.lift_seconds:.1f} + 对准 {self.align_seconds:.1f}"
            f" + 下降 {self.descend_seconds:.1f} + 后缩 {self.retract_seconds:.1f}）"
        )


def align_and_descend(
    arm,
    link: FrameSource,
    selector: ObjectSelector,
    target: ServoTarget,
    jacobian: ServoJacobian,
    config: ServoConfig,
    *,
    lift_scale: Optional[LiftScale] = None,
    z_pulses: Optional[int] = None,
    retract_pulses: Optional[int] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> GraspReadyResult:
    """**一次调用走完抓取前的就位**：抬到安全高度 → x/y 对准 → z 下降 → 伸缩后缩。

    顺序不是随便定的：

    1. **先抬到安全高度**：x/y 的每一步都要转底盘，低位转过去会撞到台面上的物块。
       整个流程只在这里查一次——后面三步都不改高度。
    2. **x/y 对准**（闭环，只看画面）：把物块伺服到 ``(tx, ty)``，也就是"爪子抓取点正在
       物块正上方"。粗调阶段用 ``coarse_settle_tolerance_mm`` 判停稳，少等几帧。
    3. **z 下降**：平面已对准才允许下降，按 :attr:`ServoConfig.z_descend_pulses` 走固定脉冲。
    4. **伸缩后缩**：再按 :attr:`ServoConfig.extension_retract_pulses` 后缩到夹持位。

    第 3、4 步是开环的（爪子压下去之后就看不到物块了），**第 2 步的结果就是它们唯一的安全
    依据**：平面误差超过 ``z_center_tolerance_mm`` 时 :func:`descend_z` 会拒绝下降、一个
    动作帧都不发。

    比"自己按顺序调四个函数"快在哪：安全高度按灵敏度**一步算到位**（不是 500 脉冲一段地
    试）；已经抬够时只花一次测量；对准阶段不再重复查安全高度。每一步的耗时都记在
    :class:`GraspReadyResult` 里，慢在哪一段一眼能看出来。

    :param lift_scale: 已标定的升降灵敏度；``None`` 表示没标定，抬升退回固定步长。
    :param z_pulses: 覆盖下降脉冲数（只这一次）；``None`` 用配置值。
    :param retract_pulses: 覆盖后缩脉冲数（只这一次）；``None`` 用配置值。
    :raises ServoError: 任何一步失败；异常信息里会写清卡在哪一步。
    """
    say = on_message if on_message is not None else (lambda _message: None)

    started = time.monotonic()
    say("① 转底盘前先确认安全高度")
    raised = raise_before_rotating(
        arm, link, selector, config, scale=lift_scale, on_message=say
    )
    lift_seconds = time.monotonic() - started

    started = time.monotonic()
    say("② x/y 对准（闭环，只动底座与伸缩）")
    aligned = align_x_y(
        arm,
        link,
        selector,
        target,
        jacobian,
        config,
        ensure_safe_height=False,
        lift_scale=lift_scale,
        on_message=say,
    )
    align_seconds = time.monotonic() - started

    started = time.monotonic()
    say("③ z 下降（平面已对准，按固定脉冲）")
    descent = descend_z(
        arm, link, selector, config, pulses=z_pulses, scale=lift_scale, on_message=say
    )
    descend_seconds = time.monotonic() - started

    started = time.monotonic()
    say("④ 伸缩后缩")
    retracted = retract_extension(arm, config, pulses=retract_pulses, on_message=say)
    retract_seconds = time.monotonic() - started

    return GraspReadyResult(
        raised_pulses=raised,
        aligned=aligned,
        descent=descent,
        retracted_pulses=retracted,
        lift_seconds=lift_seconds,
        align_seconds=align_seconds,
        descend_seconds=descend_seconds,
        retract_seconds=retract_seconds,
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


def _boolean(value: object, name: str) -> bool:
    """严格取布尔值：``true``/``false`` 才认，``1``/``"true"`` 不认。

    为什么不用 ``bool(value)``：那样任何非空字符串都成了"开"——配置文件里写错成
    ``"false"`` 会静默变成开，而这几个开关管的正是"要不要走快的那条路"。
    """
    if type(value) is not bool:
        raise ServoError(f"{name} 必须是 true 或 false")
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


def _non_negative_integer(value: object, name: str) -> int:
    """非负整数；``0`` 常用来表示"这个保护没开"。"""
    if type(value) is not int or value < 0:
        raise ServoError(f"{name} 必须是非负整数")
    return value


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

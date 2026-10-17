"""抓取伺服的配置、数值模型、结果对象与配置读写。

## 这个模块在整条链路里的位置

``eye_in_hand`` 原先一个文件里住着四件事：配置与数值模型、观测判据、脉冲原语、实际控制
算法。**只按行数切文件不会降低任何一处的复杂度**，所以这里按"谁改谁"来分：

- ``servo_model``（本模块）：配置、目标、雅可比、升降尺度、结果对象、``ServoError``、
  以及配置文件的读写。**共用类型只有这一份定义**，谁也不许再造一个副本。
- :mod:`modules.grasp.drive.servo_observation`：观测源协议、物料选择、停稳与可见性判据。
  它**不发电机动作**。
- :mod:`modules.grasp.drive.servo_motion`：真的把脉冲发给机械臂并等它走完。
  它**不读视觉、不做策略选择**。
- :mod:`modules.grasp.drive.eye_in_hand`：实际控制算法——探针、安全抬升、悬停、x/y
  闭环、z 下降与后缩，外加原来的自测 CLI。

## 依赖方向（单向，不成环）

```
eye_in_hand      → servo_model + servo_observation + servo_motion
servo_observation → servo_model（+ vision 的观测类型，只在注解里）
servo_motion      → servo_model（+ 现有机械臂/电机能力）
servo_model       → 只依赖标准库
```

``servo_observation`` 与 ``servo_motion`` **互不导入**；本模块不导入任何控制算法，
所以 ``import modules.grasp.drive.servo_model`` 不需要模型、相机或串口。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple



DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eye_in_hand.json"


#: 判定雅可比奇异的门槛（行列式绝对值）。行列式是"两个关节在画面里位移方向"张成的
#: 面积；小到接近零时 2×2 求逆会把噪声放大成巨大的脉冲指令。
MIN_JACOBIAN_DETERMINANT = 1e-6


BASE_AXIS = "base"


RADIAL_AXIS = "radial"


#: 升降轴的名字。它与 ``BASE_AXIS`` / ``RADIAL_AXIS`` 一样是**轴名字符串**，不是配置项；
#: 放在这里是为了让三个轴名只有一个定义处，而不是散在控制器里。
LIFT_AXIS = "lift"


class ServoError(RuntimeError):
    """伺服无法继续（看不到物块、雅可比奇异、不收敛、超出行程预算）。"""


class ObjectMissing(ServoError):
    """**这一帧里一个物块都没有**——不是"链路坏了"，也不是"有好几个目标"。

    单独一个类型，是因为调用方对它的处理与别的 :class:`ServoError` 不同：x/y 对准在
    "已经测到过、而且已经进入可下降精度"之后看不见物块，是**爪子自己悬到了物块上方把光
    挡住**的典型现象，可以按最后一次读数收工（见
    :func:`modules.grasp.drive.eye_in_hand._result_when_occluded`）；而"多个目标"和
    "等不到新观测"必须停下来问人。用异常消息的字符串去区分这三件事太脆，所以用类型。
    """


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

    @classmethod
    def from_want(cls, want) -> "ObjectSelector":
        """把 ``vision.locate("red")`` 那套说法变成一个判别条件。

        - 字符串 = 类别名（``"red"``，大小写与空格无关）→ ``color``；
        - 整数 = 编号（从 1 开始）→ ``material_id``；两边是**同一个编号**：
          ``_sightings_from_report`` 就是按 ``MaterialReport.faces`` 从 1 数出来的，
          与报告里打印的「物料 N:」一致。

        校验直接复用 :func:`modules.vision.reports.normalize_want`——**只有一处**定义
        "want 长什么样"，所以 ``locate`` 与 ``align_xy`` 不可能对同一种写法有两种态度。
        """
        from modules.vision.reports import normalize_want

        normalized = normalize_want(want)
        if normalized is None:
            return cls()
        if isinstance(normalized, int):
            return cls(material_id=normalized)
        return cls(color=normalized)

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
    #: **开始时就位时要悬停在物块上方多高**（毫米，镜头到物块顶面的距离）；``null`` = 不主动调整。
    #:
    #: 为什么需要它：x/y 对准**不动高度**，所以"对准时爪子在哪一层"完全取决于机械臂当时
    #: 停在哪——只由 ``safe_lift_z_mm`` 兜一个下限。而下降是**开环的固定脉冲数**
    #: （:attr:`z_descend_pulses`），起点高度不固定，落点就不固定。
    #:
    #: 配上这个值之后，就位的第一步会把镜头到物块的距离**伺服到这个数**（高了降、低了抬），
    #: 于是每次下降都从同一层开始。两条硬约束：
    #:
    #: 1. 必须 ``>= safe_lift_z_mm``——低位转底盘会撞到物块，配置不对时直接拒绝加载；
    #: 2. 往下降只会降到 ``safe_lift_z_mm`` 为止（护栏），而且要**先知道一步走多远**：
    #:    有标定的 :class:`LiftScale` 就用它，没有就先抬一段量出真实灵敏度再降。
    #:
    #: **改了它就要重测 ``z_descend_pulses``**：下降是开环的，起点换了多少毫米，
    #: 落点就跟着换多少毫米。想保持现在的行为，把它设成跟 ``safe_lift_z_mm`` 同一个数。
    hover_z_mm: Optional[float] = None
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
    #: "每一轮都要查一次安全高度"的场景（``--follow``）用**单帧**快判时要求的余量（毫米）。
    #:
    #: 只有"单帧读到的 z ≥ 安全高度 + 这个余量"才直接放行，否则退回停稳测量。
    #: 余量是必须的：运动模糊的一帧可能把 z **读大**，而"以为够高"正是低位转底盘的成因。
    safe_lift_quick_margin_mm: float = 5.0
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
        if self.safe_lift_quick_margin_mm < 0:
            raise ValueError("safe_lift_quick_margin_mm 不能是负数")
        if self.hover_z_mm is not None:
            if not self.hover_z_mm > 0:
                raise ValueError("hover_z_mm 必须是正数或 null")
            if self.hover_z_mm < self.safe_lift_z_mm:
                raise ValueError(
                    f"hover_z_mm={self.hover_z_mm} 低于 safe_lift_z_mm={self.safe_lift_z_mm}："
                    "悬停得比「转底盘前的安全高度」还低，等于低位转底盘，会撞到台面上的物块"
                )
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
    #: 是不是**因遮挡收工**的：物块被爪子（或别的结构）挡住之后，按已经测到的那次读数
    #: 收工，而不是继续要新帧。它表示"没有再验证过"，**不是**"已经对准"——调用方要据此
    #: 决定后面还能不能做开环动作（例如下降）。
    occluded: bool = False
    #: 是不是**单帧一步**（:func:`modules.grasp.drive.eye_in_hand.align_x_y` 的
    #: ``single_step``）：只测了一帧、走了一步就返回，**动作之后没有再看过**。
    #: 所以 ``readings`` / ``error_x`` / ``error_y`` 说的是**动作之前**那一次，
    #: 不代表现在的位置——调用方本来就会立刻再测一次（跟随循环就是这样）。
    unverified: bool = False

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
        if self.occluded:
            verdict += "（物块被挡住，按最后一次读数收工，未再验证）"
        if self.unverified:
            verdict += "（单帧一步：动作之后没有再验证）"
        return (
            f"x/y 对准{verdict}："
            f"最终读数 ({self.readings[-1][0]:+.2f}, {self.readings[-1][1]:+.2f}) mm，"
            f"误差 {self.error_mm:.2f} mm（容差 {self.tolerance_mm:.1f} mm）；"
            f"底座 {self.base_pulses:+d} 脉冲、伸缩 {self.radial_pulses:+d} 脉冲；"
            f"测量 {self.iterations} 次"
        )


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


@dataclass(frozen=True)
class HoverResult:
    """一次"悬停到配置高度"（:func:`hover_above_object`）的结果。"""

    #: 这一步真的跑了吗；没跑（没配 ``hover_z_mm``，或调用方要求不动高度）就是"跳过"，
    #: ``pulses`` 必为 0。
    configured: bool
    #: 实际走的升降脉冲（正 = 抬升、负 = 下降）。
    pulses: int = 0
    #: 起始 / 结束时的物块深度（毫米）；跳过时都是 ``None``。
    start_z_mm: Optional[float] = None
    end_z_mm: Optional[float] = None
    #: 需要往下走，但**完全没有灵敏度信息**（没标定、这一段也还没量出来），所以没敢降；
    #: 只保证不低于 ``safe_lift_z_mm``。
    stopped_without_scale: bool = False
    #: 需要往下走，但已经贴到 ``safe_lift_z_mm`` 这道护栏，不能再降。
    stopped_at_safe_floor: bool = False

    def summary(self) -> str:
        if not self.configured:
            return "悬停高度：跳过（没配 hover_z_mm，或调用方要求不动高度；起始高度只由安全高度兜下限）"
        if not self.pulses:
            return f"悬停高度：本来就在 z={self.start_z_mm:.1f} mm，没动"
        return (
            f"悬停高度：{self.pulses:+d} 脉冲，z {self.start_z_mm:.1f} -> "
            f"{self.end_z_mm:.1f} mm"
        )


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
    #: 就位第一步"悬停到 hover_z_mm"走的脉冲（正=抬、负=降）；没配这一项就是 0。
    hover_pulses: int = 0

    @property
    def total_seconds(self) -> float:
        return (
            self.lift_seconds
            + self.align_seconds
            + self.descend_seconds
            + self.retract_seconds
        )

    def summary(self) -> str:
        hover = f"悬停 {self.hover_pulses:+d} 脉冲、" if self.hover_pulses else ""
        return (
            f"就位完成：{hover}抬升 {self.raised_pulses} 脉冲、x/y 误差 "
            f"{self.aligned.error_mm:.2f} mm"
            f"（{self.aligned.iterations} 次测量）、下降 {self.descent.pulses} 脉冲、后缩 "
            f"{self.retracted_pulses} 脉冲；共 {self.total_seconds:.1f} s"
            f"（抬升 {self.lift_seconds:.1f} + 对准 {self.align_seconds:.1f}"
            f" + 下降 {self.descend_seconds:.1f} + 后缩 {self.retract_seconds:.1f}）"
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
        safe_lift_quick_margin_mm=_non_negative(
            document.get("safe_lift_quick_margin_mm", defaults.safe_lift_quick_margin_mm),
            "safe_lift_quick_margin_mm",
        ),
        hover_z_mm=_optional_positive(document.get("hover_z_mm"), "hover_z_mm"),
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


def _optional_positive(value: object, name: str) -> Optional[float]:
    """正数或 ``null``；``null`` 表示"这一项没配"。"""
    if value is None:
        return None
    return _positive(value, name)

"""Load the editable hardware settings used by the robot-arm controller.

这里既读 ``motion``（速度、加速度、脉冲上限）和 ``axes``（驱动器地址、传动常数），
也读**标定结果**：每轴的软行程范围和抓取前的安全位。

``limits`` 与 ``safe_pose`` 都是可选的。缺失等于"还没标定"，此时
:func:`modules.arm.drive.kinematics.check_reachable` 不做检查——**没标定就等于没有软件
行程保护**，调用方必须清楚这一点，所以这里用 ``None`` 如实表达，而不是塞一个假的默认范围。

``safe_pose`` 用的是绝对圆柱坐标，看起来和"只有相对位置模式"矛盾，其实不矛盾：只要有一
个绝对参考（回零，或相机测量），同一个物理姿态永远对应同一组 ``(θ, r, h)``，绝对安全位
就是可复现的。

``initial_motor_degrees`` 也是可选的，缺省等于"程序启动那一刻三根轴都在原点上"。它描述的
是**状态**（此刻轴在哪），不是传动常数：换了爪子、重新对齿或重装电机之后，某根轴的机械
初始位可能离原点一整个角度（例如伸缩轴停在 270° = 2400 脉冲）。单位与
``move_axis_to_motor_degrees`` 相同（``3200 脉冲 = 360°``），在这里如实声明它。回零 /
绝对归零会重新定义原点，这条声明随之失效（见 :meth:`ArmConfig.initial_motor_pulses`）。

## 回零（``homing``）

``homing`` 也是可选的。它的语义是「三个轴各自回到驱动器里存的那个零点」，所以**零点必须
在安全姿态上设定**（先用 ``O_Set → Set O`` 或 ``0x93`` 命令把当前姿态设成单圈零点）：
这样回零结束时机械臂就停在 ``safe_pose``，不需要再额外走一段。

单圈回零（``nearest`` / ``direction``）的零点建立在那颗**单圈**绝对编码器上，一圈只有一个。
所以只有当轴在断电时距零点不到**半个电机圈**时，回零才会落到正确的物理位置。
:meth:`ArmConfig.homing_warnings` 会把这个条件与 ``limits`` 对一遍并给出提示。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

from modules.arm.drive.kinematics import (
    CylindricalPose,
    TravelLimit,
    motor_pulses_for_degrees,
)
from modules.motor.config import DEFAULT_HARDWARE_PATH as DEFAULT_BUS_HARDWARE_PATH
from modules.motor.config import load_pulses_per_revolution
from modules.motor.drive.emm42 import Emm42Driver, HomeMode

#: 运动调参（rpm / acceleration / 脉冲上限）：手改。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

#: 硬件事实（轴地址、减速比、导程、行程、安全位、回零方式）：换机器/换接线时改。
DEFAULT_HARDWARE_PATH = Path(__file__).resolve().parent / "hardware.json"
AXIS_NAMES = ("base", "lift", "extension")

#: ``limits`` / ``safe_pose`` 里三个量的名字，对应 ``CylindricalPose`` 的字段。
POSE_FIELDS = ("base_degrees", "radial_mm", "height_mm")

#: 回零模式的字符串名（配置文件里用它，比裸数字可读）。
HOMING_MODES = {
    "nearest": HomeMode.SINGLE_TURN_NEAREST,
    "direction": HomeMode.SINGLE_TURN_DIRECTION,
    "collision": HomeMode.MULTI_TURN_COLLISION,
    "limit_switch": HomeMode.MULTI_TURN_LIMIT_SWITCH,
}

#: 单圈回零模式：零点建立在一圈只有一个的编码器读数上，所以行程超过一个电机圈就不唯一。
SINGLE_TURN_MODES = frozenset({HomeMode.SINGLE_TURN_NEAREST, HomeMode.SINGLE_TURN_DIRECTION})


@dataclass(frozen=True)
class HomingConfig:
    """回零触发方式与等待时长。

    ``expected_seconds`` 是**保守等待时间**：手册里回零本身有超时和失败标志位，但机械臂的
    5/6/7 号驱动器是 ``Response=None``，回零结束不会回任何包，上位机只能等一个够长的时长。
    等停稳后可以用 ``0x36`` 读位置核对；把驱动器菜单的 ``Response`` 改成 ``Reached`` /
    ``Both`` 并解析到位回包 ``ID FD 9F 6B`` 之后，这里就能换成真正的到位等待。
    """

    mode: HomeMode
    expected_seconds: float = 12.0
    settle_seconds: float = 0.5

    @property
    def is_single_turn(self) -> bool:
        return self.mode in SINGLE_TURN_MODES

    @property
    def wait_seconds(self) -> float:
        return self.expected_seconds + self.settle_seconds


@dataclass(frozen=True)
class AxisConfig:
    """One physical axis: its driver address and its transmission constant.

    - ``reduction_ratio``（仅底座）：电机转几圈输出轴转一圈；
    - ``mm_per_motor_rev``（仅升降/伸缩）：电机转一圈走多少毫米，即丝杠导程与减速比之商。
    """

    motor_id: int
    reduction_ratio: Optional[int] = None
    mm_per_motor_rev: Optional[float] = None
    #: 这台驱动器的 ``Response`` 菜单是否会回 4 字节 ACK。``False`` 就是"发完不管"。
    #: 它是**每台驱动器的菜单状态**，不是协议事实，所以必须在 hardware.json 里如实声明；
    #: 这个默认值只服务于手工构造的夹具，加载器在没有该键时会直接报错。
    expect_ack: bool = False
    #: 伸缩轴的实测半径参考，固定在电机坐标系内，不随程序启动位置变化。
    radius_reference_motor_degrees: Optional[float] = None
    radius_reference_mm: Optional[float] = None


@dataclass(frozen=True)
class ArmConfig:
    """Validated arm settings shared by the controller and hardware-test CLI."""

    pulses_per_revolution: int
    rpm: int
    acceleration: int
    default_pulses: int
    max_pulses: int
    axes: Mapping[str, AxisConfig]
    base_limit: Optional[TravelLimit] = None
    radial_limit: Optional[TravelLimit] = None
    height_limit: Optional[TravelLimit] = None
    safe_pose: Optional[CylindricalPose] = None
    homing: Optional[HomingConfig] = None
    #: 程序启动那一刻每根轴相对**原点**的电机角度；没写的轴按 0。
    #: 见 :meth:`initial_motor_pulses`：它是状态声明，不是传动常数。
    initial_motor_degrees: Mapping[str, float] = field(default_factory=dict)

    def motor_id(self, axis_name: str) -> int:
        """Return the configured Emm42 address for a named axis."""
        return self.axes[axis_name].motor_id

    @property
    def base_reduction(self) -> int:
        """Return the confirmed motor-turns/output-turn ratio of the base axis."""
        reduction_ratio = self.axes["base"].reduction_ratio
        if reduction_ratio is None:
            raise ValueError("axes.base.reduction_ratio is required")
        return reduction_ratio

    def base_degrees_per_motor_rev(self) -> float:
        """Return how many output degrees the base turns per motor revolution."""
        return 360.0 / self.base_reduction

    def require_homing(self) -> HomingConfig:
        """Return the homing settings or explain what is missing."""
        if self.homing is None:
            raise ValueError(
                "homing 还没配置；要么在 config.json 里加 homing 块，"
                "要么让位姿来源一直走相机测量（见 modules/grasp/docs/抓取坐标链路.md）"
            )
        return self.homing

    def homing_warnings(self) -> Tuple[str, ...]:
        """Return configuration combinations that make homing unreliable.

        两条独立的要求：

        1. **``safe_pose`` 必须有**——对**所有**回零模式都成立。回零只是让三个轴各自回到
           驱动器里存的零点，而那个姿态在圆柱坐标里是多少必须用相机测出来；没有它，
           抓取流程回完零仍然不知道自己站在哪。
        2. **单圈模式下行必须在一圈以内**——单圈零点建立在一圈只有一个的编码器读数上，
           行程一旦超过一个电机圈，同一个零点就对应不止一个物理位置。

        在打开串口之前就把问题说出来，而不是等回零落到错误的圈上。
        """
        homing = self.homing
        if homing is None:
            return ()
        problems = []
        if self.safe_pose is None:
            problems.append(
                "homing 已配置但 safe_pose 没填：回零只能让三个轴各自回到零点，"
                "而零点姿态在圆柱坐标里是多少必须用相机测出来填进 safe_pose"
            )
        if not homing.is_single_turn:
            return tuple(problems)
        span = self.base_degrees_per_motor_rev()
        if self.base_limit is None:
            problems.append(
                f"homing.mode 是单圈回零，但 limits.base_degrees 没填："
                f"无法确认底座行程是否在 {span:.1f}°（一个电机圈）以内"
            )
        elif self.base_limit.maximum - self.base_limit.minimum > span + 1e-9:
            problems.append(
                f"homing.mode 是单圈回零，而底座行程 "
                f"{self.base_limit.maximum - self.base_limit.minimum:.1f}° 超过一个电机圈 "
                f"{span:.1f}°：回零可能落到错误的圈上。"
                "要么把行程压进一圈，要么改用多圈有限位开关回零，"
                "要么严格保证每次断电前停在零点附近并用相机校验"
            )
        return tuple(problems)

    def mm_per_motor_rev(self, axis_name: str) -> Optional[float]:
        """Return the linear transmission constant, or ``None`` when uncalibrated."""
        return self.axes[axis_name].mm_per_motor_rev

    def require_mm_per_motor_rev(self, axis_name: str) -> float:
        """Return the linear constant or explain which experiment calibrates it."""
        value = self.axes[axis_name].mm_per_motor_rev
        if value is None:
            raise ValueError(
                f"axes.{axis_name}.mm_per_motor_rev 还没标定；"
                "先用实验 B/C 测出电机一圈走多少毫米（见 modules/grasp/docs/抓取坐标链路.md）"
            )
        return value

    def extension_radius_mm(self, pulses: int) -> float:
        """由当前伸缩脉冲和实测参考求夹持中心到底座转轴的水平距离。"""
        axis = self.axes["extension"]
        reference_degrees = axis.radius_reference_motor_degrees
        reference_mm = axis.radius_reference_mm
        if reference_degrees is None or reference_mm is None:
            raise ValueError("axes.extension 缺少 radius_reference 半径标定")
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("extension pulses must be an integer")
        lead = self.require_mm_per_motor_rev("extension")
        # 启程读数只声明开机位置；几何参考始终使用标定时的电机坐标。
        degrees = pulses * 360.0 / self.pulses_per_revolution
        radius = reference_mm + (degrees - reference_degrees) * lead / 360.0
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(
                f"extension radius must be finite and positive, got {radius!r}"
            )
        return radius

    def require_safe_pose(self) -> CylindricalPose:
        """Return the configured safe pose or explain what is missing."""
        if self.safe_pose is None:
            raise ValueError(
                "safe_pose 还没配置；抓取流程必须先有一个已标定的安全位"
                "（见 modules/grasp/docs/抓取坐标链路.md）"
            )
        return self.safe_pose

    # ------------------------------------------------------- 启程读数（初始电机角）
    def initial_motor_pulses(self, axis_name: str) -> int:
        """该轴在**程序启动那一刻**相对原点的电机角，换算成带符号脉冲；没配就是 0。

        给 :class:`modules.arm.arm_state.MotorTurnsState` 用的启程读数。它是"此刻轴在哪"
        的**声明**（例如换爪子后伸缩停在 270° = 2400 脉冲），不是传动常数、也不是动作——
        声明错了，这条路上所有绝对目标就会整体偏一个常数。回零 / 绝对归零会重新定义原点。
        """
        if axis_name not in self.axes:
            raise KeyError(
                f"{axis_name!r} 不是已知的轴；已知的是 {sorted(self.axes)}"
            )
        return motor_pulses_for_degrees(
            self.initial_motor_degrees.get(axis_name, 0.0),
            pulses_per_revolution=self.pulses_per_revolution,
        )

    # ------------------------------------------------------- 绝对参考的唯一性
    def motor_turn_span(self, axis_name: str) -> Optional[float]:
        """一个**电机圈**对应这个轴的多少物理量（度 / 毫米）。

        底座是 ``360 / reduction_ratio`` 度；直线轴就是 ``mm_per_motor_rev`` 毫米。
        ``None`` 表示这个轴的标定还没填，无法判断单圈回零是否唯一。
        """
        if axis_name == "base":
            try:
                return self.base_degrees_per_motor_rev()
            except ValueError:
                return None
        axes = getattr(self, "axes", None)
        if not axes or axis_name not in axes:
            return None
        return axes[axis_name].mm_per_motor_rev

    def axis_travel_span(self, axis_name: str) -> Optional[TravelLimit]:
        """该轴的实测行程；``None`` 表示还没标定。"""
        return {
            "base": self.base_limit,
            "lift": self.height_limit,
            "extension": self.radial_limit,
        }.get(axis_name)

    def travel_motor_degrees(self, axis_name: str) -> Optional[float]:
        """该轴的总行程折成**电机角度**；缺标定时返回 ``None``。

        与 :meth:`motor_turn_span` 互为倒数关系：那个说"一个电机圈值多少物理量"，
        这个说"整段行程值多少电机角度"。绝对值大于它的初始电机角不可能对应真实位置。
        """
        limit = self.axis_travel_span(axis_name)
        if limit is None:
            return None
        span = limit.maximum - limit.minimum
        if axis_name == "base":
            try:
                reduction = self.base_reduction
            except ValueError:
                return None
            return abs(span) * reduction
        per_revolution = self.mm_per_motor_rev(axis_name)
        if per_revolution is None or per_revolution <= 0:
            return None
        return abs(span) / per_revolution * 360.0

    def initial_state_problems(self) -> Tuple[str, ...]:
        """启程读数与行程是否自洽；空元组表示可以这么声明。

        只做**上界**检查：原点落在行程里的哪儿并不由这份配置给出（那是 ``safe_pose`` 与
        现场核对的事），所以这里只能要求 ``|初始电机角|`` 不超过该轴总行程。缺行程标定的
        轴跳过（与 ``limits`` 缺失等于没有软件保护的既有约定一致）。
        """
        labels = {"base": "底座", "lift": "升降", "extension": "伸缩"}
        problems = []
        for axis_name, degrees in sorted(self.initial_motor_degrees.items()):
            span = self.travel_motor_degrees(axis_name)
            if span is None:
                continue
            if abs(degrees) > span + 1e-9:
                problems.append(
                    f"{labels[axis_name]}（{axis_name}）的初始电机角 {degrees:.1f}° "
                    f"超过该轴总行程 {span:.1f}°：这个启程读数对应不到任何真实位置"
                )
        return tuple(problems)

    def homing_reference_problems(self) -> Tuple[str, ...]:
        """建立**绝对参考**之前必须满足的条件；空元组表示可以建立。

        与 :meth:`homing_warnings` 的区别：那个是"提示"（缺 safe_pose、行程超一圈），
        这个是"建立绝对参考的硬前提"。上层在**第一条回零帧之前**调用它，
        不满足就明确拒绝，而不是回完零拿一个不知道是哪一圈的位姿继续定位。

        单圈回零（``nearest`` / ``direction``）的条件比"行程小于一圈"更严格：

        1. 行程端点相对参考位（``safe_pose``）的距离，两端都必须**严格小于半个电机圈**；
        2. 恰好等于半圈也拒绝——等距而方向未明确时无法判定会落到哪一边；
        3. 三个轴都要检查，不能只看底座。

        例：40 mm/圈、参考 0、行程 ``[0, 30]``——虽然不足一圈，但从 30 就近回零会走到
        40 而不是 0，所以必须预检失败。
        """
        homing = self.homing
        if homing is None:
            return ("homing 还没配置：无法建立绝对参考（不知道驱动器会回到哪个物理姿态）",)
        safe_pose = self.safe_pose
        if safe_pose is None:
            return (
                "safe_pose 还没配置：回零只能让三个轴各自回到驱动器零点，"
                "而零点姿态处的夹持点坐标必须用相机测出来填进 safe_pose",
            )

        problems = []
        reference = {
            "base": safe_pose.base_degrees,
            "lift": safe_pose.height_mm,
            "extension": safe_pose.radial_mm,
        }
        labels = {"base": "底座", "lift": "升降", "extension": "伸缩"}
        units = {"base": "°", "lift": " mm", "extension": " mm"}

        for axis_name in AXIS_NAMES:
            label, unit = labels[axis_name], units[axis_name]
            limit = self.axis_travel_span(axis_name)
            if limit is None:
                problems.append(
                    f"{label}（{axis_name}）的行程还没标定（limits），"
                    "无法确认回零会落到哪一圈"
                )
                continue
            if not homing.is_single_turn:
                # 多圈模式：不靠单圈编码器，但仍要求参考位落在实测行程内，
                # 否则"回到参考位"这件事本身没有意义。
                if not limit.contains(reference[axis_name]):
                    problems.append(
                        f"{label}的 safe_pose {reference[axis_name]:.3f}{unit} 不在行程 "
                        f"{limit.describe()}{unit} 内，回零参考无法成立"
                    )
                continue
            half_turn = self.motor_turn_span(axis_name)
            if half_turn is None:
                problems.append(
                    f"{label}（{axis_name}）缺少转换常数，无法判断单圈回零是否唯一"
                )
                continue
            half_turn = abs(half_turn) / 2.0
            for name, endpoint in (("最小", limit.minimum), ("最大", limit.maximum)):
                distance = abs(endpoint - reference[axis_name])
                if distance >= half_turn:
                    problems.append(
                        f"单圈回零下{label}行程{name}端 {endpoint:.3f}{unit} 距参考位 "
                        f"{reference[axis_name]:.3f}{unit} 有 {distance:.3f}{unit}，"
                        f"已达到或超过半个电机圈（{half_turn:.3f}{unit}）："
                        "从那里就近回零可能落到相邻的圈上，绝对参考不唯一"
                    )
        return tuple(problems)


def load_arm_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    hardware_path: Path = DEFAULT_HARDWARE_PATH,
    bus_hardware_path: Optional[Path] = None,
) -> ArmConfig:
    """Read **tuning** from ``config.json`` and **hardware** from ``hardware.json``.

    - ``rpm`` / ``acceleration`` / ``default_pulses`` / ``max_pulses`` 是调参，留在
      ``config.json`` 的 ``motion`` 段；
    - 轴地址、减速比、导程、启程读数（`initial_motor_degrees`）是**硬件事实**，全在
      ``hardware.json``；``limits`` / ``safe_pose`` / ``homing`` 只剩加载校验，
      代码不再用它们做运动决策（见 ``modules/arm/docs/机械臂Emm42三轴控制说明.md``）；
    - 脉冲/圈是驱动器细分，只有一份，读 ``modules/motor/hardware.json``（``bus`` 段）。

    返回值仍然是**合并后的** :class:`ArmConfig` 快照：``runtime`` /
    测试都不需要知道它来自两个文件，只有写配置的人需要。

    任何一项缺失都报出具体的**文件与键名**，不退回代码默认值——默认值会让"漏写"看起来
    像"跑通了"。
    """
    tuning = _read_json(config_path)
    hardware = _read_json(hardware_path)
    pulses_per_revolution = load_pulses_per_revolution(
        DEFAULT_BUS_HARDWARE_PATH if bus_hardware_path is None else bus_hardware_path
    )

    motion = _mapping(tuning.get("motion"), "motion")
    axes_document = _mapping(hardware.get("axes"), "axes")
    if set(axes_document) != set(AXIS_NAMES):
        raise ValueError(f"axes must define exactly: {', '.join(AXIS_NAMES)}")

    limits = hardware.get("limits")
    limits = {} if limits is None else _mapping(limits, "limits")

    config = ArmConfig(
        pulses_per_revolution=pulses_per_revolution,
        rpm=_integer(motion.get("rpm"), "modules.motor.drive.motion.rpm", minimum=0, maximum=Emm42Driver.MAX_RPM),
        acceleration=_integer(
            motion.get("acceleration"), "modules.motor.drive.motion.acceleration", minimum=0, maximum=255),
        default_pulses=_integer(
            motion.get("default_pulses"),
            "modules.motor.drive.motion.default_pulses",
            minimum=1,
            maximum=0xFFFFFFFF,
        ),
        max_pulses=_integer(
            motion.get("max_pulses"),
            "modules.motor.drive.motion.max_pulses",
            minimum=1,
            maximum=0xFFFFFFFF,
        ),
        axes={
            axis_name: _axis_config(
                _mapping(axes_document[axis_name], f"axes.{axis_name}"),
                axis_name,
            )
            for axis_name in AXIS_NAMES
        },
        base_limit=_optional_limit(limits, "base_degrees"),
        radial_limit=_optional_limit(limits, "radial_mm"),
        height_limit=_optional_limit(limits, "height_mm"),
        safe_pose=_optional_safe_pose(hardware.get("safe_pose")),
        homing=_optional_homing(hardware.get("homing")),
        initial_motor_degrees=_initial_motor_degrees(
            hardware.get("initial_motor_degrees")
        ),
    )
    if config.default_pulses > config.max_pulses:
        raise ValueError("modules.motor.drive.motion.default_pulses must not exceed modules.motor.drive.motion.max_pulses")
    if len({axis.motor_id for axis in config.axes.values()}) != len(AXIS_NAMES):
        raise ValueError("axes motor_id values must be unique")
    problems = config.initial_state_problems()
    if problems:
        raise ValueError(
            "initial_motor_degrees 与行程不自洽：" + "；".join(problems)
        )
    return config


def _read_json(path: Path) -> Mapping[str, object]:
    """读一个配置文件并保证顶层是对象；错误信息带上文件名。"""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取配置：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"配置不是合法 JSON：{path}") from error
    return _mapping(document, str(path))


def _axis_config(document: Mapping[str, object], axis_name: str) -> AxisConfig:
    reduction_ratio = document.get("reduction_ratio")
    mm_per_motor_rev = document.get("mm_per_motor_rev")
    reference_degrees = document.get("radius_reference_motor_degrees")
    reference_mm = document.get("radius_reference_mm")
    if reference_degrees is not None or reference_mm is not None:
        if axis_name != "extension":
            raise ValueError(f"axes.{axis_name}.radius_reference is not supported")
        reference_degrees = _number(
            reference_degrees, "axes.extension.radius_reference_motor_degrees"
        )
        reference_mm = _number(reference_mm, "axes.extension.radius_reference_mm")
        if (
            not math.isfinite(reference_degrees)
            or not math.isfinite(reference_mm)
            or reference_mm <= 0
        ):
            raise ValueError(
                "axes.extension.radius_reference must be finite with positive radius"
            )
    if axis_name == "base":
        reduction_ratio = _integer(
            reduction_ratio,
            "axes.base.reduction_ratio",
            minimum=1,
            maximum=0xFFFFFFFF,
        )
        if mm_per_motor_rev is not None:
            raise ValueError("axes.base.mm_per_motor_rev is not supported")
    else:
        if reduction_ratio is not None:
            raise ValueError(f"axes.{axis_name}.reduction_ratio is not supported")
        mm_per_motor_rev = _optional_positive_number(
            mm_per_motor_rev, f"axes.{axis_name}.mm_per_motor_rev"
        )
    return AxisConfig(
        motor_id=_integer(document.get("motor_id"), f"axes.{axis_name}.motor_id", minimum=1, maximum=255),
        reduction_ratio=reduction_ratio,
        mm_per_motor_rev=mm_per_motor_rev,
        radius_reference_motor_degrees=reference_degrees,
        radius_reference_mm=reference_mm,
        expect_ack=_boolean(
            document.get("expect_ack", False), f"axes.{axis_name}.expect_ack"
        ),
    )


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是 true 或 false，实际 {value!r}")
    return value


def _optional_homing(value: object) -> Optional[HomingConfig]:
    """Read the ``homing`` block; ``mode`` accepts a readable name or the raw enum value."""
    if value is None:
        return None
    document = _mapping(value, "homing")
    raw_mode = document.get("mode")
    if isinstance(raw_mode, str):
        key = raw_mode.strip().lower()
        if key not in HOMING_MODES:
            raise ValueError(
                f"homing.mode must be one of {', '.join(sorted(HOMING_MODES))} "
                f"(or the raw 0..3 enum value), got {raw_mode!r}"
            )
        mode = HOMING_MODES[key]
    else:
        mode = HomeMode(_integer(raw_mode, "homing.mode", minimum=0, maximum=3))
    return HomingConfig(
        mode=mode,
        expected_seconds=_non_negative_number(
            document.get("expected_seconds", 12.0), "homing.expected_seconds"
        ),
        settle_seconds=_non_negative_number(
            document.get("settle_seconds", 0.5), "homing.settle_seconds"
        ),
    )


def _initial_motor_degrees(value: object) -> Mapping[str, float]:
    """Read the optional ``initial_motor_degrees`` block (轴名 -> 电机角度)。

    没写这个块就等于"三根轴都在原点上"，所以返回**空映射**而不是 ``None``：调用方不必
    再判空，缺的轴由 :meth:`ArmConfig.initial_motor_pulses` 按 0 处理。
    """
    if value is None:
        return {}
    document = _mapping(value, "initial_motor_degrees")
    unknown = sorted(set(document) - set(AXIS_NAMES))
    if unknown:
        raise ValueError(
            "initial_motor_degrees 只能给这些轴："
            + "、".join(AXIS_NAMES)
            + "；多出来的是 "
            + "、".join(unknown)
        )
    return {
        axis_name: _number(item, f"initial_motor_degrees.{axis_name}")
        for axis_name, item in document.items()
    }


def _optional_limit(document: Mapping[str, object], field: str) -> Optional[TravelLimit]:
    value = document.get(field)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"limits.{field} must be a two-element [min, max] list or null")
    return TravelLimit(
        minimum=_number(value[0], f"limits.{field}[0]"),
        maximum=_number(value[1], f"limits.{field}[1]"),
    )


def _optional_safe_pose(value: object) -> Optional[CylindricalPose]:
    if value is None:
        return None
    document = _mapping(value, "safe_pose")
    fields = {field: document.get(field) for field in POSE_FIELDS}
    missing = [field for field, item in fields.items() if item is None]
    if len(missing) == len(POSE_FIELDS):
        return None
    if missing:
        raise ValueError(
            "safe_pose 的 " + " / ".join(POSE_FIELDS) + " 必须同时给出或同时为 null"
        )
    return CylindricalPose(
        base_degrees=_number(fields["base_degrees"], "safe_pose.base_degrees"),
        radial_mm=_number(fields["radial_mm"], "safe_pose.radial_mm"),
        height_mm=_number(fields["height_mm"], "safe_pose.height_mm"),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _non_negative_number(value: object, name: str) -> float:
    number = _number(value, name)
    if number < 0:
        raise ValueError(f"{name} must not be negative")
    return number


def _optional_positive_number(value: object, name: str) -> Optional[float]:
    if value is None:
        return None
    number = _number(value, name)
    if not number > 0:
        raise ValueError(f"{name} must be positive when given")
    return number

"""机械臂上层：统一生命周期、相对脉冲原语与两个**绝对坐标**入口。

真实定义就在这里（原 ``modules/arm/drive/robot_arm.py`` 已退出）。驱动层
(:mod:`modules.motor.drive.emm42`) 只负责协议帧，本模块负责"爪子该去哪"。

## 生命周期

```python
with RobotArm.open() as arm:          # 读配置 + 开（或借）一条串口
    arm.move_up(pulses=800)           # 相对脉冲，含义与单位不变
```

上层不再自己 ``initialize()`` / 使能 / 收尾：``open`` 已经做过，``close`` 或 ``with``
负责收尾。``stop`` 只是"停住执行器"，不关连接、不额外失能。

## 两套坐标，同一个夹持点

- :meth:`RobotArm.move_cylindrical`：``base_degrees`` / ``radial_mm`` / ``height_mm``；
- :meth:`RobotArm.move_to`：基座 XYZ 毫米，逆解后进入同一套规划与执行。

两者都是**绝对位置**（不是本次增量），都定位**爪口中心（实际夹持点）**，参考系是：

- ``height_mm`` / ``z_mm``：台面为 0，向上为正；
- XY 原点：基座旋转轴与台面参考平面的交点（不是摄像头光心，也不是"回零后所在位置"）；
- ``base_degrees``：底座实际角位置，``+X`` 是标定的 0°，``+Y`` 是沿 ``rotate_cw``
  正方向转 90° 后的方向；角度符号沿用 CW 为正。

绝对定位要求"位姿已知"（显式回零或用已标定测量写入），并且要在**发第一条运动帧之前**
完成全部预检——目标、每一段中间位姿、每一帧的整数脉冲都在
:mod:`modules.arm.motion_plan` 里算完。

## 相对原语仍然直接可用

``move_up`` / ``move_down`` / ``move_forward`` / ``move_backward`` / ``rotate_cw`` /
``rotate_ccw`` / ``move_base_degrees`` / ``move_lift_mm`` / ``move_radial_mm`` 的含义、
单位与参数**都不变**。它们不自行绕安全廊道，也不改变调用者指定的动作；``wait=True``
时按实际下发的整数脉冲推进位姿，``wait=False`` 会让位姿变成未知。

``move_cylindrical_by`` / ``move_by`` 是同一条路上的**一次多轴组合**：各走一个增量
（度 / 毫米），同样不需要位姿已知——极坐标的每个轴本来就能相对走。唯一的例外是
``move_by`` 的 x/y：X-Y 到 (底座角, 半径) 的换算随所在方位变化，所以那一对增量需要
当前位姿才能分解，详见那里的说明。

## 一次让多根轴动：``move_axes``

``move_cylindrical_by`` 是**串行**的（抬升 → 伸缩 → 底座，每段等完再发下一段）；
:meth:`RobotArm.move_axes` 是**并行**的那一个，默认走 Emm42 的多机同步三步法：

```
逐轴发 0xFD（多机同步标志 = 01）缓存住 → 帧间留 DEFAULT_INTER_FRAME_SECONDS
→ 最后一条广播 00 FF 66 6B 让它们一起起步
```

于是三根轴是**真的同时起步**（轮子早就是这么走的，见 ``MecanumDrive``），
而不是"发完第一条再发第二条"那种差几十毫秒的伪并行。代价是它们**各自到位**：
谁走的路短谁先停。想让它们同时停要按轴改 rpm（联动/插补），那是另一件事，
本模块刻意不做——``motion_plan`` 也写着"不增加多轴插补"。

并行**不是无条件的**：底座要转的角度超过
:data:`modules.motor.drive.motion.DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES` 时不许和
伸缩并行（伸出状态下横扫会把整条臂扫过工作区），要显式改用
:data:`AXES_SERIAL`——它按 抬升 → 伸缩 → 底座 → 下降 的安全顺序串行走完。

## ``wait=False`` 返回票据，不返回报告

报告的含义是"这次动作的结论"，而 ``wait=False`` 的时候**还没有结论**。所以从
:meth:`RobotArm.move_axes` 起，所有 ``wait=False`` 的动作方法返回
:class:`modules.motion.MotionTicket`（票据）：

```python
ticket = arm.move_up(pulses=800, wait=False)   # 帧已经发出去了
ticket.wait()                                   # 到点 → 推进位姿 → 返回 MotionReport
```

票据的 ``estimated_seconds`` 就是过去那份报告里同名字段，"还要等多久"因此在这条路上
一样看得见；而位姿从"永远未知"变成"等完就对上"——**不调用 ``wait`` 的话和以前完全一样**。
多轴的那一份票据会把三根轴的位姿一起推进，并返回一个报告元组。

## 等一次动作走完：两种策略

``wait=True`` 必须知道"这次动作什么时候结束"，有两种做法：

| ``wait_mode`` | 做法 | 代价 |
|---|---|---|
| :data:`WAIT_ESTIMATE`（默认） | 按公式估算耗时后 ``sleep`` | 估算；**堵转了照睡不误** |
| :data:`WAIT_IN_POSITION` | 轮询 ``0x3A`` 到位位 | 每次动作多几次串口往返 |

默认仍是 :data:`WAIT_ESTIMATE`：它已经在真机上跑通，而轮询要在真机上确认
"读命令在 ``Response=None`` 的 5/6/7 号轴上确实回包"之后才值得打开。
**注意轮询是"发完帧才开始读"**：一旦 ``0x3A`` 读不到或帧不对，本方法会抛异常，
而那时动作帧已经发出去了——轴会继续走完，只是上层拿不到结果。所以先在一根轴上
单步验证过再切。
切换方式：

```python
with RobotArm.open(wait_mode=WAIT_IN_POSITION) as arm:
    report = arm.move_up(pulses=800)
    if report.motor_status is not None and report.motor_status.stalled:
        ...   # 轴撞上了东西——这是估算模式永远给不出的信息
```

轮询模式还会把**实际等待时长**和**最后一次读到的状态**放进 :class:`MotionReport`，
所以"估算偏了多少"和"是不是撞上了"都能事后看到，不必再猜。

## 两个状态：爪子在哪，电机转了多少

- ``pose_state``（:class:`modules.arm.arm_state.ArmPose`）：爪口在圆柱坐标里的位置。
  依赖标定（``mm_per_motor_rev`` 等），缺标定就整体变未知，而且底座角被归一化。
- ``turns_state``（:class:`modules.arm.arm_state.MotorTurnsState`）：三轴的**命令脉冲
  里程表**。一个标定都不需要，所以永远有值；底座角**不归一**。

后者是为"底座能转整圈"这件事存在的：``base_degrees`` 归一到 ``(-180, 180]``，连续同向
转十圈和原地没动看起来一模一样，而线缆会绕死。里程表不归一，所以看得见——但它是**下发
的脉冲**，丢步和打滑不会体现，要和实测对账就读 ``0x36``。

它记的是**相对原点**的读数，而"程序启动那一刻轴在哪"由配置的 ``initial_motor_degrees``
声明（缺省表示三根轴都在原点上）。换爪子、重新对齿或重装电机之后某根轴的机械初始位不再
落在原点上时（例如伸缩停在 270° = 2400 脉冲），就在那里如实声明，而不是让软件继续以为
它在原点；回零 / 绝对归零重新定义原点之后，读数照旧回到 0。

两张账都只在 ``_send`` 一处推进（回零/绝对归位另把原点清零），所以方向原语、单位制方法
和绝对规划帧走的是同一条路，既不会漏记也不会把同一次位移记两遍。每次
:class:`MotionReport` 都会带上该轴累计到那一刻的读数。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple, Union

from modules.arm.arm_state import ArmPose, MotorTurns, MotorTurnsState
from modules.arm.config import AXIS_NAMES, ArmConfig, load_arm_config
from modules.arm.drive.kinematics import (
    BASE_AXIS,
    EXTENSION_AXIS,
    LIFT_AXIS,
    CylindricalPose,
    base_degrees_for_pulses,
    base_pulses_for_degrees,
    linear_pulses_for_mm,
    mm_for_linear_pulses,
    motor_degrees_for_pulses,
    motor_pulses_for_degrees,
    normalize_degrees,
)
from modules.arm.motion_plan import PlanError, plan_absolute, plan_xyz
from modules.motor.drive.emm42 import Direction, Emm42Driver, MotorStatus, SerialPort
from modules.motor.drive.motion import (
    DEFAULT_INTER_FRAME_SECONDS,
    DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES,
    DEFAULT_SETTLE_SECONDS,
    estimate_move_seconds,
    in_position_timeout_seconds,
)
from modules.motion import MotionTicket

#: 等待策略：按曲线加减速公式估算耗时后 ``sleep``。当前默认，已在真机上跑通。
WAIT_ESTIMATE = "estimate"

#: 等待策略：轮询 ``0x3A`` 到位位。更准，而且是堵转的唯一软件证据；
#: 需要真机确认读命令可用后再切。
WAIT_IN_POSITION = "in_position"

#: 允许的等待策略，见模块文档字符串的"等一次动作走完"一节。
WAIT_MODES = (WAIT_ESTIMATE, WAIT_IN_POSITION)

#: 多轴模式：各轴先缓存、再一条广播让它们**同时起步**（默认）。
AXES_SAME_START = "same_start"

#: 多轴模式：按安全顺序**逐轴**走完再走下一轴（与 :meth:`RobotArm.move_cylindrical_by` 同序）。
AXES_SERIAL = "serial"

#: 允许的多轴模式，见模块文档字符串的"一次让多根轴动"一节。
AXES_MODES = (AXES_SAME_START, AXES_SERIAL)


def _require_wait_mode(value: str) -> str:
    if value not in WAIT_MODES:
        raise ValueError(f"wait_mode must be one of {WAIT_MODES}, got {value!r}")
    return value


def _require_axes_mode(value: str) -> str:
    if value not in AXES_MODES:
        raise ValueError(f"mode must be one of {AXES_MODES}, got {value!r}")
    return value


def _serial_order(moves: "List[Tuple[str, int]]") -> "List[Tuple[str, int]]":
    """按``AXES_SERIAL`` 的安全顺序重排：抬升 → 伸缩 → 底座 → 下降。

    与 :meth:`RobotArm.move_cylindrical_by` 同一条规矩：要抬升就先抬，要下降就放到最后
    （免得在低位转动底座或前伸），只走平面时先伸缩、再转底座。
    """
    by_axis = dict(moves)
    ordered: List[Tuple[str, int]] = []
    lift = by_axis.get(LIFT_AXIS, 0)
    if lift > 0:
        ordered.append((LIFT_AXIS, lift))
    radial = by_axis.get(EXTENSION_AXIS, 0)
    if radial != 0:
        ordered.append((EXTENSION_AXIS, radial))
    base = by_axis.get(BASE_AXIS, 0)
    if base != 0:
        ordered.append((BASE_AXIS, base))
    if lift < 0:
        ordered.append((LIFT_AXIS, lift))
    return ordered


@dataclass(frozen=True)
class MotionReport:
    """一次相对动作的命令量；直线轴没有毫米标定时 ``output_degrees`` 为 ``None``。"""

    motor_id: int
    pulses: int
    motor_degrees: float
    output_degrees: Optional[float]
    #: 本次动作的**估算**耗时（秒）；``wait=False`` 时调用方可以自己用它来等。
    estimated_seconds: float = 0.0
    #: 实际等待时长（秒）。估算模式等于 ``estimated_seconds``；轮询模式是实测值。
    waited_seconds: float = 0.0
    #: 轮询到位时最后一次读到的电机状态；估算模式恒为 ``None``
    #: （那条路根本不读驱动器，编不出一个状态来）。
    motor_status: Optional[MotorStatus] = None
    #: 这一帧发完之后，该轴相对**原点**的累计脉冲与角度（命令里程表，不是实测位置）。
    turns: Optional[MotorTurns] = None


#: 相对 XYZ 分解允许的最小半径（毫米）。再小下去切向分量会被 ``1/r`` 放大：1 mm 半径上
#: 一个 1 mm 的 XY 增量就是 57°，那不是"走一点"，是甩大半圈。
_MIN_RADIUS_MM = 1.0


def _relative_number(value: float, name: str) -> float:
    """相对量的入口校验：必须是**有限**数字（``bool`` 不算数字）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数字")
    return number


class RobotArm:
    """控制 5/6/7 号轴的机械臂上层：只发有界的相对位置命令。"""

    BASE_AXIS = BASE_AXIS
    LIFT_AXIS = LIFT_AXIS
    EXTENSION_AXIS = EXTENSION_AXIS

    #: 回零顺序：先缩回、再升降、最后转底座——和运动时的安全廊道是同一条规矩。
    HOME_ORDER = (EXTENSION_AXIS, LIFT_AXIS, BASE_AXIS)

    #: 轴名 -> 配置里的行程区间字段名（行程是按"量"存的，不是按轴）。
    AXIS_LIMIT_FIELDS = {
        BASE_AXIS: "base_limit",
        LIFT_AXIS: "height_limit",
        EXTENSION_AXIS: "radial_limit",
    }

    def __init__(
        self,
        config: ArmConfig,
        *,
        motors: Mapping[str, Emm42Driver],
        bus: Optional[SerialPort] = None,
        owns_bus: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        settle_seconds: float = DEFAULT_SETTLE_SECONDS,
        wait_mode: str = WAIT_ESTIMATE,
        clock: Callable[[], float] = time.monotonic,
        inter_frame_seconds: float = DEFAULT_INTER_FRAME_SECONDS,
        parallel_rotation_limit_pulses: int = DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES,
    ) -> None:
        """只保存**已经装配好的**依赖：本构造器不做任何设备 I/O。

        上层代码用 :meth:`open` 创建；这里显式接收 ``motors`` / ``bus`` 是模块内部的
        装配接缝，供 :meth:`open` 与假设备测试使用，不是给业务调用者的参数。

        :param owns_bus: 这条串口是不是**本实例打开**的。只有自己开的才在 ``close``
            里物理关闭；借来的（同一个 Robot 会话里 arm 与 chassis 共用一条）只失效自身。
        :param wait_mode: ``wait=True`` 时怎么知道动作走完了，
            取值见 :data:`WAIT_MODES`。
        :param clock: 单调时钟；轮询模式下由它和 ``sleep`` 一起判定超时，
            测试注入假实现。
        :param inter_frame_seconds: 连续两条**单帧**命令之间该留的间隔：
            :meth:`move_axes` 的缓存帧，以及使能/失能/停止这类逐轴命令都用它；
            见 :data:`modules.motor.drive.motion.DEFAULT_INTER_FRAME_SECONDS`。
        :param parallel_rotation_limit_pulses: 底座旋转与伸缩可并行的脉冲上限；
            见 :data:`modules.motor.drive.motion.DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES`。
        """
        if inter_frame_seconds < 0:
            raise ValueError("inter_frame_seconds must not be negative")
        if parallel_rotation_limit_pulses < 0:
            raise ValueError("parallel_rotation_limit_pulses must not be negative")
        self.config = config
        self.pulses_per_revolution = config.pulses_per_revolution
        self.settle_seconds = settle_seconds
        self.wait_mode = _require_wait_mode(wait_mode)
        self.inter_frame_seconds = inter_frame_seconds
        self.parallel_rotation_limit_pulses = parallel_rotation_limit_pulses
        self._sleep = sleep
        self._clock = clock
        self._motors = dict(motors)
        self._bus = bus
        self._owns_bus = bool(owns_bus) and bus is not None
        self._closed = False
        #: 爪口中心的位姿；**唯一所有者是这里**，完成回零或测量写入之前是"未知"。
        self.pose_state = ArmPose()
        #: 三轴的命令脉冲里程表；与位姿分开，因为它一个标定都不需要（见 arm_state）。
        #: **启程读数**取自配置的 ``initial_motor_degrees``：程序启动那一刻轴若不在原点上
        #: （换爪子、重新对齿之后就是这种情况），就按那里的声明起算；回零 / 绝对归零会把
        #: 原点重新定义，届时读数照旧清成 0。
        self.turns_state = MotorTurnsState(
            config.pulses_per_revolution,
            AXIS_NAMES,
            initial_pulses={
                axis_name: config.initial_motor_pulses(axis_name)
                for axis_name in AXIS_NAMES
            },
        )

    # ------------------------------------------------------------- 生命周期
    @classmethod
    def open(
        cls,
        *,
        config_path: Optional[Path] = None,
        motor_config_path: Optional[Path] = None,
        bus: Optional[SerialPort] = None,
        owns_bus: Optional[bool] = None,
        wait_mode: str = WAIT_ESTIMATE,
        inter_frame_seconds: Optional[float] = None,
        parallel_rotation_limit_pulses: Optional[int] = None,
    ) -> "RobotArm":
        """按配置开一条步进连接并返回就绪实例。

        :param config_path: arm 主配置；``None`` 用 :data:`modules.arm.config.DEFAULT_CONFIG_PATH`。
        :param motor_config_path: 步进总线配置；``None`` 用 motor 模块的默认配置。
        :param bus: 由调用方**已经打开**的共享连接（``Robot`` 装配 arm+chassis 时用）。
            给了它就借用，不再自己开第二条、也不会在 ``close`` 里关掉它。
        :param owns_bus: 显式声明这条 ``bus`` 归不归本实例；默认按"是不是自己开的"判断。
            传 ``bus`` 时默认 ``False``（借来的）。
        :param wait_mode: ``wait=True`` 时怎么知道动作走完了，
            取值见 :data:`WAIT_MODES`。
        :param inter_frame_seconds: 连续两条单帧命令之间该留的间隔（缓存位置帧、逐轴使能/失能/停止都用它）；
            ``None`` 用 :data:`modules.motor.drive.motion.DEFAULT_INTER_FRAME_SECONDS`。
            这是**USB-TTL 转接板的脾气**，换板子才需要动它。
        :param parallel_rotation_limit_pulses: 底座旋转与伸缩可并行的脉冲上限；
            ``None`` 用 :data:`modules.motor.drive.motion.DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES`。
        """
        from modules.motor.config import DEFAULT_HARDWARE_PATH as motor_default
        from modules.motor.config import load_bus_config

        arm_path = Path(config_path) if config_path is not None else None
        config = load_arm_config(**({} if arm_path is None else {"config_path": arm_path}))
        bus_path = Path(motor_config_path) if motor_config_path is not None else motor_default
        port = load_bus_config(hardware_path=bus_path).port
        tuning = {}
        if inter_frame_seconds is not None:
            tuning["inter_frame_seconds"] = inter_frame_seconds
        if parallel_rotation_limit_pulses is not None:
            tuning["parallel_rotation_limit_pulses"] = parallel_rotation_limit_pulses
        session = cls._from_settings(
            config=config,
            port=port,
            bus=bus,
            owns_bus=owns_bus,
            sleep=time.sleep,
            wait_mode=wait_mode,
            **tuning,
        )
        return session

    @classmethod
    def _from_settings(
        cls,
        *,
        config: ArmConfig,
        port: str,
        bus: Optional[SerialPort] = None,
        owns_bus: Optional[bool] = None,
        sleep: Callable[[float], None] = time.sleep,
        settle_seconds: float = DEFAULT_SETTLE_SECONDS,
        wait_mode: str = WAIT_ESTIMATE,
        clock: Callable[[], float] = time.monotonic,
        inter_frame_seconds: float = DEFAULT_INTER_FRAME_SECONDS,
        parallel_rotation_limit_pulses: int = DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES,
    ) -> "RobotArm":
        """按**已经读好的配置快照**装配：Robot 与独立 ``open`` 共用这一处。

        ``bus`` 为 ``None`` 时自己开一条串口（并声明所有权）；给了就借用。
        构造中途失败时，自己开的串口由本方法释放，外部注入的**不擅自关闭**
        （那会切断别的模块的连接）。
        """
        # 先把纯参数校验做完再碰设备：wait_mode 写错不该以"漏一条打开的串口"收场。
        _require_wait_mode(wait_mode)
        borrowing = bus is not None
        owner = (not borrowing) if owns_bus is None else bool(owns_bus)
        opened = bus
        try:
            if opened is None:
                opened = cls._open_serial(port)
            motors = {
                axis_name: Emm42Driver(
                    port,
                    config.motor_id(axis_name),
                    pulses_per_revolution=config.pulses_per_revolution,
                    # 每台驱动器的 Response 菜单状态，来自 hardware.json 的 axes.<name>.expect_ack。
                    expect_ack=config.axes[axis_name].expect_ack,
                    serial_port=opened,
                )
                for axis_name in AXIS_NAMES
            }
        except BaseException as primary:
            # 构造没返回，只能由构造器自己收尾。借来的串口留给它的所有者。
            if not borrowing and opened is not None:
                try:
                    opened.close()
                except Exception as cleanup:
                    raise primary.with_traceback(primary.__traceback__) from cleanup
            raise
        session = cls(
            config,
            motors=motors,
            bus=opened,
            owns_bus=owner and not borrowing,
            sleep=sleep,
            settle_seconds=settle_seconds,
            wait_mode=wait_mode,
            clock=clock,
            inter_frame_seconds=inter_frame_seconds,
            parallel_rotation_limit_pulses=parallel_rotation_limit_pulses,
        )
        try:
            # 使能属于会话生命周期（对齐旧菜单脚本的 initialize + enable_all），
            # 但不做任何运动：回零、开合爪、抬升都由上层显式调用。
            session.enable_all()
        except BaseException as primary:
            try:
                session.close()
            except Exception as cleanup:
                raise primary.with_traceback(primary.__traceback__) from cleanup
            raise
        return session

    @staticmethod
    def _open_serial(port: str) -> SerialPort:
        from modules.serial_link import open_serial

        return open_serial(port, baudrate=Emm42Driver.BAUDRATE, timeout=0.1)

    @property
    def closed(self) -> bool:
        """会话是否已经关闭；关闭后动作方法统一抛 ``RuntimeError``，不自动重开。"""
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RobotArm 已关闭")

    def close(self) -> None:
        """按所有权释放资源：停 → 失能 → 物理关闭（只关自己开的那条）。

        每项独立尝试：停报错不该让失能与关串口一个都不做。第一项错误不阻断其他项，
        全部试完才报告，错误清单含每个失败阶段、以第一项为 ``__cause__``。
        重复调用不再发任何命令。位姿立即变成未知（连接没了，谁也不知道轴在哪）。
        """
        if self._closed:
            return
        self._closed = True
        self.pose_state.mark_unknown()
        errors = []
        for stage, action in (
            ("stop", self._stop_axes),
            ("disable", self._disable_axes),
        ):
            try:
                action()
            except Exception as error:
                errors.append((stage, error))
        if self._owns_bus and self._bus is not None:
            try:
                self._bus.close()
            except Exception as error:
                errors.append(("bus.close", error))
        if errors:
            details = "; ".join("{}: {}".format(stage, error) for stage, error in errors)
            raise RuntimeError("RobotArm 清理失败：" + details) from errors[0][1]

    def __enter__(self) -> "RobotArm":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """收尾失败不吞掉：没有业务异常就直接抛清理错误，有则把清理错误挂成 cause。"""
        try:
            self.close()
        except Exception as cleanup:
            if exc is not None:
                raise exc.with_traceback(traceback) from cleanup
            raise
        return False

    def stop(self) -> None:
        """停住三个轴；不关连接、不额外失能。

        机械结构上的位置随停止失效（急停之后驱动器可能丢步），所以位姿立即变成未知：
        上层若还要用绝对坐标，必须显式重新回零。
        """
        self._require_open()
        self._stop_axes()
        self.pose_state.mark_unknown()

    def _each_axis(self, action: Callable[["Emm42Driver"], None]) -> None:
        """按地址顺序逐轴发**单帧**命令，帧间留 ``inter_frame_seconds``。

        ``enable`` / ``disable`` / ``stop`` 这类单帧命令原来是一条接一条发出去的。
        这台机器的 USB-TTL 上背靠背发帧会让帧失效（见
        :data:`modules.motor.drive.motion.DEFAULT_INTER_FRAME_SECONDS`）：三轴里
        5 号会回包、6/7 号不回，弄坏的帧只留下一条 ``地址 00 EE 6B``（错误命令）
        没人读走，下一个用总线的人（底盘）就会把这条错误帧当成自己的 ACK。
        """
        for index, axis_name in enumerate(AXIS_NAMES):
            if index:
                self._sleep(self.inter_frame_seconds)
            action(self._motors[axis_name])

    def _stop_axes(self) -> None:
        self._each_axis(lambda motor: motor.stop())

    def _disable_axes(self) -> None:
        self._each_axis(lambda motor: motor.enable(False))

    def enable_all(self) -> None:
        """按地址顺序给三个轴使能扭矩。``open`` 已经做过；需要重新使能时才调。"""
        self._require_open()
        self._each_axis(lambda motor: motor.enable(True))

    def disable_all(self) -> None:
        """释放三个轴的扭矩。位姿随之失效（松开之后轴可以被外力推动）。"""
        self._require_open()
        self._disable_axes()
        self.pose_state.mark_unknown()

    # ------------------------------------------------------------- 位姿
    @property
    def pose(self) -> Optional[CylindricalPose]:
        """夹持点的当前推算位姿；未知时为 ``None``。**只读**，调用者不能自行写值。"""
        return self.pose_state.pose

    @property
    def pose_source(self) -> str:
        """位姿来源：``unknown`` / ``homed`` / ``measured`` / ``dead_reckoned``。"""
        return self.pose_state.source

    def record_measurement(self, pose: CylindricalPose) -> None:
        """写入一次**已标定测量**得到的夹持点位姿（迁移自旧管线的观测消费者）。

        测量必须先通过数值、夹持点与行程校验；相机只给方向角时，调用方必须保证它与
        已知连续角位置对应，或在行程内有唯一等价角——**猜圈数是禁止的**。
        """
        self._require_open()
        if not isinstance(pose, CylindricalPose):
            raise TypeError("record_measurement 需要 CylindricalPose")
        for name, value in (
            ("base_degrees", pose.base_degrees),
            ("radial_mm", pose.radial_mm),
            ("height_mm", pose.height_mm),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数字")
        if pose.radial_mm < 0:
            raise ValueError("radial_mm 不能是负数")
        problems = []
        for name, value, limit, unit in (
            ("底座角", pose.base_degrees, self.config.base_limit, "°"),
            ("半径", pose.radial_mm, self.config.radial_limit, " mm"),
            ("高度", pose.height_mm, self.config.height_limit, " mm"),
        ):
            if limit is not None and not limit.contains(value):
                problems.append(f"{name} {value:.3f}{unit} 超出 {limit.describe()}{unit}")
        if problems:
            raise ValueError("测量位姿不可用：" + "；".join(problems))
        self.pose_state.record_measurement(
            CylindricalPose(
                normalize_degrees(pose.base_degrees), pose.radial_mm, pose.height_mm
            )
        )

    def mark_pose_unknown(self) -> None:
        """主动把位姿打回未知（发帧中断、被外力动过、怀疑丢步时用）。"""
        self.pose_state.mark_unknown()

    # ------------------------------------------------------------- 绝对坐标
    def move_cylindrical(
        self,
        *,
        base_degrees: float,
        radial_mm: float,
        height_mm: float,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
    ) -> CylindricalPose:
        """走到**绝对圆柱目标**（度 / 毫米）；返回按实际下发脉冲推算的位置。

        :raises RuntimeError: 位姿未知（先显式 ``home_all`` 或写入一次测量）。
        :raises PlanError: 缺标定、越界，或角度在行程内没有合法路线。
        :returns: 走完之后的夹持点圆柱位姿——**按实际整数脉冲推算**，不宣称物理到位。
        """
        self._require_open()
        current = self._require_pose()
        target = CylindricalPose(
            normalize_degrees(base_degrees), radial_mm, height_mm
        )
        plan = plan_absolute(config=self.config, current=current, target=target)
        return self._run_plan(plan)

    def move_to(
        self,
        *,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
    ) -> CylindricalPose:
        """走到**绝对基座 XYZ**（毫米）；逆解后进入与圆柱入口同一套规划与执行。

        ``x=y=0`` 时没有确定的方向角：**保留当前角度**（不会隐式转到 0°）。

        :raises RuntimeError: 位姿未知。
        :raises PlanError: 缺标定、越界，或 r=0 不在行程内。
        """
        self._require_open()
        current = self._require_pose()
        plan, _target = plan_xyz(
            config=self.config, current=current, x_mm=x_mm, y_mm=y_mm, z_mm=z_mm
        )
        return self._run_plan(plan)

    def _require_pose(self) -> CylindricalPose:
        if self.pose is None:
            raise RuntimeError(
                "机械臂位姿未知：先显式回零（home_all），"
                "或让已标定的测量写一次位置；不要凭猜测发运动指令"
            )
        return self.pose

    def _run_plan(self, plan) -> CylindricalPose:
        """执行一份**已经在规划阶段定好**的动作帧序列。

        执行阶段不再做任何决策：帧的顺序、脉冲数、落点都是
        :mod:`modules.arm.motion_plan` 算好的。中途失败就停止后续帧、把位姿置为未知，
        并抛出原始失败。
        """
        for frame in plan.moving_segments:
            try:
                self._execute_frame(frame)
            except BaseException:
                # 发帧或等待被打断：位置已经不可信，宁可重新对账。
                self.pose_state.mark_unknown()
                raise
        return plan.final_pose

    def _execute_frame(self, frame) -> None:
        """发一帧并等它走完（按估算耗时），然后把位姿推进到该帧的落点。"""
        signed = frame.pulses
        direction = Direction.CW if signed > 0 else Direction.CCW
        self._send(
            frame.axis,
            direction,
            abs(signed),
            rpm=None,
            acceleration=None,
            wait=True,
        )
        self.pose_state.record_move(frame.resulting_pose)

    # ------------------------------------------------------------- 回零
    def home_axis(self, axis_name: str, *, wait: bool = True) -> float:
        """触发一根轴的单圈回零；返回等了多久。

        回零耗时不确定，而 5/6/7 号驱动器是 ``Response=None``，回零结束不会回任何包，
        所以 ``wait=True`` 只能等一个保守时长（``homing.expected_seconds + settle_seconds``）。

        **单轴回零不建立三轴参考**：位姿立即变成未知。
        """
        self._require_open()
        homing = self.config.require_homing()
        self.pose_state.mark_unknown()
        self._motors[axis_name].home(homing.mode)
        if wait:
            self._sleep(homing.wait_seconds)
            # 回零让这根轴停在驱动器零点上，所以原点就定义在这里。
            # ``wait=False`` 不重置：不知道何时到位，调用方确认后自己调 mark_origin。
            self.turns_state.mark_origin(axis_name)
        return homing.wait_seconds

    def home_all(self, *, wait: bool = True) -> Tuple[float, ...]:
        """按 伸缩 → 升降 → 底座 的顺序回零，成功等待后建立绝对参考。

        建立参考的门槛（在**第一条回零帧之前**检查，见
        :meth:`modules.arm.config.ArmConfig.homing_reference_problems`）：``homing``、
        ``safe_pose``、三轴行程与转换常数齐全，且单圈回零下行程两端距参考位都严格小于
        半个电机圈。缺少判断所需标定时**明确拒绝**，而不是回完零拿一个不知道是哪一圈的
        位姿继续定位。

        ``wait=False`` 不建立参考（不知道什么时候到位）。中途失败同样保持未知。
        """
        self._require_open()
        if not wait:
            # 异步回零：连参考位都不能记录，位姿直接失效。
            self.pose_state.mark_unknown()
            return tuple(
                self.home_axis(axis_name, wait=False) for axis_name in self.HOME_ORDER
            )
        problems = self.config.homing_reference_problems()
        if problems:
            raise ValueError(
                "还不能建立绝对参考（回零会落在一个不知道是哪一圈的姿态上）：\n  - "
                + "\n  - ".join(problems)
            )
        homing = self.config.require_homing()
        # 开始前先让旧位姿失效：中途失败时绝不能留下一个"看起来还有效"的位置。
        self.pose_state.mark_unknown()
        waited = []
        try:
            for axis_name in self.HOME_ORDER:
                self._motors[axis_name].home(homing.mode)
                self._sleep(homing.wait_seconds)
                waited.append(homing.wait_seconds)
        except BaseException:
            self.pose_state.mark_unknown()
            raise
        # 只有三轴都发完并等待完成，才按配置的 safe_pose 记录参考。
        self.pose_state.record_home(self.config.require_safe_pose())
        # 三根轴现在都停在驱动器零点上，原点同样定义在这里。
        self.turns_state.mark_origin()
        return tuple(waited)

    # ------------------------------------------------------------- 归位
    def move_axis_to_zero(
        self,
        axis_name: str,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> float:
        """绝对模式把一根轴送回脉冲位置 0；返回本次等待的秒数。

        绝对模式**不知道当前位置**，所以耗时估不出来。``wait=True`` 等的是按该轴行程
        跨度算出的**最坏情况**。归位不建立参考：位姿立即变成未知。
        """
        self._require_open()
        speed = self.config.rpm if rpm is None else rpm
        accel = self.config.acceleration if acceleration is None else acceleration
        seconds = 0.0
        if wait:
            seconds = estimate_move_seconds(
                self._zero_travel_pulses(axis_name),
                speed,
                self.pulses_per_revolution,
                accel,
                self.settle_seconds,
            )
        self.pose_state.mark_unknown()
        self._motors[axis_name].move_absolute_pulses(0, speed, accel)
        if wait:
            self._sleep(seconds)
            # 绝对归位和回零停在同一个物理点上（驱动器位置计数器 0），原点同此。
            self.turns_state.mark_origin(axis_name)
        return seconds

    def move_all_to_zero(
        self,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> Tuple[float, ...]:
        """按 伸缩 → 升降 → 底座 的顺序把三轴送回零点（归位）。

        必须在**回过零之后**调用——绝对模式依赖驱动器内部的位置计数器。
        整臂归位只有在既有参考有效、全部预检与等待完成之后才更新位姿；这里保守地
        一律置为未知，要求调用方重新建立参考。
        """
        self._require_open()
        return tuple(
            self.move_axis_to_zero(
                axis_name, rpm=rpm, acceleration=acceleration, wait=wait
            )
            for axis_name in self.HOME_ORDER
        )

    # --------------------------------------------- 按里程表走绝对位置
    def move_axis_to_motor_pulses(
        self,
        axis_name: str,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> Union[MotionReport, MotionTicket]:
        """把一根轴走到**相对原点**的绝对电机脉冲位置。

        ``pulses`` 是**目标**不是增量：增量由 :attr:`turns_state` 的当前累计量算出来，
        所以同一个目标反复调用是幂等的——已经在目标上就一帧都不发。

        和驱动器自己的绝对模式（``0xFD … 01``）**不是一回事**：那个的零点在驱动器计数器
        里，这里的零点由 :meth:`modules.arm.arm_state.MotorTurnsState.mark_origin` 定义
        （回零与绝对归位会自动清零它）。所以这条路仍然只发相对帧，与本模块其余部分一致。

        增量超过 ``motion.max_pulses`` 时自动**拆成多段**，每段都不超过单帧上限。
        拆段需要 ``wait=True``：不等待的话后一帧会在前一帧还没走完时改写目标。
        """
        self._require_open()
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        delta = pulses - self.turns_state.of(axis_name).pulses
        return self._move_delta(
            axis_name, delta, rpm=rpm, acceleration=acceleration, wait=wait
        )

    def move_axis_to_motor_degrees(
        self,
        axis_name: str,
        degrees: float,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> Union[MotionReport, MotionTicket]:
        """同上，用**电机角度**给目标：``3200 脉冲 = 360°``，
        三个轴同一把尺，与机构无关。

        **不取模**：这台机器三个轴的行程都超过一个电机圈（底座 1800°、升降 1024°、
        伸缩 1057°），所以 ``move_axis_to_motor_degrees("base", 1048.0)`` 的意思是
        "走到累计 1048° 那个位置"，而不是 1048 mod 360。

        这里的 0 是**原点**（回零 / 绝对归零定义的那个点），不是"程序启动位置"：启动时轴若
        停在别处，第一次绝对定位会先把那段差额走掉——差额由配置的 ``initial_motor_degrees``
        声明（见 :meth:`modules.arm.config.ArmConfig.initial_motor_pulses`）。

        注意别和 :meth:`move_base_degrees` 搞混：那个是**输出轴**角度、按增量走。
        """
        return self.move_axis_to_motor_pulses(
            axis_name,
            motor_pulses_for_degrees(
                degrees, pulses_per_revolution=self.pulses_per_revolution
            ),
            rpm=rpm,
            acceleration=acceleration,
            wait=wait,
        )

    def _move_delta(
        self,
        axis_name: str,
        delta: int,
        *,
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
    ) -> Union[MotionReport, MotionTicket]:
        """发一段带符号增量；一次走不完就拆段，每段等完再发下一段。"""
        if delta == 0:
            # 已经在目标上：走 _signed_move 的空操作分支，一帧都不发。
            return self._signed_move(axis_name, 0, rpm, acceleration, wait)
        cap = self.config.max_pulses
        if abs(delta) <= cap:
            return self._signed_move(axis_name, delta, rpm, acceleration, wait)
        if not wait:
            raise ValueError(
                f"这次绝对移动要走 {abs(delta)} 脉冲，超过单帧上限 {cap}；"
                "拆段必须 wait=True，否则后一帧会在前一帧还没走完时改写目标"
            )
        reports: List[MotionReport] = []
        remaining = delta
        while remaining != 0:
            step = max(-cap, min(cap, remaining))
            report = self._signed_report(axis_name, step, rpm, acceleration)
            reports.append(report)
            remaining -= step
            # 轮询模式下这一段没到位就别再往下发了（估算模式没有状态可看，照旧走完）。
            if report.motor_status is not None and not report.motor_status.in_position:
                break
        return self._merge_reports(axis_name, delta, reports)

    def _signed_report(
        self,
        axis_name: str,
        pulses: int,
        rpm: Optional[int],
        acceleration: Optional[int],
    ) -> MotionReport:
        """``_signed_move`` 的**同步**版本：等完才返回，所以拿到的必然是报告。

        拆段和串行多轴都要"每段等完再发下一段"，这个窄化只在这里做一次。
        """
        report = self._signed_move(axis_name, pulses, rpm, acceleration, wait=True)
        if not isinstance(report, MotionReport):  # pragma: no cover - wait=True 恒为报告
            raise RuntimeError("内部错误：同步动作拿到了票据")
        return report

    def _merge_reports(
        self, axis_name: str, delta: int, reports: List[MotionReport]
    ) -> MotionReport:
        """把拆开的几段合成一份报告：脉冲数与角度是**总量**，不是最后一段的量。"""
        axis = self.config.axes[axis_name]
        motor_degrees = abs(
            motor_degrees_for_pulses(
                delta, pulses_per_revolution=self.pulses_per_revolution
            )
        )
        return MotionReport(
            motor_id=axis.motor_id,
            pulses=abs(delta),
            motor_degrees=motor_degrees,
            output_degrees=(
                motor_degrees / axis.reduction_ratio
                if axis.reduction_ratio is not None
                else None
            ),
            estimated_seconds=sum(item.estimated_seconds for item in reports),
            waited_seconds=sum(item.waited_seconds for item in reports),
            motor_status=reports[-1].motor_status,
            turns=self.turns_state.of(axis_name),
        )

    def _zero_travel_pulses(self, axis_name: str) -> int:
        """归位最坏要走多少脉冲：按该轴配置的行程跨度算。"""
        field = self.AXIS_LIMIT_FIELDS[axis_name]
        limit = getattr(self.config, field)
        if limit is None:
            raise ValueError(
                f"limits 里对应 {axis_name} 的区间还没标定（{field}），估不出归位耗时；"
                "请传 wait=False 自己等待"
            )
        span = limit.maximum - limit.minimum
        if axis_name == self.BASE_AXIS:
            pulses = base_pulses_for_degrees(
                span,
                pulses_per_revolution=self.pulses_per_revolution,
                reduction_ratio=self.config.base_reduction,
            )
        else:
            pulses = linear_pulses_for_mm(
                span,
                pulses_per_revolution=self.pulses_per_revolution,
                mm_per_motor_rev=self.config.require_mm_per_motor_rev(axis_name),
            )
        return max(1, abs(pulses))

    # ------------------------------------------------------------- 方向原语
    def rotate_cw(self, pulses: int, *, rpm: Optional[int] = None,
                  acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """底座顺时针走 ``pulses`` 个脉冲（相对）。"""
        return self._signed_move(BASE_AXIS, pulses, rpm, acceleration, wait)

    def rotate_ccw(self, pulses: int, *, rpm: Optional[int] = None,
                   acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """底座逆时针走 ``pulses`` 个脉冲（相对）。"""
        return self._signed_move(BASE_AXIS, -pulses, rpm, acceleration, wait)

    def move_up(self, pulses: int, *, rpm: Optional[int] = None,
                acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """升降轴上升 ``pulses`` 个脉冲（CW 是实测的正方向）。"""
        return self._signed_move(LIFT_AXIS, pulses, rpm, acceleration, wait)

    def move_down(self, pulses: int, *, rpm: Optional[int] = None,
                  acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """升降轴下降 ``pulses`` 个脉冲（CCW 是实测的负方向）。"""
        return self._signed_move(LIFT_AXIS, -pulses, rpm, acceleration, wait)

    def move_forward(self, pulses: int, *, rpm: Optional[int] = None,
                     acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """伸缩轴前伸 ``pulses`` 个脉冲（CW 是实测的正方向）。"""
        return self._signed_move(EXTENSION_AXIS, pulses, rpm, acceleration, wait)

    def move_backward(self, pulses: int, *, rpm: Optional[int] = None,
                      acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """伸缩轴后缩 ``pulses`` 个脉冲（CCW 是实测的负方向）。"""
        return self._signed_move(EXTENSION_AXIS, -pulses, rpm, acceleration, wait)

    # ------------------------------------------------- 单位制相对动作（带符号）
    def move_base_degrees(self, degrees: float, *, rpm: Optional[int] = None,
                          acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """底座相对转 ``degrees`` 度；正数对应 CW，负数对应 CCW。"""
        pulses = base_pulses_for_degrees(
            degrees,
            pulses_per_revolution=self.pulses_per_revolution,
            reduction_ratio=self.config.base_reduction,
        )
        return self._signed_move(BASE_AXIS, pulses, rpm, acceleration, wait)

    def move_lift_mm(self, millimeters: float, *, rpm: Optional[int] = None,
                     acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """升降轴相对抬升（正）/ 下降（负）``millimeters`` 毫米。"""
        pulses = linear_pulses_for_mm(
            millimeters,
            pulses_per_revolution=self.pulses_per_revolution,
            mm_per_motor_rev=self.config.require_mm_per_motor_rev(LIFT_AXIS),
        )
        return self._signed_move(LIFT_AXIS, pulses, rpm, acceleration, wait)

    def move_radial_mm(self, millimeters: float, *, rpm: Optional[int] = None,
                       acceleration: Optional[int] = None, wait: bool = True) -> Union[MotionReport, MotionTicket]:
        """伸缩轴相对前伸（正）/ 后缩（负）``millimeters`` 毫米。"""
        pulses = linear_pulses_for_mm(
            millimeters,
            pulses_per_revolution=self.pulses_per_revolution,
            mm_per_motor_rev=self.config.require_mm_per_motor_rev(EXTENSION_AXIS),
        )
        return self._signed_move(EXTENSION_AXIS, pulses, rpm, acceleration, wait)

    # ------------------------------------------------- 相对位姿（一次多轴）
    def move_cylindrical_by(
        self,
        *,
        base_degrees: float = 0.0,
        radial_mm: float = 0.0,
        height_mm: float = 0.0,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
    ) -> Tuple[MotionReport, ...]:
        """相对圆柱：三个量各走一个**增量**（度 / 毫米），不需要位姿已知。

        与 :meth:`move_cylindrical` 只差一句：那个是绝对目标（必须先有参考），这个只说
        "从此刻起再走多少"，所以三个量各自都能独立下发——极坐标的每个轴本来就能相对走。

        执行顺序按机械结构的安全规矩排，**不按参数顺序**：

        - 要**抬升**就先抬，再走"伸缩 → 底座"；
        - 要**下降**就把下降放到最后，免得在低位转动底座或前伸；
        - 只走平面时先伸缩、再转底座（与 :mod:`modules.grasp.drive.servo_motion` 的大角度
          规矩一致：伸出状态下横扫底座会把整条臂扫过工作区）。

        每段都等它走完再发下一段（这里不给 ``wait=False``：那会让后面的段在位置未知时
        接着发）。中途失败就停止后续段并抛出原始失败，**已经发出去的动作不会撤回**。

        标定要求与单轴方法相同：``base_degrees`` 要 ``axes.base.reduction_ratio``；两个
        毫米量各自要对应直线轴的 ``mm_per_motor_rev``，没标定就拒绝，不猜。

        :returns: 按**执行顺序**返回每一步的报告，且只含真的发了帧的那几步（增量不足半个
            脉冲的段取整成 0，不发帧也不出现在返回值里）。``motor_id`` 就是轴：
            5 底座 / 6 升降 / 7 伸缩。
        :raises ValueError: 参数不是有限数字，或该轴还没标定。
        """
        base = _relative_number(base_degrees, "base_degrees")
        radial = _relative_number(radial_mm, "radial_mm")
        height = _relative_number(height_mm, "height_mm")
        # 按**执行顺序**排好的 (轴, 增量)：抬升在最前、下降在最后，平面先伸缩后底座。
        steps: List[Tuple[str, float]] = []
        if height > 0.0:
            steps.append((LIFT_AXIS, height))
        if radial != 0.0:
            steps.append((EXTENSION_AXIS, radial))
        if base != 0.0:
            steps.append((BASE_AXIS, base))
        if height < 0.0:
            steps.append((LIFT_AXIS, height))
        reports: List[MotionReport] = []
        for axis_name, amount in steps:
            if axis_name == LIFT_AXIS:
                result = self.move_lift_mm(amount, rpm=rpm, acceleration=acceleration)
            elif axis_name == EXTENSION_AXIS:
                result = self.move_radial_mm(amount, rpm=rpm, acceleration=acceleration)
            else:
                result = self.move_base_degrees(amount, rpm=rpm, acceleration=acceleration)
            # 这一段是**串行**的：每段都等完才走下一段，所以拿到的必然是报告。
            if not isinstance(result, MotionReport):  # pragma: no cover - 同步调用恒为报告
                raise RuntimeError("内部错误：串行多轴拿到了票据")
            if result.pulses != 0:
                reports.append(result)
        return tuple(reports)

    def move_by(
        self,
        *,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_mm: float = 0.0,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
    ) -> Tuple[MotionReport, ...]:
        """相对基座 XYZ：在**当前位姿**上再走 ``(x_mm, y_mm, z_mm)``，单位毫米。

        ``z_mm`` 是高度增量（台面为 0、向上为正），与位姿无关（等价于
        :meth:`move_lift_mm`）。

        ``x_mm`` / ``y_mm`` **必须知道当前位姿**。这不是保守，是极坐标臂的几何事实：水平
        方向只有"转底座 + 伸缩"两个自由度，同一对 ``(dx, dy)`` 在不同方位对应不同的
        ``(Δ底座角, Δ半径)``，而且切向那部分还要除以当前半径。所以 x/y 表达的是"当前所在
        方位上的径向 / 切向分量"——没有当前 ``(底座角, 半径)`` 就分解不出来，猜一个方位
        正是这条路上最危险的事。

        不需要位姿的两条路：只要高度（``move_by(z_mm=…)`` 或 :meth:`move_lift_mm`），以及
        相机读数的相对修正（走 :mod:`modules.grasp` 的脉冲闭环，那里的每个机器常数都已经
        换成"每脉冲让画面挪多少"）。

        :returns: 与 :meth:`move_cylindrical_by` 同一套报告（按执行顺序，只含真的发帧的段）。
        :raises RuntimeError: x/y 非零但位姿未知（先 ``home_all`` 或写一次测量）。
        :raises ValueError: 参数不是有限数字，或当前半径太小、X/Y 方向没有定义。
        """
        dx = _relative_number(x_mm, "x_mm")
        dy = _relative_number(y_mm, "y_mm")
        dz = _relative_number(z_mm, "z_mm")
        if dx == 0.0 and dy == 0.0:
            return self.move_cylindrical_by(
                height_mm=dz, rpm=rpm, acceleration=acceleration
            )
        # 直接对**局部**取值判空：属性返回 Optional，类型检查器不会据此缩窄。
        current = self.pose
        if current is None:
            raise RuntimeError(
                "相对基座 XYZ 的 x/y 需要当前位姿：极坐标分解依赖此刻的底座角与半径。"
                "先显式回零（home_all），或让已标定的测量写一次位置（record_measurement）；"
                "只走高度（move_by(z_mm=…) / move_lift_mm）不需要位姿"
            )
        radius = current.radial_mm
        if radius < _MIN_RADIUS_MM:
            raise ValueError(
                f"当前半径 {radius:.3f} mm 小于 {_MIN_RADIUS_MM:.1f} mm："
                "基座轴线附近的点没有可用的 X/Y 方向，这一个增量分解不出底座角；"
                "先伸出去一点，或改用圆柱入口 move_cylindrical_by"
            )
        radians = math.radians(current.base_degrees)
        cos, sin = math.cos(radians), math.sin(radians)
        # 圆柱 -> 基座 XYZ 是 X = r·cosθ、Y = r·sinθ，所以增量反解就是这两行。
        radial_mm = cos * dx + sin * dy
        base_degrees = math.degrees((-sin * dx + cos * dy) / radius)
        return self.move_cylindrical_by(
            base_degrees=base_degrees,
            radial_mm=radial_mm,
            height_mm=dz,
            rpm=rpm,
            acceleration=acceleration,
        )

    # --------------------------------------------- 一次多轴（同时启动）
    def move_axes(
        self,
        targets: Mapping[str, int],
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
        mode: str = AXES_SAME_START,
    ) -> Union[Tuple[MotionReport, ...], MotionTicket]:
        """一次给多根轴各下一个**相对增量**，默认让它们**同时起步**。

        ```python
        arm.move_axes({"lift": 4000, "extension": -8000})    # 一起动、一起等
        ticket = arm.move_axes({...}, wait=False)             # 一起动、稍后再等
        reports = ticket.wait()
        ```

        与 :meth:`move_cylindrical_by` 的分工：那个是**串行**的（按安全顺序逐段走完），
        这个是**并行**的（默认 Emm42 多机同步三步法：逐轴缓存 → 广播触发一起起步）。
        并行的代价是**各自到位**——谁走的路短谁先停；想让它们同时停要按轴改 rpm
        （联动/插补），本模块刻意不做。

        :param targets: ``{轴名: 带符号脉冲}``；取 0 的轴不下发（协议不允许 0 位移）。
        :param mode: :data:`AXES_SAME_START`（默认，同时起步）或 :data:`AXES_SERIAL`
            （按 抬升 → 伸缩 → 底座 → 下降 的安全顺序逐轴走完），见模块文档字符串。
        :param wait: ``True`` 等整组走完，返回报告元组；``False`` 返回票据。
        :returns: ``wait=True`` 时是按 :data:`modules.arm.config.AXIS_NAMES` 顺序的报告元组
            （只含真发了帧的轴）；``wait=False`` 时是 :class:`~modules.motion.MotionTicket`，
            它的 ``wait()`` 返回同样的元组并把这几根轴的位姿一起推进。
        :raises ValueError: 轴名/脉冲非法、``mode`` 不认识、底座与伸缩的并行超限，
            或者 ``mode=AXES_SERIAL`` 配 ``wait=False``（那两者互相矛盾）。
        """
        self._require_open()
        mode = _require_axes_mode(mode)
        if mode == AXES_SERIAL and not wait:
            raise ValueError(
                f'mode="{AXES_SERIAL}" 要逐轴等完再走下一轴，和 wait=False 互相矛盾'
                f'（不等就等于让它们重叠）；要并发就用默认的 mode="{AXES_SAME_START}"'
            )
        moves = self._normalize_axes(targets)
        if not moves:
            # 全是 0 脉冲：调用方的意图是"已经到位"，一帧都不发。
            return () if wait else self._instant_ticket("arm:move_axes", ())
        if mode == AXES_SERIAL:
            reports: List[MotionReport] = []
            for axis_name, pulses in _serial_order(moves):
                reports.append(
                    self._signed_report(axis_name, pulses, rpm, acceleration)
                )
            return tuple(reports)
        return self._move_axes_together(moves, rpm, acceleration, wait)

    def _normalize_axes(self, targets: Mapping[str, int]) -> List[Tuple[str, int]]:
        """把 ``{轴名: 脉冲}`` 收成 ``(轴名, 带符号脉冲)`` 列表：查轴名、查整数、丢掉 0。

        顺序固定按 :data:`AXIS_NAMES`（而不是调用方写字典的顺序），这样同一组目标
        永远发出同一串帧，测试与日志里看到的顺序也是稳定的。
        """
        try:
            given = dict(targets.items())
        except AttributeError:
            raise ValueError("targets 必须是 {轴名: 带符号脉冲} 这样的映射") from None
        unknown = sorted(name for name in given if name not in self._motors)
        if unknown:
            raise ValueError(
                f"{'、'.join(unknown)} 不是已知的轴；已知的是 {list(AXIS_NAMES)}"
            )
        moves: List[Tuple[str, int]] = []
        for axis_name in AXIS_NAMES:
            if axis_name not in given:
                continue
            pulses = given[axis_name]
            if isinstance(pulses, bool) or not isinstance(pulses, int):
                raise ValueError(f"{axis_name} 的脉冲数必须是整数，收到 {pulses!r}")
            if pulses == 0:
                continue
            self._require_safe_pulses(abs(pulses))
            moves.append((axis_name, pulses))
        return moves

    def _move_axes_together(
        self,
        moves: List[Tuple[str, int]],
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
    ) -> Union[Tuple[MotionReport, ...], MotionTicket]:
        """多机同步三步法：逐轴缓存 → 广播触发 → 一起等。

        **缓存与触发之间必须自成一段**：广播触发帧会点燃总线上所有缓存命令，
        所以这一段里不许插入别人的缓存（本项目的调用都是"缓存完立刻触发"）。
        """
        self._require_axes_can_run_together(moves)
        speed = self.config.rpm if rpm is None else rpm
        accel = self.config.acceleration if acceleration is None else acceleration
        # 位姿在动作在途时会变成未知，所以先把**发帧之前**那一份位姿收好：
        # 收尾时必须从它起算，否则异步动作就再也算不回位置了。
        started_pose = self.pose_state.pose
        buffered: List[str] = []
        sent: List[MotionReport] = []
        try:
            for index, (axis_name, pulses) in enumerate(moves):
                direction = Direction.CW if pulses > 0 else Direction.CCW
                sent.append(
                    self._send(
                        axis_name,
                        direction,
                        abs(pulses),
                        rpm=rpm,
                        acceleration=acceleration,
                        wait=False,
                        synchronized=True,
                    )
                )
                buffered.append(axis_name)
                if index + 1 < len(moves):
                    # CH340 上背靠背两条位置帧会双双失效，所以帧间留间隔。
                    self._sleep(self.inter_frame_seconds)
            self._motors[moves[0][0]].trigger_sync_motion()
        except BaseException:
            # 缓存了却没触发是危险状态：下一次别人的广播会让这些轴突然动起来。
            self._clear_buffered(buffered)
            self.pose_state.mark_unknown()
            raise
        started = self._clock()
        if wait:
            try:
                return self._finish_axes(
                    moves, sent, speed, accel, started, started_pose, already_waited=False
                )
            except BaseException:
                # 帧已经发出去了、结论却拿不到：位置不可信，宁可重新对账。
                self.pose_state.mark_unknown()
                raise
        # 只发了帧：什么时候到位不知道，位置不可信——等票据收尾时才推进。
        self.pose_state.mark_unknown()
        return MotionTicket(
            "arm:" + "+".join(axis_name for axis_name, _ in moves),
            max(item.estimated_seconds for item in sent),
            finalize=lambda: self._finish_axes(
                moves, sent, speed, accel, started, started_pose, already_waited=True
            ),
            cancel=lambda: self._stop_axes_named([name for name, _ in moves]),
            clock=self._clock,
            sleep=self._sleep,
        )

    def _finish_axes(
        self,
        moves: List[Tuple[str, int]],
        sent: List[MotionReport],
        speed: int,
        accel: int,
        started: float,
        started_pose: Optional[CylindricalPose],
        *,
        already_waited: bool,
    ) -> Tuple[MotionReport, ...]:
        """等一组已触发的多轴动作走完，把这一组的位移一起记进位姿，返回补全的报告。

        ``already_waited=True`` 表示"估算那一段已经睡过了"（票据自己的 ``wait`` 睡的），
        所以这里只补轮询；``False`` 表示还没等过（同步调用），按 :attr:`wait_mode` 等。

        三根轴是同时走的，所以它们共用同一个"等了多久"和同一份最后状态：这一组什么时候
        算走完由**最慢的那根**决定，逐轴报不同的时长只会让人以为它们是串行走的。
        """
        status: Optional[MotorStatus] = None
        if self.wait_mode == WAIT_IN_POSITION:
            for (axis_name, pulses) in moves:
                status = self._poll_axis(axis_name, abs(pulses), speed, accel, started)
        elif not already_waited:
            self._sleep(max(item.estimated_seconds for item in sent))
        waited = self._clock() - started
        self._record_pose_after(started_pose, tuple(moves))
        return tuple(
            replace(item, waited_seconds=waited, motor_status=status) for item in sent
        )

    def _require_axes_can_run_together(self, moves: List[Tuple[str, int]]) -> None:
        """底座大角度旋转不许和伸缩并行——伸出状态下横扫会把整条臂扫过工作区。

        与 :mod:`modules.grasp.drive.servo_motion` 是同一条规矩，阈值见
        :data:`modules.motor.drive.motion.DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES`。
        """
        by_axis = dict(moves)
        base = by_axis.get(BASE_AXIS)
        if base is None or EXTENSION_AXIS not in by_axis:
            return
        if abs(base) <= self.parallel_rotation_limit_pulses:
            return
        raise ValueError(
            f"底座这一次要转 {abs(base)} 脉冲，超过并行上限 "
            f"{self.parallel_rotation_limit_pulses}：伸出状态下横扫会把整条臂扫过工作区。"
            f'改用 mode="{AXES_SERIAL}"（先伸缩到位、再转底座），或把旋转拆成几次小角度。'
        )

    def _clear_buffered(self, axes: List[str]) -> None:
        """清掉"缓存了却没触发"的危险状态：**先点火、再停住**。

        广播触发帧会点燃总线上所有缓存命令，所以缓存与触发之间失败时不能就这么算了。
        手册没写"立即停止能不能清掉待触发的缓存"，所以这里先补一条触发把缓存消费掉
        （轴开始按原命令走），再对每根轴发停止——顺序反过来就可能留下一条待触发的缓存。

        清理阶段的异常一律吞掉：**原始错误**才是要报给调用方的那一个。
        """
        if not axes:
            return
        try:
            self._motors[axes[0]].trigger_sync_motion()
        except Exception:
            pass
        for axis_name in axes:
            try:
                self._motors[axis_name].stop()
            except Exception:
                pass

    def _stop_axes_named(self, axes: List[str]) -> None:
        """尽力停住一组成员；停不下来的会抛出去，由票据记成失败。"""
        for axis_name in axes:
            self._stop_axis(axis_name)

    # ------------------------------------------------------------- 内部
    def _signed_move(
        self,
        axis_name: str,
        pulses: int,
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
    ) -> Union[MotionReport, MotionTicket]:
        """发一条带符号的相对动作，并按**实际下发的整数脉冲**推进位姿。

        位姿更新只有这一处：方向原语与单位制方法都走它，所以"相对毫米再调底层动作"
        不会把同一次位移累计两遍。

        ``wait=True`` 返回 :class:`MotionReport`；``wait=False`` 返回
        :class:`~modules.motion.MotionTicket`——那时动作**还没有结论**，
        报告要等 ``ticket.wait()``（见模块文档字符串的"``wait=False`` 返回票据"一节）。
        """
        self._require_open()
        if pulses == 0:
            # 位移不足半个脉冲：调用方的意图是"已经到位"，既不发帧也不改位置。
            axis = self.config.axes[axis_name]
            report = MotionReport(
                motor_id=axis.motor_id,
                pulses=0,
                motor_degrees=0.0,
                output_degrees=0.0 if axis.reduction_ratio is not None else None,
                estimated_seconds=0.0,
                turns=self.turns_state.of(axis_name),
            )
            return report if wait else self._instant_ticket(f"arm:{axis_name}", report)
        direction = Direction.CW if pulses > 0 else Direction.CCW
        report = self._send(
            axis_name, direction, abs(pulses), rpm=rpm, acceleration=acceleration, wait=wait
        )
        if wait:
            self._advance_pose(axis_name, pulses)
            return report
        # 只发了帧、没等它走完：什么时候到位不知道，位置不可信——等票据收尾时才推进。
        # 起点必须是**发帧之前**那一份位姿：下面这一句之后就再也读不到了。
        started_pose = self.pose_state.pose
        self.pose_state.mark_unknown()
        return self._move_ticket(axis_name, pulses, rpm, acceleration, report, started_pose)

    def _instant_ticket(self, subject: str, result: object) -> MotionTicket:
        """给"没有东西可等"的动作建票据：``wait()`` 立刻给出结论，不发帧也不睡。"""
        return MotionTicket(
            subject, 0.0, finalize=lambda: result, clock=self._clock, sleep=self._sleep
        )

    def _move_ticket(
        self,
        axis_name: str,
        signed_pulses: int,
        rpm: Optional[int],
        acceleration: Optional[int],
        sent: MotionReport,
        started_pose: Optional[CylindricalPose],
    ) -> MotionTicket:
        """给一条**已经发出**的单轴动作建票据。

        到点之后：按 :attr:`wait_mode` 确认走完（轮询模式才需要，估算那一段票据自己
        已经睡过了）→ 从 ``started_pose`` 起算推进位姿 → 返回补全了 ``waited_seconds``
        与 ``motor_status`` 的报告。

        ``cancel`` 只停这一根轴，而且**不收尾**：动作没走完就推进位姿会记错位置，
        所以取消之后位姿保持未知。
        """
        speed = self.config.rpm if rpm is None else rpm
        accel = self.config.acceleration if acceleration is None else acceleration
        started = self._clock()

        def finalize() -> MotionReport:
            status: Optional[MotorStatus] = None
            if self.wait_mode == WAIT_IN_POSITION:
                status = self._poll_axis(
                    axis_name, abs(signed_pulses), speed, accel, started
                )
            waited = self._clock() - started
            self._record_pose_after(started_pose, ((axis_name, signed_pulses),))
            return replace(sent, waited_seconds=waited, motor_status=status)

        return MotionTicket(
            f"arm:{axis_name}",
            sent.estimated_seconds,
            finalize=finalize,
            cancel=lambda: self._stop_axis(axis_name),
            clock=self._clock,
            sleep=self._sleep,
        )

    def _stop_axis(self, axis_name: str) -> None:
        """尽力停住一根轴；会话已经关掉时什么都不做（那时没什么可停的）。"""
        if self._closed:
            return
        self._motors[axis_name].stop()

    def _send(
        self,
        axis_name: str,
        direction: Direction,
        pulses: int,
        *,
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
        synchronized: bool = False,
    ) -> MotionReport:
        """真正发帧（并可选等待）。参数校验在发帧之前完成。

        ``synchronized=True`` 时这一帧只是**缓存住**（多机同步标志 = 01），
        要等一条广播 ``00 FF 66 6B`` 才会动——那是 :meth:`move_axes` 让多根轴
        同时起步的办法，所以它必然配 ``wait=False``（缓存帧还没开始走，没什么可等的）。
        """
        if synchronized and wait:
            # 缓存帧发出时轴还没动，睡一觉只是在浪费估算；等整组是调用方的事。
            raise ValueError("缓存帧（synchronized=True）不能配 wait=True")
        self._require_safe_pulses(pulses)
        speed = self.config.rpm if rpm is None else rpm
        accel = self.config.acceleration if acceleration is None else acceleration
        # 先算耗时再发命令：参数非法时抛出，串口里不会出现半条动作。
        seconds = estimate_move_seconds(
            pulses, speed, self.pulses_per_revolution, accel, self.settle_seconds
        )
        signed_pulses = pulses if direction is Direction.CW else -pulses
        self._motors[axis_name].move_relative_pulses(
            signed_pulses, speed, accel, synchronized=synchronized
        )
        # 累计里程就在这里推进，而且**只在这里**：所有相对帧——方向原语、单位制方法、
        # 以及绝对规划出来的 _execute_frame——都经过 _send，所以既不会漏也不会把同一次
        # 位移记两遍。放在发帧之后：发帧抛异常时不该记账。
        turns = self.turns_state.record(axis_name, signed_pulses)
        waited = 0.0
        status: Optional[MotorStatus] = None
        if wait:
            waited, status = self._wait_for_move(
                axis_name, pulses, speed, accel, seconds
            )
        motor_degrees = pulses * 360 / self.pulses_per_revolution
        axis = self.config.axes[axis_name]
        return MotionReport(
            motor_id=axis.motor_id,
            pulses=pulses,
            motor_degrees=motor_degrees,
            output_degrees=(
                motor_degrees / axis.reduction_ratio
                if axis.reduction_ratio is not None
                else None
            ),
            estimated_seconds=seconds,
            waited_seconds=waited,
            motor_status=status,
            turns=turns,
        )

    def _wait_for_move(
        self,
        axis_name: str,
        pulses: int,
        speed: int,
        accel: int,
        estimate: float,
    ) -> Tuple[float, Optional[MotorStatus]]:
        """按 :data:`WAIT_MODES` 等这一次动作走完，返回 (实际等待秒数, 最后读到的状态)。

        估算模式返回的时长就是估算值本身——它是"睡了多久"，不是"走了多久"。
        轮询模式返回实测时长，并带上**最后一次** ``0x3A`` 读数：超时那一刻的
        ``stalled`` / ``stall_protected`` 就是"轴没走到位"的软件证据。
        """
        if self.wait_mode == WAIT_ESTIMATE:
            # 驱动器不回 ACK / 到位包，只能按估算耗时等它走完。
            self._sleep(estimate)
            return estimate, None
        started = self._clock()
        status = self._poll_axis(axis_name, pulses, speed, accel, started)
        return self._clock() - started, status

    def _poll_axis(
        self,
        axis_name: str,
        pulses: int,
        speed: int,
        accel: int,
        started: float,
    ) -> MotorStatus:
        """轮询一根轴的 ``0x3A`` 到位位；``started`` 是**这一组动作起步的时刻**。

        分给这根轴的是"它自己的超时预算减去已经花掉的时间"：多轴同时起步时，
        后问的那根不该再从头等一遍，否则整组的等待会变成各轴预算之和。
        """
        budget = in_position_timeout_seconds(
            pulses, speed, self.pulses_per_revolution, accel, self.settle_seconds
        )
        return self._motors[axis_name].wait_until_in_position(
            max(0.0, budget - (self._clock() - started)),
            sleep=self._sleep,
            clock=self._clock,
        )

    def _advance_pose(self, axis_name: str, signed_pulses: int) -> None:
        """按实际下发的带符号整数脉冲推进位姿。

        位姿本来就未知时什么都不做（不能凭空造一个绝对位置）；已知时按标定换算成
        增量，角度照测量域归一。
        """
        # 直接对**局部**取值判空：``pose_state.is_known`` 是个布尔属性，类型检查器
        # 不会据此把 ``pose`` 缩窄成非空，而"未知时不推进"这条业务规则必须靠这个判断。
        self._record_pose_after(self.pose_state.pose, ((axis_name, signed_pulses),))

    def _record_pose_after(
        self,
        started_pose: Optional[CylindricalPose],
        moves: Tuple[Tuple[str, int], ...],
    ) -> None:
        """把一组增量叠加到**发帧那一刻**的位姿上；只有全程算得出来才写回位姿。

        为什么要传"那时"的位姿：动作在途时位姿一律是未知的（见
        :meth:`modules.arm.arm_state.ArmPose.mark_unknown` 的调用点），所以收尾如果
        从"当前读数"起算，读到的永远是 ``None``，异步动作就再也没有位姿了。
        缺标定（毫米换算不出来）时保持未知——不猜一个位置出来。

        位姿推进只有这一处：同步动作、单轴票据、多轴票据全走它。
        """
        if self._closed:
            # 会话已经关了：连接没了，谁也不知道轴在哪，迟到的收尾不该把位姿"复活"。
            return
        pose = started_pose
        if pose is not None:
            for axis_name, signed_pulses in moves:
                pose = self._pose_after(pose, axis_name, signed_pulses)
                if pose is None:
                    break
        if pose is None:
            self.pose_state.mark_unknown()
        else:
            # 起点是**发帧之前**那一份，所以现在这一刻的"未知"不该挡住回写。
            self.pose_state.record_dead_reckoning(pose)

    def _pose_after(
        self, current: CylindricalPose, axis_name: str, signed_pulses: int
    ) -> Optional[CylindricalPose]:
        """从 ``current`` 走一个增量之后的位姿；缺标定算不出来时返回 ``None``。"""
        if axis_name == BASE_AXIS:
            degrees = base_degrees_for_pulses(
                signed_pulses,
                pulses_per_revolution=self.pulses_per_revolution,
                reduction_ratio=self.config.base_reduction,
            )
            return CylindricalPose(
                normalize_degrees(current.base_degrees + degrees),
                current.radial_mm,
                current.height_mm,
            )
        mm_per_rev = self.config.mm_per_motor_rev(axis_name)
        if mm_per_rev is None:
            # 没有毫米标定就算不出位移，只能承认不知道在哪。
            return None
        millimeters = mm_for_linear_pulses(
            signed_pulses,
            pulses_per_revolution=self.pulses_per_revolution,
            mm_per_motor_rev=mm_per_rev,
        )
        if axis_name == EXTENSION_AXIS:
            return CylindricalPose(
                current.base_degrees,
                current.radial_mm + millimeters,
                current.height_mm,
            )
        return CylindricalPose(
            current.base_degrees,
            current.radial_mm,
            current.height_mm + millimeters,
        )

    def _require_safe_pulses(self, pulses: int) -> None:
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        if not 1 <= pulses <= self.config.max_pulses:
            raise ValueError(f"pulses must be in 1..{self.config.max_pulses}")

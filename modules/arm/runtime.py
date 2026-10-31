"""机械臂上层：统一生命周期、相对脉冲原语与绝对脉冲目标。

驱动层
(:mod:`modules.motor.drive.emm42`) 只负责协议帧，本模块负责"这条轴走多少"。

## 生命周期

```python
with RobotArm.open() as arm:          # 读配置 + 开（或借）一条串口
    arm.move_up(pulses=800)           # 相对脉冲
```

上层不再自己 ``initialize()`` / 使能 / 收尾：``open`` 已经做过，``close`` 或 ``with``
负责收尾。``stop`` 只是"停住执行器"，不关连接、不额外失能。

## 两种给目标的方式

- **相对原语**：``move_up`` / ``move_down`` / ``move_forward`` / ``move_backward`` /
  ``rotate_cw`` / ``rotate_ccw``，单位是脉冲（正数 = 该轴正方向）；
- **绝对脉冲目标**：:meth:`RobotArm.move_axis_to_motor_degrees` /
  :meth:`RobotArm.move_axis_to_motor_pulses`，给的是"相对原点走到哪个电机位置"
  （``3200 脉冲 = 360°``，不取模），增量由 :attr:`turns_state` 的读数算出来，
  所以同一个目标反复调用是幂等的。超出单帧上限会自动拆段。
  零点由配置的 ``initial_motor_degrees`` 声明，回零之后即 0。

抓取走的是**脉冲闭环**（相机看画面挪多少像素 → 每像素多少脉冲），不需要毫米级绝对
位姿；hardware.json 里的 ``homing`` / ``limits`` / ``safe_pose`` 只是历史标定记录，代码不读。

## 一次让多根轴动：``move_axes``

:meth:`RobotArm.move_axes` 默认是**并行**的那一个，默认走 Emm42 的多机同步三步法：

```
逐轴发 0xFD（多机同步标志 = 01）缓存住 → 帧间留 DEFAULT_INTER_FRAME_SECONDS
→ 最后一条广播 00 FF 66 6B 让它们一起起步
```

于是三根轴是**真的同时起步**（轮子早就是这么走的，见 ``MecanumDrive``），
而不是"发完第一条再发第二条"那种差几十毫秒的伪并行。代价是它们**各自到位**：
谁走的路短谁先停。想让它们同时停要按轴改 rpm（联动/插补），那是另一件事，
本模块刻意不做。

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
ticket.wait()                                   # 到点 → 返回 MotionReport
```

票据的 ``estimated_seconds`` 就是过去那份报告里同名字段，"还要等多久"因此在这条路上
一样看得见。多轴的那一份票据 ``wait()`` 之后给出这一组的报告元组。

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

## 一个状态：电机转了多少

- ``turns_state``（:class:`modules.arm.arm_state.MotorTurnsState`）：三轴的**命令脉冲
  里程表**。一个标定都不需要，所以永远有值；底座角**不归一**。

它是为"底座能转整圈"这件事存在的：连续同向转十圈和原地没动看起来一模一样，而线缆会绕死。
里程表不归一，所以看得见——但它是**下发
的脉冲**，丢步和打滑不会体现，要和实测对账就读 ``0x36``。

它记的是**相对原点**的读数，而"程序启动那一刻轴在哪"由配置的 ``initial_motor_degrees``
声明（缺省表示三根轴都在原点上）。换爪子、重新对齿或重装电机之后某根轴的机械初始位不再
落在原点上时（例如伸缩停在 270° = 2400 脉冲），就在那里如实声明，而不是让软件继续以为
它在原点；回零之后读数照旧回到 0。

这一本账只在 ``_send`` 一处推进，所以方向原语、单位制方法
和绝对脉冲目标走的是同一条路，既不会漏记也不会把同一次位移记两遍。每次
:class:`MotionReport` 都会带上该轴累计到那一刻的读数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple, Union

from modules.arm.arm_state import MotorTurns, MotorTurnsState
from modules.arm.config import AXIS_NAMES, ArmConfig, load_arm_config
from modules.arm.drive.kinematics import (
    BASE_AXIS,
    EXTENSION_AXIS,
    LIFT_AXIS,
    motor_degrees_for_pulses,
    motor_pulses_for_degrees,
)
from modules.motor.drive.emm42 import (
    POSITION_UNITS_PER_REVOLUTION,
    Direction, Emm42Driver, MotorStatus, SerialPort,
)
from modules.motor.drive.motion import (
    DEFAULT_INTER_FRAME_SECONDS,
    DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES,
    DEFAULT_SETTLE_SECONDS,
    IN_POSITION_TIMEOUT_FACTOR,
    IN_POSITION_TIMEOUT_MARGIN_SECONDS,
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

#: 多轴模式：按安全顺序**逐轴**走完再走下一轴（与 :meth:`_serial_order` 同序）。
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

    与 :func:`_serial_order` 同一条规矩：要抬升就先抬，要下降就放到最后
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
class RobotArm:
    """控制 5/6/7 号轴的机械臂上层：只发有界的相对位置命令。"""

    BASE_AXIS = BASE_AXIS
    LIFT_AXIS = LIFT_AXIS
    EXTENSION_AXIS = EXTENSION_AXIS

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
        self._planar_velocity_session = None
        self._planar_position_valid = True
        #: 三轴的命令脉冲里程表：一个标定都不需要，所以永远有值（见 arm_state）。
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
        from modules.motor.config import DEFAULT_HARDWARE_PATH as MOTOR_DEFAULT_HARDWARE_PATH
        from modules.motor.config import load_bus_config

        arm_path = Path(config_path) if config_path is not None else None
        config = load_arm_config(**({} if arm_path is None else {"config_path": arm_path}))
        bus_path = (
            Path(motor_config_path)
            if motor_config_path is not None
            else MOTOR_DEFAULT_HARDWARE_PATH
        )
        port = load_bus_config(hardware_path=bus_path).port
        tuning = {}
        if inter_frame_seconds is not None:
            tuning["inter_frame_seconds"] = inter_frame_seconds
        if parallel_rotation_limit_pulses is not None:
            tuning["parallel_rotation_limit_pulses"] = parallel_rotation_limit_pulses
        session = cls.from_settings(
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
    def from_settings(
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
        重复调用不再发任何命令。
        """
        if self._closed:
            return
        velocity_error = None
        if self._planar_velocity_session is not None:
            try:
                self._planar_velocity_session.close()
            except Exception as error:
                velocity_error = error
        self._closed = True
        errors = (
            [] if velocity_error is None else [("planar_velocity", velocity_error)]
        )
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

        急停之后驱动器可能丢步，里程表只记下发过的脉冲，与真实位置未必一致。
        """
        self._require_open()
        if self._planar_velocity_session is not None:
            self._planar_velocity_session.stop()
            return
        self._stop_axes()

    def planar_velocity(
        self,
        *,
        max_rpm: int,
        motor_limits: Mapping[str, Tuple[float, float]],
        acceleration: int = 200,
        command_timeout: float = 0.35,
    ):
        """借出平面两轴的有界速度会话；范围为不取模的绝对电机角度。"""
        from modules.arm.drive.planar_velocity import PlanarVelocitySession

        self._require_position_ready()
        session = PlanarVelocitySession(
            self,
            max_rpm=max_rpm,
            motor_limits=motor_limits,
            acceleration=acceleration,
            command_timeout=command_timeout,
        )
        self._planar_velocity_session = session
        return session

    def _require_position_ready(self) -> None:
        """速度模式独占总线；编码器失联后禁止使用未对账的命令位置。"""
        self._require_open()
        if self._planar_velocity_session is not None:
            raise RuntimeError("planar velocity session owns arm motion")
        if not self._planar_position_valid:
            raise RuntimeError(
                "arm position invalid after planar encoder/stop failure; "
                "reopen after referencing position"
            )

    def _each_axis(self, action: Callable[["Emm42Driver"], None]) -> None:
        """按地址顺序逐轴发**单帧**命令，帧间留 ``inter_frame_seconds``。

        ``enable`` / ``disable`` / ``stop`` 这类单帧命令原来是一条接一条发出去的。
        这台机器的 USB-TTL 上背靠背发帧会让帧失效（见
        :data:`modules.motor.drive.motion.DEFAULT_INTER_FRAME_SECONDS`）：三轴里
        5 号会回包、6/7 号不回，弄坏的帧只留下一条 ``地址 00 EE 6B``（错误命令）
        没人读走，下一个用总线的人（底盘）就会把这条错误帧当成自己的 ACK。
        """
        if self._planar_velocity_session is not None:
            raise RuntimeError("planar velocity session owns arm bus")
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
        """释放三个轴的扭矩（松开之后轴可以被外力推动）。"""
        self._require_open()
        self._disable_axes()

    def read_abs_state(self, axis_name: str) -> MotorTurns:
        """**一根轴现在在哪**：绝对电机角度（与 ``move_axis_to_motor_degrees`` 同一把尺）。

        ```python
        state = arm.read_abs_state("lift")
        state.motor_degrees     # 现在停的绝对角度，可以直接喂回 move_axis_to_motor_degrees
        state.pulses            # 同一件事的脉冲口径
        print(state.summary())  # "lift: +8900 脉冲 (+1001.25° / +2.781 圈)"
        ```

        读的是**命令里程表**（下发过多少脉冲），不是编码器实测：丢步、打滑、撞停都不会
        体现在这里，``stop`` / 急停之后也一样。要实测就在同一个会话里比 ``0x36``
        （``Emm42Driver.read_position()``，单位 1/65536 电机圈）；两者差多少就是丢了
        多少步——这也是"回零之后必须重新对高度"的原因。

        :raises KeyError: 轴名不认识（已知的是 ``AXIS_NAMES``）。
        """
        self._require_open()
        return self.turns_state.of(axis_name)

    def read_measured_pulses(self, axis_name: str) -> float:
        """读取编码器位置并换算为命令脉冲；只比较增量，不与命令里程表混用原点。"""
        self._require_position_ready()
        return (
            self._motors[axis_name].read_position()
            * self.pulses_per_revolution / POSITION_UNITS_PER_REVOLUTION
        )

    # ------------------------------------------------- 按里程表走绝对脉冲位置
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
        self._require_position_ready()
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

    def move_axis_to_motor_degrees_monitored(
        self,
        axis_name: str,
        degrees: float,
        *,
        keep_moving: Callable[[], bool],
        rpm: Optional[int] = None,
    ) -> bool:
        """边走边检查，检查失败立即刹停；返回是否实测到位。

        回调只做观测，不得另发运动命令。退出时用编码器增量修正命令里程表，
        避免取消半途动作后，下一次绝对定位仍从原目标位置计算位移。
        """
        target = motor_pulses_for_degrees(
            degrees, pulses_per_revolution=self.pulses_per_revolution
        )
        self._require_position_ready()
        start = self.turns_state.of(axis_name).pulses
        measured_start = self.read_measured_pulses(axis_name)
        started = self._clock()
        ticket = self.move_axis_to_motor_degrees(
            axis_name, degrees, rpm=rpm, wait=False
        )
        # 与常规到位轮询使用相同的估时倍率和余量，堵转不能无限等待。
        deadline = (
            self._clock() + ticket.estimated_seconds * IN_POSITION_TIMEOUT_FACTOR
            + IN_POSITION_TIMEOUT_MARGIN_SECONDS
        )
        status: Optional[MotorStatus] = None

        def diagnosis() -> str:
            # 使用本轮已读到的数据，不再查询，避免故障后的状态覆盖超时现场。
            flags = "尚未读取" if status is None else (
                f"使能={int(status.enabled)}、到位={int(status.in_position)}、"
                f"堵转={int(status.stalled)}、堵转保护={int(status.stall_protected)}"
            )
            return (
                f"指令增量 {target - start:+.1f} 脉冲，"
                f"实测增量 {measured - start:+.1f} 脉冲，"
                f"剩余 {target - measured:+.1f} 脉冲；"
                f"估算 {ticket.estimated_seconds:.3f} s，"
                f"已等待 {self._clock() - started:.3f} s；0x3A：{flags}"
            )

        try:
            while True:
                if not keep_moving():
                    return False
                measured = start + self.read_measured_pulses(axis_name) - measured_start
                if ticket.remaining <= 0:
                    status = self._motors[axis_name].read_status()
                    if status.stalled or status.stall_protected:
                        raise RuntimeError(f"{axis_name} 监控移动发生堵转；{diagnosis()}")
                    # 到位窗口由驱动器决定；还须实测离开起点，排除旧的到位位。
                    if status.in_position and (target == start or abs(measured - start) >= 1):
                        return True
                if self._clock() >= deadline:
                    raise TimeoutError(f"{axis_name} 监控移动未实测到位；{diagnosis()}")
                self._sleep(0.01)
        finally:
            # 先停再读；编码器与软件原点可能不同，因此只使用本段的实测增量。
            ticket.cancel()
            actual = start + round(self.read_measured_pulses(axis_name) - measured_start)
            self.turns_state.record(
                axis_name, actual - self.turns_state.of(axis_name).pulses
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

        与 :meth:`_serial_order` 的分工：那个是**串行**的（按安全顺序逐段走完），
        这个是**并行**的（默认 Emm42 多机同步三步法：逐轴缓存 → 广播触发一起起步）。
        并行的代价是**各自到位**——谁走的路短谁先停；想让它们同时停要按轴改 rpm
        （联动/插补），本模块刻意不做。

        :param targets: ``{轴名: 带符号脉冲}``；取 0 的轴不下发（协议不允许 0 位移）。
        :param mode: :data:`AXES_SAME_START`（默认，同时起步）或 :data:`AXES_SERIAL`
            （按 抬升 → 伸缩 → 底座 → 下降 的安全顺序逐轴走完），见模块文档字符串。
        :param wait: ``True`` 等整组走完，返回报告元组；``False`` 返回票据。
        :returns: ``wait=True`` 时是按 :data:`modules.arm.config.AXIS_NAMES` 顺序的报告元组
            （只含真发了帧的轴）；``wait=False`` 时是 :class:`~modules.motion.MotionTicket`，
            它的 ``wait()`` 返回同样的报告元组。
        :raises ValueError: 轴名/脉冲非法、``mode`` 不认识、底座与伸缩的并行超限，
            或者 ``mode=AXES_SERIAL`` 配 ``wait=False``（那两者互相矛盾）。
        """
        self._require_open()
        self._require_position_ready()
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
            raise
        started = self._clock()
        if wait:
            try:
                return self._finish_axes(
                    moves, sent, speed, accel, started, already_waited=False
                )
            except BaseException:
                # 帧已经发出去了、结论却拿不到：位置不可信，宁可重新对账。
                raise
        # 只发了帧：什么时候到位不知道，位置不可信——等票据收尾时才推进。
        return MotionTicket(
            "arm:" + "+".join(axis_name for axis_name, _ in moves),
            max(item.estimated_seconds for item in sent),
            finalize=lambda: self._finish_axes(
                moves, sent, speed, accel, started, already_waited=True
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
        *,
        already_waited: bool,
    ) -> Tuple[MotionReport, ...]:
        """等一组已触发的多轴动作走完，返回补全了等待时长与状态\u7684报告。

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
        return tuple(
            replace(item, waited_seconds=waited, motor_status=status) for item in sent
        )

    def _require_axes_can_run_together(self, moves: List[Tuple[str, int]]) -> None:
        """底座大角度旋转不许和伸缩并行——伸出状态下横扫会把整条臂扫过工作区。

        与 :mod:`modules.grasp.drive.visual_servo_motion` 是同一条规矩，阈值见
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
        """发一条带符号的相对动作，并按**实际下发的整数脉冲**记账。

        记账只有 ``_send`` 一处：方向原语与绝对脉冲目标都走它，所以同一次位移
        不会被累计两遍。

        ``wait=True`` 返回 :class:`MotionReport`；``wait=False`` 返回
        :class:`~modules.motion.MotionTicket`——那时动作**还没有结论**，
        报告要等 ``ticket.wait()``（见模块文档字符串的"``wait=False`` 返回票据"一节）。
        """
        self._require_open()
        self._require_position_ready()
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
            return report
        # 只发了帧、没等它走完：结论要等票据到点才有。
        return self._move_ticket(axis_name, pulses, rpm, acceleration, report)

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
    ) -> MotionTicket:
        """给一条**已经发出**的单轴动作建票据。

        到点之后：按 :attr:`wait_mode` 确认走完（轮询模式才需要，估算那一段票据自己
        已经睡过了）→ 返回补全了 ``waited_seconds`` 与 ``motor_status`` 的报告。

        ``cancel`` 只停这一根轴，而且**不收尾**：命令已经发出去了，停在哪里不知道。
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
        self._require_position_ready()
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
    def _require_safe_pulses(self, pulses: int) -> None:
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        if not 1 <= pulses <= self.config.max_pulses:
            raise ValueError(f"pulses must be in 1..{self.config.max_pulses}")

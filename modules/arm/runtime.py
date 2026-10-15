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
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

from modules.arm.arm_state import ArmPose
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
    normalize_degrees,
)
from modules.arm.motion_plan import PlanError, plan_absolute, plan_xyz
from modules.motor.drive.emm42 import Direction, Emm42Driver, SerialPort
from modules.motor.drive.motion import DEFAULT_SETTLE_SECONDS, estimate_move_seconds


@dataclass(frozen=True)
class MotionReport:
    """一次相对动作的命令量；直线轴没有毫米标定时 ``output_degrees`` 为 ``None``。"""

    motor_id: int
    pulses: int
    motor_degrees: float
    output_degrees: Optional[float]
    #: 本次动作的估算耗时（秒）；``wait=False`` 时调用方可以自己用它来等。
    estimated_seconds: float = 0.0


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
    ) -> None:
        """只保存**已经装配好的**依赖：本构造器不做任何设备 I/O。

        上层代码用 :meth:`open` 创建；这里显式接收 ``motors`` / ``bus`` 是模块内部的
        装配接缝，供 :meth:`open` 与假设备测试使用，不是给业务调用者的参数。

        :param owns_bus: 这条串口是不是**本实例打开**的。只有自己开的才在 ``close``
            里物理关闭；借来的（同一个 Robot 会话里 arm 与 chassis 共用一条）只失效自身。
        """
        self.config = config
        self.pulses_per_revolution = config.pulses_per_revolution
        self.settle_seconds = settle_seconds
        self._sleep = sleep
        self._motors = dict(motors)
        self._bus = bus
        self._owns_bus = bool(owns_bus) and bus is not None
        self._closed = False
        #: 爪口中心的位姿；**唯一所有者是这里**，完成回零或测量写入之前是"未知"。
        self.pose_state = ArmPose()

    # ------------------------------------------------------------- 生命周期
    @classmethod
    def open(
        cls,
        *,
        config_path: Optional[Path] = None,
        motor_config_path: Optional[Path] = None,
        bus: Optional[SerialPort] = None,
        owns_bus: Optional[bool] = None,
    ) -> "RobotArm":
        """按配置开一条步进连接并返回就绪实例。

        :param config_path: arm 主配置；``None`` 用 :data:`modules.arm.config.DEFAULT_CONFIG_PATH`。
        :param motor_config_path: 步进总线配置；``None`` 用 motor 模块的默认配置。
        :param bus: 由调用方**已经打开**的共享连接（``Robot`` 装配 arm+chassis 时用）。
            给了它就借用，不再自己开第二条、也不会在 ``close`` 里关掉它。
        :param owns_bus: 显式声明这条 ``bus`` 归不归本实例；默认按"是不是自己开的"判断。
            传 ``bus`` 时默认 ``False``（借来的）。
        """
        from modules.motor.config import DEFAULT_CONFIG_PATH as motor_default
        from modules.motor.config import load_bus_config

        arm_path = Path(config_path) if config_path is not None else None
        config = load_arm_config(**({} if arm_path is None else {"config_path": arm_path}))
        bus_path = Path(motor_config_path) if motor_config_path is not None else motor_default
        port = load_bus_config(config_path=bus_path).port
        session = cls._from_settings(
            config=config, port=port, bus=bus, owns_bus=owns_bus, sleep=time.sleep
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
    ) -> "RobotArm":
        """按**已经读好的配置快照**装配：Robot 与独立 ``open`` 共用这一处。

        ``bus`` 为 ``None`` 时自己开一条串口（并声明所有权）；给了就借用。
        构造中途失败时，自己开的串口由本方法释放，外部注入的**不擅自关闭**
        （那会切断别的模块的连接）。
        """
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
                    # 已读取的 5/6/7 配置均为 Response=None，控制帧不回 ACK。
                    expect_ack=False,
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
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required on Jetson. Install it in the target Python environment first."
            ) from error
        return serial.Serial(port, baudrate=Emm42Driver.BAUDRATE, timeout=0.1)

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

    def _stop_axes(self) -> None:
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].stop()

    def _disable_axes(self) -> None:
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].enable(False)

    def enable_all(self) -> None:
        """按地址顺序给三个轴使能扭矩。``open`` 已经做过；需要重新使能时才调。"""
        self._require_open()
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].enable(True)

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
                  acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """底座顺时针走 ``pulses`` 个脉冲（相对）。"""
        return self._signed_move(BASE_AXIS, pulses, rpm, acceleration, wait)

    def rotate_ccw(self, pulses: int, *, rpm: Optional[int] = None,
                   acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """底座逆时针走 ``pulses`` 个脉冲（相对）。"""
        return self._signed_move(BASE_AXIS, -pulses, rpm, acceleration, wait)

    def move_up(self, pulses: int, *, rpm: Optional[int] = None,
                acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """升降轴上升 ``pulses`` 个脉冲（CW 是实测的正方向）。"""
        return self._signed_move(LIFT_AXIS, pulses, rpm, acceleration, wait)

    def move_down(self, pulses: int, *, rpm: Optional[int] = None,
                  acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """升降轴下降 ``pulses`` 个脉冲（CCW 是实测的负方向）。"""
        return self._signed_move(LIFT_AXIS, -pulses, rpm, acceleration, wait)

    def move_forward(self, pulses: int, *, rpm: Optional[int] = None,
                     acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """伸缩轴前伸 ``pulses`` 个脉冲（CW 是实测的正方向）。"""
        return self._signed_move(EXTENSION_AXIS, pulses, rpm, acceleration, wait)

    def move_backward(self, pulses: int, *, rpm: Optional[int] = None,
                      acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """伸缩轴后缩 ``pulses`` 个脉冲（CCW 是实测的负方向）。"""
        return self._signed_move(EXTENSION_AXIS, -pulses, rpm, acceleration, wait)

    # ------------------------------------------------- 单位制相对动作（带符号）
    def move_base_degrees(self, degrees: float, *, rpm: Optional[int] = None,
                          acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """底座相对转 ``degrees`` 度；正数对应 CW，负数对应 CCW。"""
        pulses = base_pulses_for_degrees(
            degrees,
            pulses_per_revolution=self.pulses_per_revolution,
            reduction_ratio=self.config.base_reduction,
        )
        return self._signed_move(BASE_AXIS, pulses, rpm, acceleration, wait)

    def move_lift_mm(self, millimeters: float, *, rpm: Optional[int] = None,
                     acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """升降轴相对抬升（正）/ 下降（负）``millimeters`` 毫米。"""
        pulses = linear_pulses_for_mm(
            millimeters,
            pulses_per_revolution=self.pulses_per_revolution,
            mm_per_motor_rev=self.config.require_mm_per_motor_rev(LIFT_AXIS),
        )
        return self._signed_move(LIFT_AXIS, pulses, rpm, acceleration, wait)

    def move_radial_mm(self, millimeters: float, *, rpm: Optional[int] = None,
                       acceleration: Optional[int] = None, wait: bool = True) -> MotionReport:
        """伸缩轴相对前伸（正）/ 后缩（负）``millimeters`` 毫米。"""
        pulses = linear_pulses_for_mm(
            millimeters,
            pulses_per_revolution=self.pulses_per_revolution,
            mm_per_motor_rev=self.config.require_mm_per_motor_rev(EXTENSION_AXIS),
        )
        return self._signed_move(EXTENSION_AXIS, pulses, rpm, acceleration, wait)

    # ------------------------------------------------------------- 内部
    def _signed_move(
        self,
        axis_name: str,
        pulses: int,
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
    ) -> MotionReport:
        """发一条带符号的相对动作，并按**实际下发的整数脉冲**推进位姿。

        位姿更新只有这一处：方向原语与单位制方法都走它，所以"相对毫米再调底层动作"
        不会把同一次位移累计两遍。
        """
        self._require_open()
        if pulses == 0:
            # 位移不足半个脉冲：调用方的意图是"已经到位"，既不发帧也不改位置。
            axis = self.config.axes[axis_name]
            return MotionReport(
                motor_id=axis.motor_id,
                pulses=0,
                motor_degrees=0.0,
                output_degrees=0.0 if axis.reduction_ratio is not None else None,
                estimated_seconds=0.0,
            )
        direction = Direction.CW if pulses > 0 else Direction.CCW
        report = self._send(
            axis_name, direction, abs(pulses), rpm=rpm, acceleration=acceleration, wait=wait
        )
        if wait:
            self._advance_pose(axis_name, pulses)
        else:
            # 只发了帧、没等它走完：什么时候到位不知道，位置不可信。
            self.pose_state.mark_unknown()
        return report

    def _send(
        self,
        axis_name: str,
        direction: Direction,
        pulses: int,
        *,
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
    ) -> MotionReport:
        """真正发帧（并可选等待）。参数校验在发帧之前完成。"""
        self._require_safe_pulses(pulses)
        speed = self.config.rpm if rpm is None else rpm
        accel = self.config.acceleration if acceleration is None else acceleration
        # 先算耗时再发命令：参数非法时抛出，串口里不会出现半条动作。
        seconds = estimate_move_seconds(
            pulses, speed, self.pulses_per_revolution, accel, self.settle_seconds
        )
        signed_pulses = pulses if direction is Direction.CW else -pulses
        self._motors[axis_name].move_relative_pulses(signed_pulses, speed, accel)
        if wait:
            # 驱动器只收命令、不回 ACK / 到位包，只能按估算耗时等它走完。
            self._sleep(seconds)
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
        )

    def _advance_pose(self, axis_name: str, signed_pulses: int) -> None:
        """按实际下发的带符号整数脉冲推进位姿。

        位姿本来就未知时什么都不做（不能凭空造一个绝对位置）；已知时按标定换算成
        增量，角度照测量域归一。
        """
        # 直接对**局部**取值判空：``pose_state.is_known`` 是个布尔属性，类型检查器
        # 不会据此把 ``pose`` 缩窄成非空，而"未知时不推进"这条业务规则必须靠这个判断。
        current = self.pose_state.pose
        if current is None:
            return
        if axis_name == BASE_AXIS:
            degrees = base_degrees_for_pulses(
                signed_pulses,
                pulses_per_revolution=self.pulses_per_revolution,
                reduction_ratio=self.config.base_reduction,
            )
            moved = CylindricalPose(
                normalize_degrees(current.base_degrees + degrees),
                current.radial_mm,
                current.height_mm,
            )
        else:
            mm_per_rev = self.config.mm_per_motor_rev(axis_name)
            if mm_per_rev is None:
                # 没有毫米标定就算不出位移，只能承认不知道在哪。
                self.pose_state.mark_unknown()
                return
            millimeters = mm_for_linear_pulses(
                signed_pulses,
                pulses_per_revolution=self.pulses_per_revolution,
                mm_per_motor_rev=mm_per_rev,
            )
            if axis_name == EXTENSION_AXIS:
                moved = CylindricalPose(
                    current.base_degrees,
                    current.radial_mm + millimeters,
                    current.height_mm,
                )
            else:
                moved = CylindricalPose(
                    current.base_degrees,
                    current.radial_mm,
                    current.height_mm + millimeters,
                )
        self.pose_state.record_move(moved)

    def _require_safe_pulses(self, pulses: int) -> None:
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        if not 1 <= pulses <= self.config.max_pulses:
            raise ValueError(f"pulses must be in 1..{self.config.max_pulses}")

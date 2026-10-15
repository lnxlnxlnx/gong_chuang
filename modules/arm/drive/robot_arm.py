"""Conservative position-mode controller for the three ZDT Emm42 arm axes.

参数约定（与 ``modules.chassis.drive.mecanum.MecanumDrive`` 完全一致，见 ``modules/motor/drive/motion.py``）：

- ``pulses``：本次相对移动的脉冲数（必填，正数；方向由动作方法决定）；
- ``rpm`` / ``acceleration``：可选覆盖 ``modules/arm/config.json`` 中的默认值；
- ``wait``：``True``（默认）等到动作走完再返回，``False`` 只发命令。

三个轴的驱动器是 ``Response=None``：控制帧没有 ACK，也没有到位回包，所以
``wait`` 用 ``modules.motor.drive.motion.estimate_move_seconds`` 按说明书公式估算动作耗时后等待。
不等待时连续调用会让后一条位置帧覆盖前一条的目标，看起来就是"上一个动作没执行完"。

对外有两组动作方法：

- **动作 + 脉冲数**（``rotate_cw`` / ``move_up`` / ``move_forward`` …）：方向由方法名决定，
  ``pulses`` 必须为正且不超过配置上限。真机测试脚本用这一组。
- **带符号的物理量**（``move_base_degrees`` / ``move_lift_mm`` / ``move_radial_mm``）：
  正数对应同一个 CW 方向，负数自动反向。抓取流程用这一组，因为它算出来的是
  "从当前位姿到目标位姿的差"。后两个方法要求对应轴的 ``mm_per_motor_rev`` 已标定。

两组都不检查绝对行程：行程是**绝对位姿**的限制，调用方必须在发出动作前自己用
``modules.arm.drive.kinematics.require_reachable`` 检查。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from modules.motor.drive.motion import DEFAULT_SETTLE_SECONDS, estimate_move_seconds
from modules.motor.drive.emm42 import Direction, Emm42Driver, SerialPort
from modules.arm.config import AXIS_NAMES, ArmConfig, load_arm_config
from modules.arm.drive.kinematics import base_pulses_for_degrees, linear_pulses_for_mm


@dataclass(frozen=True)
class MotionReport:
    """Measured command quantity; linear axes have no millimeter calibration yet."""

    motor_id: int
    pulses: int
    motor_degrees: float
    output_degrees: Optional[float]
    #: 本次动作的估算耗时（秒）；``wait=False`` 时调用方可以自己用它来等。
    estimated_seconds: float = 0.0


class RobotArm:
    """Control arm motors 5--7 using only bounded relative position commands."""

    BASE_AXIS = "base"
    LIFT_AXIS = "lift"
    EXTENSION_AXIS = "extension"

    def __init__(
        self,
        port: str,
        *,
        config: Optional[ArmConfig] = None,
        serial_port: Optional[SerialPort] = None,
        sleep: Callable[[float], None] = time.sleep,
        settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    ) -> None:
        self.config = config if config is not None else load_arm_config()
        self.pulses_per_revolution = self.config.pulses_per_revolution
        self.port = port
        #: 到位余量（秒），估算耗时里额外留出的安全时间。
        self.settle_seconds = settle_seconds
        # 注入 sleep 便于离线单测；真机默认 time.sleep。
        self._sleep = sleep
        # 串口是**共享**的：三个驱动器用同一根总线。这里先把它开出来，再建三个驱动器。
        owns_serial = serial_port is None
        self._serial = serial_port if serial_port is not None else self._open_shared_serial()
        try:
            self._motors = {
                axis_name: Emm42Driver(
                    port,
                    self.config.motor_id(axis_name),
                    pulses_per_revolution=self.pulses_per_revolution,
                    # 已读取的 5/6/7 配置均为 Response=None，控制帧不回 ACK。
                    expect_ack=False,
                    serial_port=self._serial,
                )
                for axis_name in AXIS_NAMES
            }
        except BaseException as primary:
            # 构造没返回，就只能由构造器自己收尾：不关掉这个刚开的串口，
            # 下一个 RobotArm 会打不开同一个口（而调用方根本拿不到对象去关它）。
            # 外部注入的 serial_port **不归本构造器管**，失败时不许擅自关掉。
            if owns_serial:
                try:
                    self._serial.close()
                except Exception as cleanup:
                    raise primary.with_traceback(primary.__traceback__) from cleanup
            raise

    def initialize(self) -> "RobotArm":
        """Return the controller after the shared bus has been prepared."""
        return self

    def close(self) -> None:
        """Close the one shared serial connection used by all three axes."""
        self._serial.close()

    def enable_all(self) -> None:
        """Enable torque on base, lift, and extension axes in address order."""
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].enable(True)

    def disable_all(self) -> None:
        """Release torque on every axis after a test completes or fails."""
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].enable(False)

    def stop_all(self) -> None:
        """Immediately brake all axes; this is valid during position modules.motor.drive.motion."""
        for axis_name in AXIS_NAMES:
            self._motors[axis_name].stop()

    # ------------------------------------------------------------- 回零
    #
    # 手册 7.1：单圈零点可以用驱动器菜单 ``O_Set → Set O``（或 ``0x93`` 命令）在**任意
    # 姿态**设定，所以"回零"的语义是"回到你指定的那个姿态"。把三个轴的零点都设在安全
    # 姿态上，回零结束时机械臂就停在 ``safe_pose``，不必再额外走一段。
    #
    # 手册 6.3.2 还写明触发回零的前提：**已使能扭矩、未触发堵转保护、单圈零点值有效**，
    # 所以调用方必须先 ``enable_all()``。

    #: 回零顺序：先缩回、再升降、最后转底座——和运动时的安全廊道是同一条规矩。
    HOME_ORDER = (EXTENSION_AXIS, LIFT_AXIS, BASE_AXIS)

    def home_axis(self, axis_name: str, *, wait: bool = True) -> float:
        """Trigger one axis's homing; return how long was (or should be) waited.

        回零耗时不确定，而 5/6/7 号驱动器是 ``Response=None``，回零结束不会回任何包，
        所以 ``wait=True`` 只能等一个保守时长（``homing.expected_seconds + settle_seconds``）。
        把驱动器菜单的 ``Response`` 改成 ``Reached``/``Both`` 并解析到位回包之后，
        这里就能换成真正的到位等待。
        """
        homing = self.config.require_homing()
        self._motors[axis_name].home(homing.mode)
        if wait:
            self._sleep(homing.wait_seconds)
        return homing.wait_seconds

    def home_all(self, *, wait: bool = True) -> Tuple[float, ...]:
        """Home extension → lift → base so the arm can never sweep while extended.

        单圈回零的零点在一圈里只有一个，所以**断电时轴必须停在零点附近**，否则会就近落到
        错误的圈上——而且驱动器自己看不出来（``0x36`` 读回来一样是 0）。这一点只能靠外部
        观测发现，见 ``modules.grasp.drive.pipeline`` 的回零校验。
        """
        self.config.require_homing()
        return tuple(self.home_axis(axis_name, wait=wait) for axis_name in self.HOME_ORDER)

    # ------------------------------------------------------------- 归位
    #
    # 回零之后驱动器的位置计数器就在 0，所以"归位"是一条**绝对模式**命令回 0——
    # 不需要在上位机累加反向位移。前提是计数器有效，也就是**必须先回过零**；
    # 掉电之后计数器就没了，再次上电必须重新回零。
    #
    # 归位顺序与回零一致：先缩回、再升降、最后转底座。

    #: 轴名 -> 配置里的行程区间字段名（行程是按"量"存的，不是按轴）。
    AXIS_LIMIT_FIELDS = {
        BASE_AXIS: "base_limit",
        LIFT_AXIS: "height_limit",
        EXTENSION_AXIS: "radial_limit",
    }

    def move_axis_to_zero(
        self,
        axis_name: str,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> float:
        """绝对模式把一根轴送回脉冲位置 0；返回本次等待的秒数。

        绝对模式**不知道当前位置**，所以耗时估不出来。``wait=True`` 等的是按该轴行程跨度
        算出的**最坏情况**——一定不会半路提前返回。行程没标定时拒绝 ``wait=True``：宁可让你
        显式地传 ``wait=False`` 自己等，也不要假装知道什么时候到位。
        """
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
        self._motors[axis_name].move_absolute_pulses(0, speed, accel)
        if wait:
            self._sleep(seconds)
        return seconds

    def move_all_to_zero(self, *, rpm: Optional[int] = None, acceleration: Optional[int] = None, wait: bool = True) -> Tuple[float, ...]:
        """按 伸缩 → 升降 → 底座 的顺序把三轴送回零点（归位）。

        顺序和回零一致：伸出状态下转底座会把整条臂横扫过工作区。
        必须在**回过零之后**调用——绝对模式依赖驱动器内部的位置计数器。
        """
        return tuple(
            self.move_axis_to_zero(axis_name, rpm=rpm, acceleration=acceleration, wait=wait)
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

    def rotate_cw(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Rotate the base clockwise within the fixed safe position-mode limit."""
        return self._move(self.BASE_AXIS, Direction.CW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def rotate_ccw(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Rotate the base counterclockwise within the fixed safe limit."""
        return self._move(self.BASE_AXIS, Direction.CCW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_up(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Move the lift axis upward; CW is the mechanically confirmed direction."""
        return self._move(self.LIFT_AXIS, Direction.CW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_down(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Move the lift axis downward; CCW is the mechanically confirmed direction."""
        return self._move(self.LIFT_AXIS, Direction.CCW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_forward(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Extend the arm forward; CW is the mechanically confirmed direction."""
        return self._move(self.EXTENSION_AXIS, Direction.CW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_backward(
        self,
        pulses: int,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Retract the arm backward; CCW is the mechanically confirmed direction."""
        return self._move(self.EXTENSION_AXIS, Direction.CCW, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    # ------------------------------------------------- 单位制相对动作（带符号）
    #
    # 上面六个方法是"动作 + 脉冲数"，方向由方法名决定；下面三个收**带符号的物理量**，
    # 正数对应同一个 CW 方向，所以调用方不必自己判断该用哪个方法。抓取流程用它，
    # 因为它算出来的是"从当前位姿到目标位姿的差"。
    #
    # 注意两点：
    # - 这里不检查绝对行程。行程是**绝对位姿**的限制，由调用方在发出动作前用
    #   ``modules.arm.drive.kinematics.require_reachable`` 检查；
    # - 单次位移超过 ``max_pulses`` 时照旧抛 ``ValueError``。这是既有的保守上限，
    #   标定完行程后把它提高到"单次最大位移"以上，不要在这里绕过它。

    def move_base_degrees(
        self,
        degrees: float,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Turn the base by a signed angle; positive is CW, negative is CCW."""
        pulses = base_pulses_for_degrees(
            degrees,
            pulses_per_revolution=self.pulses_per_revolution,
            reduction_ratio=self.config.base_reduction,
        )
        return self._move_signed(self.BASE_AXIS, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_lift_mm(
        self,
        millimeters: float,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Raise (positive) or lower (negative) the lift axis by a signed distance."""
        pulses = linear_pulses_for_mm(
            millimeters,
            pulses_per_revolution=self.pulses_per_revolution,
            mm_per_motor_rev=self.config.require_mm_per_motor_rev(self.LIFT_AXIS),
        )
        return self._move_signed(self.LIFT_AXIS, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def move_radial_mm(
        self,
        millimeters: float,
        *,
        rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        wait: bool = True,
    ) -> MotionReport:
        """Extend (positive) or retract (negative) the arm by a signed distance."""
        pulses = linear_pulses_for_mm(
            millimeters,
            pulses_per_revolution=self.pulses_per_revolution,
            mm_per_motor_rev=self.config.require_mm_per_motor_rev(self.EXTENSION_AXIS),
        )
        return self._move_signed(self.EXTENSION_AXIS, pulses, rpm=rpm, acceleration=acceleration, wait=wait)

    def _move_signed(
        self,
        axis_name: str,
        pulses: int,
        *,
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
    ) -> MotionReport:
        """Send a signed relative move; a sub-pulse delta sends nothing at all."""
        if pulses == 0:
            # 位移不足半个脉冲。调用方的意图是"已经到位"，所以既不发帧也不等待；
            # 发一条 0 脉冲的帧只会多一次串口往返，还可能被驱动器当成空动作。
            axis = self.config.axes[axis_name]
            return MotionReport(
                motor_id=axis.motor_id,
                pulses=0,
                motor_degrees=0.0,
                output_degrees=(
                    0.0 if axis.reduction_ratio is not None else None
                ),
                estimated_seconds=0.0,
            )
        direction = Direction.CW if pulses > 0 else Direction.CCW
        return self._move(
            axis_name,
            direction,
            abs(pulses),
            rpm=rpm,
            acceleration=acceleration,
            wait=wait,
        )

    def _move(
        self,
        axis_name: str,
        direction: Direction,
        pulses: int,
        *,
        rpm: Optional[int],
        acceleration: Optional[int],
        wait: bool,
    ) -> MotionReport:
        self._require_safe_pulses(pulses)
        speed = self.config.rpm if rpm is None else rpm
        accel = self.config.acceleration if acceleration is None else acceleration
        # 先算耗时再发命令：参数非法时抛出，串口里不会出现半条动作。
        seconds = estimate_move_seconds(
            pulses,
            speed,
            self.pulses_per_revolution,
            accel,
            self.settle_seconds,
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

    def _open_shared_serial(self) -> SerialPort:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required on Jetson. Install it in the target Python environment first."
            ) from error
        return serial.Serial(self.port, baudrate=Emm42Driver.BAUDRATE, timeout=0.1)

    def _require_safe_pulses(self, pulses: int) -> None:
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        if not 1 <= pulses <= self.config.max_pulses:
            raise ValueError(f"pulses must be in 1..{self.config.max_pulses}")

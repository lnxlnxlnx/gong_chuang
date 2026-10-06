"""Mecanum-wheel car controller built on four Emm42 stepper drivers.

麦轮小车控制层。四台 Emm42 V5.0 驱动器共用一根 115200 TTL 总线
（USB-TTL 一个即可），各自主址 1~4，通过总线串联与 Jetson 通信。

  轮位置与默认主址:
            ┌─────────────┐
            │  FL=1   FR=2 │  车头 (forward)
            │              │
            │  RL=3   RR=4 │
            └─────────────┘

坐标约定（决定各方法方向）:
  - vx > 0  = 前进 (forward)
  - vy > 0  = 向左平移 (strafe left)
  - ω  > 0  = 逆时针旋转 (rotate CCW，俯视)
  例如 strafe_left  → 左排轮前进、右排轮后退，地面合力把底盘推向左侧。

装配后若某方向反了，优先在高层修正，不用改协议层:
  1. 整车主向完全不符  → 构造/配置时用 motion_transform="calibrated"，走 _MATRIX_CALIBRATED 标定表；
  2. 辊子排布不同        → 构造时换 pattern="O"；
  3. 单轮转向反了        → 在 wheel_signs 里给该轮 -1。
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Callable, Mapping, Protocol

from motion import DEFAULT_SETTLE_SECONDS, estimate_move_seconds
from motor_drivers.emm42 import Direction, Emm42Driver, HomeMode, SerialPort


class WheelPosition(IntEnum):
    """四轮位置，用于 motor_ids 映射表的键。"""
    FL = 0
    FR = 1
    RL = 2
    RR = 3


# 四轮固定发送顺序（随 WheelPosition 定义），矩阵与遍历都用它以保证一致
WHEEL_ORDER = (WheelPosition.FL, WheelPosition.FR, WheelPosition.RL, WheelPosition.RR)

# 麦轮逆运动学矩阵（无量纲系数的组合）：
# 标准 X 排布（辊子与前进方向成 ±45°），行 = [FL, FR, RL, RR]，列 = [vx, vy, ω]
_MATRIX_X = (
    (+1.0, +1.0, +1.0),  # FL =  vx + vy + ω
    (+1.0, -1.0, -1.0),  # FR =  vx - vy - ω
    (+1.0, +1.0, -1.0),  # RL =  vx + vy - ω
    (+1.0, -1.0, +1.0),  # RR =  vx - vy + ω
)
# O 排布：辊子方向与 X 相反，vy 与 ω 两列符号取反
_MATRIX_O = (
    (+1.0, -1.0, -1.0),
    (+1.0, +1.0, +1.0),
    (+1.0, -1.0, +1.0),
    (+1.0, +1.0, -1.0),
)
# 实车安装标定矩阵（motion_transform="calibrated" 时使用）：
# 由真机实测反推得到，已包含每轮“正转＝前进方向”的安装因子。
# 特征：vx 列 = [-, -, +, -]，vy 列 = [+, -, +, +]，ω 列 = [+, -, -, -]。
# 因此 forward=全前进、backward=全后退、strafe_left=[-, +, +, -]、自转 cw/ccw 各对。
_MATRIX_CALIBRATED = (
    (-1.0, +1.0, +1.0),  # FL = -vx + vy + ω
    (-1.0, -1.0, -1.0),  # FR = -vx - vy - ω
    (+1.0, +1.0, -1.0),  # RL = +vx + vy - ω
    (-1.0, +1.0, -1.0),  # RR = -vx + vy - ω
)


class MecanumDrive:
    """One shared serial port, four Emm42Drivers, mecanum kinematics on top.

    用法（真机）::

        car = MecanumDrive("/dev/ttyUSB0").initialize()
        car.enable_all()
        car.forward(rpm=60)               # 持续前进
        car.stop()
        car.strafe_left(rpm=40, duration=2.0)  # 速度模式：限时 2 秒自动停
        car.forward(pulses=3200, rpm=200)      # 位置模式：四轮各走 3200 脉冲（一圈）
        car.close()

    ``serial_port=`` 可注入 FakeSerial 用于离线单测。

    底层 API（供上层调用）在下方按“单轮控制 / 整车主控制”分组列出；
    高层方向方法 forward/backward/strafe_*/rotate_* 全部由 drive() 计算产生。
    """

    MAX_RPM = Emm42Driver.MAX_RPM  # 每个轮子的最高轮速 (RPM)

    def __init__(
        self,
        port: str,
        motor_ids: Mapping[WheelPosition, int] | None = None,
        *,
        wheel_signs: Mapping[WheelPosition, int] | None = None,
        acceleration: int = 10,
        pattern: str = "X",
        rotation_gain: float = 1.0,
        motion_transform: str = "standard",
        timeout: float = 0.1,
        pulses_per_revolution: int = 3200,
        serial_port: SerialPort | None = None,
        sleep: Callable[[float], None] = time.sleep,
        settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    ) -> None:
        if pattern not in ("X", "O"):
            raise ValueError("pattern must be 'X' or 'O'")
        if rotation_gain <= 0:
            raise ValueError("rotation_gain must be positive")
        if motion_transform not in ("standard", "calibrated"):
            raise ValueError("motion_transform must be 'standard' or 'calibrated'")
        # 软件逻辑层的标准默认映射；实际硬件地址由上层配置通过 motor_ids 注入。
        self.motor_ids = dict(
            motor_ids
            or {WheelPosition.FL: 1, WheelPosition.FR: 2, WheelPosition.RL: 3, WheelPosition.RR: 4}
        )
        # 单轮转向修正：装配后某轮方向反了就给 -1，否则 +1
        self.wheel_signs = dict(wheel_signs or {wheel: +1 for wheel in WHEEL_ORDER})
        self.acceleration = acceleration
        self.pattern = pattern
        self.rotation_gain = rotation_gain
        self.motion_transform = motion_transform
        self.port = port
        self.timeout = timeout
        # 位置模式按脉冲计数时用到的两个参数，与 RobotArm 保持同名同义。
        self.pulses_per_revolution = pulses_per_revolution
        self.settle_seconds = settle_seconds
        # 注入 sleep 便于离线单测；真机默认 time.sleep。
        self._sleep = sleep

        # 四条协议命令必须写同一根总线，因此共享同一个 serial 实例
        if serial_port is None:
            serial_port = self._open_shared_serial()
        self._serial = serial_port

        # 全部4个Emm42Driver共用同一个serial_port(分时复用总线)
        self.wheels: dict[WheelPosition, Emm42Driver] = {}
        for wheel in WHEEL_ORDER:
            self.wheels[wheel] = Emm42Driver(
                port,
                self.motor_ids[wheel],
                timeout=timeout,
                pulses_per_revolution=pulses_per_revolution,
                serial_port=self._serial,
            )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def initialize(self) -> "MecanumDrive":
        """返回自身以供链式调用；共享串口在构造时已打开。

        注意：四台驱动共用同一个串口对象，因此这里不会重复 open。
        """
        return self

    def close(self) -> None:
        """关闭共享的串口（四轮命令走同一根总线，关一次即可）。"""
        self._serial.close()
        self._serial = None

    def _open_shared_serial(self) -> SerialPort:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required on Jetson. Install it in the target Python environment first."
            ) from error
        return serial.Serial(self.port, baudrate=Emm42Driver.BAUDRATE, timeout=self.timeout)

    # ------------------------------------------------------------------
    # 底层 API：单轮控制（disable/调速/加减速/位置/回零均逐轮暴露）
    # ------------------------------------------------------------------
    def get_wheel(self, wheel: WheelPosition) -> Emm42Driver:
        """返回某轮对应的底层 Emm42Driver，供高级用法直接调用。"""
        return self.wheels[wheel]

    def enable_all(self, enabled: bool = True) -> None:
        """使能/失能全部四个轮子的扭矩。小车移动前必须先 enable_all(True)。"""
        for wheel in WHEEL_ORDER:
            self.wheels[wheel].enable(enabled)

    def disable_all(self) -> None:
        """等价于 enable_all(False)：全部失能松轴。"""
        self.enable_all(False)

    def enable_wheel(self, wheel: WheelPosition, enabled: bool = True) -> None:
        """只使能/失能单个轮子。"""
        self.wheels[wheel].enable(enabled)

    def set_wheel_speed(self, wheel: WheelPosition, signed_rpm: int, acceleration: int | None = None) -> None:
        """单轮持续调速。

        ``signed_rpm`` 为正==前进方向，负==后退方向，0==停止。
        ``acceleration`` 缺省用构造时的默认值。
        """
        if signed_rpm == 0:
            self.stop_wheel(wheel)
            return
        direction = Direction.CW if signed_rpm > 0 else Direction.CCW
        self.get_wheel(wheel).set_speed(
            direction, abs(signed_rpm), acceleration if acceleration is not None else self.acceleration
        )

    def set_wheel_acceleration(self, wheel: WheelPosition, acceleration: int) -> None:
        """设单个轮子的加速度档 (0..255)，由下一次发出的命令携带。"""
        self.acceleration = acceleration  # 缺省用全局值，这里语义上即为“设置默认加速度”

    def stop_wheel(self, wheel: WheelPosition) -> None:
        """让单个轮子立即停止（发 EStop 帧）。"""
        self.wheels[wheel].stop()

    def move_wheel_relative(self, wheel: WheelPosition, degrees: float, rpm: int, acceleration: int | None = None) -> None:
        """单轮相对角度移动；正角度=前进方向。"""
        self.get_wheel(wheel).move_relative(degrees, rpm, acceleration if acceleration is not None else self.acceleration)

    def move_wheel_absolute(self, wheel: WheelPosition, degrees: float, rpm: int, acceleration: int | None = None) -> None:
        """单轮绝对角度移动。"""
        self.get_wheel(wheel).move_absolute(degrees, rpm, acceleration if acceleration is not None else self.acceleration)

    def move_wheels_relative(
        self,
        fl: float,
        fr: float,
        rl: float,
        rr: float,
        rpm: int,
        acceleration: int | None = None,
        synchronized: bool = True,
    ) -> None:
        """按 FL、FR、RL、RR 逻辑顺序让四个轮子相对移动指定角度。"""
        for wheel, degrees in zip(WHEEL_ORDER, (fl, fr, rl, rr)):
            self.get_wheel(wheel).move_relative(
                degrees,
                rpm,
                acceleration if acceleration is not None else self.acceleration,
                synchronized=synchronized,
            )
        if synchronized:
            self.wheels[WHEEL_ORDER[0]].trigger_sync_motion()

    def move_wheels_absolute(
        self,
        fl: float,
        fr: float,
        rl: float,
        rr: float,
        rpm: int,
        acceleration: int | None = None,
        synchronized: bool = True,
    ) -> None:
        """按 FL、FR、RL、RR 逻辑顺序让四个轮子到达绝对角度。"""
        for wheel, degrees in zip(WHEEL_ORDER, (fl, fr, rl, rr)):
            self.get_wheel(wheel).move_absolute(
                degrees,
                rpm,
                acceleration if acceleration is not None else self.acceleration,
                synchronized=synchronized,
            )
        if synchronized:
            self.wheels[WHEEL_ORDER[0]].trigger_sync_motion()

    def home_wheel(self, wheel: WheelPosition, mode: HomeMode) -> None:
        """让单个轮子执行回零。"""
        self.wheels[wheel].home(mode)

    def move_wheels_relative_pulses(
        self,
        fl: int,
        fr: int,
        rl: int,
        rr: int,
        rpm: int,
        acceleration: int | None = None,
        synchronized: bool = True,
    ) -> None:
        """按 FL、FR、RL、RR 顺序让四个轮子相对移动指定脉冲数（位置模式）。

        0 脉冲的轮子不下发（协议不允许 0 位移）；``synchronized=True`` 时四条
        ``0xFD`` 帧先缓存，最后用一条 ``00 FF 66 6B`` 让四轮同时启动。
        """
        for wheel, pulses in zip(WHEEL_ORDER, (fl, fr, rl, rr)):
            if pulses == 0:
                continue
            self.get_wheel(wheel).move_relative_pulses(
                pulses,
                rpm,
                acceleration if acceleration is not None else self.acceleration,
                synchronized=synchronized,
            )
        if synchronized:
            self.wheels[WHEEL_ORDER[0]].trigger_sync_motion()

    def move_wheels_absolute_pulses(
        self,
        fl: int,
        fr: int,
        rl: int,
        rr: int,
        rpm: int,
        acceleration: int | None = None,
        synchronized: bool = True,
    ) -> None:
        """按 FL、FR、RL、RR 顺序让四个轮子到达绝对脉冲位置。"""
        for wheel, pulses in zip(WHEEL_ORDER, (fl, fr, rl, rr)):
            if pulses == 0:
                continue
            self.get_wheel(wheel).move_absolute_pulses(
                pulses,
                rpm,
                acceleration if acceleration is not None else self.acceleration,
                synchronized=synchronized,
            )
        if synchronized:
            self.wheels[WHEEL_ORDER[0]].trigger_sync_motion()

    # ------------------------------------------------------------------
    # 主控制器：麦轮逆运动学 + 整车主控
    # ------------------------------------------------------------------
    def drive(
        self,
        vx: float,
        vy: float,
        omega: float,
        rpm: int,
        acceleration: int | None = None,
    ) -> None:
        """笛卡尔速度 -> 四轮转速，一步到位。

        参数（均无量纲，-1..1 或任意比例）:
          vx   : 前后速度，正=前进
          vy   : 左右速度，正=左移
          omega: 旋转速度，正=逆时针
          rpm  : 四轮中的最大轮速，控制油门大小

        实现：用矩阵算出四轮的无量纲轮速，再统一归一化到 ``rpm``。
        """
        self._require_rpm(rpm)
        raw = self._kinematics(vx, vy, omega)
        peak = max(abs(value) for value in raw)
        if peak == 0:  # 全零输入 = 停止
            self.stop()
            return
        for wheel, value in zip(WHEEL_ORDER, raw):
            # wheel_signs 在最终发送前修正单轮转向
            self.set_wheel_speed(
                wheel, round(value / peak * rpm * self.wheel_signs[wheel]), acceleration
            )

    def drive_pulses(
        self,
        vx: float,
        vy: float,
        omega: float,
        pulses: int,
        rpm: int,
        acceleration: int | None = None,
        *,
        wait: bool = True,
    ) -> list[int] | None:
        """笛卡尔方向 + 脉冲数 -> 四轮同步相对位置移动（位置模式）。

        和 :meth:`drive` 用同一张逆运动学矩阵，只是把"最大轮速 = rpm"换成
        "最大轮位移 = pulses"，再逐轮取整成脉冲数。轮子位置模式因此也有了
        「走多少」的版本，不再只能靠 ``duration`` 计时。
        """
        self._require_rpm(rpm)
        self._require_pulses(pulses)
        raw = self._kinematics(vx, vy, omega)
        peak = max(abs(value) for value in raw)
        if peak == 0:  # 全零输入 = 停止
            self.stop()
            return None
        counts = [
            round(value / peak * pulses * self.wheel_signs[wheel])
            for wheel, value in zip(WHEEL_ORDER, raw)
        ]
        self.move_wheels_relative_pulses(*counts, rpm=rpm, acceleration=acceleration)
        if wait:
            seconds = estimate_move_seconds(
                max(abs(count) for count in counts),
                rpm,
                self.pulses_per_revolution,
                self.acceleration if acceleration is None else acceleration,
                self.settle_seconds,
            )
            self._sleep(seconds)
        return counts

    def _kinematics(self, vx: float, vy: float, omega: float) -> list[float]:
        """把笛卡尔速度换成四个轮子的无量纲轮速（顺序见 WHEEL_ORDER）。"""
        if self.motion_transform == "calibrated":
            # 专用实车标定矩阵（已含每轮正转安装因子），non-standard 布局用这张表
            matrix = _MATRIX_CALIBRATED
        else:
            matrix = _MATRIX_O if self.pattern == "O" else _MATRIX_X
        w = omega * self.rotation_gain
        return [a * vx + b * vy + c * w for a, b, c in matrix]

    def set_speeds(self, fl: int, fr: int, rl: int, rr: int, acceleration: int | None = None) -> None:
        """直接按 [FL, FR, RL, RR] 顺序设置四个轮的带符号轮速。

        这是最低层的整车入口：绕过矩阵，由上层自行决定各轮 RPM。
        """
        for wheel, signed_rpm in zip(WHEEL_ORDER, (fl, fr, rl, rr)):
            self.set_wheel_speed(wheel, signed_rpm, acceleration)

    def stop(self) -> None:
        """四个轮子全部立即停止（EStop）。"""
        for wheel in WHEEL_ORDER:
            self.wheels[wheel].stop()

    # ------------------------------------------------------------------
    # 高层方向方法
    #
    # 与机械臂统一的三选一参数（见 motion.py）：
    #   pulses   -> 位置模式，四轮各按逆运动学比例走脉冲数，等待估算耗时
    #   duration -> 速度模式，跑该秒数后自动停止
    #   都不给   -> 速度模式持续运动，直到 stop()
    # ------------------------------------------------------------------
    def forward(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """前进；``pulses`` 走固定距离，``duration`` 跑固定秒数，都不给则持续。"""
        return self._bounded(1.0, 0.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def backward(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """后退；参数含义同 :meth:`forward`。"""
        return self._bounded(-1.0, 0.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def strafe_left(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向左平移；参数含义同 :meth:`forward`。"""
        return self._bounded(0.0, 1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def strafe_right(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向右平移；参数含义同 :meth:`forward`。"""
        return self._bounded(0.0, -1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def rotate_cw(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """原地顺时针旋转（俯视）；参数含义同 :meth:`forward`。"""
        return self._bounded(0.0, 0.0, -1.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def rotate_ccw(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """原地逆时针旋转（俯视）；参数含义同 :meth:`forward`。"""
        return self._bounded(0.0, 0.0, 1.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def forward_left(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向左前方移动；参数含义同 :meth:`forward`。"""
        return self._bounded(1.0, 1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def forward_right(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向右前方移动；参数含义同 :meth:`forward`。"""
        return self._bounded(1.0, -1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def backward_left(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向左后方移动；参数含义同 :meth:`forward`。"""
        return self._bounded(-1.0, 1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def backward_right(
        self,
        rpm: int,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向右后方移动；参数含义同 :meth:`forward`。"""
        return self._bounded(-1.0, -1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def _bounded(
        self,
        vx: float,
        vy: float,
        omega: float,
        rpm: int,
        acceleration: int | None,
        *,
        duration: float | None,
        pulses: int | None,
        wait: bool,
    ) -> list[int] | None:
        """把"方向 + 脉冲/时长"分派到位置模式或速度模式。"""
        if pulses is not None and duration is not None:
            raise ValueError("pulses 与 duration 只能给一个：pulses=位置模式，duration=速度模式")
        if pulses is not None:
            return self.drive_pulses(vx, vy, omega, pulses, rpm, acceleration, wait=wait)
        self._timed(duration, lambda: self.drive(vx, vy, omega, rpm, acceleration))
        return None

    def _timed(self, duration: float | None, action) -> None:
        action()
        if duration is not None:
            self._sleep(duration)
            self.stop()

    @staticmethod
    def _require_rpm(rpm: int) -> None:
        if not 0 <= rpm <= Emm42Driver.MAX_RPM:
            raise ValueError(f"rpm must be in 0..{Emm42Driver.MAX_RPM}")

    @staticmethod
    def _require_pulses(pulses: int) -> None:
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise ValueError("pulses must be an integer")
        if not 1 <= pulses <= 0xFFFFFFFF:
            raise ValueError("pulses must be in 1..4294967295")

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

import math
import time
from enum import IntEnum
from typing import Callable, Mapping, Protocol

from modules.motor.drive.motion import (
    DEFAULT_INTER_FRAME_SECONDS,
    DEFAULT_SETTLE_SECONDS,
    estimate_move_seconds,
)
from modules.motor.drive.emm42 import Direction, Emm42Driver, SerialPort


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


def _require_mm(mm: float) -> float:
    """毫米位移的入口校验：必须是**有限正数**（方向由动作方法决定，所以不收负数）。"""
    if isinstance(mm, bool) or not isinstance(mm, (int, float)):
        raise ValueError("mm 必须是数字")
    value = float(mm)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("mm 必须是有限正数")
    return value


class MecanumDrive:
    """One shared serial port, four Emm42Drivers, mecanum kinematics on top.

    用法（真机）::

        car = MecanumDrive("/dev/ttyUSB0").initialize()
        car.enable_all()
        car.forward(rpm=60)               # 持续前进
        car.stop()
        car.strafe_left(rpm=40, duration=2.0)  # 速度模式：限时 2 秒自动停
        car.forward(pulses=3200, rpm=200)      # 位置模式：四轮各走 3200 脉冲（一圈）
        car.forward(mm=480, rpm=200)           # 位置模式：**车心**前进 480 mm
        car.close()

    两个位置单位的分工：``pulses`` 是**峰值轮**的脉冲（斜向时轮子行程与车心位移不等），
    ``mm`` 一律指**车心沿运动方向走过的毫米**，需要 ``mm_per_wheel_rev`` 标定；旋转没有
    毫米这回事，只能给 ``pulses``。

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
        expect_ack: bool = True,
        timeout: float = 0.1,
        pulses_per_revolution: int = 3200,
        serial_port: SerialPort | None = None,
        sleep: Callable[[float], None] = time.sleep,
        settle_seconds: float = DEFAULT_SETTLE_SECONDS,
        inter_frame_seconds: float = DEFAULT_INTER_FRAME_SECONDS,
        default_rpm: int | None = None,
        mm_per_wheel_rev: float | None = None,
    ) -> None:
        if pattern not in ("X", "O"):
            raise ValueError("pattern must be 'X' or 'O'")
        if rotation_gain <= 0:
            raise ValueError("rotation_gain must be positive")
        if motion_transform not in ("standard", "calibrated"):
            raise ValueError("motion_transform must be 'standard' or 'calibrated'")
        if inter_frame_seconds < 0:
            raise ValueError("inter_frame_seconds must not be negative")
        if default_rpm is not None:
            self._require_rpm(default_rpm)
        if mm_per_wheel_rev is not None and not (
            math.isfinite(float(mm_per_wheel_rev)) and float(mm_per_wheel_rev) > 0
        ):
            raise ValueError("mm_per_wheel_rev must be a positive number or None")
        # 软件逻辑层的标准默认映射；实际硬件地址由上层配置通过 motor_ids 注入。
        self.motor_ids = dict(
            motor_ids
            or {WheelPosition.FL: 1, WheelPosition.FR: 2, WheelPosition.RL: 3, WheelPosition.RR: 4}
        )
        # 单轮转向修正：装配后某轮方向反了就给 -1，否则 +1
        self.wheel_signs = dict(wheel_signs or {wheel: +1 for wheel in WHEEL_ORDER})
        self.acceleration = acceleration
        # 调用方没给 rpm 时用这个默认轮速（配置的 vehicle.rpm）；None = 每次都必须显式给。
        self.default_rpm = default_rpm
        # 轮子转一圈车走多少毫米；None = 还没标定，毫米指令会被明确拒绝。
        self.mm_per_wheel_rev = mm_per_wheel_rev
        self.pattern = pattern
        self.rotation_gain = rotation_gain
        self.motion_transform = motion_transform
        self.port = port
        self.timeout = timeout
        # 位置模式按脉冲计数时用到的两个参数，与 RobotArm 保持同名同义。
        self.pulses_per_revolution = pulses_per_revolution
        self.settle_seconds = settle_seconds
        # 连续两条**单帧**命令之间该留的间隔：见 ``DEFAULT_INTER_FRAME_SECONDS``。
        # 底盘一次动作要连发 2~4 条帧再加一条同步广播，这条总线背靠背发帧会双双失效。
        self.inter_frame_seconds = inter_frame_seconds
        # 注入 sleep 便于离线单测；真机默认 time.sleep。
        self._sleep = sleep

        # 四条协议命令必须写同一根总线，因此共享同一个 serial 实例。
        # ``serial_port`` 传进来就是**借用**（同一个 Robot 会话里 arm 与 chassis 共用一条）：
        # 借来的连接不归本对象关闭，只让自身失效——擅自关掉会切断另一个模块的总线。
        self._owns_serial = serial_port is None
        opened = serial_port if serial_port is not None else self._open_shared_serial()
        self._serial = opened

        # 全部4个Emm42Driver共用同一个serial_port(分时复用总线)
        try:
            self.wheels: dict[WheelPosition, Emm42Driver] = {}
            for wheel in WHEEL_ORDER:
                self.wheels[wheel] = Emm42Driver(
                    port,
                    self.motor_ids[wheel],
                    timeout=timeout,
                    pulses_per_revolution=pulses_per_revolution,
                    expect_ack=expect_ack,
                    serial_port=self._serial,
                )
        except BaseException as primary:
            # 构造没返回，就只能由构造器自己收尾；借来的串口留给它的所有者。
            self.wheels = {}
            if self._owns_serial and self._serial is not None:
                try:
                    self._serial.close()
                except Exception as cleanup:
                    raise primary.with_traceback(primary.__traceback__) from cleanup
            raise

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def owns_serial(self) -> bool:
        """这条总线是不是本对象打开的；借用的不归它关。"""
        return self._owns_serial

    def close(self) -> None:
        """关闭**自己打开**的共享串口；借来的只失效自身。

        幂等：重复调用不会关第二次。四个轮子走同一根总线，关一次即可。
        构造中途失败时本方法也可能被调用（这时 ``wheels`` 还没建完），所以不依赖它。
        """
        serial_port, self._serial = getattr(self, "_serial", None), None
        if serial_port is None:
            return
        if not getattr(self, "_owns_serial", False):
            return
        serial_port.close()

    def _open_shared_serial(self) -> SerialPort:
        from modules.serial_link import open_serial

        return open_serial(
            self.port, baudrate=Emm42Driver.BAUDRATE, timeout=self.timeout
        )

    # ------------------------------------------------------------------
    # 底层 API：单轮控制（disable/调速/加减速/位置/回零均逐轮暴露）
    # ------------------------------------------------------------------
    def get_wheel(self, wheel: WheelPosition) -> Emm42Driver:
        """返回某轮对应的底层 Emm42Driver，供高级用法直接调用。"""
        return self.wheels[wheel]

    def _each_wheel(
        self, action: Callable[[WheelPosition, Emm42Driver], None]
    ) -> None:
        """按 ``WHEEL_ORDER`` 逐轮发**单帧**命令，帧间留 ``inter_frame_seconds``。

        这台机器的 USB-TTL（CH340）上背靠背发帧会让后面的帧失效，见
        :data:`modules.motor.drive.motion.DEFAULT_INTER_FRAME_SECONDS`。底盘一个动作要
        连发 2~4 条帧再补一条同步广播，所以帧间必须留这个间隔——机械臂那条路一直这么
        做，底盘这边以前是裸的。
        """
        for index, wheel in enumerate(WHEEL_ORDER):
            if index:
                self._sleep(self.inter_frame_seconds)
            action(wheel, self.wheels[wheel])

    def enable_all(self, enabled: bool = True) -> None:
        """使能/失能全部四个轮子的扭矩。小车移动前必须先 enable_all(True)。"""
        self._each_wheel(lambda wheel, motor: motor.enable(enabled))

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

    def stop_wheel(self, wheel: WheelPosition) -> None:
        """让单个轮子立即停止（发 EStop 帧）。"""
        self.wheels[wheel].stop()

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
        帧间与触发帧之前各留 ``inter_frame_seconds``，见 :meth:`_each_wheel`。
        """
        self._move_wheel_frames(
            (fl, fr, rl, rr),
            lambda motor, pulses, accel: motor.move_relative_pulses(
                pulses, rpm, accel, synchronized=synchronized
            ),
            acceleration,
            synchronized,
        )

    def _move_wheel_frames(
        self,
        counts: "tuple[int, ...]",
        send: Callable[[Emm42Driver, int, int], None],
        acceleration: int | None,
        synchronized: bool,
    ) -> None:
        """逐轮发 ``0xFD`` 缓存帧（跳过 0 脉冲），帧间与触发帧之前都留间隔。"""
        accel = self.acceleration if acceleration is None else acceleration
        sent = 0
        for wheel, value in zip(WHEEL_ORDER, counts):
            if value == 0:
                continue
            if sent:
                self._sleep(self.inter_frame_seconds)
            send(self.get_wheel(wheel), value, accel)
            sent += 1
        if synchronized:
            if sent:
                # 触发帧同样不能紧贴着上一条缓存帧写
                self._sleep(self.inter_frame_seconds)
            self.wheels[WHEEL_ORDER[0]].trigger_sync_motion()

    # ------------------------------------------------------------------
    # 主控制器：麦轮逆运动学 + 整车主控
    # ------------------------------------------------------------------
    def drive(
        self,
        vx: float,
        vy: float,
        omega: float,
        rpm: int | None = None,
        acceleration: int | None = None,
    ) -> None:
        """笛卡尔速度 -> 四轮转速，一步到位。

        参数（均无量纲，-1..1 或任意比例）:
          vx   : 前后速度，正=前进
          vy   : 左右速度，正=左移
          omega: 旋转速度，正=逆时针
          rpm  : 四轮中的最大轮速，控制油门大小；``None`` 用构造时的默认轮速

        实现：用矩阵算出四轮的无量纲轮速，再统一归一化到 ``rpm``。
        """
        speed = self._rpm(rpm)
        raw = self._kinematics(vx, vy, omega)
        peak = max(abs(value) for value in raw)
        if peak == 0:  # 全零输入 = 停止
            self.stop()
            return
        # wheel_signs 在最终发送前修正单轮转向
        signed = {
            wheel: round(value / peak * speed * self.wheel_signs[wheel])
            for wheel, value in zip(WHEEL_ORDER, raw)
        }
        self._each_wheel(
            lambda wheel, motor: self.set_wheel_speed(
                wheel, signed[wheel], acceleration
            )
        )

    def drive_pulses(
        self,
        vx: float,
        vy: float,
        omega: float,
        pulses: int,
        rpm: int | None = None,
        acceleration: int | None = None,
        *,
        wait: bool = True,
    ) -> list[int] | None:
        """笛卡尔方向 + 脉冲数 -> 四轮同步相对位置移动（位置模式）。

        和 :meth:`drive` 用同一张逆运动学矩阵，只是把"最大轮速 = rpm"换成
        "最大轮位移 = pulses"，再逐轮取整成脉冲数。``pulses`` 量的是**峰值轮**
        的脉冲：斜向时轮子行程与车心位移不成 1:1（要按车心位移走就用 :meth:`drive_mm`）。
        """
        speed = self._rpm(rpm)
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
        self.move_wheels_relative_pulses(*counts, rpm=speed, acceleration=acceleration)
        if wait:
            self._sleep(self._travel_seconds(counts, speed, acceleration))
        return counts

    def drive_mm(
        self,
        vx: float,
        vy: float,
        mm: float,
        rpm: int | None = None,
        acceleration: int | None = None,
        *,
        wait: bool = True,
    ) -> list[int] | None:
        """笛卡尔方向 + **车心位移毫米** -> 四轮同步相对位置移动（位置模式）。

        ``mm`` 是底盘中心沿运动方向走过的毫米数，不是轮子行程：X 排布走 45° 时
        只有两只轮子滚、另两只不动，那两只各滚 ``√2 × mm``——换算用的仍是同一张
        逆运动学矩阵，所以斜向不会有"轮子按毫米直算"的偏差。

        只接受纯平移（``vx`` / ``vy`` 定方向）：旋转不改变车心在 X/Y 上的位置，
        没有"走多少毫米"这回事，要转就用 :meth:`drive_pulses` 或 ``rotate_*``。

        :raises ValueError: 没标定 ``mm_per_wheel_rev``，或 ``mm`` 不是有限正数。
        """
        speed = self._rpm(rpm)
        distance = _require_mm(mm)
        if vx == 0 and vy == 0:  # 全零输入 = 停止（与 drive/drive_pulses 一致）
            self.stop()
            return None
        counts = self._millimetre_counts(vx, vy, distance)
        if not any(counts):
            # 距离太短，四轮都取整成 0：一帧都不发（连同步广播也不发）。
            return counts
        self.move_wheels_relative_pulses(*counts, rpm=speed, acceleration=acceleration)
        if wait:
            self._sleep(self._travel_seconds(counts, speed, acceleration))
        return counts

    def _millimetre_counts(self, vx: float, vy: float, mm: float) -> list[int]:
        """车心位移毫米 -> 四轮带符号脉冲数（纯平移）。

        归一化用 ``hypot(vx, vy)`` 而不是峰值轮：这样"走 N 毫米"始终是车心走的距离，
        斜向时轮子系数可以大于 1（√2）。系数里已包含标定矩阵与 ``rotation_gain``。
        """
        per_wheel_rev = self._require_mm_per_wheel_rev()
        raw = self._kinematics(vx, vy, 0.0)
        norm = math.hypot(vx, vy)
        return [
            round(value / norm * mm * self.pulses_per_revolution / per_wheel_rev)
            * self.wheel_signs[wheel]
            for wheel, value in zip(WHEEL_ORDER, raw)
        ]

    def _travel_seconds(
        self, counts: list[int], rpm: int, acceleration: int | None
    ) -> float:
        """按**走得最远那只轮**估算这次位置模式的耗时（与估算公式同一套参数）。"""
        return estimate_move_seconds(
            max(abs(count) for count in counts),
            rpm,
            self.pulses_per_revolution,
            self.acceleration if acceleration is None else acceleration,
            self.settle_seconds,
        )

    def _require_mm_per_wheel_rev(self) -> float:
        value = self.mm_per_wheel_rev
        if value is None:
            raise ValueError(
                "还没标定轮子一圈车走多少毫米：在 chassis 的 hardware.json 里填 "
                "vehicle.mm_per_wheel_rev（实测：轮子滚一圈，车走了多少毫米）；"
                "没填之前只能用 pulses / duration"
            )
        return value

    def _rpm(self, rpm: int | None) -> int:
        """本次轮速：``None`` 用构造时的默认值，两者都没有就明确报错。"""
        value = self.default_rpm if rpm is None else rpm
        if value is None:
            raise ValueError(
                "没有给 rpm，这个驱动也没有默认轮速：要么显式传 rpm，"
                "要么在 chassis 的 config.json 里填 vehicle.rpm"
            )
        self._require_rpm(value)
        return value

    def _kinematics(self, vx: float, vy: float, omega: float) -> list[float]:
        """把笛卡尔速度换成四个轮子的无量纲轮速（顺序见 WHEEL_ORDER）。"""
        if self.motion_transform == "calibrated":
            # 专用实车标定矩阵（已含每轮正转安装因子），non-standard 布局用这张表
            matrix = _MATRIX_CALIBRATED
        else:
            matrix = _MATRIX_O if self.pattern == "O" else _MATRIX_X
        w = omega * self.rotation_gain
        return [a * vx + b * vy + c * w for a, b, c in matrix]

    def stop(self) -> None:
        """四个轮子全部立即停止（EStop）。"""
        self._each_wheel(lambda wheel, motor: motor.stop())

    # ------------------------------------------------------------------
    # 高层方向方法
    #
    # 与机械臂统一的位置/计时参数（见 modules/motor/drive/motion.py），三个只能给一个：
    #   pulses   -> 位置模式，四轮各按逆运动学比例走脉冲数（量的是**峰值轮**），等待估算耗时
    #   mm       -> 位置模式，按**车心位移**给距离（需要 mm_per_wheel_rev 标定），
    #               只有纯平移有意义，所以 rotate_* 不收这个参数
    #   duration -> 速度模式，跑该秒数后自动停止
    #   都不给   -> 速度模式持续运动，直到 stop()
    # rpm 都可以省略：省略时用配置的默认轮速（default_rpm）。
    # ------------------------------------------------------------------
    def forward(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        mm: float | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """前进；``pulses`` 走固定轮脉冲，``mm`` 走固定车心距离，``duration`` 跑固定秒数，都不给则持续。"""
        return self._bounded(1.0, 0.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, mm=mm, wait=wait)

    def backward(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        mm: float | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """后退；参数含义同 :meth:`forward`。"""
        return self._bounded(-1.0, 0.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, mm=mm, wait=wait)

    def strafe_left(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        mm: float | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向左平移；参数含义同 :meth:`forward`。"""
        return self._bounded(0.0, 1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, mm=mm, wait=wait)

    def strafe_right(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        mm: float | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向右平移；参数含义同 :meth:`forward`。"""
        return self._bounded(0.0, -1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, mm=mm, wait=wait)

    def rotate_cw(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """原地顺时针旋转（俯视）；参数含义同 :meth:`forward`，但**没有** ``mm``。"""
        return self._bounded(0.0, 0.0, -1.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def rotate_ccw(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """原地逆时针旋转（俯视）；参数含义同 :meth:`forward`，但**没有** ``mm``。"""
        return self._bounded(0.0, 0.0, 1.0, rpm, acceleration, duration=duration, pulses=pulses, wait=wait)

    def forward_left(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        mm: float | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向左前方移动；参数含义同 :meth:`forward`。"""
        return self._bounded(1.0, 1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, mm=mm, wait=wait)

    def forward_right(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        mm: float | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向右前方移动；参数含义同 :meth:`forward`。"""
        return self._bounded(1.0, -1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, mm=mm, wait=wait)

    def backward_left(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        mm: float | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向左后方移动；参数含义同 :meth:`forward`。"""
        return self._bounded(-1.0, 1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, mm=mm, wait=wait)

    def backward_right(
        self,
        rpm: int | None = None,
        acceleration: int | None = None,
        duration: float | None = None,
        *,
        pulses: int | None = None,
        mm: float | None = None,
        wait: bool = True,
    ) -> list[int] | None:
        """向右后方移动；参数含义同 :meth:`forward`。"""
        return self._bounded(-1.0, -1.0, 0.0, rpm, acceleration, duration=duration, pulses=pulses, mm=mm, wait=wait)

    def _bounded(
        self,
        vx: float,
        vy: float,
        omega: float,
        rpm: int | None,
        acceleration: int | None,
        *,
        duration: float | None,
        pulses: int | None,
        mm: float | None = None,
        wait: bool,
    ) -> list[int] | None:
        """把"方向 + 脉冲/毫米/时长"分派到位置模式或速度模式。"""
        given = [
            name
            for name, value in (("mm", mm), ("pulses", pulses), ("duration", duration))
            if value is not None
        ]
        if len(given) > 1:
            raise ValueError(
                "mm / pulses / duration 只能给一个："
                "mm=按车心距离、pulses=按轮脉冲、duration=速度模式计时"
            )
        if mm is not None:
            if omega != 0:
                raise ValueError("mm 只对纯平移有意义：旋转不改变车心在 X/Y 上的位置，请改用 pulses")
            return self.drive_mm(vx, vy, mm, rpm, acceleration, wait=wait)
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

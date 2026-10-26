"""转盘舵机封装：把裸舵机位置值收成"分度到某个工位"的语义接口。

转盘和夹爪挂的是**同一条总线舵机总线**（幻尔半双工，115200 bps），两个舵机只靠 ID 区分：
夹爪 ``gripper.servo_id``、转盘 ``turntable.servo_id``。夹爪的上层现在在
:mod:`modules.servo.runtime`（原 ``modules/servo/drive/gripper.py`` 已退出），
两者是同一层的兄弟：共用 ``BusServo``，各自补上自己的语义。

``BusServo`` 只知道"走到 0..1000 的某个位置"，而转盘的三个位置是**物料放取工位**。
这一层补上那个语义：位置来自 ``modules/servo/config.json`` 的 ``turntable.stations``，
所以标定完改配置即可，不必改代码。所有动作都是**绝对位置**命令（``MOVE_TIME_WRITE``），
没有"相对转动多少次"这种接口——分度盘最怕累计误差，绝对位置天然免疫。

## 为什么没有"触发回零"这个方法

机械臂的 Emm42 步进驱动器需要 ``0x9A`` 触发回零来建立绝对参考：它的位置计数器掉电即失效，
所以"回零"是一个**动作**。

**总线舵机不是这样**：它内部一直是绝对位置（``0x1C`` 读回来的就是当前真实角度），上电就
知道自己在哪，没有"回零"这个动作可以触发。因此转盘的"回零"就是走到 ``home`` 工位这**一个**
动作，:meth:`TurntableServo.go_home` 是它的语义入口，:meth:`TurntableServo.go_to` 是同一个
动作的通用形式。

真要把某个姿态定义成新的零点，那是舵机的**偏差标定**（指令 17/18，掉电保存），属于装机时
做一次的事，本层不封装——它会把"转盘现在认为哪是 0"永久改掉，不适合在流程里随手调用。

## 到位判据用的是读回值，不是估算

舵机的 ``POS_READ`` 会回当前真实位置，所以 :meth:`TurntableServo.go_to_station` 等完转动
时长后**读回来核对**，``TurntableReport.reached`` 是一个测量结论。这和机械臂不同：Emm42 的
位置命令没有到位回包，那边只能按说明书的曲线加减速公式估算耗时。转盘挡住/卡住时读回值会
明显偏离目标，``reached`` 为 ``False``——这比"等够时间就当到了"可信得多。

## ``wait=False`` 返回票据，不返回报告

报告的含义是"这次分度的结论"，而 ``wait=False`` 的时候**还没有结论**。所以
``wait=False`` 返回 :class:`modules.motion.MotionTicket`：

```python
ticket = turntable.go_to_station("station_2", wait=False)   # 命令已经发出去了
report = ticket.wait()                                       # 到点读回位置 → 给结论
```

票据到点前**不读串口**（读回是结论的一部分）。要多个模块一起动，就把票据丢进
``robot.motion`` 的动作组，在同一个 join 点上收尾。

## 和夹爪共用一条总线

夹爪与转盘挂的是**同一条**半双工舵机总线（靠 ID 区分），所以 :meth:`TurntableServo.open`
接受一条**已经打开**的 ``bus``：给了就借用，不再自己开第二条（同一个 COM 口开两次会失败），
也不会在 ``close`` 里关掉它——谁开的谁关。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

from modules.motion import MotionTicket
from modules.servo.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_HARDWARE_PATH,
    HOME_STATION,
    TurntableConfig,
    load_turntable_config,
)
from modules.servo.drive.bus_servo import (
    BusServo,
    BusServoConfig,
    SerialPort,
    load_bus_config,
)

#: 舵机位置量程：0..1000。一个单位对应多少度由 ``turntable.travel_degrees`` 给出
#: （本机 240°），那是装机事实，从配置读。
POSITION_UNITS = 1000

#: ``TurntableReport`` 单独构造时（测试、离线打印）用的兜底换算，运行时一律用配置值。
_FALLBACK_DEGREES_PER_UNIT = 240.0 / POSITION_UNITS


@dataclass(frozen=True)
class TurntableReport:
    """一次转盘动作的结果。"""

    #: 工位名；``go_to`` 直接给位置时是 ``"set"``。
    command: str
    target: int
    #: 动作结束后读回的实际位置。
    position: int
    reached: bool
    #: 一个位置单位对应多少度；由配置的 ``turntable.travel_degrees`` 决定。
    degrees_per_unit: float = _FALLBACK_DEGREES_PER_UNIT

    @property
    def error(self) -> int:
        """读回位置减目标位置（舵机量程单位）。"""
        return self.position - self.target

    @property
    def degrees(self) -> float:
        """读回位置对应的名义转角（度）。直连时可当作转盘转角。"""
        return self.position * self.degrees_per_unit

    def summary(self) -> str:
        state = "到位" if self.reached else f"未到位（差 {self.error:+d}）"
        return (
            f"转盘{self.command}：目标 {self.target}"
            f"（{self.target * self.degrees_per_unit:.1f}°），"
            f"读回 {self.position}（{self.degrees:.1f}°），{state}"
        )


class TurntableServo:
    """转盘舵机的分度运动、读回与串口生命周期。"""

    def __init__(
        self,
        port: str,
        *,
        config: Optional[TurntableConfig] = None,
        bus_config: Optional[BusServoConfig] = None,
        servo: Optional[BusServo] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        position_tolerance: Optional[int] = None,
        settle_seconds: Optional[float] = None,
    ) -> None:
        self.config = config if config is not None else load_turntable_config()
        bus = bus_config if bus_config is not None else load_bus_config()
        self.port = port or bus.port
        # 到位容差与停稳余量都来自配置；显式传参只用于测试或单次覆盖。
        position_tolerance = (
            self.config.position_tolerance if position_tolerance is None else position_tolerance
        )
        settle_seconds = (
            self.config.settle_seconds if settle_seconds is None else settle_seconds
        )
        if position_tolerance < 0:
            raise ValueError("position_tolerance must not be negative")
        if settle_seconds < 0:
            raise ValueError("settle_seconds must not be negative")
        self.position_tolerance = position_tolerance
        self.settle_seconds = settle_seconds
        self.degrees_per_unit = self.config.degrees_per_unit
        self._sleep = sleep
        self._clock = clock
        self._servo = (
            servo
            if servo is not None
            else BusServo(
                self.port,
                self.config.servo_id,
                baudrate=bus.baudrate,
            )
        )
        self._closed = False

    # ------------------------------------------------------------- 生命周期
    @classmethod
    def open(
        cls,
        *,
        config_path: Optional[Path] = None,
        hardware_path: Optional[Path] = None,
        bus: Optional[SerialPort] = None,
        owns_bus: Optional[bool] = None,
    ) -> "TurntableServo":
        """读转盘配置 + 舵机总线配置，开总线并使能扭矩。

        :param config_path: 运动调参（``config.json``）；``None`` 用模块默认路径。
        :param hardware_path: 硬件事实（``hardware.json``）；``None`` 用模块默认路径。
        :param bus: 由调用方**已经打开**的舵机总线（``Robot`` 装配夹爪+转盘时用）。
            给了就借用，不再自己开第二条、也不会在 ``close`` 里关掉它。
        :param owns_bus: 显式声明这条 ``bus`` 归不归本转盘；默认"借来的不归"。
        """
        path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        hardware = DEFAULT_HARDWARE_PATH if hardware_path is None else Path(hardware_path)
        config = load_turntable_config(path, hardware)
        # 注意 load_bus_config 的第一个位置参数是 platform，路径只能走关键字。
        bus_config = load_bus_config(hardware_path=hardware)
        if bus is None:
            session = cls(bus_config.port, config=config, bus_config=bus_config)
        else:
            session = cls(
                "",
                config=config,
                bus_config=bus_config,
                servo=BusServo(
                    bus_config.port,
                    config.servo_id,
                    baudrate=bus_config.baudrate,
                    serial_port=bus,
                    owns_serial=bool(owns_bus),
                ),
            )
        try:
            session.initialize()
        except BaseException as primary:
            try:
                session.close()
            except Exception as cleanup:
                raise primary.with_traceback(primary.__traceback__) from cleanup
            raise
        return session

    @property
    def closed(self) -> bool:
        """会话是否已经关闭；关闭后分度与读回统一抛 ``RuntimeError``。"""
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TurntableServo 已关闭")

    def initialize(self) -> "TurntableServo":
        """Open the bus and hold torque so the loaded turntable cannot be back-driven."""
        self._require_open()
        self._servo.initialize()
        self._servo.set_torque(True)
        return self

    def close(self) -> None:
        """Close the bus only; the turntable stays where it is.

        幂等：重复调用不再关第二次。借来的总线（``owns_serial=False``）只让自身失效。
        """
        if self._closed:
            return
        self._closed = True
        self._servo.close()

    def __enter__(self) -> "TurntableServo":
        return self.initialize()

    def __exit__(self, *_exc_info) -> None:
        self.close()

    # ------------------------------------------------------------- 工位
    @property
    def stations(self) -> Mapping[str, int]:
        """Read the configured station table (name → position)."""
        return self.config.stations

    def go_home(
        self, *, duration_ms: Optional[int] = None, wait: bool = True
    ) -> Union[TurntableReport, MotionTicket]:
        """回零：走到 ``home`` 工位（配置里通常是位置 0）。"""
        return self.go_to_station(HOME_STATION, duration_ms=duration_ms, wait=wait)

    def go_to_station(
        self,
        name: str,
        *,
        duration_ms: Optional[int] = None,
        wait: bool = True,
    ) -> Union[TurntableReport, MotionTicket]:
        """走到一个具名工位；名字不认识时在发帧之前就报错。"""
        return self._go_to(name, self.config.station(name), duration_ms=duration_ms, wait=wait)

    def go_to(
        self,
        position: int,
        *,
        duration_ms: Optional[int] = None,
        wait: bool = True,
    ) -> Union[TurntableReport, MotionTicket]:
        """走到任意舵机位置（0..1000）；标定工位时用它。

        这是**原始**入口：它不检查 ``position`` 是不是某个已配置工位，因为标定的时候
        要试的正是"还没成为工位"的位置。流程代码请用 :meth:`go_to_station`。
        """
        return self._go_to("set", position, duration_ms=duration_ms, wait=wait)

    # ------------------------------------------------------------- 状态
    def read_position(self) -> int:
        """Read the servo's current position (absolute; no homing required)."""
        self._require_open()
        return self._servo.read_position()

    def is_at(self, position: int, *, tolerance: Optional[int] = None) -> bool:
        """读回判断：转盘现在是否停在 ``position`` 附近。"""
        return abs(self.read_position() - position) <= self._tolerance(tolerance)

    def at_station(self, name: str, *, tolerance: Optional[int] = None) -> bool:
        """读回判断：转盘现在是否停在具名工位上。"""
        return self.is_at(self.config.station(name), tolerance=tolerance)

    def stop(self) -> None:
        """Brake at the current position."""
        self._require_open()
        self._servo.stop()

    def set_torque(self, enabled: bool) -> None:
        """Load or unload holding torque."""
        self._require_open()
        self._servo.set_torque(enabled)

    # ------------------------------------------------------------- 内部
    def _go_to(
        self,
        command: str,
        target: int,
        *,
        duration_ms: Optional[int],
        wait: bool,
    ) -> Union[TurntableReport, MotionTicket]:
        """发一条定点命令；``wait=True`` 等到点读回给报告，``wait=False`` 给票据。

        两条路都**先发命令再决定怎么等**：``wait=False`` 时一帧都不多读——读回是结论的
        一部分，提前读只会读到命令前的位置。
        """
        self._require_open()
        duration = self.config.duration_ms if duration_ms is None else duration_ms
        # BusServo 会先校验位置和时长再组帧，所以非法取值不会留下半条动作。
        self._servo.set_position(target, duration)
        seconds = self._duration_seconds(duration)
        if not wait:
            return MotionTicket(
                f"turntable:{command}",
                seconds,
                finalize=lambda: self._report(command, target),
                cancel=self._stop_if_open,
                clock=self._clock,
                sleep=self._sleep,
            )
        self._sleep(seconds)
        return self._report(command, target)

    def _duration_seconds(self, duration_ms: int) -> float:
        """这条命令从发出到可以读回结果要等多久：转动时长 + 停稳余量。"""
        return duration_ms / 1000.0 + self.settle_seconds

    def _report(self, command: str, target: int) -> TurntableReport:
        """读回位置并给出结论；``wait=True`` 与票据收尾走的是同一处。"""
        position = self.read_position()
        return TurntableReport(
            command=command,
            target=target,
            position=position,
            reached=abs(position - target) <= self.position_tolerance,
            degrees_per_unit=self.degrees_per_unit,
        )

    def _stop_if_open(self) -> None:
        """尽力停住转盘；会话已经关掉时什么都不做（那时没什么可停的）。"""
        if self._closed:
            return
        self._servo.stop()

    def _tolerance(self, tolerance: Optional[int]) -> int:
        value = self.position_tolerance if tolerance is None else tolerance
        if value < 0:
            raise ValueError("tolerance must not be negative")
        return value


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要硬件：用一个假舵机验证工位分度、回零、读回到位判据与异常路径。"""

    class FakeServo:
        """记录调用，并按脚本返回位置。"""

        def __init__(self, positions) -> None:
            self.calls: list = []
            self._positions = list(positions)

        def initialize(self):
            self.calls.append(("initialize",))
            return self

        def set_torque(self, enabled):
            self.calls.append(("set_torque", enabled))

        def set_position(self, position, duration_ms):
            self.calls.append(("set_position", position, duration_ms))

        def read_position(self):
            self.calls.append(("read_position",))
            return self._positions.pop(0) if self._positions else 0

        def stop(self):
            self.calls.append(("stop",))

        def close(self):
            self.calls.append(("close",))

    config = TurntableConfig(
        servo_id=2,
        duration_ms=1500,
        stations={"home": 0, "station_2": 500, "station_3": 1000},
    )
    # 显式给出总线参数，自测不去读 modules/servo/config.json，避免依赖本机端口配置。
    bus = BusServoConfig(port="FAKE", baudrate=115200, timeout_seconds=0.1)
    slept: list = []

    # 场景一：回零走到 home 工位，读回到位。
    servo = FakeServo([0])
    turntable = TurntableServo(
        "", config=config, servo=servo, sleep=slept.append, bus_config=bus
    )
    turntable.initialize()
    report = turntable.go_home()
    assert report.target == 0 and report.reached, report
    assert ("set_position", 0, 1500) in servo.calls, servo.calls
    assert ("set_torque", True) in servo.calls, servo.calls
    # 到位余量是转动时长 + 停稳余量：1.5 s + 0.2 s。
    assert slept == [1.7], slept
    # 0 位置就是 0°；500 位置是 120°——三个工位正好相隔 120°。
    assert report.degrees == 0.0, report

    # 场景二：其余两个工位，读回值各自对得上，角度换算正确。
    servo = FakeServo([500, 1000])
    turntable = TurntableServo(
        "", config=config, servo=servo, sleep=slept.append, bus_config=bus
    )
    second = turntable.go_to_station("station_2")
    third = turntable.go_to_station("station_3")
    assert (second.target, second.reached) == (500, True), second
    assert (third.target, third.reached) == (1000, True), third
    assert abs(second.degrees - 120.0) < 1e-9, second
    assert abs(third.degrees - 240.0) < 1e-9, third
    assert servo.calls[0] == ("set_position", 500, 1500), servo.calls

    # 场景三：转盘被物料或机构挡住，读回没到目标 —— reached 必须是 False，且这不是异常。
    servo = FakeServo([900])
    turntable = TurntableServo(
        "", config=config, servo=servo, sleep=slept.append, bus_config=bus
    )
    blocked = turntable.go_to_station("station_3")
    assert not blocked.reached, blocked
    assert blocked.error == -100, blocked
    assert "未到位" in blocked.summary(), blocked

    # 容差边界是闭区间，差 30 还算到位、差 31 不算。
    def reaches(reading: int) -> bool:
        return TurntableServo(
            "", config=config, servo=FakeServo([reading]), sleep=slept.append, bus_config=bus
        ).go_to_station("station_3").reached

    assert reaches(970), "差 30 在容差边界上，算到位"
    assert not reaches(969), "差 31 已经出界"

    # 场景四：wait=False 只发命令、不等待、也不读回；结论要等票据到点才有。
    slept.clear()
    now = [0.0]

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    servo = FakeServo([480])
    turntable = TurntableServo(
        "", config=config, servo=servo, sleep=sleep, clock=lambda: now[0], bus_config=bus
    )
    ticket = turntable.go_to_station("station_2", wait=False)
    assert slept == [], "发命令本身不该等"
    assert servo.calls == [("set_position", 500, 1500)], "wait=False 不该读回位置"
    # 票据到点才读回、才给结论：读到的 480 差目标 20，在容差内。
    report = ticket.wait()
    assert report.reached and report.position == 480, report
    assert abs(sum(slept) - 1.7) < 1e-9, slept

    # 场景五：工位名写错时在发帧之前就报错，串口里不会出现半条动作。
    servo = FakeServo([])
    turntable = TurntableServo("", config=config, servo=servo, bus_config=bus)
    try:
        turntable.go_to_station("station_9")
    except ValueError as error:
        assert "station_9" in str(error) and "home" in str(error), error
    else:
        raise AssertionError("未知工位必须报 ValueError")
    assert servo.calls == [], servo.calls

    # 场景六：is_at / at_station 是读回判断，不是"我发过命令"。
    turntable = TurntableServo(
        "", config=config, servo=FakeServo([500, 460]), bus_config=bus
    )
    assert turntable.at_station("station_2"), "500 就在 station_2 上"
    assert not turntable.at_station("station_2"), "460 已经偏出 30 个单位的容差"

    # 场景七：标定用的原始入口能走到任意位置，包括还不是工位的位置。
    servo = FakeServo([250])
    turntable = TurntableServo(
        "", config=config, servo=servo, sleep=slept.append, bus_config=bus
    )
    raw = turntable.go_to(250)
    assert (raw.command, raw.target, raw.reached) == ("set", 250, True), raw
    assert ("set_position", 250, 1500) in servo.calls, servo.calls

    print("转盘封装自测通过：工位分度、回零、读回到位判据与角度换算都不依赖硬件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

"""机械臂夹持点的位姿与位姿来源：整个系统里唯一回答"爪子现在在哪"的地方。

## 为什么单独一个模块

位姿必须**只有一个所有者**：谁要用谁拿到同一个实例。这里采用"唯一所有者 + 显式注入"——
:class:`modules.arm.runtime.RobotArm` 创建并唯一更新它，别的代码只读。

刻意**不做成模块级全局**：全局可变状态在测试里没法隔离，也没法在一个进程里同时管两套硬件；
显式传递让"谁动了电机"在代码里看得见。

## 这一层只记状态，不做判断

它记录位姿**以及位姿是怎么来的**（:data:`UNKNOWN` / :data:`HOMED` / :data:`MEASURED` /
:data:`DEAD_RECKONED`），因为三者的可信度不一样。但"要不要回零""要不要用相机对账"
这些**策略不在这里**：那是调用方的事，这里只把事实摆出来。

## 范围与边界

- 这里的坐标是**爪口中心（实际夹持点）**的基座圆柱坐标，不是相机光心，也不是靶标位置；
- **不保存夹爪业务状态**：夹爪是否有东西属于抓取流程，放进 arm 会让 arm 反向依赖 grasp；
- 进程重启、掉电、急停或手动拨动之后一律是"未知"：新实例从 :data:`UNKNOWN` 开始。

本模块**没有** ``python -m`` 自测入口：它是包导入路径的一部分，不能被当作可执行入口
（否则 ``python -m`` 会报 runpy 的 "found in sys.modules" 警告）。它的行为由
``modules/arm/test/unit/test_robot_arm.py`` 覆盖。

## 两个状态各管一件事

:class:`ArmPose` 回答"爪口在**哪**"（圆柱坐标，依赖标定）；
:class:`MotorTurnsState` 回答"电机**转了多少**"（命令脉冲里程表，一个标定都不需要）。
两者是**兄弟不是子集**：并在一起会让"脉冲未知"跟着"毫米没标定"一起出现，那是错的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from modules.arm.drive.kinematics import CylindricalPose, motor_degrees_for_pulses

#: 位姿来源。
UNKNOWN = "unknown"
#: 回零得到的绝对参考（驱动器零点姿态处、按配置 ``safe_pose`` 记录的夹持点坐标）。
HOMED = "homed"
#: 已标定的相机/靶标测量写进来的。
MEASURED = "measured"
#: 从上一个已知位姿按**实际下发的整数脉冲**累加出来的。
DEAD_RECKONED = "dead_reckoned"

#: 所有合法来源，便于校验与打印。
SOURCES = (UNKNOWN, HOMED, MEASURED, DEAD_RECKONED)


@dataclass
class ArmPose:
    """爪口中心在基座圆柱坐标系里的位置 + 它是怎么来的。

    ``pose`` 为 ``None`` 表示**还不知道**——此时一切绝对定位都必须拒绝，不能凭空造一个
    原点。"命令推算"（:data:`DEAD_RECKONED`）仍然可以连续执行绝对定位，它只是可信度
    比回零/测量低，并不是"不能用"。
    """

    #: 当前的夹持点圆柱位姿；``None`` 表示还不知道。
    pose: Optional[CylindricalPose] = None
    #: 上面这个位姿的来源，取 :data:`UNKNOWN` / :data:`HOMED` / :data:`MEASURED` /
    #: :data:`DEAD_RECKONED`。
    source: str = UNKNOWN

    # ------------------------------------------------------------- 查询
    @property
    def is_known(self) -> bool:
        """位姿是否可用（回零、测量或推算出来的都算，未知不算）。"""
        return self.pose is not None and self.source != UNKNOWN

    @property
    def is_absolute(self) -> bool:
        """位姿是否来自绝对参考（回零或测量），而不是靠命令量累加出来的。

        注意这**不是**"能不能做绝对定位"的开关：按命令推算出来的位姿照样可以继续走
        绝对目标，只是误差会累加。
        """
        return self.pose is not None and self.source in (HOMED, MEASURED)

    def require_pose(self) -> CylindricalPose:
        """返回位姿，或者说明为什么现在不能动。"""
        if not self.is_known or self.pose is None:
            raise RuntimeError(
                "机械臂位姿未知：先显式回零（home_all），"
                "或让已标定的测量写一次位置；不要凭猜测发运动指令"
            )
        return self.pose

    def summary(self) -> str:
        if self.pose is None:
            return "位姿未知（来源 unknown）"
        return f"{self.pose.summary()}（来源 {self.source}）"

    # ------------------------------------------------------------- 更新
    def mark_unknown(self) -> None:
        """把位姿打回未知。

        在任何"有东西动过电机而软件不知道"的时刻调用：急停后、手动拨动后、重新上电后、
        单轴回零后、``wait=False`` 发帧后、发帧中途异常后，或者关闭会话时。
        宁可重新对账，也不要拿一个可能已经错的位姿继续走。
        """
        self.pose = None
        self.source = UNKNOWN

    def record_home(self, pose: CylindricalPose) -> None:
        """记下**整臂回零并等待完成**之后的夹持点位姿。

        只有三轴都回到同一个物理参考时才允许调用；单轴或异步回零不建立参考。
        """
        self.pose = pose
        self.source = HOMED

    def record_measurement(self, pose: CylindricalPose) -> None:
        """记下已标定测量写进来的夹持点位姿。"""
        self.pose = pose
        self.source = MEASURED

    def record_move(self, pose: CylindricalPose) -> None:
        """记下一次动作完成后的位姿（按**实际下发的整数脉冲**推算）。

        只在已经知道位姿时才推进；本来就未知就必须保持未知，不能凭空造一个位置出来。
        """
        if not self.is_known:
            return
        self.pose = pose
        self.source = DEAD_RECKONED


@dataclass(frozen=True)
class MotorTurns:
    """一根电机轴相对**原点**的累计量。

    ``pulses`` 带符号，正号是该轴的正方向（``Direction.CW``）；可以增大也可以减小。
    这里刻意**不做归一**：底座能转整圈，而"一共转了多少度"正是要看的那件事——
    归一化之后连续同向转十圈和没转看起来一样，线缆会绕死而软件不知道。
    """

    axis_name: str
    pulses: int
    pulses_per_revolution: int

    @property
    def motor_degrees(self) -> float:
        return motor_degrees_for_pulses(
            self.pulses, pulses_per_revolution=self.pulses_per_revolution
        )

    @property
    def motor_revolutions(self) -> float:
        return self.pulses / self.pulses_per_revolution

    def summary(self) -> str:
        return (
            f"{self.axis_name}: {self.pulses:+d} 脉冲 "
            f"({self.motor_degrees:+.2f}° / {self.motor_revolutions:+.3f} 圈)"
        )


class MotorTurnsState:
    """三根轴的**命令脉冲里程表**：相对原点一共下发过多少脉冲。

    为什么和 :class:`ArmPose` 分开：``ArmPose`` 需要 ``mm_per_motor_rev`` 这类标定，
    标定缺失就整体变成未知；而脉冲**永远知道**，所以它要独立存活。

    **它记的是下发的脉冲，不是走到的位置。** 丢步、皮带打滑、撞上东西都不会在这里
    体现——所以它有用，但不能当实测用。要和实测对账就比一比 ``0x36``：
    ``Emm42Driver.read_position()`` 给的是编码器测出来的真实位置。

    "什么时候算原点"**不由这里决定**：回零、绝对归位、还是程序启动那一刻，是调用方的
    策略（和 :class:`ArmPose` 同一条规矩：这一层只记状态）。调用方在合适的时刻调
    :meth:`mark_origin`。
    """

    def __init__(self, pulses_per_revolution: int, axis_names: Iterable[str]) -> None:
        if pulses_per_revolution <= 0:
            raise ValueError("pulses_per_revolution must be positive")
        self.pulses_per_revolution = pulses_per_revolution
        self._pulses: Dict[str, int] = {name: 0 for name in axis_names}

    def of(self, axis_name: str) -> MotorTurns:
        """读一根轴的当前累计量。"""
        if axis_name not in self._pulses:
            raise KeyError(
                f"{axis_name!r} 不是已知的轴；已知的是 {sorted(self._pulses)}"
            )
        return MotorTurns(
            axis_name, self._pulses[axis_name], self.pulses_per_revolution
        )

    def record(self, axis_name: str, signed_pulses: int) -> MotorTurns:
        """把一次**已经下发成功**的带符号整数脉冲累加进去，返回累加后的读数。

        负数就是减回去，所以它天然支持"转了再转回来"。
        """
        if isinstance(signed_pulses, bool) or not isinstance(signed_pulses, int):
            raise ValueError("signed_pulses must be an integer")
        self.of(axis_name)
        self._pulses[axis_name] += signed_pulses
        return self.of(axis_name)

    def mark_origin(self, axis_name: Optional[str] = None) -> None:
        """把现在这一刻定义成原点，清零累计量；给轴名就只清那一根。

        回零或绝对归位之后调用（那时轴都在驱动器零点上）。**什么时候算原点由调用方
        决定**，这里只提供动作、不提供策略。
        """
        if axis_name is None:
            for name in self._pulses:
                self._pulses[name] = 0
            return
        self.of(axis_name)
        self._pulses[axis_name] = 0

    def summary(self) -> str:
        return "；".join(
            self.of(name).summary() for name in sorted(self._pulses)
        )

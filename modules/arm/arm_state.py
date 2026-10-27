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

## 只记账，不做判断

:class:`MotorTurnsState` 回答"电机**转了多少**"（命令脉冲里程表，一个标定都不需要），
所以它永远有值。绝对坐标那套（爪口在哪）已经退出：抓取走的是脉冲闭环，不需要位姿推算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

from modules.arm.drive.kinematics import motor_degrees_for_pulses

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


    **它记的是下发的脉冲，不是走到的位置。** 丢步、皮带打滑、撞上东西都不会在这里
    体现——所以它有用，但不能当实测用。要和实测对账就比一比 ``0x36``：
    ``Emm42Driver.read_position()`` 给的是编码器测出来的真实位置。

    "什么时候算原点"**不由这里决定**：回零、绝对归位、还是程序启动那一刻，是调用方的
    策略。调用方在合适的时刻调 :meth:`mark_origin`。

    ## 启程读数（``initial_pulses``）

    默认每根轴都从 0 开始，意思是"程序启动那一刻它就在原点上"。这个默认在结构上并不
    总是成立：换了爪子、重新对齿或重装电机之后，某根轴的**机械初始位**可能离原点一整个
    角度（例如伸缩轴停在 270° = 2400 脉冲 = 45.75 mm 处）。这时由调用方在构造时**如实
    声明**启程读数，而不是让软件继续以为它在原点。

    声明的是**事实，不是动作**：它既不发电也不动轴，错了就是这条路上所有绝对目标整体
    偏一个常数。回零或绝对归位一旦重新定义原点（:meth:`mark_origin`），启程读数照旧被
    清成 0——那时轴真的停在驱动器零点上，启程读数已经没有意义。
    """

    def __init__(
        self,
        pulses_per_revolution: int,
        axis_names: Iterable[str],
        initial_pulses: Optional[Mapping[str, int]] = None,
    ) -> None:
        """``initial_pulses`` 是"程序启动那一刻每根轴在哪"的声明；没给的轴按 0。"""
        if pulses_per_revolution <= 0:
            raise ValueError("pulses_per_revolution must be positive")
        self.pulses_per_revolution = pulses_per_revolution
        self._pulses: Dict[str, int] = {name: 0 for name in axis_names}
        declared = {} if initial_pulses is None else initial_pulses
        for axis_name, pulses in declared.items():
            self.of(axis_name)  # 轴名非法就 KeyError，与 record 一致
            if isinstance(pulses, bool) or not isinstance(pulses, int):
                raise ValueError(f"initial_pulses[{axis_name!r}] must be an integer")
            self._pulses[axis_name] = pulses

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

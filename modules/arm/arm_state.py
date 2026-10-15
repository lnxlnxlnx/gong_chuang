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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.arm.drive.kinematics import CylindricalPose

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


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要硬件：验证来源标记、未知保护与"未知不推进"。"""
    state = ArmPose()

    # 新实例必须从"未知"开始：状态只在内存里，掉电即失效。
    assert not state.is_known and not state.is_absolute
    assert state.source == UNKNOWN
    try:
        state.require_pose()
    except RuntimeError as error:
        assert "未知" in str(error), error
    else:
        raise AssertionError("未知位姿必须拒绝提供")

    # 未知状态下推进位姿是无效的：不能凭空造一个。
    state.record_move(CylindricalPose(10.0, 50.0, 30.0))
    assert not state.is_known, state

    # 回零给出绝对参考。
    state.record_home(CylindricalPose(0.0, 40.0, 60.0))
    assert state.is_known and state.is_absolute and state.source == HOMED

    # 走一步之后变成推算：可信度下降，但仍可继续做绝对定位。
    state.record_move(CylindricalPose(5.0, 45.0, 55.0))
    assert state.source == DEAD_RECKONED and not state.is_absolute and state.is_known

    # 测量重新给出绝对参考。
    state.record_measurement(CylindricalPose(5.0, 45.0, 55.0))
    assert state.is_absolute and state.source == MEASURED

    # 打回未知之后拒绝继续运动。
    state.mark_unknown()
    assert not state.is_known and state.pose is None

    print("机械臂位姿自测通过：来源标记、未知保护、未知不推进都正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

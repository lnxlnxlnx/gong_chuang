"""机械臂与夹爪的当前状态：整个系统里唯一能回答"现在在哪"的地方。

电机和舵机在别处也会被驱动，所以状态**只该有一个所有者**，谁要用谁拿到同一个实例。
这里用"唯一所有者 + 显式注入"：

```python
state = ArmState()
pipeline = GraspPipeline(..., state=state)     # 抓取流程往里写
...                                            # 别的控制代码读同一个 state
```

刻意**不做成模块级全局**：全局可变状态在测试里没法隔离，也没法在一个进程里同时管两套硬件；
显式传递则让"谁动了电机"在代码里看得见。

## 这一层只记状态，不做判断

它记录位姿**以及位姿是怎么来的**（:data:`HOMED` / :data:`MEASURED` /
:data:`DEAD_RECKONED`），因为三者的可信度不一样，调用方需要据此决定下一步怎么做。
但"要不要回零""要不要用相机对账""够不够信"这些**策略不在这里**——
那是 main.py 的事，这里只把事实摆出来。

## 进程重启后必须是"未知"

状态只在内存里，掉电即失效。所以新实例一律从 :data:`UNKNOWN` 开始；
:meth:`ArmState.mark_unknown` 用来在"有人动过电机而我不知道"时把它打回未知
（急停后、手动拨动后、捕捉到异常之后）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.arm.drive.kinematics import CylindricalPose

#: 位姿来源。
UNKNOWN = "unknown"
#: 回零得到的绝对参考。
HOMED = "homed"
#: 相机测到的。
MEASURED = "measured"
#: 从上一个已知位姿按命令量累加出来的。
DEAD_RECKONED = "dead_reckoned"


@dataclass
class ArmState:
    """机械臂位姿与夹爪状态的唯一真相源。"""

    #: 当前圆柱位姿；``None`` 表示还不知道。
    pose: Optional[CylindricalPose] = None
    #: 上面这个位姿的来源，取 :data:`UNKNOWN` / :data:`HOMED` / :data:`MEASURED` /
    #: :data:`DEAD_RECKONED`。
    source: str = UNKNOWN
    #: 夹爪舵机最近一次读回的位置值。
    gripper_position: Optional[int] = None
    #: 夹爪里是否有东西；``None`` 表示还没闭合过。
    gripper_holds: Optional[bool] = None

    # ------------------------------------------------------------- 查询
    @property
    def is_known(self) -> bool:
        """位姿是否可用（回零、测量或推算出来的都算，未知不算）。"""
        return self.pose is not None and self.source != UNKNOWN

    @property
    def is_absolute(self) -> bool:
        """位姿是否来自绝对参考（回零或测量），而不是靠命令量累加出来的。"""
        return self.pose is not None and self.source in (HOMED, MEASURED)

    def require_pose(self) -> CylindricalPose:
        """返回位姿，或者说明为什么现在不能动。"""
        if not self.is_known or self.pose is None:
            raise RuntimeError(
                "机械臂位姿未知；先回零或用相机测一次靶标，不要凭猜测发运动指令"
            )
        return self.pose

    def summary(self) -> str:
        if self.pose is None:
            located = "位姿未知"
        else:
            located = f"{self.pose.summary()}（来源 {self.source}）"
        if self.gripper_holds is None:
            jaws = "夹爪未动作"
        else:
            jaws = (
                f"夹爪 {self.gripper_position}，"
                + ("夹着东西" if self.gripper_holds else "空夹")
            )
        return f"{located}；{jaws}"

    # ------------------------------------------------------------- 更新
    def mark_unknown(self) -> None:
        """把位姿打回未知。

        在任何"有东西动过电机而软件不知道"的时刻调用：急停后、手动拨动后、重新上电后、
        或者捕捉到异常之后。宁可重新对账，也不要拿一个可能已经错的位姿继续走。
        """
        self.pose = None
        self.source = UNKNOWN

    def record_home(self, pose: CylindricalPose) -> None:
        """记下回零之后机械臂所在的位姿。"""
        self.pose = pose
        self.source = HOMED

    def record_measurement(self, pose: CylindricalPose) -> None:
        """记下相机测到的位姿。"""
        self.pose = pose
        self.source = MEASURED

    def record_move(self, pose: CylindricalPose) -> None:
        """记下一次动作完成后的位姿（按命令量推算）。

        只在已经知道位姿时才推进；本来就未知就必须保持未知，不能凭空造一个。
        """
        if not self.is_known:
            return
        self.pose = pose
        self.source = DEAD_RECKONED

    def record_gripper(self, *, position: int, reached: bool) -> None:
        """记下夹爪读回值。

        ``reached=False`` 说明爪口没走到闭合位，也就是**夹到了东西**：这是总线舵机的
        位置读回白送的抓取检测。
        """
        self.gripper_position = int(position)
        self.gripper_holds = not reached


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要硬件：验证来源标记、共享语义与"未知"保护。"""
    state = ArmState()

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
    assert state.pose == CylindricalPose(0.0, 40.0, 60.0)

    # 走一步之后变成推算，可信度下降但位姿仍然可用。
    state.record_move(CylindricalPose(5.0, 45.0, 55.0))
    assert state.source == DEAD_RECKONED and not state.is_absolute and state.is_known

    # 相机测量重新给出绝对参考。
    state.record_measurement(CylindricalPose(5.0, 45.0, 55.0))
    assert state.is_absolute and state.source == MEASURED

    # 夹爪读回：没到闭合位 = 夹到了东西。
    state.record_gripper(position=430, reached=False)
    assert state.gripper_holds is True
    state.record_gripper(position=500, reached=True)
    assert state.gripper_holds is False

    # 打回未知之后拒绝继续运动，但夹爪事实保留（那是已经发生的事）。
    state.mark_unknown()
    assert not state.is_known and state.gripper_holds is False

    # 同一个实例注入到别处就是同一份状态（这就是"唯一所有者"的意思）。
    shared = ArmState()
    other_reference = shared
    shared.record_home(CylindricalPose(0.0, 10.0, 20.0))
    assert other_reference.pose == CylindricalPose(0.0, 10.0, 20.0)

    print("机械臂状态自测通过：来源标记、共享语义、未知保护都正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

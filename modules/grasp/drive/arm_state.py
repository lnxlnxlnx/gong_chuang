"""机械臂与夹爪的当前状态：整个抓取系统里唯一能回答"现在在哪"的地方。

## 为什么需要一个状态对象

位姿有三个来源，可信度完全不同：

| 来源 | 怎么来的 | 可信度 |
|---|---|---|
| ``homed`` | 回零，绝对参考 | 高；但"就近落到错误的圈"时驱动器自己看不出来 |
| ``measured`` | 相机测爪上靶标 | 受标定残差限制，但能发现上面那种情况 |
| ``dead_reckoned`` | 回零/测量之后按命令量累加 | 会随背隙和堵转累积，走多了必须对账 |

所以这里记的不是"位姿 + 一个对不对的布尔值"，而是**位姿 + 它是怎么来的**。
调用方据此决定下一步要不要用相机对账（:attr:`ArmState.needs_verification`）。

单圈回零最阴的地方正在于：**回零落到错误的圈上时，驱动器给不出任何异常信号**——
``0x36`` 读回来一样是 0（它只有一圈的读数）。只有外部观测（相机看靶标）能发现。

## 为什么不做成全局单例

模块级全局状态在测试里没法隔离，也没法在一个进程里同时管两套硬件。这里用
**唯一所有者 + 显式注入**：状态只活在一个 :class:`ArmState` 实例里，谁要用谁拿到引用。
``GraspPipeline`` 默认自己建一个；需要跨模块共享时把它传进去。

## 进程重启后必须是"未知"

状态只在内存里，掉电即失效。所以新实例一律从 :data:`UNKNOWN` 开始，
:meth:`ArmState.mark_unknown` 用来在"有人动过电机而我不知道"时把它打回未知。
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

#: 连续死推算多少步之后建议做一次相机对账。默认每步都对账：抓取流程本来就是每轮取帧，
#: 不额外花时间；把它调大就是"要速度、容忍漂移"。
DEFAULT_VERIFICATION_INTERVAL = 1


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
    #: 上次相机对账之后又走了多少步。
    moves_since_verification: int = 0
    #: 死推算多少步之后建议对账。
    verification_interval: int = DEFAULT_VERIFICATION_INTERVAL

    # ------------------------------------------------------------- 查询
    @property
    def is_known(self) -> bool:
        """位姿是否可用（回零、测量或推算出来的都算，未知不算）。"""
        return self.pose is not None and self.source != UNKNOWN

    @property
    def is_absolute(self) -> bool:
        """位姿是否来自绝对参考（回零或测量），而不是累加出来的。"""
        return self.pose is not None and self.source in (HOMED, MEASURED)

    @property
    def needs_verification(self) -> bool:
        """是否该用相机对一次账。

        绝对参考本身不用对账；只有死推算走够步数之后才需要——这一步同时兜住
        "回零落到错误的圈"和"被挡住却没发现"两种驱动器看不见的失效。
        """
        if not self.is_known:
            return False
        if self.source != DEAD_RECKONED:
            return False
        return self.moves_since_verification >= self.verification_interval

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

        在任何"有东西动过电机而软件没数"的时刻调用：急停后、手动拨动后、
        重新上电后、或者捕捉到异常之后。宁可重新对账，也不要拿一个可能已经错的位姿继续走。
        """
        self.pose = None
        self.source = UNKNOWN
        self.moves_since_verification = 0

    def record_home(self, pose: CylindricalPose) -> None:
        """记下回零之后机械臂所在的位姿（应当就是 ``safe_pose``）。"""
        self.pose = pose
        self.source = HOMED
        self.moves_since_verification = 0

    def record_measurement(self, pose: CylindricalPose) -> None:
        """记下相机测到的位姿。"""
        self.pose = pose
        self.source = MEASURED
        self.moves_since_verification = 0

    def record_move(self, pose: CylindricalPose) -> None:
        """记下一次动作完成后的位姿（按命令量推算）。

        只在已经知道位姿时才推进；本来就未知就必须保持未知，不能凭空造一个。
        """
        if not self.is_known:
            return
        self.pose = pose
        self.source = DEAD_RECKONED
        self.moves_since_verification += 1

    def record_gripper(self, *, position: int, reached: bool) -> None:
        """记下夹爪读回值。

        ``reached=False`` 说明爪口没走到闭合位，也就是**夹到了东西**：这是总线舵机的
        位置读回白送的抓取检测。
        """
        self.gripper_position = int(position)
        self.gripper_holds = not reached


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要硬件：验证来源跟踪、对账策略和"未知"保护。"""
    state = ArmState()

    # 新实例必须从"未知"开始：状态只在内存里，掉电即失效。
    assert not state.is_known
    assert not state.needs_verification
    try:
        state.require_pose()
    except RuntimeError as error:
        assert "未知" in str(error), error
    else:
        raise AssertionError("未知位姿必须拒绝提供")

    # 未知状态下推进位姿是无效的：不能凭空造一个。
    state.record_move(CylindricalPose(10.0, 50.0, 30.0))
    assert not state.is_known, state

    # 回零给出绝对参考，本身不需要对账。
    state.record_home(CylindricalPose(0.0, 40.0, 60.0))
    assert state.is_known and state.is_absolute and state.source == HOMED
    assert not state.needs_verification

    # 走一步之后变成死推算，默认下一步就该对账。
    state.record_move(CylindricalPose(5.0, 45.0, 55.0))
    assert state.source == DEAD_RECKONED and not state.is_absolute
    assert state.needs_verification

    # 对账之后回到"已测量"，计数清零。
    state.record_measurement(CylindricalPose(5.0, 45.0, 55.0))
    assert state.source == MEASURED and not state.needs_verification

    # 把对账间隔调大就不会每步都要求对账。
    lazy = ArmState(pose=CylindricalPose(0.0, 40.0, 60.0), source=HOMED, verification_interval=3)
    for _ in range(2):
        lazy.record_move(CylindricalPose(0.0, 40.0, 60.0))
    assert not lazy.needs_verification, lazy
    lazy.record_move(CylindricalPose(0.0, 40.0, 60.0))
    assert lazy.needs_verification, lazy

    # 夹爪读回：没到闭合位 = 夹到了东西。
    state.record_gripper(position=430, reached=False)
    assert state.gripper_holds is True
    state.record_gripper(position=500, reached=True)
    assert state.gripper_holds is False

    # 打回未知之后一切归零，且拒绝继续运动。
    state.mark_unknown()
    assert not state.is_known and state.gripper_holds is False

    print("机械臂状态自测通过：来源跟踪、对账策略、未知保护都正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

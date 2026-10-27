"""转动/停稳门控：**先看见它真的动过，才承认它停稳了**。

## 为什么不能只看"连着几帧读数一样"

``modules.vision.drive.sequence`` 的 :attr:`Fusion.moving` 回答的是"这一批连拍里读数有没有
系统性漂移"。它答不了另一个问题：

    转盘**本来就静止**时，读数连续几帧一模一样——看起来和"转完停稳了"完全一致。

把这两种情况混为一谈，现场表现就是"上一轮的残留状态被当成这一轮的结果上报"：转盘还没
开始转，上报已经发出去了。这不是精度问题，是**逻辑错误**，而且它只在"刚好赶上静止的
那一瞬间"才复现，最难查。

所以本模块把"动过"变成一个**必须显式观察到的事实**（:attr:`MotionVerdict.ever_moved`），
而不是从"现在很静"反推出来的结论。判据分两级：

1. **动过**：连续若干帧速度超过 ``moving_speed``（默认 30 px/s），才置位。置位之后
   **只有** :meth:`MotionGate.reset` 能清掉——一次漏检不抹掉"这一轮确实转过"这个事实。
2. **停稳**：动过之后，连续若干帧速度低于 ``near_stop_speed``（默认 25 px/s）**且**
   帧间间隔落在合理范围内，才算停稳。

两个阈值差一截（30 对 25）是刻意的：迟滞（hysteresis）让"33→28→32 px/s"这种在阈值上
抖动的读数不会把状态机来回翻面。

## 速度从哪来

相邻两帧的位置差除以时间差。位置是**语义量**（物料的画面坐标、圆环圆心、任何可比的东西），
由调用方给出像素坐标；本模块不关心它是什么，只关心它变不变、变多快。

**刻意不做跨帧的目标匹配**（那是"≤3 件时穷举 3! 排列"那一类工作）：匹配是另一件事，
需要知道颜色、位置、数量，属于调用方。本模块只处理"同一个目标在两个时刻的位置"——
调用方负责保证两次给的确实是同一个目标。这样本模块才只依赖标准库、能完全离线测试。

命令行自测::

    python -m modules.vision.drive.motion_gate        # 离线自测，不需要摄像头
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

#: 状态名。用字符串而不是 Enum：它是给人看的（打印、日志），而且要被测试直接比对。
STATE_IDLE = "idle"
STATE_MOVING = "moving"
STATE_SETTLED = "settled"

#: 认为"确实在动"的速度下限（像素/秒）。
DEFAULT_MOVING_SPEED_PX_S = 30.0

#: 认为"已经接近停下"的速度上限（像素/秒）。必须**小于** ``moving_speed`` 才有迟滞。
DEFAULT_NEAR_STOP_SPEED_PX_S = 25.0

#: "确实在动"要连续满足几帧。
DEFAULT_REQUIRED_MOVING_FRAMES = 2

#: "已经停下"要连续满足几帧。
DEFAULT_REQUIRED_LOW_SPEED_FRAMES = 3

#: 两帧间隔的合理范围（秒）。太短（同一帧被处理两次）算不出速度；太长（中间掉了一段）
#: 算出来的平均速度会把"绕着走了一大圈又回来"当成"没动"。
DEFAULT_MIN_INTERVAL_S = 0.02
DEFAULT_MAX_INTERVAL_S = 0.25

#: 一个测量周期结束（刚上报过）之后的状态名。
STATE_REPORTED = "reported"


@dataclass(frozen=True)
class MotionVerdict:
    """一次 :meth:`MotionGate.observe` 的结论。"""

    #: ``idle`` / ``moving`` / ``settled`` / ``reported``。
    state: str
    #: **本周期内曾经**确认过转动。置位后不会因为漏检而清掉，只有 ``reset()`` 才清。
    ever_moved: bool
    #: 这一帧是否应当上报。**同一个周期只会为 True 一次**——上报过就锁存到下一轮转动。
    report_now: bool
    #: 本帧的速度（像素/秒）；算不出速度（第一帧、间隔不合理、漏检）时是 ``None``。
    speed_px_s: Optional[float]
    #: 连续满足"在动"的帧数。
    moving_frames: int
    #: 连续满足"接近停下"的帧数。
    low_speed_frames: int

    @property
    def settled(self) -> bool:
        """是不是"动过并且已经停稳"——可以按这个结果取值了。"""
        return self.state in (STATE_SETTLED, STATE_REPORTED)

    def summary(self) -> str:
        """一行中文结论，供终端打印。"""
        speed = "算不出速度" if self.speed_px_s is None else f"{self.speed_px_s:.0f} px/s"
        if self.state == STATE_IDLE:
            state = "还没看见它动过（静止不等于停稳）"
        elif self.state == STATE_MOVING:
            state = "正在动"
        elif self.state == STATE_SETTLED:
            state = "已停稳"
        else:
            state = "已停稳（本周期已上报）"
        return (
            f"{state}：{speed}，在动 {self.moving_frames} 帧 / 低速 {self.low_speed_frames} 帧，"
            f"本周期动过={self.ever_moved}"
        )


class MotionGate:
    """把"位置 + 时刻"的序列收成"动过没有 / 停稳没有 / 该不该上报"。

    Parameters:
        moving_speed_px_s / near_stop_speed_px_s: 两个速度阈值（见模块文档的迟滞说明）。
        required_moving_frames / required_low_speed_frames: 各自要连续满足几帧。
        min_interval_s / max_interval_s: 帧间隔的合理范围；超出范围的间隔**算不出速度**
            （返回 ``None``），但那一次观测的位置仍然成为下一次的基准——宁可少算一次速度，
            也不要拿一段跨了掉帧的间隔去算平均速度。
        key: 可选的目标标识（颜色名、编号）。换了目标就**重新开始**计数，
            并且立刻清掉 ``ever_moved``：这一轮动过的是**上一个**目标。

    **状态机**（``idle`` → ``moving`` → ``settled`` → ``reported`` → ``idle``）::

        静止时一直 idle（ever_moved=False，永不上报）
        速度连续 2 帧 ≥30 px/s  -> moving，ever_moved=True
        速度连续 3 帧 ≤25 px/s  -> settled，report_now=True，随后 reported
        再次转动                -> 重新开始，下一轮又能上报一次
    """

    def __init__(
        self,
        *,
        moving_speed_px_s: float = DEFAULT_MOVING_SPEED_PX_S,
        near_stop_speed_px_s: float = DEFAULT_NEAR_STOP_SPEED_PX_S,
        required_moving_frames: int = DEFAULT_REQUIRED_MOVING_FRAMES,
        required_low_speed_frames: int = DEFAULT_REQUIRED_LOW_SPEED_FRAMES,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        max_interval_s: float = DEFAULT_MAX_INTERVAL_S,
        key: Optional[str] = None,
    ) -> None:
        if near_stop_speed_px_s > moving_speed_px_s:
            # 反过来的话状态机会在阈值附近来回翻面，"停稳"永远等不到。
            raise ValueError(
                "near_stop_speed_px_s 不能大于 moving_speed_px_s"
                f"（{near_stop_speed_px_s} > {moving_speed_px_s}，两者之间要有迟滞）"
            )
        for name, value in (
            ("moving_speed_px_s", moving_speed_px_s),
            ("near_stop_speed_px_s", near_stop_speed_px_s),
            ("min_interval_s", min_interval_s),
            ("max_interval_s", max_interval_s),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负的有限数，实际是 {value!r}")
        if max_interval_s <= min_interval_s:
            raise ValueError("max_interval_s 必须大于 min_interval_s")
        for name, value in (
            ("required_moving_frames", required_moving_frames),
            ("required_low_speed_frames", required_low_speed_frames),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须是 ≥1 的整数，实际是 {value!r}")

        self.moving_speed_px_s = float(moving_speed_px_s)
        self.near_stop_speed_px_s = float(near_stop_speed_px_s)
        self.required_moving_frames = required_moving_frames
        self.required_low_speed_frames = required_low_speed_frames
        self.min_interval_s = float(min_interval_s)
        self.max_interval_s = float(max_interval_s)
        self.key = key
        self.reset()

    # ------------------------------------------------------------- 生命周期
    def reset(self) -> None:
        """回到"什么都没发生过"：``ever_moved`` 清掉，下一轮要重新看见转动。

        这是**唯一**能清掉 ``ever_moved`` 的入口——漏检、换目标以外的任何情况都不该清，
        否则"刚转过就漏了两帧"会被当成"没转过"。
        """
        self._last_stamp: Optional[float] = None
        self._last_position: Optional[Tuple[float, float]] = None
        self._ever_moved = False
        self._moving_frames = 0
        self._low_speed_frames = 0
        self._reported = False
        self._last_speed: Optional[float] = None

    @property
    def ever_moved(self) -> bool:
        """本周期内是否确认过转动。"""
        return self._ever_moved

    @property
    def reported(self) -> bool:
        """本周期是否已经上报过（锁存标志，下一轮转动才复位）。"""
        return self._reported

    # --------------------------------------------------------------- 观测
    def observe(
        self,
        position: Optional[Tuple[float, float]],
        stamp: float,
        *,
        key: Optional[str] = None,
    ) -> MotionVerdict:
        """喂一帧观测：``position`` 是目标位置（像素），``stamp`` 是单调时刻（秒）。

        ``position`` 为 ``None`` 表示**这一帧没看见目标**（运动模糊、被挡住、还没进画面）。
        漏检的处理是刻意的：速度算不出来，测速基准被重建，但 ``ever_moved`` **保持不变**——
        "转过"是一个已经发生过的事实，不该因为后来漏了两帧就被撤销。

        ``key`` 与构造时的 ``key`` 不同，表示**换了目标**：重新开始计数并清掉
        ``ever_moved``，因为这一轮动过的是上一个目标。
        """
        if key is not None and key != self.key:
            self.key = key
            self.reset()
            # reset() 已经把 stamp 清掉了，这一帧就是新目标的基准帧。
        if not math.isfinite(stamp):
            raise ValueError(f"stamp 必须是有限的时刻，实际是 {stamp!r}")

        if position is None:
            # 漏检：重建测速基准，但不清 ever_moved（见 docstring）。
            self._last_stamp = None
            self._last_position = None
            self._last_speed = None
            self._low_speed_frames = 0
            return self._verdict()

        point = (float(position[0]), float(position[1]))

        if self._last_stamp is None or self._last_position is None:
            # 第一帧（或刚漏检过）：只立基准，算不出速度。
            self._last_stamp = stamp
            self._last_position = point
            self._last_speed = None
            return self._verdict()

        interval = stamp - self._last_stamp
        if not self.min_interval_s <= interval <= self.max_interval_s:
            # 间隔不合理：不拿它算速度，但位置仍然成为下一次的基准。
            self._last_stamp = stamp
            self._last_position = point
            self._last_speed = None
            return self._verdict()

        travel = math.hypot(point[0] - self._last_position[0], point[1] - self._last_position[1])
        speed = travel / interval
        self._last_stamp = stamp
        self._last_position = point
        self._last_speed = speed

        if speed >= self.moving_speed_px_s:
            self._moving_frames += 1
            self._low_speed_frames = 0
        elif speed <= self.near_stop_speed_px_s:
            self._low_speed_frames += 1
            self._moving_frames = 0
        else:
            # 迟滞带以内：两边都不算满足，已有的连续计数**不清零**。
            pass

        if self._moving_frames >= self.required_moving_frames:
            self._ever_moved = True
            # 重新转起来：本周期已经上报过的话，这里就是"下一轮"的开始。
            self._reported = False

        report_now = False
        if self._ever_moved and self._low_speed_frames >= self.required_low_speed_frames:
            if not self._reported:
                report_now = True
                self._reported = True

        return self._verdict(report_now=report_now)

    # ------------------------------------------------------------- 内部
    def _verdict(self, *, report_now: bool = False) -> MotionVerdict:
        if self._reported:
            state = STATE_REPORTED
        elif not self._ever_moved:
            state = STATE_IDLE
        elif self._low_speed_frames >= self.required_low_speed_frames:
            state = STATE_SETTLED
        else:
            state = STATE_MOVING
        return MotionVerdict(
            state=state,
            ever_moved=self._ever_moved,
            report_now=report_now,
            speed_px_s=self._last_speed,
            moving_frames=self._moving_frames,
            low_speed_frames=self._low_speed_frames,
        )


def feed(
    gate: MotionGate,
    samples: Sequence[Tuple[Optional[Tuple[float, float]], float]],
) -> MotionVerdict:
    """把一串 ``(位置, 时刻)`` 依次喂给 ``gate``，返回**最后一个**结论。

    便利函数，只为让调用方（和自测）少写一层循环；语义与逐帧调用 :meth:`MotionGate.observe`
    完全一致。
    """
    verdict = MotionVerdict(
        state=STATE_IDLE,
        ever_moved=False,
        report_now=False,
        speed_px_s=None,
        moving_frames=0,
        low_speed_frames=0,
    )
    for position, stamp in samples:
        verdict = gate.observe(position, stamp)
    return verdict


# ---------------------------------------------------------------------- 自测
def _uniform_samples(
    count: int,
    *,
    step_px: float,
    interval_s: float = 0.1,
    start_stamp: float = 0.0,
    start_position: Tuple[float, float] = (0.0, 0.0),
) -> list:
    """合成一串匀速直线运动的观测：位置每帧走 ``step_px``，时刻每帧走 ``interval_s``。"""
    return [
        (
            (start_position[0] + index * step_px, start_position[1]),
            start_stamp + index * interval_s,
        )
        for index in range(count)
    ]


def _self_test() -> int:
    problems: list[str] = []

    # ---- 核心：本来就静止时，连着几帧一样也**不许**上报 ----
    gate = MotionGate()
    still = [(  (10.0, 10.0), index * 0.1) for index in range(10)]
    verdict = feed(gate, still)
    if verdict.ever_moved:
        problems.append("画面一直静止，不该被认为动过")
    if verdict.settled or verdict.report_now:
        problems.append("画面一直静止时绝不能上报（静止 ≠ 停稳）")
    if verdict.state != STATE_IDLE:
        problems.append(f"一直静止的状态应为 {STATE_IDLE}，实际 {verdict.state}")

    # ---- 先动后停：动过之后停稳，应当上报**恰好一次** ----
    gate = MotionGate()
    moving = _uniform_samples(8, step_px=10.0, interval_s=0.1)      # 100 px/s
    stopping = [((80.0 + index * 0.5, 0.0), 0.8 + (index + 1) * 0.1) for index in range(6)]
    verdicts = [gate.observe(position, stamp) for position, stamp in moving + stopping]
    reports = [index for index, item in enumerate(verdicts) if item.report_now]
    if len(reports) != 1:
        problems.append(f"一个周期应恰好上报一次，实际上报了 {len(reports)} 次")
    if not verdicts[-1].ever_moved:
        problems.append("转过之后 ever_moved 应为 True")
    if not verdicts[-1].settled:
        problems.append("动过并停稳之后应为 settled")

    # ---- 再次转动：锁存要能复位，下一轮又能上报一次 ----
    # 时刻从 1.5 接上（上一批停在 1.4）：接不上的话第一帧会因为"间隔太长"算不出速度，
    # 白白吃掉一帧——那正是 max_interval_s 该做的事，不是这里的被测对象。
    second_moving = _uniform_samples(
        8, step_px=10.0, interval_s=0.1, start_stamp=1.5, start_position=(0.0, 0.0)
    )
    second_stopping = [
        ((80.0 + index * 0.5, 0.0), 2.3 + (index + 1) * 0.1) for index in range(6)
    ]
    second = [gate.observe(position, stamp) for position, stamp in second_moving + second_stopping]
    second_reports = [index for index, item in enumerate(second) if item.report_now]
    if len(second_reports) != 1:
        problems.append(f"第二轮应恰好再上报一次，实际上报了 {len(second_reports)} 次")
    if not second[-1].settled:
        problems.append("第二轮动过并停稳之后应为 settled")

    # ---- 漏检不撤销"动过" ----
    gate = MotionGate()
    feed(gate, _uniform_samples(8, step_px=10.0, interval_s=0.1))
    if not gate.ever_moved:
        problems.append("匀速 100 px/s 连续 8 帧应确认动过")
    feed(gate, [(None, 1.0), (None, 1.1)])
    if not gate.ever_moved:
        problems.append("漏检不该清掉 ever_moved（转过是已发生的事实）")
    if gate.reported:
        problems.append("还没停稳就不该处于已上报状态")

    # ---- 重置才清得掉 ----
    gate.reset()
    if gate.ever_moved or gate.reported:
        problems.append("reset() 之后应当回到什么都没发生过")

    # ---- 迟滞：阈值上下来回抖，不该把状态翻来翻去 ----
    gate = MotionGate()
    feed(gate, _uniform_samples(4, step_px=5.0, interval_s=0.1))          # 50 px/s 动过
    jitter = [((20.0 + (index % 2) * 2.7, 0.0), 0.4 + index * 0.1) for index in range(8)]
    verdict = feed(gate, jitter)                                          # 27 px/s，落在 25–30 之间
    if verdict.state == STATE_REPORTED:
        problems.append("速度落在迟滞带（25–30 px/s）以内时不该判停稳")

    # ---- 间隔不合理：算不出速度，也不能拿它当速度 ----
    gate = MotionGate()
    verdict = feed(gate, [((0.0, 0.0), 0.0), ((100.0, 0.0), 1.0)])        # 隔了 1 s
    if verdict.speed_px_s is not None:
        problems.append("超出 max_interval_s 的间隔应当算不出速度")
    if verdict.ever_moved:
        problems.append("一次跨了掉帧的间隔不该被判成动过")

    # ---- 换目标：重新开始，且清掉 ever_moved ----
    gate = MotionGate(key="red")
    feed(gate, _uniform_samples(8, step_px=10.0, interval_s=0.1))
    if not gate.ever_moved:
        problems.append("red 目标匀速运动应确认动过")
    gate.observe((0.0, 0.0), 1.0, key="blue")
    if gate.ever_moved:
        problems.append("换了目标之后 ever_moved 必须清掉（动过的是上一个目标）")

    # ---- 构造参数校验 ----
    for bad_call, name in (
        (lambda: MotionGate(moving_speed_px_s=20.0, near_stop_speed_px_s=30.0), "停稳阈值高于转动阈值"),
        (lambda: MotionGate(required_moving_frames=0), "转动帧数为 0"),
        (lambda: MotionGate(required_low_speed_frames=0), "停稳帧数为 0"),
        (lambda: MotionGate(min_interval_s=0.3, max_interval_s=0.2), "间隔上下限颠倒"),
        (lambda: MotionGate(moving_speed_px_s=-1.0), "转动阈值为负"),
    ):
        try:
            bad_call()
        except ValueError:
            continue
        problems.append(f"{name} 应被拒绝，实际被接受了")

    print("合成序列（先动 100 px/s，再降到 5 px/s）：")
    gate = MotionGate()
    for position, stamp in _uniform_samples(4, step_px=10.0, interval_s=0.1):
        print(f"  {gate.observe(position, stamp).summary()}")
    for position, stamp in [((40.0 + index * 0.5, 0.0), 0.4 + index * 0.1) for index in range(5)]:
        verdict = gate.observe(position, stamp)
        print(f"  {verdict.summary()}{'  <-- 上报' if verdict.report_now else ''}")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：静止不误报、动过才认停稳、报告只发一次、漏检不撤销事实。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="转动/停稳门控：动过才算停稳")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

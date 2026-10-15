"""观测侧：观测源协议、物料选择，以及"这一帧值不值得信"的判据。

这里全部是**关于观测**的事，一行机械臂动作都没有：

- :class:`FrameSource`：观测源要提供的那一个方法（``CameraLink`` 与本机相机会话都满足它）；
- :func:`select_object`：从一堆观测里认出"哪条是物块"；
- :func:`_settled_sighting` / :func:`measure_settled`：等到画面**停稳**再取值；
- :func:`_probe_visible`：抬升前后判断物块还在不在画面里。

为什么"停稳"这么重要：机械臂刚停下的那一两帧里，运动模糊会把坐标读偏，而伺服容差
只有几毫米。拿模糊帧去解算，下一步就会朝错误方向走。

依赖方向见 :mod:`modules.grasp.drive.servo_model` 的模块说明：本模块只依赖
``servo_model`` 和 vision 的观测类型，不导入控制器，也不导入脉冲原语。
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence, Tuple

from modules.grasp.drive.servo_model import ObjectSelector, ServoConfig, ServoError

if TYPE_CHECKING:
    from modules.vision.drive.sighting import MaterialSighting


class FrameSource(Protocol):
    """观测源需要提供的那一件事：取一组观测。

    故意只要这一个方法，于是两种来源都能插进来，而这里不需要知道是哪种：

    - :class:`modules.camera_link.drive.link.CameraLink`（远端镜头走 TCP）；
    - :class:`modules.vision.runtime.Vision`（本机 USB 摄像头 + 模型，默认走这条）。

    ``min_received_at`` 是"只要不早于这个时刻的观测"——机械臂动完之后
    **必须**带上它，否则可能读到动作**之前**的画面（详见模块开头）。
    """

    def frame(
        self,
        *,
        timeout: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
        min_received_at: Optional[float] = None,
    ) -> Optional[Sequence[MaterialSighting]]: ...


def select_object(
    sightings: Sequence[MaterialSighting], selector: ObjectSelector
) -> MaterialSighting:
    """从一帧里挑出物块。

    :raises ServoError: 一个都没有，或者没有一个明确的。
    """
    matched = [item for item in sightings if selector.matches(item)]
    if not matched:
        listing = "、".join(
            f"物料 {item.material_id}({item.color})" for item in sightings
        ) or "（空）"
        raise ServoError(
            f"这一帧里没有物块（{selector.describe()}）；收到 {len(sightings)} 条：{listing}"
        )
    if len(matched) > 1 and selector.is_open:
        listing = "、".join(f"物料 {item.material_id}({item.color})" for item in matched)
        raise ServoError(
            f"画面里有多个目标（{listing}），不知道该对准哪一个；"
            "请在 modules/grasp/eye_in_hand.json 的 object 里填 material_id 或 color"
        )
    return matched[-1]


def _read_object(link: FrameSource, selector: ObjectSelector, config: ServoConfig, *, after: Optional[float] = None) -> MaterialSighting:
    """等一帧，并返回物块那一观测。``after`` 给出"只要不早于这个时刻"的过滤。"""
    frame = link.frame(
        timeout=config.frame_timeout_seconds,
        min_received_at=after,
    )
    if frame is None:
        error = getattr(link, "error", None)
        raise ServoError(
            "等不到新的观测"
            + (f"；接收链路报错：{error}" if error else "；确认镜头端已连上并正在发数据")
        )
    return select_object(frame, selector)


def measure_offset(
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    after: Optional[float] = None,
) -> Tuple[float, float]:
    """读一帧，返回物块在相机系的水平偏移 ``(x, y)``（毫米）。"""
    sighting = _read_object(link, selector, config, after=after)
    return (sighting.x_mm, sighting.y_mm)


def _settled_sighting(
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    after: Optional[float] = None,
    tolerance: Optional[float] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> MaterialSighting:
    """等到画面不再变，返回那一刻的整条观测（不只是坐标）。

    返回观测而不只是 ``(x, y)``，是因为**像素直径**也要用：爪瓣扫过物块会把它挡住一部分，
    那是闭环唯一能自己发现的"机械干涉"信号。

    ``tolerance`` 覆盖一致性判据：粗调阶段（离目标还有十几毫米）用
    ``coarse_settle_tolerance_mm`` 就够，不必为了 0.4 mm 多等好几帧——每一帧都要跑一次
    定位，本机实测 178 ms。
    """
    say = on_message if on_message is not None else (lambda _message: None)
    allowed = config.settle_tolerance_mm if tolerance is None else tolerance
    deadline = time.monotonic() + config.settle_timeout_seconds
    previous: Optional[Tuple[float, float]] = None
    last: Optional[MaterialSighting] = None
    while True:
        try:
            current = _read_object(link, selector, config, after=after)
        except ServoError:
            # 运动模糊时检不到物块是常态，不该当失败——接着等。
            previous = None
        else:
            last = current
            point = (current.x_mm, current.y_mm, current.z_mm)
            if previous is not None and math.sqrt(
                (point[0] - previous[0]) ** 2
                + (point[1] - previous[1]) ** 2
                + (point[2] - previous[2]) ** 2
            ) <= allowed:
                return current
            previous = point
        if time.monotonic() >= deadline:
            if last is None:
                raise ServoError(
                    f"等了 {config.settle_timeout_seconds:.1f} s 也没能稳定看到物块；"
                    "确认物块在画面里、没有被人或线材挡住"
                )
            say(
                f"  等停稳超时（{config.settle_timeout_seconds:.1f} s）："
                f"画面仍在变，先按最后读数 ({last.x_mm:+.2f}, {last.y_mm:+.2f}) mm 继续"
            )
            return last
        time.sleep(config.settle_poll_seconds)


def measure_settled(
    link: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    *,
    after: Optional[float] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> Tuple[float, float]:
    """测到画面**不再变**为止，返回那个读数。

    为什么不能"按估算耗时等一等就读"：实测 ``wait=True`` 在 0.652 s 就返回了，
    可读数在那之后还在大幅变化（``-23.65,+39.77`` → ``-33.38,+23.03``），
    中间还反复因为运动模糊整个检不到物块。拿那种数去解算，等于用噪声当观测——
    闭环会振荡，探针会把雅可比算大好几倍。

    判据是**连续两次读数足够接近**：把"等多久"交给观测本身，而不是交给一个估算公式。
    """
    sighting = _settled_sighting(link, selector, config, after=after, on_message=on_message)
    return (sighting.x_mm, sighting.y_mm)


VISIBILITY_PROBE_FRAMES = 2


def _probe_visible(link: FrameSource, selector: ObjectSelector, config: ServoConfig) -> bool:
    """快速回答"物块在不在画面里"，**不等停稳**。

    为什么可以不等：这一步只问"在不在"，不问"在哪"——差几毫米无所谓。而等停稳
    （连续两帧一致）至少要 537 ms；看不到时还要一直等到 ``settle_timeout_seconds``
    （默认 4 s）才认输，乘上"抬起来找"的步数就是几十秒。

    代价是可能被一帧运动模糊骗到：所以连读 :data:`VISIBILITY_PROBE_FRAMES` 帧，
    任何一帧认出来就算"在"。
    """
    for _ in range(VISIBILITY_PROBE_FRAMES):
        try:
            _read_object(link, selector, config)
        except ServoError:
            continue
        return True
    return False

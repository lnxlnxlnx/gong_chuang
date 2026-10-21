"""画面级停稳：连续几帧的**像素差**够小，才认为画面静止。

## 为什么要它

``modules.vision.drive.motion_gate`` 判的是**目标位置**稳不稳：它要求先看见目标动过，
再看位置速度掉下来。它回答不了"整幅画面这一刻到底动没动"——尤其当目标已经不在画面里
（刚被爪子挡住、或转盘把它带走了），位置门控没有输入，只能说"没看见"。

抓取前真正要回答的是另一个问题：

    现在按下下降，爪子落下去的那几百毫秒里，物块会不会沿着转盘走掉？

所以这里直接比**像素**：取连续几帧，两两做灰度绝对差。逐像素差超过
``pixel_threshold`` 的算"变了"，再看两个汇总指标：

- ``changed_ratio``：变化像素占全画面的比例；
- ``mean_abs_diff``：整幅图的平均灰度差。

两个都在阈值内才算静止。**所有相邻帧对都要通过**——只比首尾两帧会被"转过去又转回来"
骗过去。

## 阈值怎么定

默认值面向 1280×720 的 USB 摄像头（30 fps、室内常亮）：

- ``pixel_threshold = 30``：低于 30 个灰阶的差多半是传感器噪声 / 自动曝光；
- ``max_changed_ratio = 0.005``：0.5% ≈ 4600 像素。一个 30 mm 物料在画面里约
  270–400 px 直径（6–12 万像素），所以它每秒挪十几毫米以上就会越过这个比例。
  **不要再收紧到 0.2% 那一档**：桌面 / 相机只要轻轻抖一两个像素，边缘像素就足以
  超限，于是转盘明明已经停了、判据却一直报"还在动"（现场遇到的就是这个误判）。
- ``max_mean_diff = 4.0``：兜住"整幅画面一起轻微变亮 / 变暗"（自动曝光、反光），
  同一个数也吸收整幅轻微晃动带来的平均灰度变化。

放宽的代价是**很慢的移动**会被放过：0.5% 粗算对应物块每帧挪七八个像素（十几 mm/s）。
从"判稳定"到"夹爪合上"只有半秒左右，这个量级仍在爪瓣能兜住的范围内。要更严就把
``max_changed_ratio`` 调回 0.002，并先看 ``summary()`` 里的两个实测值再决定，
而不是凭感觉改 ``pixel_threshold``。

## 与 ``measure_settled`` 的分工

| 问的问题 | 用哪个 |
|---|---|
| 检测到的物料位置稳不稳？ | ``Vision.measure_settled(...)`` |
| 整幅画面有没有在动？ | ``Vision.scene_stable()`` |

后者不依赖检测：画面里没有物料也能回答。这正是"下降之前最后看一眼"需要的性质。

命令行自测::

    python -m modules.vision.drive.scene_stability    # 合成帧，不需要摄像头
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

#: 一次判断连拍几帧。3 帧只有两组比较，4 帧有三组，多一组冗余换来的时间约 33 ms。
DEFAULT_STABLE_FRAMES = 4

#: 逐像素灰度差超过它才算"这个像素变了"。低于它的差多是传感器噪声 / 自动曝光。
DEFAULT_PIXEL_DIFF_THRESHOLD = 30

#: 变化像素占全画面的比例上限。
DEFAULT_MAX_CHANGED_RATIO = 0.005

#: 整幅画面的平均灰度差上限。
DEFAULT_MAX_MEAN_DIFF = 4.0


@dataclass(frozen=True)
class FrameDifference:
    """相邻两帧的像素差指标。"""

    #: 整幅图的平均绝对灰度差（0–255 灰阶）。
    mean_abs_diff: float
    #: 逐像素差超过 ``pixel_threshold`` 的像素占比（0–1）。
    changed_ratio: float

    def summary(self) -> str:
        return f"平均差 {self.mean_abs_diff:.2f} 灰阶，变化像素 {self.changed_ratio * 100:.3f}%"


@dataclass(frozen=True)
class SceneStability:
    """一次"画面停稳没有"的结论，连同全部相邻帧对的实测值。"""

    #: 每一对相邻帧的差；长度为帧数 − 1。
    pairs: Tuple[FrameDifference, ...]
    #: 判"变了"的逐像素灰度差门槛。
    pixel_threshold: int
    #: 平均灰度差的允许上限。
    mean_diff_limit: float
    #: 变化像素比例的允许上限。
    changed_ratio_limit: float

    @property
    def frames(self) -> int:
        """这次判断一共看了几帧。"""
        return len(self.pairs) + 1

    @property
    def max_mean_abs_diff(self) -> float:
        """所有相邻帧对里最大的平均灰度差。"""
        return max((pair.mean_abs_diff for pair in self.pairs), default=0.0)

    @property
    def max_changed_ratio(self) -> float:
        """所有相邻帧对里最大的变化像素比例。"""
        return max((pair.changed_ratio for pair in self.pairs), default=0.0)

    @property
    def stable(self) -> bool:
        """**每一对**相邻帧都够小才算静止；任何一对超限都为 ``False``。"""
        return all(
            pair.mean_abs_diff <= self.mean_diff_limit
            and pair.changed_ratio <= self.changed_ratio_limit
            for pair in self.pairs
        )

    def summary(self) -> str:
        """一行中文结论，供终端打印。"""
        if not self.pairs:
            return "帧数不够，无法判断画面是否静止"
        verdict = "静止" if self.stable else "还在动"
        return (
            f"{verdict}（{self.frames} 帧）：最大平均差 {self.max_mean_abs_diff:.2f} 灰阶"
            f"（上限 {self.mean_diff_limit:.2f}），最大变化像素 "
            f"{self.max_changed_ratio * 100:.3f}%（上限 {self.changed_ratio_limit * 100:.3f}%）"
        )


def _check_thresholds(
    *, pixel_threshold: int, max_mean_diff: float, max_changed_ratio: float
) -> None:
    if type(pixel_threshold) is not int or not 0 <= pixel_threshold <= 255:
        raise ValueError(
            f"pixel_threshold 必须是 0–255 的整数，收到 {pixel_threshold!r}"
        )
    if not math.isfinite(max_mean_diff) or max_mean_diff < 0:
        raise ValueError(f"max_mean_diff 必须是非负的有限数，收到 {max_mean_diff!r}")
    if (
        not math.isfinite(max_changed_ratio)
        or not 0.0 <= max_changed_ratio <= 1.0
    ):
        raise ValueError(
            f"max_changed_ratio 必须在 0–1 之间，收到 {max_changed_ratio!r}"
        )


def _grayscale(frame) -> np.ndarray:
    """帧 → float64 灰度图；灰度换算只有 :func:`material_face.to_grayscale` 一份。

    延迟导入是刻意的：``modules.vision.runtime`` 在包初始化时就要拿到本模块的默认值，
    而 ``material_face`` 会连带加载 ``hough_lock``——那是 ``python -m`` 自测入口，
    ``import modules.vision`` 不该预加载它（见 ``tests/test_vision_type_exports.py``）。
    第一次调用之后模块已在 ``sys.modules`` 里，后续不再有导入开销。
    """
    from modules.vision.drive.material_face import to_grayscale

    return to_grayscale(frame)


def frame_difference(
    previous,
    current,
    *,
    pixel_threshold: int = DEFAULT_PIXEL_DIFF_THRESHOLD,
) -> FrameDifference:
    """量相邻两帧的像素差。``previous`` / ``current`` 可以是 BGR 或灰度帧。

    两帧尺寸必须一致——尺寸不同是调用方把两路画面混在一起了，不能静默裁成一样大。
    """
    if type(pixel_threshold) is not int or not 0 <= pixel_threshold <= 255:
        raise ValueError(
            f"pixel_threshold 必须是 0–255 的整数，收到 {pixel_threshold!r}"
        )
    previous_gray = _grayscale(previous)
    current_gray = _grayscale(current)
    if previous_gray.shape != current_gray.shape:
        raise ValueError(
            f"两帧尺寸不一致：{previous_gray.shape} 对 {current_gray.shape}"
        )
    difference = np.abs(current_gray - previous_gray)
    if difference.size == 0:  # pragma: no cover - to_grayscale 已经挡掉空帧
        raise ValueError("frame is empty")
    changed = float(np.count_nonzero(difference > pixel_threshold)) / difference.size
    return FrameDifference(
        mean_abs_diff=float(difference.mean()),
        changed_ratio=changed,
    )


def evaluate_stability(
    frames: Sequence[np.ndarray],
    *,
    pixel_threshold: int = DEFAULT_PIXEL_DIFF_THRESHOLD,
    max_mean_diff: float = DEFAULT_MAX_MEAN_DIFF,
    max_changed_ratio: float = DEFAULT_MAX_CHANGED_RATIO,
) -> SceneStability:
    """把一串连续帧收成"画面停稳没有"的结论。

    :raises ValueError: 少于两帧，或任何阈值不合法。
    """
    _check_thresholds(
        pixel_threshold=pixel_threshold,
        max_mean_diff=max_mean_diff,
        max_changed_ratio=max_changed_ratio,
    )
    if len(frames) < 2:
        raise ValueError(f"至少需要两帧才能比较，收到 {len(frames)} 帧")
    pairs = tuple(
        frame_difference(
            frames[index], frames[index + 1], pixel_threshold=pixel_threshold
        )
        for index in range(len(frames) - 1)
    )
    return SceneStability(
        pairs=pairs,
        pixel_threshold=pixel_threshold,
        mean_diff_limit=max_mean_diff,
        changed_ratio_limit=max_changed_ratio,
    )


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    problems: list[str] = []

    def blank(value: int = 0) -> np.ndarray:
        return np.full((48, 64, 3), value, dtype=np.uint8)

    def with_square(x: int, *, value: int = 220) -> np.ndarray:
        frame = blank()
        frame[16:32, x:x + 16] = value
        return frame

    # ---- 静止：完全相同的帧 ----
    still = evaluate_stability([blank()] * DEFAULT_STABLE_FRAMES)
    if not still.stable:
        problems.append(f"完全相同的帧应当判静止：{still.summary()}")
    if still.max_mean_abs_diff != 0.0 or still.max_changed_ratio != 0.0:
        problems.append("相同帧的像素差应当为 0")

    # ---- 运动：方块在挪 ----
    moving = evaluate_stability(
        [with_square(x) for x in (4, 12, 20, 28)]
    )
    if moving.stable:
        problems.append(f"方块在挪应当判还在动：{moving.summary()}")
    if moving.max_changed_ratio <= DEFAULT_MAX_CHANGED_RATIO:
        problems.append("方块挪动 8 px 的变化比例应当超过默认上限")

    # ---- 每一对都要通过：第一对静止、第二对运动 ----
    half = evaluate_stability([blank(), blank(), with_square(24)])
    if half.stable:
        problems.append("只要有一对在动，整段就不能判静止")

    # ---- 阈值：同一组帧，放宽“变化比例”就能通过 ----
    relaxed = evaluate_stability(
        [with_square(x) for x in (4, 12)], max_changed_ratio=1.0, max_mean_diff=255.0
    )
    if not relaxed.stable:
        problems.append("阈值放宽到极限时，同一组帧应当判静止")

    # ---- 灰度帧与 BGR 帧都接受 ----
    gray_pair = evaluate_stability([blank()[:, :, 0], blank()[:, :, 0]])
    if not gray_pair.stable:
        problems.append("二维灰度帧应当照常比较")

    # ---- 形状不一致：必须拒绝 ----
    try:
        frame_difference(blank(), np.zeros((24, 32, 3), dtype=np.uint8))
    except ValueError:
        pass
    else:
        problems.append("两帧尺寸不一致应当被拒绝")

    # ---- 参数校验 ----
    for bad_call, name in (
        (lambda: evaluate_stability([blank()]), "只有一帧"),
        (lambda: evaluate_stability([blank()] * 2, pixel_threshold=-1), "负门槛"),
        (lambda: evaluate_stability([blank()] * 2, max_mean_diff=-1.0), "负平均差上限"),
        (
            lambda: evaluate_stability([blank()] * 2, max_changed_ratio=1.5),
            "变化比例上限超过 1",
        ),
    ):
        try:
            bad_call()
        except ValueError:
            continue
        problems.append(f"{name} 应被拒绝，实际被接受了")

    print(f"完全相同的帧：{still.summary()}")
    print(f"方块每帧挪 8 px：{moving.summary()}")
    print(f"前两组静止、最后一组在动：{half.summary()}")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：静止不误报、每一对都要通过、阈值与尺寸校验都正确。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="画面级停稳：连续几帧的像素差")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

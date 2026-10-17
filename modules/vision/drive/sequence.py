"""多帧融合：把一串观测收成一个可信值，并且**分得清"在动"和"在抖"**。

单帧观测有两种完全不同的坏法，混在一起处理就会两头不讨好：

- **抖（静态噪声）**：机械臂刚停、转盘刚到位，画面里读数在真值附近跳。这时应当
  多取几帧、按一致性剔掉野值，再取一个稳健的中心值。
- **动（系统性漂移）**：物料在转盘上还没停稳，读数在时间上**单调地**走。这时
  取平均值是错的——它给出一个"哪一时刻都不是"的位置；而对着一组真值持续变化的
  读数按"离中心太远"剔野值，会把**所有**帧都剔掉。

所以这里的做法是先拟合一条**时间上的直线**（Theil–Sen，中位数斜率），再按
"离这条线多远"剔野值：

- 静止时斜率≈0，这条线就是中心值，剔野值的行为和普通稳健平均一样；
- 运动时斜率≠0，野值判据跟着线走，帧不会被误剔，而 :attr:`Fusion.moving` 会
  如实报出"还在动"，同时给出速度——调用方可以据此决定"等它停"还是"按这个速度
  外推到某一时刻"。

**为什么可以用直线近似圆周运动**：转盘上的物料走的是圆弧，但一次融合只覆盖很短
的一段时间（一次连拍，实测本机一帧定位 178 ms，7 帧约 1.2 s），转盘在这段时间里
只转过很小的角度，圆弧在这一小段上与直线无异。真要长时间跟踪圆周，就该在外层
做圆周拟合，而不是把窗口拉长。

本模块**只依赖标准库**：不导入 numpy、不导入 cv2、不碰相机与串口，所以能完全
离线测试，也能被 ``modules.grasp`` 那样的上层直接复用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

#: 野值判据：残差超过 ``reject_k × 稳健标准差`` 的点丢掉。
#: 用 1.4826×MAD 当标准差估计（正态分布下与 ``std`` 同尺度，但不受野值影响）。
DEFAULT_REJECT_K = 3.0

#: 残差判据的绝对下限（毫米）。MAD 在"读数几乎不动"时会小到接近 0，
#: 那时任何一点抖动都会被判成野值——所以判据不能小于这个数。
#: 0.5 mm 是这套光学的量化噪声量级（70 mm 处半径差 1 px ≈ 0.3 mm）。
DEFAULT_FLOOR_MM = 0.5

#: 认为"还没停稳"的位移下限（毫米）：拟合出的直线在这段时间里走过的距离超过它，
#: :attr:`Fusion.moving` 就是 ``True``。
DEFAULT_STILL_MM = 0.4

#: 至少要几个**有效读数**才能给结论。3 个才能定一条线，而少于 3 个时
#: "剔野值"没有意义（剔掉一个就只剩猜）。
DEFAULT_MIN_SAMPLES = 3


@dataclass(frozen=True)
class TimedPoint:
    """一帧里一个目标的观测：时刻 + 相机系毫米坐标。

    ``stamp`` 用 :func:`time.monotonic` 那样的**单调时钟**即可，本模块只关心差值。
    ``diameter_px`` 与 ``color`` 是诊断量，不参与几何融合。
    """

    stamp: float
    x_mm: float
    y_mm: float
    z_mm: float
    diameter_px: Optional[float] = None
    color: Optional[str] = None

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.x_mm, self.y_mm, self.z_mm)


@dataclass(frozen=True)
class Fusion:
    """一次多帧融合的结果：值、它有多可信，以及"目标是不是还在动"。"""

    #: 融合出来的观测；``stamp`` 是**参考时刻**（默认取最后一帧的时刻）。
    point: TimedPoint
    #: 内点相对融合值的最大偏差（毫米）。这是"这一串读数有多一致"的直接读数。
    spread_mm: float
    #: 参与融合的内点帧数。
    samples: int
    #: 因为离拟合线太远被丢掉的帧数。
    rejected: int
    #: 一帧都没看到目标的次数（运动模糊、被挡住、画面里没料）。
    misses: int
    #: 读数在时间上系统性漂移、且漂移量超过阈值——**还没停稳**。
    moving: bool
    #: 拟合出的速度（毫米/秒）；静止时接近 0。
    velocity_mm_s: Tuple[float, float, float]
    #: 这批帧覆盖的时间长度（秒）。
    span_seconds: float
    #: 这一串读数里出现过的颜色名（诊断"是不是换了一个目标"）。
    colors: Tuple[str, ...]

    @property
    def frames(self) -> int:
        """这一批一共问了几帧（内点 + 野值 + 没看到）。"""
        return self.samples + self.rejected + self.misses

    @property
    def speed_mm_s(self) -> float:
        """速度的模（毫米/秒）。"""
        return math.sqrt(sum(component * component for component in self.velocity_mm_s))

    @property
    def seen_ratio(self) -> float:
        """看得到目标的那几帧里，有多少帧被采纳（0–1）。"""
        seen = self.samples + self.rejected
        return 0.0 if seen == 0 else self.samples / seen

    @property
    def confident(self) -> bool:
        """够不够格当"一次可信测量"：内点够多、没在动、且野值没占多数。

        **没有把 spread 写进来**是有意的：spread 的合理量级取决于距离与标定，
        写死一个毫米数会在近处过严、远处过松。调用方拿 :attr:`spread_mm`
        和自己的容差比。
        """
        return (
            self.samples >= DEFAULT_MIN_SAMPLES
            and not self.moving
            and self.seen_ratio >= 0.5
        )

    def summary(self) -> str:
        state = f"还在动（{self.speed_mm_s:.1f} mm/s）" if self.moving else "已停稳"
        return (
            f"{state}：xyz=({self.point.x_mm:+.1f}, {self.point.y_mm:+.1f}, "
            f"{self.point.z_mm:.1f}) mm，"
            f"内点 {self.samples}/{self.frames} 帧（丢 {self.rejected}、没看到 {self.misses}），"
            f"最大偏差 {self.spread_mm:.2f} mm，跨 {self.span_seconds * 1000:.0f} ms"
        )


def _median(values: Sequence[float]) -> float:
    """中位数；空序列抛错（调用方保证非空）。"""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mad(values: Sequence[float], center: float) -> float:
    """中位绝对偏差（未乘 1.4826）。"""
    return _median([abs(value - center) for value in values])


def _theil_sen(times: Sequence[float], values: Sequence[float]) -> Tuple[float, float]:
    """中位数斜率 + 中位数截距；时间全都相同时退化成"没有趋势"。

    Theil–Sen 只用到两两斜率的中位数，所以一两个野值改不动它——这正是这里要的：
    先拿一条**不被野值带偏**的趋势线，才能用"离这条线多远"去判野值。
    """
    slopes = []
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            delta = times[j] - times[i]
            if delta > 0:
                slopes.append((values[j] - values[i]) / delta)
    if not slopes:
        return (0.0, _median(values))
    slope = _median(slopes)
    intercept = _median([value - slope * time for time, value in zip(times, values)])
    return (slope, intercept)


def _fit_axis(times, values, *, reject_k, floor_mm):
    """一个轴的稳健直线拟合 + 野值剔除；返回 ``(内点下标, 斜率, 截距, 最大残差)``。"""
    slope, intercept = _theil_sen(times, values)
    residuals = [value - (intercept + slope * time) for time, value in zip(times, values)]
    center = _median(residuals)
    scale = 1.4826 * _mad(residuals, center)
    limit = max(reject_k * scale, floor_mm)

    keep = [index for index, value in enumerate(residuals) if abs(value - center) <= limit]
    if not keep:
        # 判据再松也不会一个都不留；真出现说明输入里有 NaN 之类的坏值。
        return ([], slope, intercept, float("inf"))
    if len(keep) < len(values):
        kept_times = [times[index] for index in keep]
        kept_values = [values[index] for index in keep]
        slope, intercept = _theil_sen(kept_times, kept_values)
    residuals = [values[index] - (intercept + slope * times[index]) for index in keep]
    spread = max((abs(value) for value in residuals), default=0.0)
    return (keep, slope, intercept, spread)


def fuse(
    samples: Sequence[Optional[TimedPoint]],
    *,
    reject_k: float = DEFAULT_REJECT_K,
    floor_mm: float = DEFAULT_FLOOR_MM,
    still_mm: float = DEFAULT_STILL_MM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    reference_stamp: Optional[float] = None,
) -> Optional[Fusion]:
    """把一串观测（``None`` 表示这一帧没看到）收成一个可信值。

    ``reference_stamp`` 给出"要在哪一时刻的值"：静止时它就是那个值，运动时它是
    按拟合速度**外推**到该时刻的值。省略时取最后一帧的时刻（也就是"现在"）。

    返回 ``None`` 表示有效读数不够（少于 ``min_samples``），调用方**不要**退回
    单帧值——那正是这个函数要挡掉的东西。

    :raises ValueError: 参数不合法。
    """
    if reject_k <= 0 or floor_mm <= 0 or still_mm < 0:
        raise ValueError("reject_k / floor_mm / still_mm must be positive")
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3")

    seen = [sample for sample in samples if sample is not None]
    misses = len(samples) - len(seen)
    if len(seen) < min_samples:
        return None

    times = [sample.stamp for sample in seen]
    axes = {
        name: [getattr(sample, f"{name}_mm") for sample in seen]
        for name in ("x", "y", "z")
    }

    # 三个轴各自拟合，但**取交集**当内点：一个点只要在某一个轴上说不通，
    # 整帧就不可信（多半是那一帧模糊了或框错了圆，三个轴会一起错）。
    fits = {
        name: _fit_axis(times, values, reject_k=reject_k, floor_mm=floor_mm)
        for name, values in axes.items()
    }
    common = set(range(len(seen)))
    for keep, _slope, _intercept, _spread in fits.values():
        common &= set(keep)
    inliers = sorted(common)
    if len(inliers) < min_samples:
        return None

    # 内点上重新拟合，再取最终值。
    kept_times = [times[index] for index in inliers]
    final = {}
    for name, values in axes.items():
        kept_values = [values[index] for index in inliers]
        slope, intercept = _theil_sen(kept_times, kept_values)
        final[name] = (slope, intercept)

    span = max(kept_times) - min(kept_times)
    if reference_stamp is None:
        reference_stamp = max(kept_times)
    origin = min(kept_times)
    point_values = {
        name: intercept + slope * reference_stamp
        for name, (slope, intercept) in final.items()
    }
    velocity = tuple(final[name][0] for name in ("x", "y", "z"))

    # 位移按**整批帧覆盖的时间**算，不是按到参考时刻的时间：判断的是"这段时间里
    # 目标动了多少"，参考时刻选在哪里不影响这个判断。
    displacement = math.sqrt(
        sum((final[name][0] * span) ** 2 for name in ("x", "y", "z"))
    )
    spread = max(
        max(
            (
                abs(axes[name][index] - (final[name][1] + final[name][0] * times[index]))
                for index in inliers
            ),
            default=0.0,
        )
        for name in ("x", "y", "z")
    )
    last = seen[-1]
    colors: Tuple[str, ...] = tuple(
        dict.fromkeys(sample.color for sample in seen if sample.color)
    )
    return Fusion(
        point=TimedPoint(
            stamp=float(reference_stamp),
            x_mm=point_values["x"],
            y_mm=point_values["y"],
            z_mm=point_values["z"],
            diameter_px=last.diameter_px,
            color=colors[0] if len(colors) == 1 else last.color,
        ),
        spread_mm=float(spread),
        samples=len(inliers),
        rejected=len(seen) - len(inliers),
        misses=misses,
        moving=displacement > still_mm,
        velocity_mm_s=(velocity[0], velocity[1], velocity[2]),
        span_seconds=max(0.0, span),
        colors=colors,
    )


# ---------------------------------------------------------------------- 自测
def _p(stamp: float, x: float, y: float = 0.0, z: float = 100.0, **rest) -> TimedPoint:
    return TimedPoint(stamp=stamp, x_mm=x, y_mm=y, z_mm=z, **rest)


def _self_test() -> int:
    problems: list[str] = []

    # 1. 静止 + 一个野值：应当剔掉它，并给出接近真值的中心。
    quiet = [_p(i * 0.2, 10.0 + jitter) for i, jitter in enumerate((0.0, 0.1, -0.1, 0.05, -0.05))]
    quiet.insert(2, _p(0.4, 40.0))                    # 一帧框错了圆
    fused = fuse(quiet, still_mm=0.4)
    if fused is None:
        problems.append("静止 + 一个野值应当给出结论")
    else:
        if fused.rejected != 1:
            problems.append(f"应当正好剔掉 1 帧，实际 {fused.rejected}")
        if abs(fused.point.x_mm - 10.0) > 0.2:
            problems.append(f"中心值应当回到 10 mm 附近：{fused.summary()}")
        if fused.moving:
            problems.append(f"静止的一串不该被判成在动：{fused.summary()}")
        print(f"静止+野值   {fused.summary()}")

    # 2. 匀速运动：不能报成一个"哪一时刻都不是"的平均值，而且要报出速度。
    moving = [_p(i * 0.2, 5.0 * i * 0.2) for i in range(6)]      # 5 mm/s 沿 x
    fused = fuse(moving, still_mm=0.4)
    if fused is None:
        problems.append("匀速运动应当给出结论")
    else:
        if not fused.moving:
            problems.append(f"5 mm/s 应当被判成在动：{fused.summary()}")
        if abs(fused.velocity_mm_s[0] - 5.0) > 1e-9:
            problems.append(f"速度应当是 5 mm/s：{fused.velocity_mm_s}")
        if abs(fused.point.x_mm - 5.0) > 1e-6:
            problems.append(f"参考时刻取最后一帧，x 应当是 5.0：{fused.point.x_mm}")
        if fused.rejected:
            problems.append(f"匀速运动不该有任何一帧被当成野值：{fused.summary()}")
        print(f"匀速运动   {fused.summary()}")

    # 3. 运动中夹一个野值：野值要剔掉，运动本身**不能**被剔掉。
    dirty = list(moving)
    dirty.insert(1, _p(0.05, 99.0))
    fused = fuse(dirty, still_mm=0.4)
    if fused is None:
        problems.append("运动 + 野值应当给出结论")
    else:
        if fused.rejected != 1:
            problems.append(f"运动里只该剔掉那一个野值，实际 {fused.rejected}")
        if abs(fused.velocity_mm_s[0] - 5.0) > 0.05:
            problems.append(f"剔野值后速度应当仍是 5 mm/s：{fused.velocity_mm_s}")
        print(f"运动+野值   {fused.summary()}")

    # 4. 外推到指定时刻：这是"按速度推算"的入口。
    fused = fuse(moving, still_mm=0.4, reference_stamp=0.0)
    if fused is None or abs(fused.point.x_mm) > 1e-6:
        problems.append(f"外推到 t=0 应当是 0：{None if fused is None else fused.point.x_mm}")

    # 5. 没看到的帧算 miss，不参与几何，但要计数。
    with_misses = [moving[0], None, moving[1], None, moving[2], moving[3], moving[4], moving[5]]
    fused = fuse(with_misses, still_mm=0.4)
    if fused is None or fused.misses != 2:
        problems.append(f"没看到的帧应当计入 misses：{None if fused is None else fused.misses}")

    # 6. 有效读数不够时必须返回 None，而不是拿单帧凑数。
    if fuse([_p(0.0, 1.0), None, None]) is not None:
        problems.append("有效读数不够时必须返回 None")
    if fuse([]) is not None:
        problems.append("空序列必须返回 None")
    if fuse([None, None, None, None]) is not None:
        problems.append("全是 miss 必须返回 None")

    # 7. 抖动小到判据下限以内时不该被当成"在动"。
    trembling = [_p(i * 0.2, 0.0 + (0.1 if i % 2 else -0.1)) for i in range(6)]
    fused = fuse(trembling, still_mm=0.4)
    if fused is None or fused.moving:
        problems.append(f"±0.1 mm 的抖动不该被判成在动：{None if fused is None else fused.summary()}")

    # 8. 单调但极慢的漂移（0.1 mm/s，整段 0.1 mm）：小于 still_mm，算静止。
    crawling = [_p(i * 0.2, 0.1 * i * 0.2) for i in range(6)]
    fused = fuse(crawling, still_mm=0.4)
    if fused is None or fused.moving:
        problems.append(f"整段只挪 0.1 mm 不该算在动：{None if fused is None else fused.summary()}")

    # 9. 时间全相同：不能除以 0，应当退化成"没有趋势"。
    same_time = [_p(1.0, 3.0), _p(1.0, 3.1), _p(1.0, 2.9), _p(1.0, 3.05)]
    fused = fuse(same_time)
    if fused is None or fused.moving or abs(fused.point.x_mm - 3.0) > 0.2:
        problems.append(f"同一时刻的一串应当退化成中位数：{None if fused is None else fused.summary()}")

    # 10. 颜色变化要如实记下来（多半是换了一个目标，而不是同一个在动）。
    two_colors = [_p(0.0, 0.0, color="green"), _p(0.2, 0.0, color="green"),
                  _p(0.4, 0.0, color="red"), _p(0.6, 0.0, color="green")]
    fused = fuse(two_colors)
    if fused is None or set(fused.colors) != {"green", "red"}:
        problems.append(f"颜色变化应当被记下来：{None if fused is None else fused.colors}")

    # 11. confident 的含义：在动 / 野值过半 都不算可信。
    if fuse(moving, still_mm=0.4).confident:
        problems.append("还在动的融合不该是 confident")
    if not fuse(quiet, still_mm=0.4).confident:
        problems.append("静止且只有一个野值的融合应当是 confident")

    # 12. 非法参数。
    for kwargs in ({"reject_k": 0.0}, {"floor_mm": -1.0}, {"still_mm": -1.0},
                   {"min_samples": 2}):
        try:
            fuse(quiet, **kwargs)
        except ValueError:
            pass
        else:
            problems.append(f"非法参数必须报错：{kwargs}")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：静止剔野值、运动报速度、外推、misses、退化输入与非法参数都正确。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="多帧观测的稳健融合（区分抖动与运动）")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

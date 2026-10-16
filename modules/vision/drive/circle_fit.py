"""从轮廓点拟合圆：给物料顶面一个不被遮挡带偏的圆心与半径。

颜色块路径原先用「连通域质心 + 面积等效半径 ``sqrt(area/π)``」。物料顶面被夹爪
挡掉一角、或者一半落进阴影里时，掩膜只剩**残缺的圆**：质心被拽向可见的那一半，
面积等效半径比真实半径小。两个数一起错，距离和 x/y 就一起错。

实测（真机帧 ``output/vision_predict/material_locate_live.png``，真值半径 145.4 px，
从左侧推进黑色遮挡，遮挡像素不属于任何调色板颜色）::

    遮挡   面积等效半径   距离误差     拟合半径   距离误差
     0%      144.6        +0.6 mm     146.4      -0.7 mm
    15%      136.9        +6.5 mm     146.0      -0.4 mm
    30%      124.8       +17.2 mm     145.6      -0.1 mm
    45%      108.2       +35.9 mm     145.1      +0.2 mm
    60%        --           --        142.2      +2.4 mm

注意 30%/45% 那两行：流程**仍然接受**这个候选，只是把距离报错了 17–36 mm。
对下游逆解来说，这比"没找到"更危险——它会照着错的数动。

算法是 **Kåsa 代数最小二乘 + RANSAC**：

- Kåsa 把 ``x² + y² = a·x + b·y + c`` 当成一次线性求解，整圈点一次算完，快；
- RANSAC 只让"同意同一个圆"的点说话，把遮挡留下的**直边（弦）**和掩膜上的毛糙点
  排除在外。没有这一步，那条直边会把半径一路拽小：实测 45% 遮挡下直接拟合半径
  从 145.1 掉到 108 附近，和面积等效半径一样错。

本模块**只依赖 numpy**：不导入 cv2、不碰硬件、不开文件，所以能完全离线测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: RANSAC 内点带宽（像素）：轮廓点到拟合圆的距离在这个范围内就算"同意这个圆"。
#: 取 2.5 是按掩膜边界的毛糙程度定的——顶面被光照分成亮暗两块时，分割边界本就有
#: 一两个像素的锯齿，带宽比它略大才不会把真边界当成外点。
DEFAULT_BAND_PX = 2.5

#: RANSAC 迭代次数。三点定圆的样本空间很小，几百次足够把弦上的点排除掉；
#: 再多也不会更准——真边界点的比例是固定的。
DEFAULT_ITERATIONS = 400

#: 三点样本的最小包围盒边长（像素）。三点接近共线时解的半径会飞到几千像素，
#: 那不是圆而是数值噪声：实测真机帧上会出现 r=1739 / r=2989 这种解，
#: 而且它们照样"解释"三四个点，靠内点数排序根本拦不住。
MIN_SAMPLE_SPREAD_PX = 2.0

#: 内点占比下限。实测真顶面 0.35–0.86（越完整越高），台面/夹爪凑出来的"圆"
#: 0.04–0.52——两者重叠，所以**主要的那道闸是半径域**（调用方给的窗口），
#: 这一项只负责挡掉"点根本不在一个圆上"的极端情况。
DEFAULT_MIN_INLIER_RATIO = 0.30

#: 拟合前把轮廓点等间隔抽到这么多点以内。RANSAC 的开销随点数线性增长，
#: 而形状信息在几百个点上已经饱和（与 ``material_blob.ENCLOSING_MAX_POINTS``
#: 同一个理由，只是那里是给 ``minEnclosingCircle`` 用的）。
DEFAULT_MAX_POINTS = 512

#: 角度覆盖下限：内点绕拟合圆心铺开的角度占整圈的比例。
#: 一条很短的弧也能"拟合"出一个大圆，但那条解没有约束力——实测 75% 遮挡时
#: 覆盖只剩 0.25 左右，半径开始失稳。
DEFAULT_MIN_COVERAGE = 0.20


@dataclass(frozen=True)
class CircleFit:
    """一次成功的圆拟合：圆心、半径，以及这个解有多可信。"""

    center_x: float
    center_y: float
    radius: float
    #: 落在 ``±band_px`` 带内的点占参与拟合点数的比例（0–1）。
    inlier_ratio: float
    #: 内点绕圆心覆盖的角度占整圈的比例（0–1）。完整圆面接近 1，被挡掉一半约 0.5。
    coverage: float
    #: 参与拟合的点数（已抽样之后）。
    points: int

    @property
    def center(self) -> tuple[float, float]:
        """Return ``(x, y)`` of the fitted centre."""
        return (self.center_x, self.center_y)

    @property
    def inliers(self) -> int:
        """落在带宽内的点数。"""
        return int(round(self.inlier_ratio * self.points))

    def summary(self) -> str:
        return (
            f"@({self.center_x:.1f},{self.center_y:.1f}) r={self.radius:.1f} px，"
            f"内点 {self.inlier_ratio * 100:.0f}%（{self.inliers}/{self.points}），"
            f"角度覆盖 {self.coverage * 100:.0f}%"
        )


def as_points(points) -> np.ndarray:
    """把各种形状的轮廓点统一成 ``(N, 2)`` 的 float64 数组。

    接受 ``(N, 1, 2)``（``cv2.findContours`` 的原样输出）、``(N, 2)`` 和
    任何能 ``reshape(-1, 2)`` 的序列。非有限值（NaN/inf）在这里就被丢掉。
    """
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 2) if array.size % 2 == 0 else np.zeros((0, 2))
    elif array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    elif array.ndim != 2 or array.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    array = array[:, :2]
    return array[np.isfinite(array).all(axis=1)]


def fit_circle_kasa(points) -> Optional[tuple[float, float, float]]:
    """Kåsa 代数最小二乘：对**全部**给定点求一个圆；解不出来返回 ``None``。

    这一条是给"点已经干净了"的场合用的（拟合最后在 RANSAC 内点上重算一次）。
    直接把整条残缺轮廓丢进来会被那条直边带偏，见模块文档。
    """
    array = as_points(points)
    if len(array) < 3:
        return None
    x = array[:, 0]
    y = array[:, 1]
    design = np.column_stack([x, y, np.ones_like(x)])
    target = x * x + y * y
    try:
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = (float(value) for value in solution[:3])
    center_x, center_y = a / 2.0, b / 2.0
    squared = c + center_x * center_x + center_y * center_y
    if not math.isfinite(squared) or squared <= 0:
        return None
    radius = math.sqrt(squared)
    if not math.isfinite(radius) or radius <= 0:
        return None
    return (center_x, center_y, radius)


def angular_coverage(points, center_x: float, center_y: float) -> float:
    """点绕 ``(center_x, center_y)`` 铺开的角度占整圈的比例（0–1）。

    完整闭合的轮廓接近 1（分割的锯齿会让它略小于 1），半圈约 0.5。
    角度上是"最大空隙"的补：把角度排序后找最大的那个空档。
    """
    array = as_points(points)
    if len(array) < 3:
        return 0.0
    angles = np.arctan2(array[:, 1] - center_y, array[:, 0] - center_x)
    angles = np.sort(np.mod(angles, 2.0 * np.pi))
    gaps = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
    return float(1.0 - gaps.max() / (2.0 * np.pi))


def _decimate(points: np.ndarray, limit: int) -> np.ndarray:
    """等间隔抽样到 ``limit`` 点以内（``limit`` 不超过 0 表示不抽）。"""
    if limit <= 0 or len(points) <= limit:
        return points
    stride = len(points) // limit + 1
    return np.ascontiguousarray(points[::stride])


def fit_circle(
    points,
    *,
    band_px: float = DEFAULT_BAND_PX,
    iterations: int = DEFAULT_ITERATIONS,
    min_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 0,
) -> Optional[CircleFit]:
    """RANSAC + Kåsa：从（可能有缺损的）轮廓点里拟合一个可信的圆。

    ``min_radius`` / ``max_radius`` 是**半径合理域**——调用方一般直接给针孔模型
    算出来的窗口。它是这里最有效的一道闸：退化样本给出的 r=1739 这类解，
    内点数可能比真解还多，只有半径域能一眼挡掉。

    返回 ``None`` 表示"这批点拟合不出可信的圆"，调用方应当退回原来的估计
    （面积等效半径），而不是拿一个不可信的解去算距离。
    """
    if band_px <= 0 or iterations < 1:
        raise ValueError("band_px and iterations must be positive")
    if min_radius is not None and min_radius <= 0:
        raise ValueError("min_radius must be positive")
    if max_radius is not None and max_radius <= 0:
        raise ValueError("max_radius must be positive")
    if min_radius is not None and max_radius is not None and max_radius <= min_radius:
        raise ValueError("max_radius must exceed min_radius")
    if not 0 < min_inlier_ratio <= 1:
        raise ValueError("min_inlier_ratio must be within (0, 1]")
    if not 0 <= min_coverage < 1:
        raise ValueError("min_coverage must be within [0, 1)")

    array = _decimate(as_points(points), max_points)
    if len(array) < 5:
        return None

    def plausible(radius: float) -> bool:
        if not math.isfinite(radius) or radius <= 0:
            return False
        if min_radius is not None and radius < min_radius:
            return False
        if max_radius is not None and radius > max_radius:
            return False
        return True

    rng = np.random.default_rng(seed)
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    for _ in range(iterations):
        sample = array[rng.choice(len(array), size=3, replace=False)]
        if float(np.min(sample.max(axis=0) - sample.min(axis=0))) < MIN_SAMPLE_SPREAD_PX:
            continue                       # 三点接近共线：解会飞到几千像素
        solved = fit_circle_kasa(sample)
        if solved is None:
            continue
        center_x, center_y, radius = solved
        if not plausible(radius):
            continue
        distances = np.hypot(array[:, 0] - center_x, array[:, 1] - center_y)
        inliers = np.abs(distances - radius) <= band_px
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < 3:
        return None
    if best_count / len(array) < min_inlier_ratio:
        return None

    # 在内点上重算一次：RANSAC 那一步只用到 3 个点，精度不够。
    solved = fit_circle_kasa(array[best_inliers])
    if solved is None:
        return None
    center_x, center_y, radius = solved
    if not plausible(radius):
        return None

    distances = np.hypot(array[:, 0] - center_x, array[:, 1] - center_y)
    inliers = np.abs(distances - radius) <= band_px
    ratio = float(inliers.sum()) / len(array)
    if ratio < min_inlier_ratio:
        return None
    coverage = angular_coverage(array[inliers], center_x, center_y)
    if coverage < min_coverage:
        return None
    return CircleFit(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        inlier_ratio=ratio,
        coverage=coverage,
        points=len(array),
    )


# ---------------------------------------------------------------------- 自测
def _circle_points(center_x, center_y, radius, count=720, *, keep=None):
    """一圈点；``keep`` 给定时只保留满足它的那些（用来造"残缺的圆"）。"""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.column_stack(
        [center_x + radius * np.cos(angles), center_y + radius * np.sin(angles)]
    )
    if keep is not None:
        points = points[keep(points)]
    return points


def _self_test() -> int:
    problems: list[str] = []

    # 完整的圆：应当几乎精确。
    full = fit_circle(_circle_points(300.0, 200.0, 120.0))
    if full is None:
        problems.append("完整的圆应当拟合得出来")
    else:
        if abs(full.center_x - 300.0) > 0.2 or abs(full.center_y - 200.0) > 0.2:
            problems.append(f"完整圆的圆心偏了：{full.summary()}")
        if abs(full.radius - 120.0) > 0.2:
            problems.append(f"完整圆的半径偏了：{full.summary()}")
        if full.coverage < 0.95:
            problems.append(f"完整圆的角度覆盖应当接近 1：{full.summary()}")
        print(f"完整圆      {full.summary()}")

    # 只剩一段弧（挡掉一侧）：半径仍然要准，覆盖要掉下来。
    keep = lambda p: p[:, 0] > 300.0 - 0.35 * 120.0  # noqa: E731
    arc_points = _circle_points(300.0, 200.0, 120.0, keep=keep)
    arc = fit_circle(arc_points)
    if arc is None:
        problems.append("弧段应当还能拟合（真值半径 120）")
    else:
        if abs(arc.radius - 120.0) > 1.0:
            problems.append(f"弧段的半径偏了：{arc.summary()}")
        print(f"弧段(截掉一侧) {arc.summary()}")

    # 面积等效半径在同样的弧上会错多少：这就是要替换掉的那个估计。
    # 用多边形面积公式算，避免这里依赖 cv2。
    polygon = _circle_points(300.0, 200.0, 120.0, count=721)
    inside = polygon[keep(polygon)]
    area = 0.5 * abs(
        np.dot(inside[:, 0], np.roll(inside[:, 1], -1))
        - np.dot(inside[:, 1], np.roll(inside[:, 0], -1))
    )
    equivalent = math.sqrt(area / np.pi)
    print(f"  同一段弧的面积等效半径 {equivalent:.1f} px（真值 120.0）——"
          f"这才是被替换掉的那个估计")
    if equivalent >= 115.0:
        problems.append("这段弧的面积等效半径本该明显偏小，自测的构造有问题")

    # 直边（遮挡留下的弦）混进来：RANSAC 应当把它当外点，半径不被拽小。
    chord = np.column_stack([
        np.full(400, 300.0 - 0.6 * 120.0),
        np.linspace(200.0 - 120.0 * 0.8, 200.0 + 120.0 * 0.8, 400),
    ])
    keep_arc = lambda p: p[:, 0] > 300.0 - 0.6 * 120.0  # noqa: E731
    with_chord = np.vstack([_circle_points(300.0, 200.0, 120.0, keep=keep_arc), chord])
    cut = fit_circle(with_chord)
    if cut is None:
        problems.append("带弦的残缺圆应当拟合得出来")
    else:
        if abs(cut.radius - 120.0) > 2.0:
            problems.append(f"带弦时半径被拽偏了（真值 120）：{cut.summary()}")
        print(f"带弦(遮挡 40%) {cut.summary()}")

    # 半径域：把真解挡在外面时应当拒绝，而不是硬给一个解。
    if fit_circle(_circle_points(300.0, 200.0, 120.0), min_radius=40.0, max_radius=100.0) is not None:
        problems.append("半径域之外的解必须拒绝")
    if fit_circle(_circle_points(300.0, 200.0, 120.0), min_radius=140.0, max_radius=400.0) is not None:
        problems.append("半径域之外的解必须拒绝（偏大的一侧）")

    # 退化输入：共线点拟合不出圆。
    collinear = np.column_stack([np.linspace(0.0, 500.0, 60), np.full(60, 100.0)])
    if fit_circle(collinear) is not None:
        problems.append("共线点不该拟合出圆")
    if fit_circle(np.zeros((0, 2))) is not None:
        problems.append("空点集不该拟合出圆")
    if fit_circle(np.zeros((3, 2))) is not None:
        problems.append("点太少不该拟合出圆")

    # 输入的三种形状都要能吃：cv2 轮廓的 (N,1,2)、列表、已经是 (N,2) 的数组。
    raw = _circle_points(300.0, 200.0, 120.0, count=120)
    for name, value in (
        ("(N,1,2)", raw.reshape(-1, 1, 2)),
        ("list", raw.tolist()),
        ("(N,2)", raw),
    ):
        got = fit_circle(value)
        if got is None or abs(got.radius - 120.0) > 0.5:
            problems.append(f"{name} 形状的输入没拟合对：{None if got is None else got.summary()}")

    # 非法参数要当场报错，而不是悄悄用一个错的带宽。
    for kwargs in ({"band_px": 0.0}, {"iterations": 0}, {"min_radius": 0.0},
                   {"min_inlier_ratio": 0.0}, {"min_coverage": 1.0},
                   {"min_radius": 200.0, "max_radius": 100.0}):
        try:
            fit_circle(_circle_points(300.0, 200.0, 120.0), **kwargs)
        except ValueError:
            pass
        else:
            problems.append(f"非法参数必须报错：{kwargs}")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：完整圆、弧段、带弦的残缺圆都拟合得准，退化输入与非法参数都被挡住。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="从轮廓点用 RANSAC + Kåsa 拟合圆")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

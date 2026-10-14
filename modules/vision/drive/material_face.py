"""锁定圆面到底像不像"一块物料顶面"：内部够不够均匀、圆周上是不是真有边缘。

霍夫圆只保证"这里能凑出一个圆"，不保证这个圆是物料顶面：夹爪的圆弧、背景上
一块平坦的区域都能凑出圆，而且凑出来的半径各不相同，距离就会跟着乱跳。
本模块给每个锁定圆面两个只看灰度、不看颜色的指标：

- ``interior_std``：内部（半径 ``interior_ratio`` 倍以内）的灰度标准差。
  物料顶面是一块单色圆面，标准差很小；夹爪弧线的内部跨过白爪和暗背景，
  标准差大一个量级（实测 4 对 16–20）。
- ``edge_support``：圆周上真正存在径向边缘的比例（0–1）。背景上一块平坦区域
  凑出来的圆没有边缘，这一项接近 0；真正的圆面接近 1（实测 0.8 对 0.08）。

两个指标都不依赖颜色，所以六种颜色的物料通用；也不依赖尺度，只跟这一帧有关。

命令行自测::

    python -m modules.vision.drive.material_face     # 合成圆面 + 干扰，不需要摄像头
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modules.vision.drive.hough_lock import LockedRegion

#: 内部取样半径占圆面半径的比例。留出边缘过渡带，避免把模糊的边算进"内部"。
DEFAULT_INTERIOR_RATIO = 0.75

#: 圆周外侧取样半径占圆面半径的比例：边缘就在这两圈之间。
DEFAULT_OUTER_RATIO = 1.25

#: 圆周上取多少个方向的样本。36 个方向（每 10°）足够分辨"整圈都是边"和"背景上一块平地"，
#: 又比 72 个省一半时间——每帧每个候选圆都要算一次。
DEFAULT_RAYS = 36

#: 内部统计的抽样步长：半径超过这个像素数就隔一个像素取一个（大圆面本来就有几万个像素）。
STRIDE_RADIUS_PX = 60

#: 内圈灰度与外侧灰度差多少才算"这里有一条径向边缘"。
DEFAULT_EDGE_THRESHOLD = 25.0

#: 超过这个内部标准差就不当"单色圆面"看（实测物料约 4，夹爪弧线约 16–20）。
DEFAULT_INTERIOR_STD_MAX = 12.0

#: 低于这个边界支撑比例就不当"有完整边界"看（实测物料约 0.8，平坦背景约 0.08）。
DEFAULT_EDGE_SUPPORT_MIN = 0.55


@dataclass(frozen=True)
class FaceQuality:
    """一个锁定圆面的"像不像物料顶面"的两个原始指标。"""

    #: 内部灰度标准差；越小越像一块单色圆面。
    interior_std: float
    #: 圆周上检出径向边缘的方向占比，0–1；越大越像真的圆面边界。
    edge_support: float
    #: 实际参与统计的方向数（贴边被跳过的方向不计）。
    sampled_rays: int

    @property
    def looks_like_face(self) -> bool:
        """够均匀、边界也够完整时才算像物料顶面。"""
        return (
            self.interior_std <= DEFAULT_INTERIOR_STD_MAX
            and self.edge_support >= DEFAULT_EDGE_SUPPORT_MIN
        )

    def summary(self) -> str:
        return (
            f"内部σ={self.interior_std:.1f}，边界支撑={self.edge_support * 100:.0f}%"
            f"（{self.sampled_rays} 个方向）"
        )


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """把一帧 BGR/BGRA/灰度图变成 float64 灰度图（纯 NumPy，不依赖 cv2）。

    **已经是 float64 的二维灰度图原样返回，不再复制。** 这一条是为
    ``MaterialLocator`` 准备的：它一帧要对每个候选圆各量一次 face，而每量一次
    都重转整帧的话，1280×720 上转一次 12.0 ms、复制一次 1.0 ms——霍夫在真实
    台面图上会给 24–87 个候选，"87 × 12 ms ≈ 1.0 s"就是这么来的（实测）。
    调用方自己转一次、把这张图传进来，**数值完全不变**：还是同一条
    ``mean(axis=2, dtype=np.float64)`` 算出来的，只是不再重复算。

    非 float64 的输入（float32 也算）仍然复制并转成 float64：float32 与 float64
    的累加结果在末位可能不同，而下游是拿这些数跟阈值比的，不能悄悄换精度。
    """
    array = np.asarray(frame)
    if array.ndim == 2:
        if array.size == 0:
            raise ValueError("frame is empty")
        if array.dtype == np.float64:
            return array
        return array.astype(np.float64)
    if array.ndim != 3:
        raise ValueError(f"unsupported frame shape: {array.shape}")
    gray = array[:, :, :3].mean(axis=2, dtype=np.float64)
    if gray.size == 0:
        raise ValueError("frame is empty")
    return gray


def measure_face(
    frame: np.ndarray,
    region: LockedRegion,
    *,
    interior_ratio: float = DEFAULT_INTERIOR_RATIO,
    outer_ratio: float = DEFAULT_OUTER_RATIO,
    rays: int = DEFAULT_RAYS,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> FaceQuality:
    """量一个锁定圆面的内部均匀度与边界支撑度。

    ``frame`` 可以是 BGR 也可以是灰度图；**同一帧要量多个圆时请自己先转一次**
    :func:`to_grayscale`，再把这个灰度图传给本函数（见 ``to_grayscale`` 说明：
    每个候选都重转一遍整帧，是 ``MaterialLocator`` 里最容易被忽略的一笔开销）。
    ``region`` 是 :class:`LockedRegion`。
    """
    if rays < 8:
        raise ValueError("rays must be at least 8")
    if not 0 < interior_ratio < 1:
        raise ValueError("interior_ratio must be between 0 and 1")
    if outer_ratio <= 1:
        raise ValueError("outer_ratio must exceed 1")
    gray = to_grayscale(frame)
    height, width = gray.shape
    return FaceQuality(
        interior_std=interior_std(gray, region, ratio=interior_ratio),
        edge_support=edge_support(
            gray, region, rays=rays, outer_ratio=outer_ratio, threshold=edge_threshold
        ),
        sampled_rays=rays,
    )


def interior_std(gray: np.ndarray, region: LockedRegion, *, ratio: float = DEFAULT_INTERIOR_RATIO) -> float:
    """圆面内部（半径 ``ratio`` 倍以内）的灰度标准差。

    大圆面按 :data:`STRIDE_RADIUS_PX` 隔点取样：标准差不需要每个像素都算，
    而 7 cm 机位下一个顶面就有十几万像素，逐点算是每帧最贵的一步。
    """
    sample_radius = max(1.0, region.radius * ratio)
    stride = 1 if region.radius < STRIDE_RADIUS_PX else 2
    x1 = int(max(0, np.floor(region.center_x - sample_radius)))
    x2 = int(min(gray.shape[1], np.ceil(region.center_x + sample_radius) + 1))
    y1 = int(max(0, np.floor(region.center_y - sample_radius)))
    y2 = int(min(gray.shape[0], np.ceil(region.center_y + sample_radius) + 1))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return float("inf")
    crop = gray[y1:y2:stride, x1:x2:stride]
    ys, xs = np.ogrid[y1:y2:stride, x1:x2:stride]
    inside = (xs - region.center_x) ** 2 + (ys - region.center_y) ** 2 <= sample_radius**2
    values = crop[inside]
    if values.size < 8:
        return float("inf")
    return float(values.std())


def edge_support(
    gray: np.ndarray,
    region: LockedRegion,
    *,
    rays: int = DEFAULT_RAYS,
    outer_ratio: float = DEFAULT_OUTER_RATIO,
    threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> float:
    """圆周上有径向边缘的方向占比。

    每个方向比较``半径处``与``外圈``的平均灰度：差超过 ``threshold`` 就算这里有
    一条边界。取样点落到画面外的方向直接跳过，所以贴着画面边缘的圆不会被冤枉。
    """
    height, width = gray.shape
    reached = 0
    supported = 0
    for angle in np.linspace(0.0, 2.0 * np.pi, rays, endpoint=False):
        cos, sin = float(np.cos(angle)), float(np.sin(angle))
        inner = _patch_mean(
            gray, region.center_x + cos * region.radius, region.center_y + sin * region.radius
        )
        outer = _patch_mean(
            gray,
            region.center_x + cos * region.radius * outer_ratio,
            region.center_y + sin * region.radius * outer_ratio,
        )
        if inner is None or outer is None:
            continue
        reached += 1
        if abs(inner - outer) > threshold:
            supported += 1
    if reached == 0:
        return 0.0
    return supported / reached


def _patch_mean(gray: np.ndarray, x: float, y: float, *, half: int = 2) -> float | None:
    """以 ``(x, y)`` 为中心取一小块的平均灰度；越界返回 ``None``。"""
    xi, yi = int(round(x)), int(round(y))
    if not (half <= xi < gray.shape[1] - half and half <= yi < gray.shape[0] - half):
        return None
    return float(gray[yi - half:yi + half + 1, xi - half:xi + half + 1].mean())


# ---------------------------------------------------------------------- 自测
def _synthetic_scene() -> tuple[np.ndarray, LockedRegion, LockedRegion, LockedRegion]:
    """一帧里有：一块单色亮圆面、一个内部不匀的干扰圆、一块平坦背景。

    返回 ``(帧, 物料圆, 干扰圆, 平坦背景)``。
    """
    frame = np.full((480, 640), 60.0)
    ys, xs = np.ogrid[:480, :640]
    # 干扰：一条更亮的横带，最后被物料圆面盖住中段，但仍然横穿干扰圆的内部
    frame[240:250, 120:520] = 250.0
    # 物料：单色圆面，内部标准差接近 0，圆周上是完整的亮暗边界
    frame[(xs - 320) ** 2 + (ys - 240) ** 2 <= 70**2] = 200.0
    material = LockedRegion(center_x=320.0, center_y=240.0, radius=70.0)
    jaw = LockedRegion(center_x=320.0, center_y=245.0, radius=140.0)
    # 平坦背景：内部很均匀，但圆周上没有任何边缘
    flat = LockedRegion(center_x=560.0, center_y=400.0, radius=60.0)
    return frame, material, jaw, flat


def _self_test() -> int:
    frame, material, jaw, flat = _synthetic_scene()
    material_quality = measure_face(frame, material)
    jaw_quality = measure_face(frame, jaw)
    flat_quality = measure_face(frame, flat)

    problems: list[str] = []
    print(f"物料圆面：{material_quality.summary()}  像顶面={material_quality.looks_like_face}")
    print(f"夹爪弧线：{jaw_quality.summary()}  像顶面={jaw_quality.looks_like_face}")
    print(f"平坦背景：{flat_quality.summary()}  像顶面={flat_quality.looks_like_face}")

    if not material_quality.looks_like_face:
        problems.append("单色圆面应当被判为像物料顶面")
    if jaw_quality.looks_like_face:
        problems.append("夹爪弧线的内部不均匀，不应被判为物料顶面")
    if flat_quality.looks_like_face:
        problems.append("平坦背景没有边界，不应被判为物料顶面")
    if not material_quality.interior_std < jaw_quality.interior_std:
        problems.append("物料圆面的内部标准差应当明显小于夹爪弧线")
    if not material_quality.edge_support > flat_quality.edge_support:
        problems.append("物料圆面的边界支撑应当明显高于平坦背景")

    # 贴边的圆不该因为取样越界而崩掉
    measure_face(frame, LockedRegion(center_x=5.0, center_y=5.0, radius=40.0))

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：内部均匀度与边界支撑能区分物料顶面 / 夹爪弧线 / 平坦背景。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="锁定圆面的内部均匀度与边界支撑")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

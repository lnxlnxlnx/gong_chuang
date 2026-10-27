"""两点像素测量：画面上点两个点，量出 ΔX / ΔY / 直线距离。

职责单一：**只有算术**。不取帧、不开窗口、不点鼠标——点从哪来是调用方的事
（真机交互在 ``modules.vision.test.manual.pixel_measure``，会话入口在
``modules.vision.runtime.Vision.pixel_distance``）。

## 为什么单独立一个模块

这条链路上真正会出错的地方只有一个：**"哪个是哪一个点"**。第一击与第二击的顺序
决定 ΔX / ΔY 的正负号，而正负号在"判断工件偏左还是偏右"时就是结论本身。所以本模块
把"有符号分量"当作主输出（``dx_px`` / ``dy_px``），把长度当作派生量（``length_px``），
并且**不做任何排序**——按传入顺序算，调用方给反了就得到反的符号，这是如实报数。

像素到毫米的换算刻意**不放在这里**：那需要焦距与工作距离，属于标定，见
:func:`to_millimetres`。本模块的核心函数在没有相机、没有标定的机器上也能跑。

## 为什么不用 cv2 的测量原语

没有这样的原语。``cv2.arcLength`` 量的是轮廓周长，``cv2.norm`` 量的是向量模长但拿不到
有符号分量。这条链路本来就只有 ``dx``、``dy``、``hypot`` 三个式子，写出来比拼 OpenCV
调用更短也更好读。

命令行自测::

    python -m modules.vision.drive.pixel_measure        # 离线自测，不需要摄像头/标定
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

#: 画面坐标（像素），左上角为原点，x 向右、y 向下。
#:
#: 与 :mod:`modules.vision.drive.hough_lock` 的 ``LockedRegion`` 用同一套图像坐标，
#: 所以圆面圆心可以直接当测点传进来。
Point = Tuple[float, float]

#: 标定时用的已知真实长度（毫米）——与相机内参无关，是"尺子量出来的那个数"。
DEFAULT_REFERENCE_MM = 30.0


@dataclass(frozen=True)
class PixelMeasurement:
    """一次两点测量：两个端点 + 有符号分量 + 直线距离。

    ``dx_px`` / ``dy_px`` **有符号**：从 :attr:`start` 指向 :attr:`end`。
    :attr:`length_px` 恒为非负。

    :attr:`image_size` 是这一帧的 ``(宽, 高)``；给了它就能查出"端点是否落在画面外"
    （:attr:`start_inside` / :attr:`end_inside`），``None`` 表示不查这一项。
    """

    start: Point
    end: Point
    dx_px: float
    dy_px: float
    length_px: float
    image_size: Optional[Tuple[int, int]] = None

    @property
    def start_inside(self) -> Optional[bool]:
        """起点是否落在画面内；没给 :attr:`image_size` 时是 ``None``（不是 ``False``）。"""
        return None if self.image_size is None else _inside(self.start, self.image_size)

    @property
    def end_inside(self) -> Optional[bool]:
        """终点是否落在画面内；没给 :attr:`image_size` 时是 ``None``。"""
        return None if self.image_size is None else _inside(self.end, self.image_size)

    @property
    def angle_degrees(self) -> float:
        """方向角，单位度：画面右方为 0、下方为 +90、左方为 ±180。

        与 :attr:`dx_px` / :attr:`dy_px` 同一套符号（y 向下），所以"工件偏向画面下方"
        在这里是正角。长度为零时返回 ``0.0`` 而不是抛错——一个退化成点的测量没有方向，
        但让调用方为了打印一行字去接异常是过度设计。
        """
        if self.length_px == 0.0:
            return 0.0
        return math.degrees(math.atan2(self.dy_px, self.dx_px))

    def summary(self) -> str:
        """一行中文结论，供终端打印。"""
        text = (
            f"两点测量：({self.start[0]:.1f}, {self.start[1]:.1f}) -> "
            f"({self.end[0]:.1f}, {self.end[1]:.1f})，"
            f"ΔX={self.dx_px:+.1f} px、ΔY={self.dy_px:+.1f} px、"
            f"距离={self.length_px:.2f} px"
        )
        if self.image_size is not None:
            text += f"，画面 {self.image_size[0]}×{self.image_size[1]}"
        return text


@dataclass(frozen=True)
class MillimetreMeasurement:
    """把 :class:`PixelMeasurement` 按已知标定换算成毫米的结果。

    :attr:`scale_mm_per_px` 是**该点当前景深处的**比例尺；换机位、换高度都会变，
    所以它跟着每次测量一起返回，而不是藏在配置里让调用方猜。
    """

    measurement: PixelMeasurement
    focal_px: float
    working_distance_mm: float
    scale_mm_per_px: float
    dx_mm: float
    dy_mm: float
    length_mm: float

    def summary(self) -> str:
        return (
            f"{self.measurement.summary()}；按 f={self.focal_px:.0f} px、"
            f"工作距离 {self.working_distance_mm:.0f} mm 换算，"
            f"比例尺 {self.scale_mm_per_px:.4f} mm/px："
            f"ΔX={self.dx_mm:+.2f} mm、ΔY={self.dy_mm:+.2f} mm、"
            f"距离={self.length_mm:.2f} mm"
        )


def _inside(point: Point, image_size: Tuple[int, int]) -> bool:
    """点在画面内：闭区间判断（正好落在最后一列/行上算在内）。"""
    width, height = image_size
    return 0.0 <= point[0] <= width and 0.0 <= point[1] <= height


def _checked_point(point, name: str) -> Point:
    """把一个测点收成 ``(x, y)`` 浮点二元组，写法不对就当场报错。

    这里**刻意不接受** ``None`` 或长度不为 2 的序列：静默补一个 0 会让"少点了一下"
    变成"量出左边缘到另一点的距离"，那种错误在现场表现为一个看起来正常的数。
    """
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"{name} 必须是 (x, y) 这样的两元素坐标，实际是 {point!r}")
    coordinates = []
    for index, value in enumerate(point):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{index}] 必须是数字，实际是 {value!r}")
        coordinates.append(float(value))
    return (coordinates[0], coordinates[1])


def pixel_distance(
    start,
    end,
    *,
    image_size: Optional[Tuple[int, int]] = None,
) -> PixelMeasurement:
    """量出两点之间的 ΔX / ΔY（有符号）与直线距离，单位像素。

    方向从 ``start`` 指向 ``end``：``dx_px = end.x - start.x``。**不排序、不取绝对值**，
    所以要"从 A 到 B"还是"从 B 到 A"由调用方决定。
    """
    first = _checked_point(start, "start")
    second = _checked_point(end, "end")
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    return PixelMeasurement(
        start=first,
        end=second,
        dx_px=dx,
        dy_px=dy,
        length_px=math.hypot(dx, dy),
        image_size=None if image_size is None else (int(image_size[0]), int(image_size[1])),
    )


def to_millimetres(
    measurement: PixelMeasurement,
    *,
    focal_px: float,
    working_distance_mm: float,
) -> MillimetreMeasurement:
    """按针孔模型把像素测量换算成毫米。

    ``毫米 = 像素 × 工作距离 / 焦距``——这就是 :mod:`modules.vision.drive.pinhole`
    那套换算，只是这里量的是**两点之差**而不是某个点到光轴的距离。

    ``working_distance_mm`` 是**被测量平面**到镜头的距离。用错这个数（比如拿了台面
    深度去量一个抬高的工件顶面）会按比例整体缩放结果，而**方向与比值不受影响**：
    所以"偏左还是偏右"永远可信，"偏了几毫米"取决于这个数。
    """
    if focal_px <= 0:
        raise ValueError("focal_px must be positive")
    if working_distance_mm <= 0:
        raise ValueError("working_distance_mm must be positive")
    scale = float(working_distance_mm) / float(focal_px)
    return MillimetreMeasurement(
        measurement=measurement,
        focal_px=float(focal_px),
        working_distance_mm=float(working_distance_mm),
        scale_mm_per_px=scale,
        dx_mm=measurement.dx_px * scale,
        dy_mm=measurement.dy_px * scale,
        length_mm=measurement.length_px * scale,
    )


def focal_px_from_known_length(
    *,
    pixel_length: float,
    real_length_mm: float,
    distance_mm: float,
) -> float:
    """由"已知真实长度 + 已知距离 + 量到的像素长度"反解焦距。

    这是标定焦距的唯一公式（``f = 距离 × 像素长度 / 真实长度``），
    ``distance_mm`` 是**被测物所在平面**到镜头的距离。多测几次取平均更稳——
    把几次结果交给 :func:`mean_focal_px`。
    """
    if pixel_length <= 0:
        raise ValueError("pixel_length must be positive")
    if real_length_mm <= 0:
        raise ValueError("real_length_mm must be positive")
    if distance_mm <= 0:
        raise ValueError("distance_mm must be positive")
    return float(distance_mm) * float(pixel_length) / float(real_length_mm)


def mean_focal_px(samples: Sequence[float]) -> float:
    """把多次反解出来的焦距取平均。

    刻意**不做**野值剔除：焦距的几次测量之间差得离谱，通常意味着某一次的距离或真实
    长度填错了，那时应当让调用方看见散布（自己打印各次结果），而不是被一个稳健平均
    悄悄抹平。空序列直接报错。
    """
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("至少要有一个焦距样本")
    for value in values:
        if value <= 0:
            raise ValueError("焦距样本必须是正数")
    return sum(values) / len(values)


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    problems: list[str] = []

    measured = pixel_distance((100.0, 200.0), (400.0, 600.0), image_size=(1280, 720))
    if (measured.dx_px, measured.dy_px) != (300.0, 400.0):
        problems.append(f"ΔX/ΔY 应为 (300, 400)，实际 {measured.dx_px, measured.dy_px}")
    if abs(measured.length_px - 500.0) > 1e-9:
        problems.append(f"3-4-5 直角三角形的距离应为 500.0，实际 {measured.length_px}")

    # 符号是主输出：反向量一次必须整体取反，长度不变。
    reversed_measure = pixel_distance((400.0, 600.0), (100.0, 200.0))
    if (reversed_measure.dx_px, reversed_measure.dy_px) != (-300.0, -400.0):
        problems.append(
            f"反向测量应有符号分量取反，实际 {reversed_measure.dx_px, reversed_measure.dy_px}"
        )
    if abs(reversed_measure.length_px - measured.length_px) > 1e-9:
        problems.append("反向测量的直线距离应与正向相同")

    if measured.start_inside is not True or measured.end_inside is not True:
        problems.append("两个端点都在 1280×720 画面内，应判为 True")
    outside = pixel_distance((100.0, 200.0), (1400.0, 600.0), image_size=(1280, 720))
    if outside.end_inside is not False:
        problems.append("x=1400 落在 1280 宽的画外，end_inside 应为 False")
    unqueried = pixel_distance((100.0, 200.0), (400.0, 600.0))
    if unqueried.start_inside is not None:
        problems.append("没给 image_size 时 inside 应为 None（表示没查），不是 False")

    # 画面右方为 0°、下方为 +90°（y 轴向下）。
    if abs(pixel_distance((0.0, 0.0), (10.0, 0.0)).angle_degrees - 0.0) > 1e-9:
        problems.append("向右的方向角应为 0°")
    if abs(pixel_distance((0.0, 0.0), (0.0, 10.0)).angle_degrees - 90.0) > 1e-9:
        problems.append("向下的方向角应为 +90°（y 轴向下）")
    if abs(pixel_distance((0.0, 0.0), (-10.0, 0.0)).angle_degrees - 180.0) > 1e-9:
        problems.append("向左的方向角应为 ±180°")
    if pixel_distance((5.0, 5.0), (5.0, 5.0)).angle_degrees != 0.0:
        problems.append("退化成一个点的测量，方向角应为 0.0 而不是抛错")

    # 换算是线性的：同一比例尺下 GF 不变、长度按比例缩放。
    millimetres = to_millimetres(measured, focal_px=500.0, working_distance_mm=100.0)
    if abs(millimetres.scale_mm_per_px - 0.2) > 1e-9:
        problems.append(f"比例尺应为 0.2 mm/px，实际 {millimetres.scale_mm_per_px}")
    if abs(millimetres.length_mm - 100.0) > 1e-9:
        problems.append(f"500 px × 0.2 mm/px 应为 100.0 mm，实际 {millimetres.length_mm}")
    if abs(millimetres.dx_mm / millimetres.dy_mm - 0.75) > 1e-9:
        problems.append("换算不该改变 ΔX:ΔY 的比值（尺度是各向同性的）")

    # 焦距反解：f = 距离 × 像素长 / 真实长；反过来用它能还原出真实长度。
    focal = focal_px_from_known_length(pixel_length=250.0, real_length_mm=30.0, distance_mm=60.0)
    if abs(focal - 500.0) > 1e-9:
        problems.append(f"焦距应反解为 500.0，实际 {focal}")
    if abs(mean_focal_px([480.0, 500.0, 520.0]) - 500.0) > 1e-9:
        problems.append("多次焦距样本的平均值应为 500.0")

    # 写法不对要当场报错，而不是静默补 0 变成一个看着正常的数。
    for bad_call, name in (
        (lambda: pixel_distance((1.0,), (2.0, 3.0)), "少一个分量"),
        (lambda: pixel_distance(None, (2.0, 3.0)), "传了 None"),
        (lambda: pixel_distance(("a", 1.0), (2.0, 3.0)), "分量不是数字"),
        (lambda: to_millimetres(measured, focal_px=0.0, working_distance_mm=100.0), "焦距为 0"),
        (lambda: to_millimetres(measured, focal_px=500.0, working_distance_mm=-1.0), "距离为负"),
        (lambda: focal_px_from_known_length(pixel_length=0.0, real_length_mm=30.0, distance_mm=60.0), "像素长为 0"),
        (lambda: mean_focal_px([]), "焦距样本为空"),
    ):
        try:
            bad_call()
        except ValueError:
            continue
        problems.append(f"{name} 应被拒绝，实际被接受了")

    print(f"合成测量：{measured.summary()}")
    print(f"毫米换算：{millimetres.summary()}")
    print(f"焦距反解：250 px / 30 mm @ 60 mm -> f={focal:.1f} px")
    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：有符号分量、直线距离、方向角、毫米换算与焦距反解都正确。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="两点像素测量：ΔX / ΔY / 直线距离")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())

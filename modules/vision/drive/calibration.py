"""相机水平面 → 机械臂基座平面的平面标定（纯数学，不碰 cv2 / 串口 / 摄像头）。

## 为什么只需要「圆心 + 转角」三个参数

``modules.vision.drive.pinhole`` 在光轴竖直向下时给出的 ``x_mm/y_mm`` 已经是**水平**
毫米量（理由见该模块说明），所以相机水平面与机械臂基座平面之间只差一个刚体变换：

```
X_arm = R(rotation_degrees) · (p_cam − center)
```

- ``center``：基座旋转轴在相机水平面上的投影，也就是 ``p_cam`` 中对应 ``r = 0`` 的点；
- ``rotation_degrees``：θ 零点偏置，即相机画面轴向与机械臂 X 轴之间的夹角。

**这里刻意没有缩放参数。** 尺度已经由像素焦距承担；再放一个缩放会和焦距互相吞掉，
两个数永远不可能同时确定。缩放固定为 1：焦距不准就去重标焦距，而不是用缩放去凑。

拿到平面坐标后，圆柱关节量是两行三角函数：

```
base_degrees = atan2(Y, X)        radial_mm = hypot(X, Y)
```

## 拟合要用到的观测

每次观测给一组 ``CalibrationSample``：**相机测到的靶标位置**加**当时命令的关节量**。

机械臂只有相对位置模式，所以命令的 ``radial_mm`` 是"相对标定起点的累计量"，
含一个未知常数偏移 ``r0``。拟合时把 ``r0`` 一起解出来（迭代两三步即收敛），
否则圆心会被这个偏移系统地拉偏。``r0`` 只用于拟合，不是机器常数，所以不写进结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Tuple

import numpy as np

#: 圆心 / 径向偏移的交替迭代次数。实测两三次就到 1e-9 以内。
MAX_FIT_ITERATIONS = 12

#: 交替迭代的收敛阈值（毫米）。
FIT_TOLERANCE_MM = 1e-9


@dataclass(frozen=True)
class CalibrationSample:
    """一次标定观测：相机测到的靶标位置，加当时命令的关节量。"""

    #: 靶标中心在相机水平面上的位置（毫米），由 ``pinhole`` 给出。
    x_mm: float
    y_mm: float
    #: 该次观测时命令的底座转角（度）。
    base_degrees: float
    #: 该次观测时命令的径向伸出量（毫米），相对标定起点累计。
    radial_mm: float


@dataclass(frozen=True)
class PlaneCalibration:
    """相机水平面到机械臂基座平面的标定结果。"""

    #: 基座旋转轴在相机水平面上的投影（毫米）。
    center_x_mm: float
    center_y_mm: float
    #: θ 零点偏置（度）。
    rotation_degrees: float
    #: 拟合残差 RMS（毫米）；由 :func:`fit_plane_calibration` 填入。
    rms_mm: float = 0.0
    #: 参与拟合的样本数。
    samples: int = 0

    def to_base_plane(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """把相机水平面上的点换算成机械臂平面坐标 ``(X, Y)``，原点在基座旋转轴。"""
        dx = float(x_mm) - self.center_x_mm
        dy = float(y_mm) - self.center_y_mm
        radians = math.radians(self.rotation_degrees)
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        return (dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a)

    def to_polar(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """把相机水平面上的点换算成 ``(base_degrees, radial_mm)``。

        ``base_degrees`` 落在 ``(-180, 180]``，与 :func:`modules.arm.drive.kinematics.inverse_cylindrical`
        的输出同域，可直接相减求最短转角。
        """
        plane_x, plane_y = self.to_base_plane(x_mm, y_mm)
        return (math.degrees(math.atan2(plane_y, plane_x)), math.hypot(plane_x, plane_y))

    def residual_mm(self, sample: CalibrationSample) -> float:
        """一个样本经本标定换算后，与命令量之间的位置误差（毫米）。"""
        plane_x, plane_y = self.to_base_plane(sample.x_mm, sample.y_mm)
        radians = math.radians(sample.base_degrees)
        expected_x = sample.radial_mm * math.cos(radians)
        expected_y = sample.radial_mm * math.sin(radians)
        return math.hypot(plane_x - expected_x, plane_y - expected_y)

    def summary(self) -> str:
        return (
            f"基座轴像点=({self.center_x_mm:.2f},{self.center_y_mm:.2f}) mm，"
            f"θ 零点={self.rotation_degrees:.2f}°，"
            f"残差 RMS={self.rms_mm:.2f} mm（{self.samples} 点）"
        )


def fit_plane_calibration(samples: Iterable[CalibrationSample]) -> PlaneCalibration:
    """用一组标定观测解出 ``center`` 与 ``rotation_degrees``。

    圆心用线性最小二乘求（把 ``(x−cx)² + (y−cy)² = r²`` 展开成对 ``cx``、``cy`` 的线性式），
    转角用圆均值求。命令径向量里的未知常数偏移 ``r0`` 与圆心交替迭代消掉。

    :raises ValueError: 样本少于 3 个、点几乎共线，或角度观测无法定出零点。
    """
    points: List[CalibrationSample] = list(samples)
    if len(points) < 3:
        raise ValueError("平面标定至少需要 3 个样本")

    radial_offset = 0.0
    center_x, center_y = 0.0, 0.0
    for _ in range(MAX_FIT_ITERATIONS):
        center_x, center_y = _fit_center(points, radial_offset)
        measured = [
            math.hypot(sample.x_mm - center_x, sample.y_mm - center_y) - sample.radial_mm
            for sample in points
        ]
        updated = sum(measured) / len(measured)
        if abs(updated - radial_offset) < FIT_TOLERANCE_MM:
            break
        radial_offset = updated

    rotation = _circular_mean_degrees(
        sample.base_degrees
        - math.degrees(math.atan2(sample.y_mm - center_y, sample.x_mm - center_x))
        for sample in points
    )
    provisional = PlaneCalibration(
        center_x_mm=center_x,
        center_y_mm=center_y,
        rotation_degrees=rotation,
    )
    rms = math.sqrt(
        sum(provisional.residual_mm(sample) ** 2 for sample in points) / len(points)
    )
    return replace(provisional, rms_mm=rms, samples=len(points))


def _fit_center(
    points: Sequence[CalibrationSample],
    radial_offset: float,
) -> Tuple[float, float]:
    """给定径向偏移，线性最小二乘求圆心。

    展开 ``(x−cx)² + (y−cy)² = r²``：

    ```
    x² + y² − r² = 2·cx·x + 2·cy·y − (cx² + cy²)
    ```

    左边全部已知，右边对 ``(cx, cy)`` 线性，所以一次 ``lstsq`` 就够。

    **``r²`` 必须留在等式左边。** 每个样本的径向不同，把 ``r² − cx² − cy²`` 当成待求常数
    （普通 Kasa 圆拟合的写法）只在所有点共圆时成立；本实验的半径是变化的，那样写会把
    圆心系统地拉偏。``r = radial + r0`` 里的未知起点偏移 ``r0`` 由外层迭代消掉。
    """
    matrix = np.array(
        [[2.0 * sample.x_mm, 2.0 * sample.y_mm, 1.0] for sample in points],
        dtype=float,
    )
    target = np.array(
        [
            sample.x_mm ** 2 + sample.y_mm ** 2 - (sample.radial_mm + radial_offset) ** 2
            for sample in points
        ],
        dtype=float,
    )
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    center_x, center_y = float(solution[0]), float(solution[1])
    if not (math.isfinite(center_x) and math.isfinite(center_y)):
        raise ValueError("平面标定点几乎共线，无法定出基座旋转轴")
    return center_x, center_y


def _circular_mean_degrees(values: Iterable[float]) -> float:
    """角度的圆均值，结果落在 ``(-180, 180]``。"""
    angles = [float(value) for value in values]
    if not angles:
        raise ValueError("没有可求圆均值的角度")
    sum_cos = sum(math.cos(math.radians(angle)) for angle in angles)
    sum_sin = sum(math.sin(math.radians(angle)) for angle in angles)
    if math.hypot(sum_cos, sum_sin) < 1e-12:
        raise ValueError("角度观测两两抵消，无法定出 θ 零点")
    return math.degrees(math.atan2(sum_sin, sum_cos))


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要相机：用合成样本验证标定能还原出真值。"""
    truth = PlaneCalibration(center_x_mm=120.0, center_y_mm=-45.0, rotation_degrees=37.0)

    samples = []
    for index in range(16):
        base_degrees = -160.0 + index * 20.0
        radial_mm = 60.0 + index * 5.0
        # 由真值反推"相机测到的"靶标位置。
        radians = math.radians(base_degrees - truth.rotation_degrees)
        samples.append(
            CalibrationSample(
                x_mm=truth.center_x_mm + radial_mm * math.cos(radians),
                y_mm=truth.center_y_mm + radial_mm * math.sin(radians),
                base_degrees=base_degrees,
                radial_mm=radial_mm,
            )
        )

    fitted = fit_plane_calibration(samples)
    assert abs(fitted.center_x_mm - truth.center_x_mm) < 1e-6, fitted
    assert abs(fitted.center_y_mm - truth.center_y_mm) < 1e-6, fitted
    assert abs(fitted.rotation_degrees - truth.rotation_degrees) < 1e-6, fitted
    assert fitted.rms_mm < 1e-6, fitted

    base_degrees, radial_mm = fitted.to_polar(120.0, -45.0 + 80.0)
    assert abs(radial_mm - 80.0) < 1e-6, (base_degrees, radial_mm)
    assert abs(base_degrees - 127.0) < 1e-6, base_degrees

    # 命令径向量带未知起点偏移时，圆心仍必须解对。
    offset_samples = [
        replace(sample, radial_mm=sample.radial_mm + 133.0) for sample in samples
    ]
    refitted = fit_plane_calibration(offset_samples)
    assert abs(refitted.center_x_mm - truth.center_x_mm) < 1e-6, refitted
    assert abs(refitted.center_y_mm - truth.center_y_mm) < 1e-6, refitted

    try:
        fit_plane_calibration(samples[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("少于 3 个样本必须报错")

    print(f"平面标定自测通过：{fitted.summary()}（起点偏移 133 mm 仍复现真值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

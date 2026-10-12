"""针孔相机模型：像素与相机系毫米之间的纯换算。

这里只有数学，不导入 cv2、不打开摄像头、不碰串口，所以能完全离线测试。
``material_locate`` 的圆面测距和抓取用的靶标定位共用这一份实现，保证
「像素 → 毫米」在整个项目里只有一个定义。

## 坐标系约定

（与 ``docs/视觉物料定位.md`` 一致）

- 原点在镜头光心；``x`` 向画面右、``y`` 向画面下、``z`` 沿光轴向前（俯视时向下）；
- 主点取画面中心，水平与竖直像素焦距相等；不校正镜头畸变。

## 为什么这直接给出了水平毫米坐标

光轴竖直向下时，``z`` 就是竖直距离。一条从光心射向物点的光线，其水平分量正比于
像素偏移，所以 ``x_mm = (u - cx) * z / f`` 就是**物点到光轴的真实水平距离**，
与该点的高度无关。抓取的平面标定（``modules.vision.drive.calibration``）依赖这一条：
不同高度的靶标和物料仍然落在同一个水平平面坐标系里。

相机一旦倾斜，这一条就不成立，水平坐标会带上与高度相关的偏置；标定前请先确认
镜头确实接近竖直向下。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

#: ``top_pic_2.jpg``（1280x720）在约 145 mm 实测距离下的一次粗标定：
#: 内侧圆面 209 px、实际直径 30 mm，145 * 209 / 30 ≈ 1012。
#: 这是**粗标定值**，正式使用前必须在真实安装高度重标（见 ``docs/抓取坐标链路.md``）。
DEFAULT_FOCAL_PX_AT_1280 = 1012.0

#: 默认焦距只对这一画幅比例成立；裁切或变焦后必须显式传入标定值。
DEFAULT_FRAME_ASPECT_RATIO = 16 / 9
ASPECT_RATIO_TOLERANCE = 0.02


@dataclass(frozen=True)
class CameraFix:
    """一个圆面中心在相机系里的位置，单位毫米。"""

    #: 画面右方为正。
    x_mm: float
    #: 画面下方为正。
    y_mm: float
    #: 沿光轴的深度；俯视时即镜头到该点的竖直距离。
    depth_mm: float

    @property
    def range_mm(self) -> float:
        """光心到该点的空间直线距离。"""
        return math.sqrt(self.x_mm ** 2 + self.y_mm ** 2 + self.depth_mm ** 2)

    def summary(self) -> str:
        return (
            f"xyz=({self.x_mm:.1f},{self.y_mm:.1f},{self.depth_mm:.1f}) mm，"
            f"直线距离约 {self.range_mm:.1f} mm"
        )


@dataclass(frozen=True)
class PinholeCamera:
    """一帧画幅上的针孔模型：像素焦距加画面尺寸。"""

    focal_px: float
    image_width: float
    image_height: float

    def __post_init__(self) -> None:
        if not self.focal_px > 0:
            raise ValueError("focal_px must be positive")
        if not self.image_width > 0 or not self.image_height > 0:
            raise ValueError("image size must be positive")

    @classmethod
    def from_shape(
        cls,
        shape: Sequence[int],
        *,
        focal_px: Optional[float] = None,
    ) -> "PinholeCamera":
        """用一帧图像的 ``shape`` 和（可选的）标定焦距建立模型。

        ``focal_px`` 省略时按 :func:`resolve_focal_px` 的规则从 1280 画幅换算，
        并且拒绝非 16:9 的裁切画幅。
        """
        height, width = int(shape[0]), int(shape[1])
        return cls(
            focal_px=resolve_focal_px(width, height, focal_px),
            image_width=float(width),
            image_height=float(height),
        )

    def depth_from_radius(self, radius_px: float, diameter_mm: float) -> float:
        """用已知真实直径的圆面像素半径反推镜头到该圆面的深度。"""
        if not radius_px > 0:
            raise ValueError("radius_px must be positive")
        if not diameter_mm > 0:
            raise ValueError("diameter_mm must be positive")
        return self.focal_px * diameter_mm / (2.0 * radius_px)

    def fix_from_circle(
        self,
        center_x_px: float,
        center_y_px: float,
        radius_px: float,
        diameter_mm: float,
    ) -> CameraFix:
        """把一个已知直径的圆面换算成相机系毫米坐标。"""
        depth_mm = self.depth_from_radius(radius_px, diameter_mm)
        return CameraFix(
            x_mm=(center_x_px - self.image_width / 2.0) * depth_mm / self.focal_px,
            y_mm=(center_y_px - self.image_height / 2.0) * depth_mm / self.focal_px,
            depth_mm=depth_mm,
        )

    def principal_point(self) -> Tuple[float, float]:
        """返回主点像素坐标（本模型固定取画面中心）。"""
        return (self.image_width / 2.0, self.image_height / 2.0)


def resolve_focal_px(width: int, height: int, focal_px: Optional[float] = None) -> float:
    """选定本画幅使用的像素焦距。

    显式传入时原样使用（调用方为该画幅标定过）。省略时只接受未裁切的 16:9
    画幅，并按宽度从 :data:`DEFAULT_FOCAL_PX_AT_1280` 等比换算；其他比例直接报错，
    而不是悄悄用一个错的焦距。
    """
    if focal_px is not None:
        if not focal_px > 0:
            raise ValueError("focal_px must be positive")
        return float(focal_px)
    if height <= 0 or abs(width / height - DEFAULT_FRAME_ASPECT_RATIO) > ASPECT_RATIO_TOLERANCE:
        raise ValueError("默认焦距只适用于无裁切 16:9 画幅；请传入该画幅标定后的 focal_px")
    return DEFAULT_FOCAL_PX_AT_1280 * width / 1280.0


def focal_px_from_known_target(
    *,
    distance_mm: float,
    pixel_diameter: float,
    real_diameter_mm: float,
) -> float:
    """用「已知真实直径的物体放在已知距离处」反推像素焦距。

    这是标定焦距的标准做法：``f = 距离 × 像素直径 / 真实直径``。
    多次测量取平均更稳。
    """
    if not distance_mm > 0:
        raise ValueError("distance_mm must be positive")
    if not pixel_diameter > 0:
        raise ValueError("pixel_diameter must be positive")
    if not real_diameter_mm > 0:
        raise ValueError("real_diameter_mm must be positive")
    return distance_mm * pixel_diameter / real_diameter_mm


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要摄像头：验证测距、投影和画幅规则。"""
    camera = PinholeCamera(focal_px=DEFAULT_FOCAL_PX_AT_1280, image_width=1280, image_height=720)

    # 文档里的实测样本：209 px 对应 30 mm，距离应回到约 145 mm。
    radius = 209 / 2.0
    depth = camera.depth_from_radius(radius, 30.0)
    assert abs(depth - 145.2) < 1.0, depth

    # 主点上的圆面：xy 必须为零。
    centred = camera.fix_from_circle(640, 360, radius, 30.0)
    assert abs(centred.x_mm) < 1e-9 and abs(centred.y_mm) < 1e-9, centred

    # 画面右移 100 px：x 应为 100 * depth / f。
    shifted = camera.fix_from_circle(740, 360, radius, 30.0)
    assert abs(shifted.x_mm - 100 * depth / camera.focal_px) < 1e-9, shifted

    # 默认焦距只在 16:9 下可用。
    assert abs(resolve_focal_px(1280, 720) - DEFAULT_FOCAL_PX_AT_1280) < 1e-9
    assert abs(resolve_focal_px(640, 360) - DEFAULT_FOCAL_PX_AT_1280 / 2) < 1e-9
    try:
        resolve_focal_px(640, 480)
    except ValueError:
        pass
    else:
        raise AssertionError("裁切画幅必须报错")

    # 焦距标定公式：145 mm 处 209 px 的 30 mm 圆面。
    assert abs(focal_px_from_known_target(distance_mm=145, pixel_diameter=209, real_diameter_mm=30) - 1010.2) < 0.5

    print("针孔模型自测通过：测距、投影、焦距标定与画幅规则都正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

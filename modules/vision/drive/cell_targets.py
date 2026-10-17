"""在同一帧里区分夹爪靶标与物料，并换算成相机系毫米。

抓取闭环必须**同时**知道两件事：物料在哪、机械臂在哪。俯视图里两者都是圆，所以这里用
圆心附近的**颜色均值**把它们分开——物料是蓝色的，靶标是另一种高对比度颜色。

分类规则刻意做得很窄，宁可少认也不认错：

- 物料的参考颜色和靶标的参考颜色**都**配好时，才启用颜色区分；否则所有圆面都当物料，
  靶标恒为 ``None``。抓取流程据此在打开摄像头之前拒绝启动，而不是拿一个猜的目标去算距离。
- 颜色都超出容差的圆面进 ``unmatched``，只留像素圆心，不编造三维坐标。

靶标有多个时 ``marker`` 取像素半径最大的那个（离镜头最近，最可能是正悬在台面上方的爪子），
完整的候选留在 ``markers`` 里供排查。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from modules.vision.config import TargetSettings
from modules.vision.drive.hough_lock import HoughRingLocator, LockedRegion
from modules.vision.drive.pinhole import CameraFix, PinholeCamera

#: 两类目标的角色名。
MATERIAL_ROLE = "material"
MARKER_ROLE = "marker"

#: 采样颜色时取半径的哪个比例，避开圆面外沿的深色边。
DEFAULT_COLOR_SAMPLE_FRACTION = 0.7


@dataclass(frozen=True)
class CellTarget:
    """一个已识别出角色的圆面目标。"""

    role: str
    region: LockedRegion
    #: 采样区域内的平均颜色（BGR）。
    mean_bgr: Tuple[float, float, float]
    #: 按该角色的已知直径换算出的相机系坐标（毫米）。
    fix: CameraFix

    @property
    def center(self) -> Tuple[float, float]:
        return self.region.center

    def summary(self) -> str:
        return f"{self.role} @({self.center[0]:.0f},{self.center[1]:.0f}) {self.fix.summary()}"


@dataclass(frozen=True)
class CellTargets:
    """一帧里识别出的全部目标。"""

    markers: Tuple[CellTarget, ...]
    materials: Tuple[CellTarget, ...]
    #: 颜色对不上任何参考值的圆面；只有像素圆心。
    unmatched: Tuple[LockedRegion, ...]

    @property
    def marker(self) -> Optional[CellTarget]:
        """像素半径最大的靶标候选；用来测机械臂当前位姿。"""
        if not self.markers:
            return None
        return max(self.markers, key=lambda target: target.region.radius)

    @property
    def material(self) -> Optional[CellTarget]:
        """像素半径最大的物料候选；用于单物料抓取。"""
        if not self.materials:
            return None
        return max(self.materials, key=lambda target: target.region.radius)

    @property
    def present(self) -> bool:
        return bool(self.materials)

    def summary(self) -> str:
        parts = [f"物料 {len(self.materials)} 个", f"靶标 {len(self.markers)} 个"]
        if self.unmatched:
            parts.append(f"无法归类 {len(self.unmatched)} 个")
        return "，".join(parts)


class CellTargetDetector:
    """霍夫圆面定位 + 颜色归类 + 针孔换算，一次给出两类目标。"""

    def __init__(
        self,
        *,
        camera: PinholeCamera,
        targets: TargetSettings,
        locator: Optional[HoughRingLocator] = None,
        color_sample_fraction: float = DEFAULT_COLOR_SAMPLE_FRACTION,
    ) -> None:
        if not 0.0 < color_sample_fraction <= 1.0:
            raise ValueError("color_sample_fraction must be in (0, 1]")
        self.camera = camera
        self.targets = targets
        self.locator = locator if locator is not None else HoughRingLocator()
        self.color_sample_fraction = float(color_sample_fraction)

    def detect(self, frame: np.ndarray) -> CellTargets:
        """在一帧 BGR 图像里找出所有目标并归类。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame is empty")
        markers = []
        materials = []
        unmatched = []
        for region in self.locator.locate_regions(frame):
            mean_bgr = self.sample_color(frame, region)
            role = self.classify(mean_bgr)
            if role is None:
                unmatched.append(region)
                continue
            diameter_mm = (
                self.targets.marker_diameter_mm
                if role == MARKER_ROLE
                else self.targets.material_diameter_mm
            )
            if diameter_mm is None:
                unmatched.append(region)
                continue
            target = CellTarget(
                role=role,
                region=region,
                mean_bgr=mean_bgr,
                fix=self.camera.fix_from_circle(
                    region.center_x, region.center_y, region.radius, diameter_mm
                ),
            )
            (markers if role == MARKER_ROLE else materials).append(target)
        return CellTargets(
            markers=tuple(markers),
            materials=tuple(materials),
            unmatched=tuple(unmatched),
        )

    def classify(self, mean_bgr: Sequence[float]) -> Optional[str]:
        """按颜色均值判断角色；分不开或都超容差时返回 ``None``。"""
        if not self.targets.can_split_by_color:
            return MATERIAL_ROLE
        to_material = _color_distance(mean_bgr, self.targets.material_bgr)
        to_marker = _color_distance(mean_bgr, self.targets.marker_bgr)
        if min(to_material, to_marker) > self.targets.color_tolerance:
            return None
        return MATERIAL_ROLE if to_material <= to_marker else MARKER_ROLE

    def sample_color(self, frame: np.ndarray, region: LockedRegion) -> Tuple[float, float, float]:
        """取圆心附近圆面内的平均颜色（BGR），避开外沿。"""
        import cv2

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        sample_radius = max(1, int(round(region.radius * self.color_sample_fraction)))
        cv2.circle(
            mask,
            (int(round(region.center_x)), int(round(region.center_y))),
            sample_radius,
            255,
            -1,
        )
        if not mask.any():
            # 圆面完全落在画面外：没有可采样的像素，如实给出黑色而不是拿整帧平均冒充。
            return (0.0, 0.0, 0.0)
        mean = cv2.mean(frame, mask=mask)
        return (float(mean[0]), float(mean[1]), float(mean[2]))

    def draw(self, frame: np.ndarray, targets: CellTargets) -> np.ndarray:
        """标出两类目标，颜色与角色一致。"""
        import cv2

        annotated = frame.copy()
        for role, color in ((MATERIAL_ROLE, (0, 0, 255)), (MARKER_ROLE, (0, 255, 0))):
            group = targets.materials if role == MATERIAL_ROLE else targets.markers
            for target in group:
                center = (
                    int(round(target.region.center_x)),
                    int(round(target.region.center_y)),
                )
                cv2.circle(annotated, center, int(round(target.region.radius)), color, 2)
                # cv2.putText 画不了中文；完整信息在 summary() 里。
                label = f"{target.role} {target.fix.depth_mm:.0f}mm"
                cv2.putText(
                    annotated,
                    label,
                    (max(center[0] - 40, 4), center[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        for region in targets.unmatched:
            cv2.circle(
                annotated,
                (int(round(region.center_x)), int(round(region.center_y))),
                int(round(region.radius)),
                (128, 128, 128),
                1,
            )
        return annotated


def _color_distance(
    sample: Sequence[float],
    reference: Optional[Sequence[float]],
) -> float:
    if reference is None:
        return float("inf")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(sample, reference)))


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要摄像头：合成一帧「蓝色物料 + 白色靶标」验证归类与测距。

    圆面画成「实心填充 + 深色外圈」：填充给出可采样的颜色，外圈给出霍夫需要的边缘。
    纯实心色块在默认霍夫阈值下检不出来，那不是本模块的问题，但不能拿它当样本，
    否则自测测的是夹具而不是代码。
    """
    import cv2

    focal_px = 1012.0
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    for center, radius, fill, rim in (
        ((500, 400), 105, (200, 120, 60), (120, 70, 30)),      # 物料：蓝色，半径 105 px
        ((900, 300), 60, (245, 245, 245), (150, 150, 150)),    # 靶标：白色，半径 60 px
    ):
        cv2.circle(frame, center, radius, fill, -1)
        cv2.circle(frame, center, radius, rim, 8)

    targets = TargetSettings(
        material_diameter_mm=30.0,
        marker_diameter_mm=20.0,
        material_bgr=(200.0, 120.0, 60.0),
        marker_bgr=(245.0, 245.0, 245.0),
        color_tolerance=60.0,
    )
    detector = CellTargetDetector(
        camera=PinholeCamera(focal_px=focal_px, image_width=1280, image_height=720),
        targets=targets,
    )

    found = detector.detect(frame)
    assert len(found.materials) == 1, found.summary()
    assert len(found.markers) == 1, found.summary()
    assert not found.unmatched, found.summary()
    assert found.material is not None and found.material.role == MATERIAL_ROLE
    assert found.marker is not None and found.marker.role == MARKER_ROLE

    # 换算必须与各自已知直径严格一致：这是本模块的接线，不是霍夫的精度。
    assert abs(
        found.material.fix.depth_mm - focal_px * 30 / (2 * found.material.region.radius)
    ) < 1e-9, found.material
    assert abs(
        found.marker.fix.depth_mm - focal_px * 20 / (2 * found.marker.region.radius)
    ) < 1e-9, found.marker

    # 霍夫应找到大致正确的圆（真值 105 / 60 px），并给出正确的左右、上下符号。
    assert abs(found.material.region.radius - 105) < 15, found.material.region
    assert abs(found.marker.region.radius - 60) < 15, found.marker.region
    assert found.material.fix.x_mm < 0 < found.marker.fix.x_mm, found.summary()
    assert found.marker.fix.y_mm < 0 < found.material.fix.y_mm, found.summary()

    # 没配颜色时不区分：所有圆面都当物料，靶标为 None。
    plain = CellTargetDetector(
        camera=detector.camera,
        targets=TargetSettings(material_diameter_mm=30.0),
    ).detect(frame)
    assert plain.marker is None and len(plain.materials) == 2, plain.summary()

    # 颜色都对不上时进 unmatched，且不编造三维坐标。
    stranger = CellTargetDetector(
        camera=detector.camera,
        targets=TargetSettings(
            material_diameter_mm=30.0,
            marker_diameter_mm=20.0,
            material_bgr=(0.0, 0.0, 0.0),
            marker_bgr=(0.0, 0.0, 0.0),
            color_tolerance=5.0,
        ),
    ).detect(frame)
    assert stranger.marker is None and not stranger.materials, stranger.summary()
    assert len(stranger.unmatched) == 2, stranger.summary()

    print(
        "同帧目标识别自测通过："
        f"{found.summary()}；物料 {found.material.fix.depth_mm:.1f} mm，"
        f"靶标 {found.marker.fix.depth_mm:.1f} mm。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

"""把「相机看到的一帧」翻译成「机械臂圆柱坐标里的位姿」。

这是视觉与机械臂之间**唯一**的接缝，方向严格单向：

```
像素 -> 相机系毫米（pinhole）
     -> 基座平面 (X, Y)（calibration 的平面标定）
     -> 离台面高度（kinematics.height_above_table）
     -> (θ, r, h)（kinematics.inverse_cylindrical）
```

所以 ``pipeline`` 只跟 :class:`CylindricalPose` 打交道，既不知道像素，也不需要导入 cv2。

两个特殊约定：

- **靶标给出的是夹持点所在的竖直轴。** 靶标装在爪上，相机测到它，就等于测到了机械臂——
  这正是"不回零、只靠相机"能成立的原因。但靶标高于实际夹持点，所以
  ``pipeline`` 会把测到的位姿按 ``h_grip = h_marker - marker_to_grasp_mm`` 换成**夹持点**
  再写进 ``arm``（计划 §5.4）；本模块只负责报出靶标自己的位姿。
- **物料位姿的 ``height_mm`` 是物料高度**（顶面离台面），不是夹持高度。夹持高度由
  ``pipeline`` 直接按 ``material_height_mm * grip_ratio`` 算出来（那个竖直偏移已经在
  写入夹持点时减过一次，生成目标时不能再加一遍）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from modules.arm.drive.kinematics import CylindricalPose, height_above_table, inverse_cylindrical
from modules.vision.config import VisionConfig
from modules.vision.drive.calibration import PlaneCalibration
from modules.vision.drive.cell_targets import CellTargetDetector, CellTargets
from modules.vision.drive.hough_lock import HoughRingLocator
from modules.vision.drive.pinhole import CameraFix, PinholeCamera


class TargetDetector(Protocol):
    """``CellTargetDetector`` 的最小接口；测试可以注入假件。"""

    def detect(self, frame) -> CellTargets: ...


@dataclass(frozen=True)
class CellObservation:
    """一帧里机械臂与物料在圆柱坐标里的位姿。"""

    #: 机械臂当前位姿（由爪上靶标测出）；看不到靶标时为 ``None``。
    marker: Optional[CylindricalPose]
    #: 物料顶面的位姿；``height_mm`` 是物料高度（顶面离台面）。
    material: Optional[CylindricalPose]
    marker_fix: Optional[CameraFix] = None
    material_fix: Optional[CameraFix] = None

    def summary(self) -> str:
        marker = self.marker.summary() if self.marker is not None else "未看到靶标"
        material = self.material.summary() if self.material is not None else "未看到物料"
        return f"机械臂 {marker}；物料 {material}"


class CellPerception:
    """一帧 → 两个圆柱位姿。"""

    def __init__(
        self,
        *,
        detector: TargetDetector,
        plane: PlaneCalibration,
        table_depth_mm: float,
    ) -> None:
        if not table_depth_mm > 0:
            raise ValueError("table_depth_mm must be positive")
        self.detector = detector
        self.plane = plane
        self.table_depth_mm = float(table_depth_mm)

    @classmethod
    def from_vision_config(
        cls,
        config: VisionConfig,
        frame_shape: Sequence[int],
        *,
        locator: Optional[HoughRingLocator] = None,
    ) -> "CellPerception":
        """用已标定的视觉配置和实际帧尺寸组装。

        ``config.require_plane()`` 与 ``require_table_depth()`` 会在这里就报错，
        而不是等到抓取抓了一半才发现没标定。
        """
        return cls(
            detector=CellTargetDetector(
                camera=PinholeCamera.from_shape(frame_shape, focal_px=config.focal_px),
                targets=config.targets,
                locator=locator,
            ),
            plane=config.require_plane(),
            table_depth_mm=config.require_table_depth(),
        )

    def observe(self, frame) -> CellObservation:
        """检测一帧并换算成两个圆柱位姿。"""
        found = self.detector.detect(frame)
        marker = found.marker
        material = found.material
        return CellObservation(
            marker=None if marker is None else self.to_pose(marker.fix),
            material=None if material is None else self.to_pose(material.fix),
            marker_fix=None if marker is None else marker.fix,
            material_fix=None if material is None else material.fix,
        )

    def to_pose(self, fix: CameraFix) -> CylindricalPose:
        """相机系毫米 → 圆柱位姿。"""
        plane_x, plane_y = self.plane.to_base_plane(fix.x_mm, fix.y_mm)
        return inverse_cylindrical(
            plane_x,
            plane_y,
            height_above_table(
                table_depth_mm=self.table_depth_mm,
                point_depth_mm=fix.depth_mm,
            ),
        )


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要摄像头：用假检测器验证「像素→相机系→基座平面→圆柱」整条链。"""
    from modules.vision.drive.cell_targets import MATERIAL_ROLE, MARKER_ROLE, CellTarget
    from modules.vision.drive.hough_lock import LockedRegion

    def target(role: str, fix: CameraFix) -> CellTarget:
        return CellTarget(
            role=role,
            region=LockedRegion(0.0, 0.0, 10.0),
            mean_bgr=(0.0, 0.0, 0.0),
            fix=fix,
        )

    class FakeDetector:
        def __init__(self, found: CellTargets) -> None:
            self.found = found

        def detect(self, _frame) -> CellTargets:
            return self.found

    # 标定：基座轴在相机水平面的原点，θ 零点为 0。
    plane = PlaneCalibration(center_x_mm=0.0, center_y_mm=0.0, rotation_degrees=0.0)
    table_depth_mm = 300.0
    found = CellTargets(
        markers=(target(MARKER_ROLE, CameraFix(x_mm=0.0, y_mm=40.0, depth_mm=290.0)),),
        materials=(target(MATERIAL_ROLE, CameraFix(x_mm=50.0, y_mm=0.0, depth_mm=285.0)),),
        unmatched=(),
    )
    perception = CellPerception(
        detector=FakeDetector(found), plane=plane, table_depth_mm=table_depth_mm
    )

    observation = perception.observe(frame=None)
    assert observation.marker is not None and observation.material is not None

    # 物料在 +X 方向 50 mm、顶面高 15 mm。
    assert abs(observation.material.base_degrees) < 1e-9, observation.material
    assert abs(observation.material.radial_mm - 50.0) < 1e-9, observation.material
    assert abs(observation.material.height_mm - 15.0) < 1e-9, observation.material

    # 靶标在 +Y 方向 40 mm、高 10 mm。
    assert abs(observation.marker.base_degrees - 90.0) < 1e-9, observation.marker
    assert abs(observation.marker.radial_mm - 40.0) < 1e-9, observation.marker
    assert abs(observation.marker.height_mm - 10.0) < 1e-9, observation.marker

    # 平面标定的转角必须真的作用到 θ 上。
    rotated = CellPerception(
        detector=FakeDetector(found),
        plane=PlaneCalibration(0.0, 0.0, rotation_degrees=10.0),
        table_depth_mm=table_depth_mm,
    ).observe(frame=None)
    assert abs(rotated.material.base_degrees - 10.0) < 1e-9, rotated.material

    print("感知适配自测通过：像素系→基座平面→圆柱位姿整条链与手算一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

"""镜头端直接给出毫米坐标时，镜头系 → 基座平面的换算（纯数学）。

## 与 :mod:`modules.vision.drive.calibration` 的关系

那份标定（``PlaneCalibration``）解决的是同一件事，对象是**相机水平面**：

```
X_arm = R(rotation_degrees) · (p_cam − center)
```

区别只在输入从哪来：

| 输入来源 | 谁把像素变成毫米 |
|---|---|
| 本仓库的摄像头（只拿到像素） | :mod:`modules.vision.drive.pinhole`，靠 ``focal_px`` |
| **镜头端直接报 xyz** | 对面已经做完了，本仓库不重复换算 |

所以这里**复用** :class:`PlaneCalibration` 的数学，只补两件它不知道的事：

1. 输入是 :class:`~modules.vision.drive.sighting.MaterialSighting`（带物料编号、颜色、置信度）；
2. ``z`` 要能反过来对账 ``table_depth_mm`` 标定得对不对——**这是唯一能发现"台面深度填错
   了"的地方**，因为平面换算只用 ``x/y``，单个点的 ``x/y`` 看不出深度错没错。

## 为什么 xy 与 z 必须分开看

``xyz`` 里的 ``x/y`` 是镜头光轴垂足到物料的**水平**距离，与物料多高无关（光轴竖直时）。
``z`` 是沿光轴的深度，俯视时向下为正。所以：

- 平面换算（这一层）**只用 x/y**；
- 高度换算要用 ``table_depth_mm − z``，那是抓取流程第三步的事，不在 ``x/y`` 对准里。

把两者混在一起写，就会出现"物块越高、机械臂越往外伸"这种谁都看不出来的系统性错误。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from modules.vision.drive.calibration import PlaneCalibration
from modules.vision.drive.sighting import MaterialSighting

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "sighting_plane.json"

#: ``z`` 与 ``table_depth_mm`` 差多少毫米以上就提示去核对台面深度标定。
DEFAULT_DEPTH_TOLERANCE_MM = 40.0


@dataclass(frozen=True)
class SightingPlane:
    """镜头系的毫米点 → 机械臂圆柱坐标，外加台面深度对账。"""

    plane: PlaneCalibration
    #: 镜头到台面的深度（毫米）；``None`` 表示还没标定，跳过对账。
    table_depth_mm: Optional[float] = None
    #: ``z`` 与 ``table_depth_mm`` 相差多少毫米以上就提示。
    depth_tolerance_mm: float = DEFAULT_DEPTH_TOLERANCE_MM

    def polar(self, sighting: MaterialSighting) -> Tuple[float, float]:
        """一条观测 → ``(base_degrees, radial_mm)``。

        ``base_degrees`` 落在 ``(-180, 180]``，与 ``atan2`` 同域，
        可以直接和 :func:`modules.arm.drive.kinematics.shortest_angle_delta_degrees` 配合。
        """
        return self.plane.to_polar(sighting.x_mm, sighting.y_mm)

    def polar_for_xy(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """任取一个镜头系毫米点 → ``(base_degrees, radial_mm)``。

        也用来换算**爪上靶标**的平面位置：靶标同样是镜头系里的一个点。
        """
        return self.plane.to_polar(x_mm, y_mm)

    def depth_problem(self, sighting: MaterialSighting) -> Optional[str]:
        """``z`` 与台面深度标定明显对不上时返回一句说明，否则返回 ``None``。

        物料浮在台面上方时 ``z`` 本来就该小于 ``table_depth_mm``，差值就是物料高度；
        所以这里**只在"物料跑到台面下面去了"（``z`` 明显大于台面深度）时报警**——
        那说明 ``table_depth_mm`` 填错了，或者镜头端给的 ``z`` 不是"沿光轴的深度"。
        """
        if self.table_depth_mm is None:
            return None
        if sighting.z_mm > self.table_depth_mm + self.depth_tolerance_mm:
            return (
                f"物料 {sighting.material_id} 的 z={sighting.z_mm:.1f} mm 比台面深度 "
                f"{self.table_depth_mm:.1f} mm 还大 {sighting.z_mm - self.table_depth_mm:.1f} mm；"
                "物料不可能在台面下面，请核对 table_depth_mm 标定，"
                "或者确认镜头端的 z 是「沿光轴向下的深度」而不是别的量"
            )
        return None

    def summary(self) -> str:
        depth = (
            "未标定"
            if self.table_depth_mm is None
            else f"{self.table_depth_mm:.1f} mm"
        )
        return f"{self.plane.summary()}；镜头到台面深度={depth}"


def load_sighting_plane(config_path: Path = DEFAULT_CONFIG_PATH) -> SightingPlane:
    """读取 ``modules/vision/sighting_plane.json``。

    :raises ValueError: 平面标定的三项没填全。**这一层刻意不给默认值**——
        编一个"基座就在镜头正下方"的假设会让机械臂朝着错误的方向转过去，
        而那个错误看起来完全像一个正常的动作。
    """
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取镜头系标定：{config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"镜头系标定不是合法 JSON：{config_path}") from error
    if not isinstance(document, dict):
        raise ValueError("镜头系标定必须是 JSON 对象")

    plane = _plane(document.get("plane"))
    table_depth = document.get("table_depth_mm")
    if table_depth is not None:
        table_depth = _number(table_depth, "table_depth_mm")
        if not table_depth > 0:
            raise ValueError("table_depth_mm 必须是正数或 null")
    tolerance = _number(
        document.get("depth_tolerance_mm", DEFAULT_DEPTH_TOLERANCE_MM),
        "depth_tolerance_mm",
    )
    if not tolerance > 0:
        raise ValueError("depth_tolerance_mm 必须是正数")
    return SightingPlane(
        plane=plane, table_depth_mm=table_depth, depth_tolerance_mm=tolerance
    )


def _plane(value: object) -> PlaneCalibration:
    if value is None:
        raise ValueError(
            "plane 还没标定：请在 modules/vision/calibration.json 里填好"
            "基座轴在镜头水平面上的投影中心（center_x_mm / center_y_mm）"
            "与 θ 零点偏置（rotation_degrees）"
        )
    if not isinstance(value, dict):
        raise ValueError("plane 必须是 JSON 对象")

    def required(name: str) -> float:
        item = value.get(name)
        if item is None:
            raise ValueError(f"plane.{name} 还没填；三项必须一起给出")
        return _number(item, f"plane.{name}")

    return PlaneCalibration(
        center_x_mm=required("center_x_mm"),
        center_y_mm=required("center_y_mm"),
        rotation_degrees=required("rotation_degrees"),
        rms_mm=_number(value.get("rms_mm", 0.0), "plane.rms_mm"),
        samples=_integer(value.get("samples", 0), "plane.samples"),
    )


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    return float(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return value


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要镜头：用合成标定验证换算与深度对账。"""
    from modules.vision.drive.sighting import parse_sighting

    plane = SightingPlane(
        plane=PlaneCalibration(center_x_mm=100.0, center_y_mm=-50.0, rotation_degrees=90.0),
        table_depth_mm=300.0,
    )

    # 镜头系 (100, -50) 就是基座旋转轴，径向必须是 0。
    assert plane.polar_for_xy(100.0, -50.0) == (0.0, 0.0), plane.polar_for_xy(100.0, -50.0)

    # 旋转 90° 之后：镜头 +x 方向的点落到基座 +y 方向（θ=90°），半径 20。
    base_degrees, radial = plane.polar_for_xy(120.0, -50.0)
    assert abs(base_degrees - 90.0) < 1e-9, base_degrees
    assert abs(radial - 20.0) < 1e-9, radial

    sighting = parse_sighting(
        "物料 4: 绿色物料（green），xyz=(110.0,-50.0,285.0) mm"
    )
    base_degrees, radial = plane.polar(sighting)
    assert abs(base_degrees - 90.0) < 1e-9 and abs(radial - 10.0) < 1e-9, (base_degrees, radial)
    # 物料在台面上方 300−285 = 15 mm，不算深度问题。
    assert plane.depth_problem(sighting) is None, plane.depth_problem(sighting)

    # 物料"跑到台面下面"必须报警：那是 table_depth_mm 或 z 语义错了。
    below = parse_sighting("物料 5: 蓝色物料（blue），xyz=(110.0,-50.0,400.0) mm")
    problem = plane.depth_problem(below)
    assert problem is not None and "台面" in problem, problem

    # 台面深度没标定时跳过对账，而不是拿一个默认值硬比。
    uncalibrated = SightingPlane(plane=plane.plane, table_depth_mm=None)
    assert uncalibrated.depth_problem(below) is None
    assert "未标定" in uncalibrated.summary(), uncalibrated.summary()

    print(f"镜头系平面换算自测通过：{plane.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

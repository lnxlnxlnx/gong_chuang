"""x/y 对准的闭环编排：让爪子正下方落到物块上（**不动高度**）。

这是用户要的抓取第一步。整条链路的每一段都已经单独存在，这一层只负责把它们接起来
并处理真机上必然出现的两件事：**动作有误差**、**镜头给的是上一时刻的位置**。

```
modules/camera_link/drive/link.py        收 TCP 行 -> MaterialSighting
        |
modules/vision/drive/sighting_plane.py   镜头系 mm -> (θ, r)
        |
modules/grasp/drive/align.py             (θ, r) 差 -> (Δθ, Δr)      <- 精确解
        |
modules/grasp/drive/run_xy.py            闭环：走一步 -> 重新测 -> 直到进容差  <- 这一层
        |
modules/arm/drive/robot_arm.py           发帧
```

## 为什么明明有精确解还要闭环

:func:`modules.grasp.drive.align.solve_planar_alignment` 在平面里给出的是**精确解**，
只要输入准确，一步就能到位。真机上输入不准确，来源有三个，而且都不是解算器能修的：

1. **命令量 ≠ 到位量**（背隙、丢步、皮带弹性）；
2. **平面标定有残差**（``plane.rms_mm``）；
3. **镜头报的 marker 与物料不是同一时刻的位置**——机械臂在动的时候读数，
   两者必然错开。

所以这一层每次都**重新测**，而不是相信"我发了 10 mm 就一定走了 10 mm"。
和 ``pipeline._visual_align`` 是同一个理由，只是那边的反馈来自另一路相机。

## 安全：要转大角度时先缩回

伸出状态下转底座会把整条臂扫过工作区。这是**结构上的**风险，与高度无关，
所以哪怕只做 x/y 对准也必须走这一步缩回。阈值和半径来自
``modules/grasp/config.json`` 的 ``correction.rotation_corridor_degrees`` 与
``retract_radius_mm``；后者为 ``null`` 时**明确不缩回**（而不是偷偷用一个默认半径）——
真机上"缩回到哪"必须由标定过的工作区决定。

## 谁负责合规

这一层只动平面（θ, r）。**高度一律不碰**：z 是下一步的事，这里改了高度会让
"先对准 x/y 再对 z"这个顺序失去意义。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from modules.arm.config import ArmConfig, load_arm_config
from modules.arm.drive.kinematics import (
    CylindricalPose,
    check_reachable,
    shortest_angle_delta_degrees,
)
from modules.camera_link.drive.link import CameraLink
from modules.grasp.config import GraspConfig, load_grasp_config
from modules.grasp.drive.align import (
    AlignmentError,
    PlanarAlignment,
    is_aligned,
    planar_error_mm,
    pose_after,
    solve_planar_alignment,
)
from modules.vision.drive.sighting import MaterialSighting
from modules.vision.drive.sighting_plane import SightingPlane

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "xy_align.json"

#: 角度小于这个值就当作"没在转"，不必走缩回廊道。
ANGLE_EPSILON_DEGREES = 0.05

#: 径向差小于这个值就当作"这一维不动"。
LENGTH_EPSILON_MM = 1e-6


class XYAlignError(RuntimeError):
    """x/y 对准无法继续（看不到爪子、看不到物料、够不着、不收敛）。"""


@dataclass(frozen=True)
class GripperMarker:
    """怎么从一堆观测里认出"哪条是爪上靶标"。

    镜头端把靶标也当成一个物料报出来（同样的文本格式），所以这里只需要一条判别规则：

    - ``material_id``：优先按编号认（最可靠，编号是稳定的）；
    - ``color``：其次按颜色认（靶标通常是显眼的单色圆片）。

    两个都是 ``None`` 时**拒绝工作**：那时没有任何依据把靶标和物料分开，
    程序会把物料当成爪子，算出一个"已经对准"的结论然后一动不动——
    这种静默错误比报错危险得多。
    """

    material_id: Optional[int] = None
    color: Optional[str] = None

    def __post_init__(self) -> None:
        if self.material_id is None and self.color is None:
            raise XYAlignError(
                "还没配置怎么认出爪上靶标：请在 modules/grasp/xy_align.json 的 "
                "gripper_marker 里填 material_id 或 color"
            )
        if self.color is not None:
            object.__setattr__(self, "color", self.color.strip().lower())

    def matches(self, sighting: MaterialSighting) -> bool:
        """这条观测是不是爪上靶标。两个条件都给时，两个都要满足。"""
        if self.material_id is not None and sighting.material_id != self.material_id:
            return False
        if self.color is not None and sighting.color != self.color:
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.material_id is not None:
            parts.append(f"编号 {self.material_id}")
        if self.color is not None:
            parts.append(f"颜色 {self.color}")
        return "爪上靶标 = " + "、".join(parts)


@dataclass(frozen=True)
class XYAlignConfig:
    """闭环对准的可调参数。"""

    #: 平面误差小于这个值就算对准（毫米）。
    tolerance_mm: float = 3.0
    #: 最多重新测量几次。每次重新测都要等一帧，所以次数直接决定耗时。
    iterations: int = 4
    #: 手头这帧数据比现在旧多少毫秒就不再信它，宁可等下一帧。
    max_frame_age_ms: float = 500.0


def load_xy_align_config(config_path: Path = DEFAULT_CONFIG_PATH) -> XYAlignConfig:
    """读取 ``modules/grasp/xy_align.json``。"""
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise XYAlignError(f"无法读取 x/y 对准配置：{config_path}") from error
    except json.JSONDecodeError as error:
        raise XYAlignError(f"x/y 对准配置不是合法 JSON：{config_path}") from error
    if not isinstance(document, dict):
        raise XYAlignError("x/y 对准配置必须是 JSON 对象")

    tolerance = document.get("tolerance_mm", XYAlignConfig.tolerance_mm)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise XYAlignError("tolerance_mm 必须是正数")
    iterations = document.get("iterations", XYAlignConfig.iterations)
    if type(iterations) is not int or iterations < 1:
        raise XYAlignError("iterations 必须是正整数")
    age = document.get("max_frame_age_ms", XYAlignConfig.max_frame_age_ms)
    if isinstance(age, bool) or not isinstance(age, (int, float)) or age <= 0:
        raise XYAlignError("max_frame_age_ms 必须是正数")
    return XYAlignConfig(
        tolerance_mm=float(tolerance), iterations=iterations, max_frame_age_ms=float(age)
    )


def load_gripper_marker(config_path: Path = DEFAULT_CONFIG_PATH) -> GripperMarker:
    """读取 ``modules/grasp/xy_align.json`` 里的 ``gripper_marker``。"""
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise XYAlignError(f"无法读取 x/y 对准配置：{config_path}") from error
    except json.JSONDecodeError as error:
        raise XYAlignError(f"x/y 对准配置不是合法 JSON：{config_path}") from error
    if not isinstance(document, dict):
        raise XYAlignError("x/y 对准配置必须是 JSON 对象")
    marker = document.get("gripper_marker")
    if marker is None:
        marker = {}
    if not isinstance(marker, dict):
        raise XYAlignError("gripper_marker 必须是 JSON 对象")

    material_id = marker.get("material_id")
    if material_id is not None and (type(material_id) is not int or material_id < 0):
        raise XYAlignError("gripper_marker.material_id 必须是非负整数或 null")
    color = marker.get("color")
    if color is not None and (not isinstance(color, str) or not color.strip()):
        raise XYAlignError("gripper_marker.color 必须是非空字符串或 null")
    return GripperMarker(material_id=material_id, color=color)


@dataclass(frozen=True)
class XYResult:
    """一次 x/y 对准的结果。"""

    #: 这次对准瞄准的物料（对准前测到的位置）。
    material: CylindricalPose
    #: 每个移动步执行完之后推算的爪子位姿（最后一个是最终位姿）。
    waypoints: Tuple[CylindricalPose, ...]
    #: 最终爪子位姿。
    pose: CylindricalPose
    #: 最终平面误差（毫米）。
    error_mm: float
    #: 判据阈值；放在这里是为了让调用方不用再传一次配置就能判断。
    tolerance_mm: float
    #: 用掉了几次重新测量（1 表示第一次测完走一步就进容差了）。
    measurements: int

    @property
    def aligned(self) -> bool:
        return self.error_mm <= self.tolerance_mm

    def summary(self) -> str:
        return (
            f"x/y 对准{'完成' if self.aligned else '未达标'}："
            f"物料 {self.material.summary()}；最终 {self.pose.summary()}；"
            f"平面误差 {self.error_mm:.2f} mm（容差 {self.tolerance_mm:.1f} mm）；"
            f"移动 {len(self.waypoints)} 步，重新测量 {self.measurements} 次"
        )


class XYAligner:
    """把镜头观测接成"测 → 算 → 走 → 再测"的闭环，直到爪子对准物料上方。"""

    def __init__(
        self,
        *,
        arm,
        link: CameraLink,
        plane: SightingPlane,
        marker: GripperMarker,
        config: Optional[XYAlignConfig] = None,
        grasp_config: Optional[GraspConfig] = None,
        arm_config: Optional[ArmConfig] = None,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.arm = arm
        self.link = link
        self.plane = plane
        self.marker = marker
        self.config = config if config is not None else XYAlignConfig()
        self.grasp_config = (
            grasp_config if grasp_config is not None else load_grasp_config()
        )
        self.arm_config = arm_config if arm_config is not None else load_arm_config()
        self._say = on_message if on_message is not None else (lambda _message: None)

    # ------------------------------------------------------------- 观测
    def identify(
        self, sightings: Sequence[MaterialSighting]
    ) -> Tuple[MaterialSighting, MaterialSighting]:
        """把一帧观测分成 ``(靶标, 物料)``。

        :raises XYAlignError: 少了任何一方。**刻意不猜**——猜错的一方会让机械臂
            朝着错误的位置动，而动作本身看起来完全正常。
        """
        markers, materials = [], []
        for sighting in sightings:
            (markers if self.marker.matches(sighting) else materials).append(sighting)

        if not markers:
            raise XYAlignError(
                f"这一帧里没有爪上靶标（{self.marker.describe()}）；"
                f"收到了 {len(sightings)} 条观测："
                f"{'、'.join(f'物料 {item.material_id}({item.color})' for item in sightings) or '（空）'}"
            )
        if not materials:
            raise XYAlignError(
                "这一帧里只有爪上靶标，没有物料；"
                "把物块放到镜头视野里，或检查 gripper_marker 的判别条件是不是把物料也认成了靶标"
            )
        if len(materials) > 1:
            summary = "、".join(f"物料 {item.material_id}({item.color})" for item in materials)
            raise XYAlignError(
                f"这一帧里有多个物料（{summary}），不知道该对准哪一个；"
                "请只放一个物块在视野里"
            )
        return markers[-1], materials[0]

    def observe(self) -> Tuple[MaterialSighting, MaterialSighting]:
        """等一帧足够新的观测，返回 ``(靶标, 物料)``。

        "足够新"是硬要求：动作之后手上那帧就过期了，拿它再解一次会算出零位移，
        看起来像"已经对准"，实际是拿旧数据骗自己。所以这里**只接受新帧**。
        """
        frame = self.link.frame(
            timeout=self.config.max_frame_age_ms / 1000.0,
            max_age_seconds=self.config.max_frame_age_ms / 1000.0,
        )
        if frame is None:
            error = getattr(self.link, "error", None)
            raise XYAlignError(
                "等不到新的观测"
                + (f"；接收链路报错：{error}" if error else "；确认镜头端已连上并正在发数据")
            )
        return self.identify(frame)

    @staticmethod
    def _age_ms(sighting: MaterialSighting, now: float) -> Optional[float]:
        """这条观测到现在的年龄（毫秒）；没有时间戳时返回 ``None``。"""
        if sighting.received_at is None:
            return None
        return (now - sighting.received_at) * 1000.0

    # ------------------------------------------------------------- 主流程
    def align(self) -> XYResult:
        """对准一次，返回最终位姿与误差。不收敛或够不着都抛 :class:`XYAlignError`。"""
        path: List[CylindricalPose] = []
        error = float("inf")
        for attempt in range(1, self.config.iterations + 1):
            marker, material = self.observe()
            self._check_frame_pairing(marker, material)
            problem = self._depth_problem(material)
            if problem is not None:
                self._say(f"警告：{problem}")

            current = self._pose_of(marker, "爪上靶标")
            target = self._pose_of(material, f"物料 {material.material_id}")
            error = planar_error_mm(current, target)
            self._say(
                f"第 {attempt} 次测量：爪子 {current.summary()}；"
                f"物料 {target.summary()}；平面误差 {error:.2f} mm"
            )
            if is_aligned(current, target, tolerance_mm=self.config.tolerance_mm):
                return XYResult(
                    material=target,
                    waypoints=tuple(path),
                    pose=current,
                    error_mm=error,
                    measurements=attempt,
                    tolerance_mm=self.config.tolerance_mm,
                )

            try:
                alignment = solve_planar_alignment(current, target)
            except AlignmentError as problem:
                raise XYAlignError(str(problem)) from problem

            waypoints = self._waypoints(current, target, alignment)
            reached = current
            for waypoint in waypoints:
                self._execute(reached, waypoint)
                reached = waypoint
                path.append(waypoint)
            self._say(
                f"  走了 {len(waypoints)} 步 -> {reached.summary()}；"
                f"（相对测量起点 {alignment.summary()}）"
            )

        # 用尽迭代仍未进容差：报错而不是继续。此时多半是标定或背隙出了问题，
        # 继续走只会让机械臂越来越偏。
        raise XYAlignError(
            f"x/y 对准 {self.config.iterations} 次仍未进入容差 "
            f"{self.config.tolerance_mm:.1f} mm；最后一次平面误差 {error:.2f} mm。"
            "先看平面标定残差 plane.rms_mm 与实验 G 的重复精度，"
            "再考虑调大 iterations 或 tolerance_mm"
        )

    # ------------------------------------------------------------- 内部
    def _check_frame_pairing(
        self, marker: MaterialSighting, material: MaterialSighting
    ) -> None:
        """同一帧里的两条观测必须来自同一时刻。

        镜头端如果先报一条、过一会儿再报另一条，机械臂在动的时候两者会错开——
        精确解就变成了"照着一个已经不成立的位置去动"。这里只做提醒，不做拦截：
        真机上本来就是两条独立的消息，拦了就没法干活了。
        """
        now = time.monotonic()
        marker_age = self._age_ms(marker, now)
        material_age = self._age_ms(material, now)
        if marker_age is None or material_age is None:
            return
        gap = abs(marker_age - material_age)
        if gap > self.config.max_frame_age_ms:
            self._say(
                f"警告：靶标与物料的观测时间差 {gap:.0f} ms，"
                "机械臂在动的时候这会让解算结果偏掉；闭环会靠下一次测量修回来"
            )

    def _depth_problem(self, material: MaterialSighting) -> Optional[str]:
        """台面深度对账的提醒；这一层只是转发，判断在 ``SightingPlane`` 里。"""
        checker = getattr(self.plane, "depth_problem", None)
        return None if checker is None else checker(material)

    def _pose_of(self, sighting: MaterialSighting, what: str) -> CylindricalPose:
        """一条观测 → 圆柱位姿；``height_mm`` 在这里没有意义，统一填 0。

        高度是下一步（z 对准）的事，这里只借 :class:`CylindricalPose` 装平面坐标。
        填 0 而不是填 ``z``，是为了让"这一步没动高度"在日志里一眼看得出来。
        """
        base_degrees, radial_mm = self.plane.polar(sighting)
        pose = CylindricalPose(base_degrees, radial_mm, 0.0)
        problems = check_reachable(
            pose,
            base_limit=self.arm_config.base_limit,
            radial_limit=self.arm_config.radial_limit,
        )
        if problems:
            raise XYAlignError(
                f"{what} 超出平面行程：{'；'.join(problems)}；"
                "先把物块放近一点，或把 limits 按实验 F 标定好"
            )
        return pose

    def _waypoints(
        self,
        current: CylindricalPose,
        target: CylindricalPose,
        alignment: PlanarAlignment,
    ) -> List[CylindricalPose]:
        """把"转 Δθ + 伸 Δr"展开成实际要走的中间位姿。

        要转的角度超过阈值、而且现在还伸在外面时，先缩到 ``retract_radius_mm``
        再转，转完再伸出去——伸出状态下横扫底座会把整条臂扫过工作区。
        """
        final = pose_after(current, alignment.delta_base_degrees, alignment.delta_radial_mm)
        corridor = self._corridor_radius()
        needs_rotation = abs(alignment.delta_base_degrees) > ANGLE_EPSILON_DEGREES
        if (
            corridor is None
            or not needs_rotation
            or abs(alignment.delta_base_degrees)
            <= self.grasp_config.correction.rotation_corridor_degrees
        ):
            # 小角度就地转是安全的（爪子本来就悬在物料上方），不必缩回一次。
            return self._planar_steps(current, final)

        retracted = CylindricalPose(current.base_degrees, corridor, current.height_mm)
        # 复用解算器给的最短转角，而不是直接跳向 final：final 的绝对角度可能反过来
        # 让这次旋转绕远路（例如 −150° 到 153° 本该走 −57°，直接跳却是 +303°）。
        turned = pose_after(
            retracted, alignment.delta_base_degrees, corridor - retracted.radial_mm
        )
        steps = self._planar_steps(current, retracted)
        steps += self._planar_steps(retracted, turned)
        steps += self._planar_steps(turned, final)
        return steps

    def _corridor_radius(self) -> Optional[float]:
        radius = self.grasp_config.retract_radius_mm
        if radius is None:
            return None
        problems = check_reachable(
            CylindricalPose(0.0, radius, 0.0), radial_limit=self.arm_config.radial_limit
        )
        if problems:
            raise XYAlignError(
                f"配置的缩回半径 {radius:.1f} mm 不在径向行程内：{'；'.join(problems)}"
            )
        return radius

    def _planar_steps(
        self, start: CylindricalPose, end: CylindricalPose
    ) -> List[CylindricalPose]:
        """两个位姿之间的实际动作序列，顺序固定为 **伸缩 → 旋转**。

        先缩回再转，和 ``pipeline._advance`` 是同一条规矩。
        """
        steps: List[CylindricalPose] = []
        if abs(end.radial_mm - start.radial_mm) > LENGTH_EPSILON_MM:
            steps.append(CylindricalPose(start.base_degrees, end.radial_mm, start.height_mm))
        if (
            abs(shortest_angle_delta_degrees(start.base_degrees, end.base_degrees))
            > ANGLE_EPSILON_DEGREES
        ):
            steps.append(end)
        return steps

    def _execute(self, start: CylindricalPose, end: CylindricalPose) -> None:
        """走一步（只含一个关节量），并在发帧前再查一次行程。"""
        delta_radial = end.radial_mm - start.radial_mm
        delta_base = shortest_angle_delta_degrees(start.base_degrees, end.base_degrees)
        problems = check_reachable(
            end,
            base_limit=self.arm_config.base_limit,
            radial_limit=self.arm_config.radial_limit,
        )
        if problems:
            raise XYAlignError(f"目标位姿越界：{'；'.join(problems)}")

        rpm = self.grasp_config.motion_rpm
        kwargs = {} if rpm is None else {"rpm": rpm}
        if abs(delta_radial) > LENGTH_EPSILON_MM:
            self.arm.move_radial_mm(delta_radial, wait=True, **kwargs)
        if abs(delta_base) > ANGLE_EPSILON_DEGREES:
            self.arm.move_base_degrees(delta_base, wait=True, **kwargs)


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要镜头、不需要串口：用假链路 + 假臂跑通"测 → 算 → 走 → 再测"。"""
    from modules.arm.config import AxisConfig
    from modules.arm.drive.kinematics import TravelLimit
    from modules.grasp.config import CorrectionSettings, GraspConfig
    from modules.vision.drive.calibration import PlaneCalibration
    from modules.vision.drive.sighting import parse_sighting

    # 镜头系里基座轴心在 (100, -50)，θ 零点偏 90°；于是镜头 +x 方向就是基座的 θ=90°。
    plane = SightingPlane(
        plane=PlaneCalibration(center_x_mm=100.0, center_y_mm=-50.0, rotation_degrees=90.0),
        table_depth_mm=300.0,
    )

    class FakeArm:
        """真值位姿 + 可选背隙；第一次径向移动少走，逼闭环去修。"""

        def __init__(self, *, base: float, radial: float, backlash: float = 0.0) -> None:
            self.base, self.radial, self.height = base, radial, 60.0
            self._backlash = backlash
            self.calls: List[Tuple[str, float]] = []

        def move_base_degrees(self, degrees, **_kwargs):
            self.calls.append(("base", round(degrees, 4)))
            self.base += degrees

        def move_radial_mm(self, millimeters, **_kwargs):
            lost = min(abs(millimeters), self._backlash) if self._backlash > 0 else 0.0
            self._backlash -= lost
            self.calls.append(("radial", round(millimeters, 4)))
            self.radial += millimeters - (lost if millimeters > 0 else -lost)

        def move_lift_mm(self, millimeters, **_kwargs):
            raise AssertionError("x/y 对准不许动高度")

    class FakeLink:
        """按假臂的**真实**位姿合成观测，不额外注入误差。"""

        def __init__(self, arm: FakeArm, material_xy: Tuple[float, float]) -> None:
            self.arm = arm
            self.material_xy = material_xy
            self.error = None

        def frame(self, **_kwargs):
            arm_x = 100.0 + self.arm.radial * math.cos(math.radians(self.arm.base - 90.0))
            arm_y = -50.0 + self.arm.radial * math.sin(math.radians(self.arm.base - 90.0))
            now = time.monotonic()
            marker = parse_sighting(
                f"物料 9: 红色物料（red），xyz=({arm_x:.4f},{arm_y:.4f},295.0) mm",
                received_at=now,
            )
            target = parse_sighting(
                f"物料 3: 绿色物料（green），xyz=({self.material_xy[0]},{self.material_xy[1]},285.0) mm",
                received_at=now,
            )
            return (marker, target)

    arm_config = ArmConfig(
        pulses_per_revolution=3200,
        rpm=10,
        acceleration=5,
        default_pulses=800,
        max_pulses=5000,
        axes={
            "base": AxisConfig(motor_id=5, reduction_ratio=5),
            "lift": AxisConfig(motor_id=6, mm_per_motor_rev=40.0),
            "extension": AxisConfig(motor_id=7, mm_per_motor_rev=40.0),
        },
        base_limit=TravelLimit(-180.0, 180.0),
        # 下限取 15 而不是 30：夹具里的物料在径向 22.4 mm 处，行程下限必须让它可达。
        radial_limit=TravelLimit(15.0, 200.0),
        height_limit=TravelLimit(-5.0, 120.0),
    )
    marker = GripperMarker(material_id=9)
    config = XYAlignConfig(tolerance_mm=1.0, iterations=4)

    # 物料放在镜头系 (110, -30)：相对基座轴心是 (10, 20) -> θ=63.43°，r=22.36。
    material_xy = (110.0, -30.0)

    # 场景一：已经对准，一次测量就该返回，不许发任何动作。
    # 物料在镜头系 (110, -30)，基座轴心在 (100, -50) -> 相对 (10, 20)；
    # θ 零点偏 90°，所以基座角 = atan2(20,10) + 90 = 153.43°，径向 = 22.3607。
    already = FakeArm(base=153.434948, radial=22.360680)
    aligner = XYAligner(
        arm=already,
        link=FakeLink(already, material_xy),
        plane=plane,
        marker=marker,
        config=config,
        grasp_config=GraspConfig(motion_rpm=None, correction=CorrectionSettings()),
        arm_config=arm_config,
    )
    result = aligner.align()
    assert result.aligned and result.measurements == 1, result
    assert not already.calls, already.calls
    assert "完成" in result.summary(), result.summary()

    # 场景二：没对准，而且第一次伸出少走 4 mm（背隙）——闭环必须靠第二次测量修回来。
    arm = FakeArm(base=0.0, radial=120.0, backlash=4.0)
    aligner = XYAligner(
        arm=arm,
        link=FakeLink(arm, material_xy),
        plane=plane,
        marker=marker,
        config=config,
        grasp_config=GraspConfig(motion_rpm=None, correction=CorrectionSettings()),
        arm_config=arm_config,
    )
    result = aligner.align()
    assert result.aligned, result
    assert result.measurements >= 2, result
    assert abs(arm.radial - 22.360680) < 1.0, arm.radial
    assert abs(arm.base - 153.434948) < 1.0, arm.base
    # 高度一次都没被碰过（FakeArm.move_lift_mm 会直接断言失败）。

    # 场景三：要转大角度时先缩回配置的安全半径。
    corridor_arm = FakeArm(base=-150.0, radial=120.0)
    corridor_aligner = XYAligner(
        arm=corridor_arm,
        link=FakeLink(corridor_arm, material_xy),
        plane=plane,
        marker=marker,
        config=config,
        grasp_config=GraspConfig(
            motion_rpm=None,
            retract_radius_mm=50.0,
            correction=CorrectionSettings(rotation_corridor_degrees=5.0),
        ),
        arm_config=arm_config,
    )
    corridor_aligner.align()
    # 廊道里那一转必须走**最短**方向：−150° → 153.43° 是 −56.57°（穿过 −180°），
    # 不是 +303.43°。直接跳向目标绝对角会绕出 300 多度，把整条臂横扫一遍。
    applied = sum(value for kind, value in corridor_arm.calls if kind == "base")
    assert abs(applied - (-56.565)) < 1.0, applied
    # 缩回动作必须出现在旋转之前：按 (半径, 角度) 记录轨迹检查。
    trace: List[Tuple[float, float]] = []
    radial, base = 120.0, -150.0
    for kind, value in corridor_arm.calls:
        if kind == "radial":
            radial += value
        else:
            base += value
        trace.append((round(radial, 4), round(base, 4)))
    turn_index = next(
        index
        for index, (_r, b) in enumerate(trace)
        if abs(b - (-150.0)) > 1e-6
    )
    assert trace[turn_index][0] == 50.0, trace

    # 场景四：视野里没有靶标 -> 报错而不是瞎动。
    class NoMarkerLink:
        error = None

        def frame(self, **_kwargs):
            item = parse_sighting(
                "物料 3: 绿色物料（green），xyz=(110,-30,285) mm",
                received_at=time.monotonic(),
            )
            return (item,)

    blind_arm = FakeArm(base=0.0, radial=100.0)
    blind = XYAligner(
        arm=blind_arm,
        link=NoMarkerLink(),
        plane=plane,
        marker=marker,
        config=config,
        grasp_config=GraspConfig(),
        arm_config=arm_config,
    )
    try:
        blind.align()
    except XYAlignError as error:
        assert "靶标" in str(error), error
    else:
        raise AssertionError("看不到靶标必须报 XYAlignError")
    assert not blind_arm.calls, blind_arm.calls

    # 场景五：够不着的物料 -> 报错，且不发任何动作。
    far_arm = FakeArm(base=0.0, radial=100.0)
    far = XYAligner(
        arm=far_arm,
        link=FakeLink(far_arm, (400.0, -50.0)),
        plane=plane,
        marker=marker,
        config=config,
        grasp_config=GraspConfig(),
        arm_config=arm_config,
    )
    try:
        far.align()
    except XYAlignError as error:
        assert "行程" in str(error), error
    else:
        raise AssertionError("越界物料必须报 XYAlignError")
    assert not far_arm.calls, far_arm.calls

    # 场景六：gripper_marker 没配置时必须拒绝构造，而不是把物料当成爪子。
    try:
        GripperMarker()
    except XYAlignError as error:
        assert "gripper_marker" in str(error), error
    else:
        raise AssertionError("没配置判别条件必须报 XYAlignError")

    print(f"x/y 对准编排自测通过：{result.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

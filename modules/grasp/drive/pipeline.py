"""抓取动作编排：观测 → 可达性 → 安全绕行 → 视觉微调 → 夹紧 → 退回。

这一层只跟 :class:`CylindricalPose` 和三个抽象接口打交道（机械臂、夹爪、观测器），
所以完全可以在没有摄像头、没有串口的情况下用假件测。

## 这一层真正负责的三件事

1. **大角度重定位前必须先缩回抬升。** 伸缩臂在伸出状态下横扫底座，会把整条臂扫过工作区。
   所以 :meth:`_command` 只要发现目标角度和当前不同，就先经
   :meth:`ArmConfig.require_safe_pose` 给出的"安全廊道"走一遍：缩回到安全半径 →
   抬到安全高度 → 转角度 → 再伸出去。这条廊道上的每一个中间位姿都是**已知量**，
   不依赖测量。
   :meth:`_visual_align` 的小角度修正可以就地转（爪子本来就悬在物料上方几十毫米），
   不受这条约束——否则每次微调都要缩回一次，太慢且没有必要。
2. **最短转角。** θ 从 170° 到 -170° 要走 20°，不是 340°。
3. **任何失败路径都要收尾。** 异常一律走 :meth:`abort`：停三轴、失能。
   刻意**不**张开夹爪——中途失败时松开爪子会把已经抓住的东西掉下去。

## 与标定的一个约定

``base_degrees`` 的正方向必须与 :meth:`RobotArm.move_base_degrees` 的正方向一致
（也就是 CW 为正）。这一条由平面标定的做法保证：标定时记录的累计角度必须就是用
``move_base_degrees`` / ``rotate_cw`` 命令出来的角度，拟合出来的 θ 才会和命令同向
（见 ``modules.vision.drive.calibration``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from modules.arm.config import ArmConfig, load_arm_config
from modules.arm.drive.kinematics import (
    CylindricalPose,
    forward_cylindrical,
    marker_height_for_grasp,
    normalize_degrees,
    require_reachable,
    shortest_angle_delta_degrees,
)
from modules.grasp.config import GraspConfig, load_grasp_config
from modules.grasp.drive.perception import CellObservation
from modules.servo.drive.gripper import GripperReport

#: 小于这个角度就当作"角度没变"，不必绕安全廊道。
ANGLE_EPSILON_DEGREES = 0.05

#: 小于这个长度就当作"这一维不用动"。
LENGTH_EPSILON_MM = 1e-6


class GraspError(RuntimeError):
    """抓取流程无法继续（看不到目标、超出可达域、闭环丢目标等）。"""


class ArmLike(Protocol):
    """``RobotArm`` 在抓取流程里用到的部分。"""

    def move_base_degrees(self, degrees: float, **kwargs) -> object: ...
    def move_lift_mm(self, millimeters: float, **kwargs) -> object: ...
    def move_radial_mm(self, millimeters: float, **kwargs) -> object: ...
    def stop_all(self) -> None: ...
    def disable_all(self) -> None: ...
    def close(self) -> None: ...


class GripperLike(Protocol):
    """``GripperServo`` 在抓取流程里用到的部分。"""

    def open_jaws(self, **kwargs) -> GripperReport: ...
    def close_jaws(self, **kwargs) -> GripperReport: ...
    def close(self) -> None: ...


class ObserverLike(Protocol):
    """``CellPerception`` 在抓取流程里用到的部分。"""

    def observe(self, frame) -> CellObservation: ...


@dataclass(frozen=True)
class PickResult:
    """一次抓取的结果。"""

    #: 抓取前测到的物料位姿；``height_mm`` 是物料高度。
    material: CylindricalPose
    approach_pose: CylindricalPose
    grasp_pose: CylindricalPose
    gripper: GripperReport
    #: 视觉微调收敛用掉的迭代次数（从 0 开始计数）。
    correction_steps: int

    @property
    def held_something(self) -> bool:
        """爪口被挡住 ⇒ 夹到了东西。

        夹紧时读回值**没**到闭合位，说明爪口之间有别的东西。空夹时爪子会一路走到闭合位。
        这是总线舵机的位置读回白送的抓取检测，不需要额外传感器。
        """
        return not self.gripper.reached

    def summary(self) -> str:
        return (
            f"物料 {self.material.summary()}；抓取位 {self.grasp_pose.summary()}；"
            f"{self.gripper.summary()}；微调 {self.correction_steps} 次；"
            f"{'夹到了东西' if self.held_something else '空夹'}"
        )


class GraspPipeline:
    """把视觉、机械臂和夹爪接成一条抓取流程。"""

    def __init__(
        self,
        *,
        arm: ArmLike,
        gripper: GripperLike,
        observer: ObserverLike,
        frames: Callable[[], object],
        config: Optional[GraspConfig] = None,
        arm_config: Optional[ArmConfig] = None,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.observer = observer
        self.frames = frames
        self.config = config if config is not None else load_grasp_config()
        self.arm_config = arm_config if arm_config is not None else load_arm_config()
        #: 最近一次已知的机械臂位姿；``None`` 表示还没测过。
        self._pose: Optional[CylindricalPose] = None

    # ------------------------------------------------------------- 观测
    def observe(self) -> CellObservation:
        """取一帧并换算成两个圆柱位姿。"""
        return self.observer.observe(self.frames())

    def measure_pose(self) -> CylindricalPose:
        """用相机重新测量机械臂当前位姿（靠爪上靶标），并刷新内部状态。"""
        observation = self.observe()
        if observation.marker is None:
            raise GraspError(
                "画面里看不到夹爪靶标；先确认 targets.marker_diameter_mm 和靶标颜色已配置，"
                "且靶标装在爪顶、朝向镜头"
            )
        self._pose = _normalized(observation.marker)
        return self._pose

    @property
    def pose(self) -> Optional[CylindricalPose]:
        """最近一次已知位姿；没测过就是 ``None``。"""
        return self._pose

    # ------------------------------------------------------------- 主流程
    def pick_up(self) -> PickResult:
        """从台面抓起物料并退回安全位。

        顺序：观测 → 可达性 → 绕到物料上方 → 视觉微调 → 下降 → 夹紧 → 抬起 → 退安全位。
        下降段是**盲走**：爪子压上去会遮住物料，所以微调必须在下降之前收敛完。
        """
        try:
            observation = self._observe_both()
            material = observation.material
            if material is None:  # _observe_both 已保证，这里只为类型收窄
                raise GraspError("画面里没有物料")

            grasp_height = marker_height_for_grasp(
                material_height_mm=material.height_mm,
                grip_ratio=self.config.grip_ratio,
                marker_to_grasp_mm=self.config.marker_to_grasp_mm,
            )
            approach_pose = CylindricalPose(
                material.base_degrees,
                material.radial_mm,
                grasp_height + self.config.clearance_mm,
            )
            self._require_reachable(approach_pose)

            self._command(approach_pose)
            correction_steps = self._visual_align()

            # 微调之后就地下降：物料没动，所以降到当前 (θ, r) 正上方即可。
            if self._pose is None:
                raise GraspError("内部状态丢失了机械臂位姿")
            grasp_pose = CylindricalPose(
                self._pose.base_degrees, self._pose.radial_mm, grasp_height
            )
            self._command(grasp_pose)
            report = self.gripper.close_jaws()

            self._command(
                CylindricalPose(
                    grasp_pose.base_degrees,
                    grasp_pose.radial_mm,
                    grasp_height + self.config.clearance_mm,
                )
            )
            self._command(self.arm_config.require_safe_pose())
        except Exception:
            self.abort()
            raise
        return PickResult(
            material=material,
            approach_pose=approach_pose,
            grasp_pose=grasp_pose,
            gripper=report,
            correction_steps=correction_steps,
        )

    def place_at(self, pose: CylindricalPose) -> GripperReport:
        """把手里拿的东西放到指定位姿：过去 → 下降 → 张开 → 退回安全位。"""
        self.measure_pose()
        self._require_reachable(pose)
        try:
            self._command(pose)
            report = self.gripper.open_jaws()
            self._command(
                CylindricalPose(
                    pose.base_degrees,
                    pose.radial_mm,
                    pose.height_mm + self.config.clearance_mm,
                )
            )
            self._command(self.arm_config.require_safe_pose())
        except Exception:
            self.abort()
            raise
        return report

    def park(self) -> CylindricalPose:
        """退到安全位并停下；每个循环开始或结束时用它。"""
        self.measure_pose()
        try:
            self._command(self.arm_config.require_safe_pose())
        except Exception:
            self.abort()
            raise
        if self._pose is None:
            raise GraspError("内部状态丢失了机械臂位姿")
        return self._pose

    def abort(self) -> None:
        """失败路径的统一收尾：停三轴、失能。

        刻意**不**动夹爪：中途失败时松开爪子会把已经抓住的物料掉下去。
        收尾本身再出错也不能盖住原始异常，所以这里吞掉异常并继续。
        """
        for action in (self.arm.stop_all, self.arm.disable_all):
            try:
                action()
            except Exception:  # noqa: BLE001 - 收尾失败不能掩盖真正的错误
                pass

    def close(self) -> None:
        """关闭两侧硬件（机械臂总线与舵机总线是两条串口）。"""
        self.arm.close()
        self.gripper.close()

    # ------------------------------------------------------------- 运动
    def _command(self, target: CylindricalPose) -> None:
        """走到 ``target``；要转角度时先经安全廊道缩回抬升。"""
        if self._pose is None:
            raise GraspError("还不知道机械臂当前位置；先 measure_pose()")
        target = _normalized(target)
        self._require_reachable(target)
        delta_base = shortest_angle_delta_degrees(
            self._pose.base_degrees, target.base_degrees
        )
        if abs(delta_base) <= ANGLE_EPSILON_DEGREES:
            # 不转角度：先对齐平面，再改高度（下降落到物料上是这条路径）。
            self._advance(
                CylindricalPose(target.base_degrees, target.radial_mm, self._pose.height_mm)
            )
            self._advance(target)
            return
        safe = self.arm_config.require_safe_pose()
        for waypoint in (
            CylindricalPose(self._pose.base_degrees, safe.radial_mm, self._pose.height_mm),
            CylindricalPose(self._pose.base_degrees, safe.radial_mm, safe.height_mm),
            CylindricalPose(target.base_degrees, safe.radial_mm, safe.height_mm),
            CylindricalPose(target.base_degrees, target.radial_mm, safe.height_mm),
            target,
        ):
            self._advance(waypoint)

    def _advance(self, waypoint: CylindricalPose) -> None:
        """走到只差一个坐标的 ``waypoint``；顺序固定为 伸缩 → 升降 → 旋转。"""
        if self._pose is None:
            raise GraspError("还不知道机械臂当前位置；先 measure_pose()")
        self._require_reachable(waypoint)
        delta_base = shortest_angle_delta_degrees(
            self._pose.base_degrees, waypoint.base_degrees
        )
        delta_radial = waypoint.radial_mm - self._pose.radial_mm
        delta_height = waypoint.height_mm - self._pose.height_mm
        if abs(delta_radial) > LENGTH_EPSILON_MM:
            self._move(self.arm.move_radial_mm, delta_radial)
        if abs(delta_height) > LENGTH_EPSILON_MM:
            self._move(self.arm.move_lift_mm, delta_height)
        if abs(delta_base) > ANGLE_EPSILON_DEGREES:
            self._move(self.arm.move_base_degrees, delta_base)
        # 命令发出且 wait=True 已等完，就把内部状态推进到 waypoint（角度照测量域归一）。
        self._pose = _normalized(waypoint)

    def _move(self, action: Callable[..., object], value: float) -> None:
        rpm = self.config.motion_rpm
        if rpm is None:
            action(value, wait=True)
        else:
            action(value, wait=True, rpm=rpm)

    def _visual_align(self) -> int:
        """把爪上靶标对到物料正上方；返回收敛用掉的迭代次数。

        只动平面对（θ, r），不动高度。固定相机看固定物料是拿不到反馈的，所以这里看的是
        **爪上的靶标**：它的平面位置就是机械臂的平面位置。

        迭代用尽仍未进容差时**报错而不继续**：此时下降多半会抓空，响亮地失败比悄悄抓空好。
        真机上偶尔收敛慢是正常的，调大 ``correction.iterations`` 或放宽
        ``tolerance_mm``，不要靠忽略这个错误来解决。
        """
        last_error = float("inf")
        for step in range(self.config.correction.iterations):
            observation = self._observe_both()
            marker, material = observation.marker, observation.material
            if marker is None or material is None:  # _observe_both 已保证
                raise GraspError("闭环微调时丢了目标")
            self._pose = _normalized(marker)
            last_error = _planar_error_mm(marker, material)
            if last_error <= self.config.correction.tolerance_mm:
                return step
            gain = self.config.correction.gain
            delta_base = (
                shortest_angle_delta_degrees(marker.base_degrees, material.base_degrees) * gain
            )
            delta_radial = (material.radial_mm - marker.radial_mm) * gain
            if abs(delta_radial) > LENGTH_EPSILON_MM:
                self._move(self.arm.move_radial_mm, delta_radial)
            if abs(delta_base) > ANGLE_EPSILON_DEGREES:
                self._move(self.arm.move_base_degrees, delta_base)
            self._pose = _normalized(
                CylindricalPose(
                    marker.base_degrees + delta_base,
                    marker.radial_mm + delta_radial,
                    marker.height_mm,
                )
            )
        raise GraspError(
            f"视觉闭环 {self.config.correction.iterations} 次仍未收敛，"
            f"最后平面误差 {last_error:.1f} mm（容差 {self.config.correction.tolerance_mm:.1f} mm）；"
            "先看标定残差 rms_mm，再考虑调大 correction.iterations 或 tolerance_mm"
        )

    # ------------------------------------------------------------- 校验
    def _observe_both(self) -> CellObservation:
        """观测并要求靶标和物料都在；把位姿刷成测量值。"""
        observation = self.observe()
        if observation.marker is None and observation.material is None:
            raise GraspError("画面里既没有靶标也没有物料")
        if observation.marker is None:
            raise GraspError("画面里看不到夹爪靶标；闭环没有反馈就无法继续")
        if observation.material is None:
            raise GraspError("画面里没有物料")
        self._pose = _normalized(observation.marker)
        return observation

    def _require_reachable(self, pose: CylindricalPose) -> None:
        try:
            require_reachable(
                pose,
                base_limit=self.arm_config.base_limit,
                radial_limit=self.arm_config.radial_limit,
                height_limit=self.arm_config.height_limit,
            )
        except ValueError as error:
            # 目标够不着是任务失败，不是编程错误，统一成 GraspError 便于调用方捕获。
            raise GraspError(str(error)) from error


def _planar_error_mm(first: CylindricalPose, second: CylindricalPose) -> float:
    """两个位姿在水平面上的距离（毫米），忽略高度差。"""
    first_x, first_y, _ = forward_cylindrical(first)
    second_x, second_y, _ = forward_cylindrical(second)
    return math.hypot(second_x - first_x, second_y - first_y)


def _normalized(pose: CylindricalPose) -> CylindricalPose:
    """把位姿的底座角归一到 ``(-180, 180]``。

    相机只给 ``atan2`` 的输出，所以**存进 ``_pose`` 的角度必须与它同域**；否则"从 170°
    再转 20°"会记成 190°，物理上就是 -170°，却通不过 ±180 的行程检查，抓取会在半路
    莫名其妙地失败。
    """
    return CylindricalPose(
        normalize_degrees(pose.base_degrees), pose.radial_mm, pose.height_mm
    )


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要硬件或摄像头：用假机械臂/假夹爪/脚本化观测跑通整条流程。"""
    from modules.arm.config import AxisConfig
    from modules.arm.drive.kinematics import TravelLimit
    from modules.grasp.config import CorrectionSettings, GraspConfig
    from modules.grasp.drive.perception import CellObservation
    from modules.servo.drive.gripper import CLOSE_COMMAND, OPEN_COMMAND, GripperReport

    safe_radial = 40.0

    class FakeArm:
        """跟踪真实位姿；第一次径向移动故意少走 ``BACKLASH_MM``，模拟背隙。

        这样"命令位姿"和"真实位姿"会真的分开——闭环要修的正是这个差，
        而不是靠脚本硬塞一个不一致的观测值。
        """

        BACKLASH_MM = 4.0

        def __init__(self) -> None:
            self.base, self.radial, self.height = 0.0, safe_radial, 60.0
            self.rotations: list = []
            self.calls: list = []
            self._backlash = self.BACKLASH_MM

        def move_base_degrees(self, degrees, **_kwargs):
            self.rotations.append((degrees, self.radial))
            self.base += degrees
            self.calls.append(("base", round(degrees, 3)))

        def move_radial_mm(self, millimeters, **_kwargs):
            lost = min(abs(millimeters), self._backlash) if self._backlash else 0.0
            self._backlash -= lost
            self.radial += millimeters - math.copysign(lost, millimeters)
            self.calls.append(("radial", round(millimeters, 3)))

        def move_lift_mm(self, millimeters, **_kwargs):
            self.height += millimeters
            self.calls.append(("lift", round(millimeters, 3)))

        def stop_all(self):
            self.calls.append(("stop_all",))

        def disable_all(self):
            self.calls.append(("disable_all",))

        def close(self):
            self.calls.append(("close",))

    class FakeGripper:
        def __init__(self, reached: bool) -> None:
            self.reached = reached
            self.calls: list = []

        def close_jaws(self, **_kwargs):
            self.calls.append("close")
            position = 500 if self.reached else 430
            return GripperReport(
                command=CLOSE_COMMAND, target=500, position=position, reached=self.reached
            )

        def open_jaws(self, **_kwargs):
            self.calls.append("open")
            return GripperReport(command=OPEN_COMMAND, target=300, position=300, reached=True)

        def close(self):
            self.calls.append("close_port")

    class CoupledObserver:
        """观测直接读假臂的**真实**位姿，不额外注入测量误差。

        这样夹具天然自洽：闭环看到的偏差就是假臂真没走够的那几毫米，
        和真机上"命令量 ≠ 到位量"是同一个来源。
        """

        def __init__(self, arm, material) -> None:
            self.arm = arm
            self.material = material

        def observe(self, _frame):
            return CellObservation(
                marker=CylindricalPose(self.arm.base, self.arm.radial, self.arm.height),
                material=self.material,
            )

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
        base_limit=TravelLimit(-110.0, 110.0),
        radial_limit=TravelLimit(40.0, 160.0),
        height_limit=TravelLimit(-5.0, 120.0),
        safe_pose=CylindricalPose(0.0, safe_radial, 60.0),
    )
    config = GraspConfig(
        clearance_mm=20.0,
        grip_ratio=0.5,
        marker_to_grasp_mm=0.0,
        motion_rpm=None,
        correction=CorrectionSettings(iterations=3, tolerance_mm=3.0, gain=0.8),
    )

    material = CylindricalPose(base_degrees=30.0, radial_mm=90.0, height_mm=20.0)
    arm = FakeArm()
    gripper = FakeGripper(reached=False)  # 读回没到闭合位 ⇒ 夹到了东西
    pipeline = GraspPipeline(
        arm=arm,
        gripper=gripper,
        observer=CoupledObserver(arm, material),
        frames=lambda: None,
        config=config,
        arm_config=arm_config,
    )

    result = pipeline.pick_up()

    # 夹持高度 = 物料高 20 × 0.5 + 靶标偏移 0 = 10 mm。
    assert abs(result.grasp_pose.height_mm - 10.0) < 1e-9, result.grasp_pose
    # 背隙让第一次伸出的 50 mm 只走了 46 mm（落在 86 而不是 90），闭环必须把它修回来。
    assert result.correction_steps == 1, result.correction_steps
    # 闭环的保证就是"下降到物料水平误差一个容差以内"。
    assert (
        _planar_error_mm(result.grasp_pose, material) <= config.correction.tolerance_mm + 1e-9
    ), result.grasp_pose
    assert result.held_something, result

    # 安全不变量：伸出去（径向不在安全半径）时只允许小角度微调，不允许大角度重定位。
    for degrees, radial in arm.rotations:
        if abs(radial - safe_radial) > 1e-6:
            assert abs(degrees) <= 5.0, f"伸出状态下转了 {degrees}°（径向 {radial}）"
    # 大角度重定位确实发生了，而且发生在缩回之后。
    assert any(abs(degrees) > 20.0 for degrees, _ in arm.rotations), arm.rotations
    assert any(abs(degrees) > 20.0 and abs(radial - safe_radial) < 1e-6 for degrees, radial in arm.rotations)

    # 结束时必须回到安全位。
    assert abs(arm.radial - safe_radial) < 1e-6, arm.radial
    assert abs(arm.height - 60.0) < 1e-6, arm.height
    assert abs(arm.base - 0.0) < 1e-6, arm.base

    # 够不着的目标：报 GraspError，并且已经收尾（停轴 + 失能），且没有发出任何动作。
    far_arm = FakeArm()
    far_pipeline = GraspPipeline(
        arm=far_arm,
        gripper=FakeGripper(reached=False),
        observer=CoupledObserver(far_arm, CylindricalPose(0.0, 200.0, 20.0)),
        frames=lambda: None,
        config=config,
        arm_config=arm_config,
    )
    try:
        far_pipeline.pick_up()
    except GraspError as error:
        assert "行程" in str(error), error
    else:
        raise AssertionError("超出径向行程的目标必须报 GraspError")
    assert ("stop_all",) in far_arm.calls and ("disable_all",) in far_arm.calls, far_arm.calls
    assert not any(call[0] in ("base", "radial", "lift") for call in far_arm.calls), far_arm.calls

    # 看不到靶标：报 GraspError，闭环不能靠猜。
    class BlindObserver:
        def observe(self, _frame):
            return CellObservation(marker=None, material=material)

    blind_arm = FakeArm()
    blind = GraspPipeline(
        arm=blind_arm,
        gripper=FakeGripper(reached=True),
        observer=BlindObserver(),
        frames=lambda: None,
        config=config,
        arm_config=arm_config,
    )
    try:
        blind.pick_up()
    except GraspError as error:
        assert "靶标" in str(error), error
    else:
        raise AssertionError("看不到靶标必须报 GraspError")

    print(f"抓取编排自测通过：{result.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

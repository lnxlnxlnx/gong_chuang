"""抓取编排：安全廊道、最短转角、闭环收敛，以及每条失败路径的收尾。

夹具刻意做成**自洽**的：观测直接读假臂的真实位姿，误差只来自假臂自己没走够的背隙。
这样"命令量 ≠ 到位量"就是真机上的那个来源，而不是靠脚本硬塞一个不一致的观测值。
"""

from __future__ import annotations

import math
import unittest

from modules.arm.config import ArmConfig, AxisConfig
from modules.arm.drive.kinematics import CylindricalPose, TravelLimit, normalize_degrees
from modules.grasp.config import CorrectionSettings, GraspConfig
from modules.grasp.drive.perception import CellObservation
from modules.grasp.drive.pipeline import GraspError, GraspPipeline, _planar_error_mm
from modules.servo.drive.gripper import CLOSE_COMMAND, OPEN_COMMAND, GripperReport

SAFE = CylindricalPose(0.0, 40.0, 60.0)
MATERIAL = CylindricalPose(30.0, 90.0, 20.0)
BACKLASH_MM = 4.0
GRASP_HEIGHT_MM = 10.0          # 物料高 20 × grip_ratio 0.5 + 靶标偏移 0
APPROACH_HEIGHT_MM = 30.0       # 夹持高度 + clearance 20


def arm_config(**overrides) -> ArmConfig:
    settings = dict(
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
        safe_pose=SAFE,
    )
    settings.update(overrides)
    return ArmConfig(**settings)


def grasp_config(**overrides) -> GraspConfig:
    settings = dict(
        clearance_mm=20.0,
        grip_ratio=0.5,
        marker_to_grasp_mm=0.0,
        motion_rpm=None,
        correction=CorrectionSettings(iterations=3, tolerance_mm=3.0, gain=0.8),
    )
    settings.update(overrides)
    return GraspConfig(**settings)


class FakeArm:
    """跟踪真实位姿；每次动作后把**绝对**位姿记进 ``poses``，动作量记进事件日志。"""

    def __init__(self, events: list, *, backlash_mm: float = BACKLASH_MM) -> None:
        self.base, self.radial, self.height = 0.0, SAFE.radial_mm, SAFE.height_mm
        self.events = events
        self.rotations: list = []
        self.calls: list = []
        self.poses: list = [(self.base, self.radial, self.height)]
        self._backlash = backlash_mm

    def _record(self, kind: str, value: float) -> None:
        self.events.append(("move", kind, value))
        self.calls.append((kind, value))
        self.poses.append((self.base, self.radial, self.height))

    def move_base_degrees(self, degrees, **_kwargs):
        self.rotations.append((degrees, self.radial))
        self.base += degrees
        self._record("base", degrees)

    def move_radial_mm(self, millimeters, **_kwargs):
        lost = min(abs(millimeters), self._backlash) if self._backlash else 0.0
        self._backlash -= lost
        self.radial += millimeters - math.copysign(lost, millimeters)
        self._record("radial", millimeters)

    def move_lift_mm(self, millimeters, **_kwargs):
        self.height += millimeters
        self._record("lift", millimeters)

    def stop_all(self):
        self.calls.append(("stop_all",))
        self.events.append(("stop_all",))

    def disable_all(self):
        self.calls.append(("disable_all",))
        self.events.append(("disable_all",))

    def close(self):
        self.calls.append(("close",))

    def trace(self) -> list:
        """每次动作后的**绝对**高度序列（不是动作量）。"""
        return [height for _, _, height in self.poses]


class CoupledObserver:
    """观测直接读假臂的真实位姿；事件日志里记下每次观测到的平面误差。"""

    def __init__(self, arm: FakeArm, material=MATERIAL, events=None) -> None:
        self.arm = arm
        self.material = material
        self.events = events if events is not None else []

    def observe(self, _frame) -> CellObservation:
        # 相机只能给出 atan2 的输出，所以测量角必须归一；夹具也照做，
        # 否则"转过 180° 之后"的位姿在夹具里是 190°，而真机上永远是 -170°。
        marker = CylindricalPose(
            normalize_degrees(self.arm.base), self.arm.radial, self.arm.height
        )
        self.events.append(("observe", _planar_error_mm(marker, self.material)))
        return CellObservation(marker=marker, material=self.material)


class FixedObserver:
    """始终返回同一组位姿；用来构造"怎么调都不收敛"的场景。"""

    def __init__(self, marker, material) -> None:
        self.marker = marker
        self.material = material

    def observe(self, _frame) -> CellObservation:
        return CellObservation(marker=self.marker, material=self.material)


class FakeGripper:
    def __init__(self, *, reached: bool) -> None:
        self.reached = reached
        self.calls: list = []

    def close_jaws(self, **_kwargs) -> GripperReport:
        self.calls.append("close")
        return GripperReport(
            command=CLOSE_COMMAND,
            target=500,
            position=500 if self.reached else 430,
            reached=self.reached,
        )

    def open_jaws(self, **_kwargs) -> GripperReport:
        self.calls.append("open")
        return GripperReport(command=OPEN_COMMAND, target=300, position=300, reached=True)

    def close(self) -> None:
        self.calls.append("close_port")


def build_pipeline(
    *,
    gripper=None,
    config=None,
    observer=None,
    events=None,
    arm=None,
    arm_settings=None,
    **arm_overrides,
):
    events = [] if events is None else events
    arm = arm if arm is not None else FakeArm(events, **arm_overrides)
    return GraspPipeline(
        arm=arm,
        gripper=gripper if gripper is not None else FakeGripper(reached=False),
        observer=observer if observer is not None else CoupledObserver(arm, MATERIAL, events),
        frames=lambda: None,
        config=config if config is not None else grasp_config(),
        arm_config=arm_settings if arm_settings is not None else arm_config(),
    ), arm, events


class GraspCorridorTests(unittest.TestCase):
    def test_the_arm_never_makes_a_large_rotation_while_extended(self) -> None:
        """伸缩臂在伸出状态下横扫底座会把整条臂扫过工作区。"""
        pipeline, arm, _ = build_pipeline()

        pipeline.pick_up()

        for degrees, radial in arm.rotations:
            if abs(radial - SAFE.radial_mm) > 1e-6:
                self.assertLessEqual(abs(degrees), 5.0, f"伸出状态下转了 {degrees}°")

    def test_large_repositioning_does_happen_and_only_after_retracting(self) -> None:
        pipeline, arm, _ = build_pipeline()

        pipeline.pick_up()

        retracted_rotations = [
            degrees for degrees, radial in arm.rotations if abs(radial - SAFE.radial_mm) < 1e-6
        ]
        self.assertGreater(len(retracted_rotations), 0, arm.rotations)
        self.assertTrue(all(abs(item) > 20.0 for item in retracted_rotations), retracted_rotations)

    def test_the_rotation_takes_the_short_way(self) -> None:
        # 物料在 -170°，起点在 170°：应当只走 20°，而不是 340°。
        # 这一条需要 ±180 的底座行程，默认夹具的 ±110 装不下。
        material = CylindricalPose(-170.0, 90.0, 20.0)
        events: list = []
        arm = FakeArm(events)
        arm.base = 170.0
        pipeline, _, _ = build_pipeline(
            arm=arm,
            events=events,
            observer=CoupledObserver(arm, material, events),
            arm_settings=arm_config(base_limit=TravelLimit(-180.0, 180.0)),
        )

        pipeline.pick_up()

        repositioning = [
            degrees for degrees, radial in arm.rotations if abs(radial - SAFE.radial_mm) < 1e-6
        ]
        self.assertTrue(repositioning)
        # 每一段缩回后的回转都走短边，绝不超过半圈。
        for degrees in repositioning:
            self.assertLessEqual(abs(degrees), 180.0 + 1e-6, repositioning)
        # 170° → -170° 只走 20°，而不是绕 340° 扫过整个工作区。
        self.assertAlmostEqual(repositioning[0], 20.0, places=6, msg=repositioning)


class GraspSequenceTests(unittest.TestCase):
    def test_the_arm_descends_to_the_grasp_height_after_the_loop_converges(self) -> None:
        pipeline, arm, events = build_pipeline()

        result = pipeline.pick_up()

        self.assertEqual(result.correction_steps, 1)
        descent = events.index(("move", "lift", GRASP_HEIGHT_MM - APPROACH_HEIGHT_MM))
        last_observation = max(
            index for index, event in enumerate(events) if event[0] == "observe" and index < descent
        )
        self.assertLessEqual(events[last_observation][1], 3.0)
        # 只下降到夹持高度一次，而且降到的正是算出来的那个高度。
        self.assertEqual(arm.trace().count(GRASP_HEIGHT_MM), 1, arm.trace())

    def test_the_grasp_pose_is_derived_from_the_measured_material(self) -> None:
        pipeline, _, _ = build_pipeline()

        result = pipeline.pick_up()

        self.assertAlmostEqual(result.grasp_pose.height_mm, GRASP_HEIGHT_MM, places=9)
        self.assertLessEqual(
            _planar_error_mm(result.grasp_pose, MATERIAL), 3.0 + 1e-9, result.grasp_pose
        )
        self.assertAlmostEqual(result.material.radial_mm, MATERIAL.radial_mm, places=9)

    def test_the_cycle_ends_at_the_safe_pose(self) -> None:
        pipeline, arm, _ = build_pipeline()

        pipeline.pick_up()

        self.assertAlmostEqual(arm.radial, SAFE.radial_mm, places=6)
        self.assertAlmostEqual(arm.height, SAFE.height_mm, places=6)
        self.assertAlmostEqual(arm.base, SAFE.base_degrees, places=6)

    def test_a_blocked_close_is_reported_as_holding_something(self) -> None:
        pipeline, _, _ = build_pipeline(gripper=FakeGripper(reached=False))
        result = pipeline.pick_up()
        self.assertTrue(result.held_something)
        self.assertIn("夹到了东西", result.summary())

    def test_an_empty_close_is_reported_as_nothing_held(self) -> None:
        pipeline, _, _ = build_pipeline(gripper=FakeGripper(reached=True))
        result = pipeline.pick_up()
        self.assertFalse(result.held_something)
        self.assertIn("空夹", result.summary())

    def test_the_configured_rpm_is_forwarded_to_every_arm_move(self) -> None:
        seen: list = []

        class RpmRecordingArm(FakeArm):
            """把每次动作收到的 rpm 关键字记下来。"""

            def _record_rpm(self, kwargs):
                seen.append(kwargs.get("rpm"))

            def move_lift_mm(self, millimeters, **kwargs):
                self._record_rpm(kwargs)
                super().move_lift_mm(millimeters, **kwargs)

            def move_radial_mm(self, millimeters, **kwargs):
                self._record_rpm(kwargs)
                super().move_radial_mm(millimeters, **kwargs)

            def move_base_degrees(self, degrees, **kwargs):
                self._record_rpm(kwargs)
                super().move_base_degrees(degrees, **kwargs)

        events: list = []
        arm = RpmRecordingArm(events)
        pipeline, _, _ = build_pipeline(
            arm=arm,
            events=events,
            observer=CoupledObserver(arm, MATERIAL, events),
            config=grasp_config(motion_rpm=8),
        )

        pipeline.pick_up()

        self.assertTrue(seen)
        self.assertTrue(all(rpm == 8 for rpm in seen), seen)

    def test_without_a_configured_rpm_the_arm_keeps_its_own_defaults(self) -> None:
        seen: list = []

        class RpmRecordingArm(FakeArm):
            def move_lift_mm(self, millimeters, **kwargs):
                seen.append(kwargs.get("rpm"))
                super().move_lift_mm(millimeters, **kwargs)

        events: list = []
        arm = RpmRecordingArm(events)
        pipeline, _, _ = build_pipeline(
            arm=arm, events=events, observer=CoupledObserver(arm, MATERIAL, events)
        )

        pipeline.pick_up()

        self.assertTrue(seen)
        self.assertTrue(all(rpm is None for rpm in seen), seen)


class GraspFailureTests(unittest.TestCase):
    def test_an_unreachable_material_fails_before_any_motion_and_still_cleans_up(self) -> None:
        events: list = []
        arm = FakeArm(events)
        pipeline, _, _ = build_pipeline(
            arm=arm,
            events=events,
            observer=CoupledObserver(arm, CylindricalPose(0.0, 200.0, 20.0), events),
        )

        with self.assertRaisesRegex(GraspError, "行程"):
            pipeline.pick_up()

        self.assertFalse(any(event[0] == "move" for event in events), events)
        self.assertIn(("stop_all",), arm.calls)
        self.assertIn(("disable_all",), arm.calls)

    def test_a_cycle_that_never_converges_fails_instead_of_descending(self) -> None:
        events: list = []
        arm = FakeArm(events)
        stuck = CylindricalPose(0.0, SAFE.radial_mm, APPROACH_HEIGHT_MM)
        pipeline, _, _ = build_pipeline(
            arm=arm,
            events=events,
            observer=FixedObserver(stuck, MATERIAL),
            config=grasp_config(correction=CorrectionSettings(iterations=2, tolerance_mm=1.0, gain=0.5)),
        )

        with self.assertRaisesRegex(GraspError, "未收敛"):
            pipeline.pick_up()

        # 只要没收敛就不许下降：所有高度都停在接近高度或以上。
        self.assertGreaterEqual(min(arm.trace()), APPROACH_HEIGHT_MM - 1e-6, arm.trace())
        self.assertIn(("stop_all",), arm.calls)
        self.assertIn(("disable_all",), arm.calls)

    def test_a_missing_marker_fails_because_the_loop_has_no_feedback(self) -> None:
        events: list = []
        arm = FakeArm(events)
        pipeline, _, _ = build_pipeline(
            arm=arm, events=events, observer=FixedObserver(None, MATERIAL)
        )

        with self.assertRaisesRegex(GraspError, "靶标"):
            pipeline.pick_up()

        self.assertFalse(any(event[0] == "move" for event in events), events)

    def test_a_missing_material_fails(self) -> None:
        events: list = []
        arm = FakeArm(events)
        pipeline, _, _ = build_pipeline(
            arm=arm, events=events, observer=FixedObserver(CylindricalPose(0.0, 40.0, 60.0), None)
        )

        with self.assertRaisesRegex(GraspError, "物料"):
            pipeline.pick_up()

    def test_abort_never_opens_the_jaws(self) -> None:
        """中途失败时松开爪子会把已经抓住的物料掉下去。"""
        gripper = FakeGripper(reached=True)
        pipeline, _, _ = build_pipeline(
            gripper=gripper, observer=FixedObserver(None, MATERIAL)
        )

        with self.assertRaises(GraspError):
            pipeline.pick_up()

        self.assertNotIn("open", gripper.calls)


class GraspPlaceAndLifecycleTests(unittest.TestCase):
    def test_place_at_moves_there_opens_the_jaws_and_retreats_to_safe(self) -> None:
        # 这一条测的是动作顺序与几何，所以用无背隙的臂，断言才能保持精确。
        events: list = []
        gripper = FakeGripper(reached=True)
        arm = FakeArm(events, backlash_mm=0.0)
        pipeline, _, _ = build_pipeline(
            arm=arm,
            events=events,
            gripper=gripper,
            observer=CoupledObserver(arm, MATERIAL, events),
        )
        target = CylindricalPose(-60.0, 70.0, 15.0)

        report = pipeline.place_at(target)

        self.assertEqual(report.command, OPEN_COMMAND)
        self.assertIn("open", gripper.calls)
        # 放置点必须真的到过：高度降到过 target.height_mm。
        self.assertAlmostEqual(min(arm.trace()), target.height_mm, places=6, msg=arm.trace())
        self.assertAlmostEqual(arm.base, SAFE.base_degrees, places=6)
        self.assertAlmostEqual(arm.radial, SAFE.radial_mm, places=6)
        self.assertAlmostEqual(arm.height, SAFE.height_mm, places=6)

    def test_park_goes_to_the_safe_pose(self) -> None:
        pipeline, arm, _ = build_pipeline()

        parked = pipeline.park()

        self.assertEqual(parked, SAFE)
        self.assertAlmostEqual(arm.base, SAFE.base_degrees, places=6)

    def test_close_closes_both_buses(self) -> None:
        gripper = FakeGripper(reached=True)
        pipeline, arm, _ = build_pipeline(gripper=gripper)

        pipeline.close()

        self.assertIn(("close",), arm.calls)
        self.assertIn("close_port", gripper.calls)

    def test_the_pipeline_reports_the_last_known_pose_or_none(self) -> None:
        pipeline, _, _ = build_pipeline()
        self.assertIsNone(pipeline.pose)

        measured = pipeline.measure_pose()

        self.assertEqual(measured, CylindricalPose(0.0, SAFE.radial_mm, SAFE.height_mm))
        self.assertEqual(pipeline.pose, measured)

    def test_placing_without_a_marker_fails_before_moving(self) -> None:
        events: list = []
        arm = FakeArm(events)
        pipeline, _, _ = build_pipeline(
            arm=arm, events=events, observer=FixedObserver(None, MATERIAL)
        )

        with self.assertRaisesRegex(GraspError, "靶标"):
            pipeline.place_at(CylindricalPose(0.0, 60.0, 20.0))

        self.assertFalse(any(event[0] == "move" for event in events), events)


if __name__ == "__main__":
    unittest.main()

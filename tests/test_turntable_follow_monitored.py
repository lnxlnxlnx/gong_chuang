"""监控下降与重抓的离线仿真：执行真实控制代码，只替换电机和相机设备。"""

import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app.gripper as app_gripper
import modules.grasp.runtime as grasp_runtime
from modules.arm.runtime import RobotArm
from modules.arm.test.unit.test_arm_axes import (
    FakeClock, FakeSerial, RecordingSleep, fixture_config,
)
from modules.grasp import Grasp
from modules.grasp.drive.visual_servo_model import (
    MaterialSelector, VisualServoConfig, VisualServoError,
    VisualServoJacobian, VisualServoTarget,
)
from modules.vision.drive.sighting import MaterialSighting


class SimulatedLift:
    """按假时钟积分位置，编码器原点故意与命令里程表相差 12345 脉冲。"""

    def __init__(self, clock, start, ppr):
        self.clock, self.position, self.ppr = clock, start, ppr
        self.target, self.stamp, self.speed = start, clock(), 0
        self.moves, self.stops = [], []
        self.stalled = False

    def update(self):
        distance = self.target - self.position
        step = 0 if self.stalled else self.speed * (self.clock() - self.stamp)
        self.position += math.copysign(min(abs(distance), step), distance)
        self.stamp = self.clock()

    def move_relative_pulses(self, pulses, rpm, acceleration, *, synchronized=False):
        self.update()
        self.target = self.position + pulses
        self.speed = rpm * self.ppr / 60
        self.moves.append((self.clock(), pulses))

    def read_position(self):
        self.update()
        return (self.position + 12345) * 65536 / self.ppr

    def stop(self):
        self.update()
        self.target = self.position
        self.stops.append((self.clock(), self.position))

    def read_status(self):
        self.update()
        return SimpleNamespace(
            in_position=abs(self.position - self.target) <= self.ppr * 0.3 / 360,
            stalled=False, stall_protected=False,
        )


class SimulatedCamera:
    def __init__(self, clock, sleep):
        self.clock, self.sleep = clock, sleep
        self.offset = lambda: (0.0, 0.0)
        self.stamp = lambda: self.clock()
        self.samples = []

    def frame_fast(self, **_kwargs):
        self.sleep(0.04)
        offset = self.offset()
        self.samples.append((self.clock(), offset))
        if offset is None:
            return ()
        return (MaterialSighting(
            material_id=1, color="red", x_mm=offset[0], y_mm=offset[1],
            z_mm=80, received_at=self.stamp(),
        ),)


class MonitoredGrabTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.sleep = RecordingSleep(self.clock)
        self.arm = RobotArm.from_settings(
            config=fixture_config(acceleration=0, initial_motor_degrees={"lift": 830}),
            port="FAKE", bus=FakeSerial(), clock=self.clock, sleep=self.sleep,
        )
        start = self.arm.read_abs_state("lift").pulses
        self.motor = SimulatedLift(self.clock, start, self.arm.pulses_per_revolution)
        self.arm._motors["lift"] = self.motor
        self.camera = SimulatedCamera(self.clock, self.sleep)
        self.grasp = Grasp(
            config=VisualServoConfig(tolerance_mm=6, z_center_tolerance_mm=6),
            selector=MaterialSelector(), target=VisualServoTarget(0, 0),
            jacobian=VisualServoJacobian(-0.08, 0, 0, 0.02), lift_scale=None,
        )
        self.grasp._arm, self.grasp._source = self.arm, self.camera
        self.robot = SimpleNamespace(arm=self.arm, grasp=self.grasp, gripper=Mock())
        fake_time = SimpleNamespace(monotonic=self.clock, sleep=self.sleep)
        for module in (app_gripper, grasp_runtime):
            context = patch.object(module, "time", fake_time)
            context.start()
            self.addCleanup(context.stop)

    def grab(self, *, seconds=5):
        return app_gripper._descend_and_grab_from_turntable_monitored(
            self.robot, "red", deadline=self.clock() + seconds,
        )

    def assert_back_at_observation_height(self):
        self.motor.update()
        self.assertAlmostEqual(self.motor.position * 360 / self.motor.ppr, 830, delta=0.2)
        self.assertAlmostEqual(self.arm.read_abs_state("lift").motor_degrees, 830, delta=0.2)
        self.robot.gripper.close_jaws.assert_not_called()

    def test_camera_keeps_reading_during_descent_and_grabs_only_at_height(self):
        grabbed_at = []
        self.robot.gripper.close_jaws.side_effect = lambda: grabbed_at.append(
            self.motor.position * 360 / self.motor.ppr
        )
        self.assertTrue(self.grab())
        self.assertGreater(len(self.camera.samples), 3)
        self.assertAlmostEqual(grabbed_at[0], 550, delta=0.2)
        self.motor.update()
        self.assertAlmostEqual(self.motor.position * 360 / self.motor.ppr, 800, delta=0.2)

    def test_missing_or_outside_target_brakes_halfway_and_returns_to_observation(self):
        started = self.clock()
        self.camera.offset = lambda: (0, 0) if self.clock() - started < 0.12 else None
        self.assertFalse(self.grab())
        stopped_height = self.motor.stops[0][1] * 360 / self.motor.ppr
        self.assertGreater(stopped_height, 550)
        self.assertLess(stopped_height, 830)
        self.assert_back_at_observation_height()

    def test_visible_target_outside_grab_range_never_closes_jaws(self):
        self.camera.offset = lambda: (6.1, 0)
        self.assertFalse(self.grab())
        self.assert_back_at_observation_height()

    def test_final_frame_after_braking_can_cancel_the_grab(self):
        self.camera.offset = lambda: None if self.motor.stops else (0, 0)
        self.assertFalse(self.grab())
        self.assert_back_at_observation_height()

    def test_other_colour_does_not_replace_requested_material(self):
        self.assertFalse(self.grasp.material_in_grab_range("blue"))

    def test_old_frames_and_invalid_coordinates_fail_instead_of_grabbing(self):
        self.camera.stamp = lambda: -1
        with self.assertRaisesRegex(VisualServoError, "时间戳"):
            self.grab()
        self.robot.gripper.close_jaws.assert_not_called()
        self.assertTrue(self.motor.stops)
        self.assertAlmostEqual(self.arm.read_abs_state("lift").pulses, self.motor.position, delta=1)
        self.camera.stamp = lambda: self.clock()
        self.camera.offset = lambda: (math.nan, 0)
        with self.assertRaisesRegex(VisualServoError, "有限数"):
            self.grasp.material_in_grab_range("red")

    def test_stalled_lift_never_grabs_and_stops_on_timeout(self):
        self.motor.stalled = True
        with self.assertRaisesRegex(TimeoutError, "实测到位"):
            self.grab()
        self.robot.gripper.close_jaws.assert_not_called()
        self.assertTrue(self.motor.stops)
        self.assertAlmostEqual(self.arm.read_abs_state("lift").motor_degrees, 830, delta=0.2)

    def test_driver_arrival_window_does_not_require_exact_encoder_equality(self):
        normal_status = self.motor.read_status
        def in_window():
            self.motor.update()
            if abs(self.motor.position - self.motor.target) < 3:
                self.motor.position = self.motor.target + 2
                self.motor.speed = 0
            return normal_status()
        self.motor.read_status = in_window
        self.assertTrue(self.grab())
        self.robot.gripper.close_jaws.assert_called_once()

    def test_unchanged_encoder_with_stale_arrival_flag_does_not_allow_grabbing(self):
        self.motor.stalled = True
        self.motor.read_status = lambda: SimpleNamespace(
            in_position=True, stalled=False, stall_protected=False,
        )
        with self.assertRaises(TimeoutError):
            self.grab()
        self.robot.gripper.close_jaws.assert_not_called()

    def test_camera_error_cancels_motion_and_propagates(self):
        def fail():
            raise RuntimeError("相机掉线")
        self.camera.offset = fail
        with self.assertRaisesRegex(RuntimeError, "相机掉线"):
            self.grab()
        self.robot.gripper.close_jaws.assert_not_called()
        self.assertTrue(self.motor.stops)
        self.assertAlmostEqual(self.arm.read_abs_state("lift").pulses, self.motor.position, delta=1)

    def test_budget_expiring_during_frame_read_returns_before_retry(self):
        self.assertFalse(self.grab(seconds=0.02))
        self.assert_back_at_observation_height()

    def test_retry_stays_on_same_colour_and_stows_only_after_success(self):
        events = []
        follow_options = []
        def follow(_robot, color, **kwargs):
            self.motor.update()
            events.append(("follow", color, round(self.motor.position * 360 / self.motor.ppr)))
            follow_options.append(kwargs)
            self.camera.offset = lambda: None if len(follow_options) == 1 else (0, 0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(app_gripper, "_follow_material_until_stable", side_effect=follow))
            stack.enter_context(patch.object(app_gripper, "_stow_material_to_repo", side_effect=lambda _r: events.append("stow")))
            stack.enter_context(patch.object(app_gripper, "move_axes_to"))
            app_gripper.grab_material_from_turntable_to_repo_follow_v2(
                self.robot, ("red",), home=None,
            )
        self.assertEqual(events, [("follow", "red", 830), ("follow", "red", 830), "stow"])
        self.assertEqual([kw["min_follow_seconds"] for kw in follow_options], [0, 0])
        self.assertEqual([kw["stable_seconds"] for kw in follow_options], [0.3, 0.3])
        self.assertLess(follow_options[1]["timeout"], follow_options[0]["timeout"])
        self.robot.gripper.close_jaws.assert_called_once()

    def test_repeated_loss_exhausts_one_budget_and_never_stows(self):
        self.camera.offset = lambda: None
        with ExitStack() as stack:
            stack.enter_context(patch.object(app_gripper, "_follow_material_until_stable"))
            stow = stack.enter_context(patch.object(app_gripper, "_stow_material_to_repo"))
            with self.assertRaises(TimeoutError):
                app_gripper.grab_material_from_turntable_to_repo_follow_v2(
                    self.robot, ("red",), home=None, follow_seconds=0.2,
                )
        stow.assert_not_called()
        self.assert_back_at_observation_height()


class ShortStableGateTests(unittest.TestCase):
    def test_a_stopped_target_is_released_after_point_three_seconds_without_four_second_gate(self):
        from modules.grasp.drive import eye_in_hand as algorithms
        from modules.grasp.test.unit.test_continuous_follow import VelocityArm, WorldLink, follow
        from modules.grasp.test.unit.test_eye_in_hand import FollowClock

        clock = FollowClock()
        arm = VelocityArm(clock)
        link = WorldLink(arm, stop_at=0)
        robot = Mock()
        robot.arm.read_abs_state.return_value.motor_degrees = 830
        robot.grasp.follow_material.side_effect = lambda _color, **kw: follow(arm, link, **kw)
        with ExitStack() as stack:
            stack.enter_context(patch.object(algorithms, "time", clock))
            stack.enter_context(patch.object(app_gripper, "_descend_and_grab_from_turntable_monitored", return_value=True))
            stack.enter_context(patch.object(app_gripper, "_stow_material_to_repo"))
            stack.enter_context(patch.object(app_gripper, "move_axes_to"))
            app_gripper.grab_material_from_turntable_to_repo_follow_v2(robot, ("red",), home=None)
        self.assertGreaterEqual(clock.now, 0.3)
        self.assertLess(clock.now, 1.0)

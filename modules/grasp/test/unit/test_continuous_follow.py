"""连续跟随的带时间、编码器和偏心转盘仿真，不连接硬件。"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.arm.config import load_arm_config
from modules.arm.runtime import RobotArm
from modules.grasp.drive.follow_trajectory import ClockwiseTrajectory, PlanarGeometry
from modules.grasp.drive import eye_in_hand as algorithms
from modules.grasp.drive.visual_servo_model import (
    MaterialSelector,
    VisualServoConfig,
    VisualServoError,
    VisualServoJacobian,
    VisualServoTarget,
)
from modules.grasp.test.unit.test_eye_in_hand import FollowClock, sighting


class VelocityArm:
    """把连续 rpm 积分成实测位置，记录采样和停止以检查交接。"""

    def __init__(self, clock):
        self.clock = clock
        self.config = load_arm_config()
        self.pulses_per_revolution = self.config.pulses_per_revolution
        self.position = [0.0, 500 * self.pulses_per_revolution / 360]
        self.velocity = (0, 0)
        self.stamp = clock.now
        self.commands = []
        self.samples = []
        self.closed = False
        self.stops = 0

    def _advance(self):
        dt = self.clock.now - self.stamp
        for axis in range(2):
            self.position[axis] += (
                self.velocity[axis] * self.pulses_per_revolution / 60 * dt
            )
        self.stamp = self.clock.now

    def planar_velocity(self, **kwargs):
        self.options = kwargs
        return self

    def sample(self):
        self._advance()
        value = SimpleNamespace(
            stamp=self.clock.now,
            base_pulses=self.position[0],
            extension_pulses=self.position[1],
        )
        self.samples.append(value)
        return value

    def position_at(self, stamp):
        for left, right in zip(self.samples, self.samples[1:]):
            if left.stamp <= stamp <= right.stamp:
                ratio = (
                    0
                    if left.stamp == right.stamp
                    else (stamp - left.stamp) / (right.stamp - left.stamp)
                )
                return (
                    left.base_pulses + ratio * (right.base_pulses - left.base_pulses),
                    left.extension_pulses
                    + ratio * (right.extension_pulses - left.extension_pulses),
                )
        raise ValueError("timestamp outside measured history")

    def set_velocity(self, base, extension):
        self._advance()
        self.velocity = tuple(int(max(-50, min(50, v))) for v in (base, extension))
        self.commands.append((self.clock.now, *self.velocity))
        return self.velocity

    def stop(self):
        self._advance()
        self.velocity = (0, 0)
        self.stops += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()
        self.closed = True


JACOBIAN = VisualServoJacobian(base_dx=-0.08, base_dy=0, radial_dx=0, radial_dy=0.02)


class WorldLink:
    """转盘中心偏离机械臂原点；观测含相机运动和 40ms 推理延迟。"""

    def __init__(self, arm, *, stop_at=2.0, missing=(), fail_at=None):
        self.arm = arm
        self.clock = arm.clock
        self.stop_at = stop_at
        self.missing = missing
        self.fail_at = fail_at
        self.errors = []
        self.reads = self.fast_reads = self.slow_reads = 0
        self.frames_while_moving = 0
        self.scene_checks = 0
        self.origin_radius = arm.config.extension_radius_mm(round(arm.position[1]))

    def frame_fast(self, **kwargs):
        self.fast_reads += 1
        self.reads += 1
        self.clock.sleep(0.04)
        self.arm._advance()
        stamp = self.clock.now
        if self.fail_at and stamp >= self.fail_at:
            raise RuntimeError("camera failed")
        theta = -0.65 * min(stamp, self.stop_at)
        x = self.origin_radius - 35 + 35 * math.cos(theta)
        y = 35 * math.sin(theta)
        ppr = self.arm.pulses_per_revolution
        target_base = (
            -math.atan2(y, x) * ppr * self.arm.config.base_reduction / (2 * math.pi)
        )
        target_ext = (
            500 * ppr / 360 + (math.hypot(x, y) - self.origin_radius) * ppr / 61
        )
        offset = JACOBIAN.displacement_for(
            self.arm.position[0] - target_base, self.arm.position[1] - target_ext
        )
        self.errors.append(math.hypot(*offset))
        if any(self.arm.velocity):
            self.frames_while_moving += 1
        self.clock.sleep(0.04)
        if any(low <= stamp <= high for low, high in self.missing):
            return ()
        return (sighting(4, *offset, stamp=stamp),)

    def frame(self, **kwargs):
        self.slow_reads += 1
        raise AssertionError("following must use fast observations")

    def scene_stable(self):
        self.scene_checks += 1
        return self.clock.now >= self.stop_at and not any(self.arm.velocity)


def follow(arm, link, **kwargs):
    return algorithms.follow_material_xy(
        arm,
        link,
        MaterialSelector(),
        VisualServoTarget(0, 0),
        JACOBIAN,
        VisualServoConfig(inter_frame_seconds=0.01),
        **kwargs,
    )


class ContinuousFollowTests(unittest.TestCase):
    def rig(self, **kwargs):
        clock = FollowClock()
        arm = VelocityArm(clock)
        return clock, arm, WorldLink(arm, **kwargs)

    def test_primary_path_tracks_and_hands_back_measured_position(self):
        clock, arm, link = self.rig()
        start = tuple(arm.position)
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=7)
        self.assertTrue(result.aligned)
        self.assertTrue(arm.closed)
        self.assertEqual(arm.velocity, (0, 0))
        self.assertEqual(result.base_pulses, round(arm.position[0] - start[0]))
        self.assertEqual(result.radial_pulses, round(arm.position[1] - start[1]))
        self.assertGreater(link.frames_while_moving, 5)
        self.assertEqual(arm.options["max_rpm"], 50)
        self.assertEqual(arm.options["motor_limits"]["extension"], (30, 990))
        self.assertEqual(arm.options["motor_limits"]["base"], (-900, 900))
        self.assertLess(max(link.errors), 20)

    def test_short_blackout_continues_observed_clockwise_curve(self):
        clock, arm, link = self.rig(missing=((0.9, 1.12),))
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=7)
        self.assertTrue(result.aligned)
        blind = [v for t, *v in arm.commands if 0.95 <= t <= 1.18]
        self.assertTrue(blind)
        self.assertTrue(all(any(v) for v in blind))
        self.assertTrue(all(v[0] > 0 for v in blind))

    def test_long_blackout_stops_and_never_claims_stability(self):
        clock, arm, link = self.rig(missing=((0.9, 9),))
        with (
            patch.object(algorithms, "time", clock),
            self.assertRaises(VisualServoError),
        ):
            follow(arm, link, timeout=2)
        late = [v for t, *v in arm.commands if t > 1.5]
        self.assertTrue(late)
        self.assertTrue(all(v == [0, 0] for v in late))
        self.assertTrue(arm.closed)

    def test_startup_invisible_does_not_guess_a_direction(self):
        clock, arm, link = self.rig(missing=((0, 9),))
        with (
            patch.object(algorithms, "time", clock),
            self.assertRaises(VisualServoError),
        ):
            follow(arm, link, timeout=0.6)
        self.assertTrue(all(b == e == 0 for _, b, e in arm.commands))

    def test_camera_failure_stops_and_propagates(self):
        clock, arm, link = self.rig(fail_at=0.5)
        with (
            patch.object(algorithms, "time", clock),
            self.assertRaisesRegex(RuntimeError, "camera failed"),
        ):
            follow(arm, link)
        self.assertTrue(arm.closed)
        self.assertEqual(arm.velocity, (0, 0))

    def test_excess_rpm_is_rejected_before_opening_session(self):
        clock, arm, link = self.rig()
        with self.assertRaisesRegex(ValueError, "50"):
            follow(arm, link, rpm=51)
        self.assertFalse(hasattr(arm, "options"))

    def test_missing_capture_timestamp_fails_and_stops(self):
        clock, arm, link = self.rig()
        link.frame_fast = lambda **_: (sighting(4, 0, 0, stamp=None),)
        with (
            patch.object(algorithms, "time", clock),
            self.assertRaisesRegex((VisualServoError, ValueError), "时间戳|timestamp"),
        ):
            follow(arm, link)
        self.assertTrue(arm.closed)

    def test_keyboard_interrupt_stops_without_position_motion(self):
        clock, arm, link = self.rig()

        def interrupt(**kwargs):
            raise KeyboardInterrupt()

        link.frame_fast = interrupt
        with (
            patch.object(algorithms, "time", clock),
            self.assertRaises(KeyboardInterrupt),
        ):
            follow(arm, link)
        self.assertTrue(arm.closed)
        self.assertEqual(arm.velocity, (0, 0))

    def test_a_scene_check_exception_stops(self):
        clock, arm, link = self.rig(stop_at=0)

        def failed_scene():
            raise RuntimeError("scene failed")

        link.scene_stable = failed_scene
        with (
            patch.object(algorithms, "time", clock),
            self.assertRaisesRegex(RuntimeError, "scene failed"),
        ):
            follow(arm, link, timeout=6)
        self.assertTrue(arm.closed)
        self.assertEqual(arm.velocity, (0, 0))

    def test_real_arm_session_tracks_with_encoder_and_serial_exchange_latency(self):
        """算法接真实速度会话，仅电机和相机使用物理运动仿真。"""
        from modules.arm.drive import planar_velocity

        clock, plant, link = self.rig(missing=((1.0, 1.18),))
        starts = tuple(plant.position)

        class Motor:
            timeout = 0.1
            expect_ack = False

            def __init__(self, axis):
                self.axis = axis
                self.speeds = []

            def read_position(self):
                clock.sleep(0.004)
                plant._advance()
                return round(
                    (plant.position[self.axis] - starts[self.axis])
                    * 65536
                    / plant.pulses_per_revolution
                )

            def set_speed(self, direction, rpm, acceleration):
                plant._advance()
                speeds = list(plant.velocity)
                speeds[self.axis] = rpm if int(direction) == 0 else -rpm
                plant.velocity = tuple(speeds)
                self.speeds.append(rpm)
                clock.sleep(0.03)

            def stop(self):
                plant._advance()
                speeds = list(plant.velocity)
                speeds[self.axis] = 0
                plant.velocity = tuple(speeds)
                clock.sleep(0.03)

        motors = {"base": Motor(0), "extension": Motor(1), "lift": Motor(0)}
        arm = RobotArm(plant.config, motors=motors, sleep=clock.sleep)
        for axis, position in zip(("base", "extension"), starts):
            arm.turns_state.record(
                axis, round(position) - arm.turns_state.of(axis).pulses
            )
        with (
            patch.object(algorithms, "time", clock),
            patch.object(planar_velocity, "time", clock),
        ):
            result = follow(arm, link, timeout=8)
        self.assertTrue(result.aligned)
        self.assertEqual(plant.velocity, (0, 0))
        self.assertGreater(link.frames_while_moving, 5)
        self.assertLess(max(link.errors), 20)
        self.assertEqual(arm.turns_state.of("base").pulses, round(plant.position[0]))
        self.assertTrue(
            all(speed <= 50 for motor in motors.values() for speed in motor.speeds)
        )


class TrajectoryTests(unittest.TestCase):
    def observed_arc(self, omega):
        track = ClockwiseTrajectory()
        for i in range(9):
            stamp = i * 0.08
            angle = math.pi + omega * stamp
            track.observe(
                stamp, (220 + 35 * math.cos(angle), 50 + 35 * math.sin(angle))
            )
        return track

    def test_clockwise_material_can_require_counterclockwise_base_motion(self):
        track = self.observed_arc(-0.7)
        point, velocity = track.predict(0.8, invisible=True)
        geometry = PlanarGeometry(load_arm_config())
        _, speeds = geometry.joints(point, velocity, reference_base=0)
        self.assertLess(speeds[0], 0, "不能把转盘顺时针硬编码成底座顺时针")
        self.assertIsNone(track.predict(1.01, invisible=True))

    def test_counterclockwise_observations_are_not_used_for_blind_motion(self):
        track = self.observed_arc(0.7)
        self.assertIsNone(track.predict(0.8, invisible=True))

    def test_straight_or_noisy_observations_do_not_invent_a_circle(self):
        track = ClockwiseTrajectory()
        for i in range(8):
            track.observe(i * 0.08, (220 + i, 50))
        self.assertIsNone(track.predict(0.65, invisible=True))

    def test_geometry_round_trip_unwraps_base_across_multiple_revolutions(self):
        geometry = PlanarGeometry(load_arm_config())
        for base in (-22000, 0, 22000):
            point = geometry.world(base, 4000)
            joints, _ = geometry.joints(point, (0, 0), reference_base=base)
            self.assertAlmostEqual(joints[0], base)
            self.assertAlmostEqual(joints[1], 4000)


if __name__ == "__main__":
    unittest.main()

"""连续跟随的带时间、编码器和偏心转盘仿真，不连接硬件。"""

import math
import json
from dataclasses import replace
from pathlib import Path
import tempfile
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
    load_visual_servo_config,
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
        limit = self.options["max_rpm"]
        self.velocity = tuple(
            int(max(-limit, min(limit, v))) for v in (base, extension)
        )
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
    kwargs.setdefault("rpm", 50)
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

    def test_short_loss_does_not_restart_the_four_second_gate(self):
        clock, arm, link = self.rig(stop_at=0, missing=((3.7, 3.85),))
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=5)
        self.assertTrue(result.aligned)
        self.assertGreaterEqual(clock.now, 4)
        self.assertLess(clock.now, 4.8)

    def test_stability_duration_is_required_and_adjustable(self):
        for duration in (0.5, 1.0):
            with self.subTest(duration=duration):
                clock, arm, link = self.rig(stop_at=0)
                with patch.object(algorithms, "time", clock):
                    result = follow(
                        arm, link, timeout=2, min_follow_seconds=0,
                        stable_seconds=duration,
                    )
                self.assertTrue(result.aligned)
                self.assertGreaterEqual(clock.now, duration)
                self.assertLess(clock.now, duration + 0.2)

    def test_retry_can_skip_the_already_satisfied_four_second_gate(self):
        clock, arm, link = self.rig(stop_at=0)
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=2, min_follow_seconds=0)
        self.assertTrue(result.aligned)
        self.assertGreaterEqual(clock.now, 0.5)
        self.assertLess(clock.now, 1.2)

    def test_late_stop_still_requires_half_a_second(self):
        clock, arm, link = self.rig(stop_at=4.2)
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=6)
        self.assertTrue(result.aligned)
        self.assertGreaterEqual(clock.now, 4.7)
        self.assertLess(clock.now, 4.95)

    def test_stability_is_accumulated_while_the_arm_is_still_aligning(self):
        clock, arm, link = self.rig(stop_at=0)
        arm.position[0] = -1500
        aligned_at = []
        original = link.frame_fast

        def observe(**kwargs):
            result = original(**kwargs)
            if math.hypot(result[0].x_mm, result[0].y_mm) <= 6 and not aligned_at:
                aligned_at.append(clock.now)
            return result

        link.frame_fast = observe
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=2, min_follow_seconds=0)
        self.assertTrue(result.aligned)
        self.assertGreater(link.frames_while_moving, 2)
        self.assertLess(clock.now - aligned_at[0], 0.25)

    def test_missing_material_restarts_only_the_stability_window(self):
        clock, arm, link = self.rig(stop_at=0, missing=((0.3, 0.4),))
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=2, min_follow_seconds=0)
        self.assertTrue(result.aligned)
        self.assertGreaterEqual(clock.now, 0.9)
        self.assertLess(clock.now, 1.2)

    def test_braking_outside_tolerance_continues_alignment_before_handoff(self):
        clock, arm, link = self.rig(stop_at=0)
        original = arm.stop
        disturbed = []

        def stop():
            original()
            if not disturbed:
                arm.position[0] += 300
                disturbed.append(clock.now)

        arm.stop = stop
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=3, min_follow_seconds=0)
        self.assertTrue(result.aligned)
        self.assertGreater(clock.now, disturbed[0])
        self.assertLess(abs(JACOBIAN.displacement_for(arm.position[0], 0)[0]), 6)
        self.assertEqual(arm.velocity, (0, 0))

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

    def test_stopped_material_with_detection_jitter_settles_without_hunting(self):
        for noise in (0.4, 0.8, 1.2):
            with self.subTest(noise=noise):
                clock, arm, link = self.rig(stop_at=0)
                original = link.frame_fast

                def noisy_frame(**kwargs):
                    frame = original(**kwargs)
                    item = frame[0]
                    return (
                        replace(
                            item,
                            x_mm=item.x_mm + noise * math.sin(link.reads * 1.7),
                            y_mm=item.y_mm + noise * math.sin(link.reads * 2.3),
                        ),
                    )

                link.frame_fast = noisy_frame
                with patch.object(algorithms, "time", clock):
                    result = follow(arm, link, timeout=6, rpm=80)
                self.assertTrue(result.aligned)
                self.assertTrue(all(b == e == 0 for _, b, e in arm.commands))
                self.assertEqual(link.scene_checks, 0)

    def test_jitter_after_a_real_turntable_move_does_not_prevent_stopping(self):
        clock, arm, link = self.rig()
        original = link.frame_fast

        def noisy_frame(**kwargs):
            frame = original(**kwargs)
            item = frame[0]
            return (
                replace(
                    item,
                    x_mm=item.x_mm + 0.8 * math.sin(link.reads * 1.7),
                    y_mm=item.y_mm + 0.8 * math.sin(link.reads * 2.3),
                ),
            )

        link.frame_fast = noisy_frame
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=7, rpm=80)
        self.assertTrue(result.aligned)
        self.assertGreater(link.frames_while_moving, 5)
        self.assertTrue(all(b == e == 0 for t, b, e in arm.commands if t > 3.0))

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
            follow(arm, link, timeout=3)
        late = [v for t, *v in arm.commands if t > 2.5]
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
        with self.assertRaisesRegex(ValueError, "80"):
            follow(arm, link, rpm=81)
        self.assertFalse(hasattr(arm, "options"))

    def test_80_rpm_reaches_the_bounded_speed_session(self):
        clock, arm, link = self.rig()
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, rpm=80, timeout=7)
        self.assertTrue(result.aligned)
        self.assertEqual(arm.options["max_rpm"], 80)

    def test_occlusion_at_farthest_point_continues_into_retraction(self):
        """最远点前开始遮挡，回收动作主要发生在丢料0.35秒之后。"""
        clock, arm, link = self.rig(stop_at=2)
        geometry = PlanarGeometry(arm.config)
        radial_start = arm.position[1]

        def rear_arc(**kwargs):
            clock.sleep(0.04)
            arm._advance()
            stamp = clock.now
            angle = 0.65 * (0.75 - min(stamp, 2.0))
            x = link.origin_radius - 35 * math.cos(0.65 * 0.75) + 35 * math.cos(angle)
            y = -35 * math.sin(0.65 * 0.75) + 35 * math.sin(angle)
            desired, _ = geometry.joints((x, y), (0, 0), arm.position[0])
            offset = JACOBIAN.displacement_for(
                arm.position[0] - desired[0], arm.position[1] - desired[1]
            )
            clock.sleep(0.04)
            if 0.76 < stamp < 1.5:
                return ()
            return (sighting(4, *offset, stamp=stamp),)

        link.frame_fast = rear_arc
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, rpm=50, timeout=8)
        hidden_retract = [e for t, b, e in arm.commands if 1.1 < t < 1.5]
        self.assertTrue(hidden_retract)
        self.assertTrue(all(e < 0 for e in hidden_retract))
        self.assertLess(arm.position[1], radial_start)
        self.assertTrue(result.aligned)

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

    def test_a_scene_check_exception_does_not_block_world_stability(self):
        clock, arm, link = self.rig(stop_at=0)

        def failed_scene():
            raise RuntimeError("scene failed")

        link.scene_stable = failed_scene
        with patch.object(algorithms, "time", clock):
            result = follow(arm, link, timeout=6)
        self.assertTrue(result.aligned)
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
    def test_consistent_slow_motion_is_not_removed_as_detection_jitter(self):
        track = ClockwiseTrajectory()
        for i in range(8):
            stamp = i * 0.09
            track.observe(stamp, (220 + 2 * stamp, 50))
        self.assertFalse(track.stationary)
        self.assertAlmostEqual(track.velocity[0], 2)

    def test_unconfirmed_startup_noise_is_not_velocity_feedforward(self):
        track = ClockwiseTrajectory()
        for stamp, x in ((0, 220), (0.09, 221), (0.18, 219)):
            track.observe(stamp, (x, 50))
        self.assertEqual(track.velocity, (0, 0))
        self.assertFalse(track.stationary, "帧数不足时不能声称已停稳")

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
        self.assertIsNone(track.predict(2.15, invisible=True))

    def test_configured_prediction_window_expires_even_for_a_valid_circle(self):
        track = ClockwiseTrajectory(prediction_seconds=0.6)
        source = self.observed_arc(-0.7)
        for stamp, x, y in source.points:
            track.observe(stamp, (x, y))
        self.assertIsNotNone(track.predict(1.2, invisible=True))
        self.assertIsNone(track.predict(1.25, invisible=True))

    def test_prediction_duration_loads_from_motion_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eye_in_hand.json"
            path.write_text(
                json.dumps({"follow_prediction_seconds": 1.2}), encoding="utf-8"
            )
            config = load_visual_servo_config(path)
            self.assertEqual(config.follow_prediction_seconds, 1.2)
            for duration in (0, -1, float("inf"), float("nan")):
                with self.subTest(duration=duration), self.assertRaises(ValueError):
                    VisualServoConfig(follow_prediction_seconds=duration)

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

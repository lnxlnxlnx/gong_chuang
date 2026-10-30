"""用带取帧延迟和运动时长的假设备检查跟随节奏，不连接硬件。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.grasp.drive import eye_in_hand as algorithms
from modules.grasp.drive.visual_servo_model import RADIAL_AXIS
from modules.grasp.drive.visual_servo_motion import (
    RADIAL_MIN_MOTOR_DEGREES,
    _send_axis,
    clamp_radial_retract,
)
from modules.grasp.test.unit import test_eye_in_hand as fixtures
from modules.grasp.drive.visual_servo_model import (
    MaterialSelector,
    VisualServoError,
    VisualServoJacobian,
    VisualServoTarget,
)
from modules.grasp.test.unit.test_eye_in_hand import (
    FakeArm,
    FollowClock,
    JumpyDepthLink,
    sighting,
)
from modules.motor.drive.motion import estimate_move_seconds


class TimedArm:
    """按时间推进两轴位置；上一段没结束就重发同轴指令会直接失败。"""

    def __init__(self, clock, jacobian):
        self.clock = clock
        self.jacobian = jacobian
        self.config = SimpleNamespace(rpm=80)
        self.pulses_per_revolution = 3200
        self.moves = []

    @property
    def settled(self):
        return all(end <= self.clock.now + 1e-9 for _, _, end, _, _ in self.moves)

    @property
    def offset(self):
        pulses = {"base": 0.0, "radial": 0.0}
        for axis, start, end, count, _rpm in self.moves:
            progress = min(1.0, max(0.0, (self.clock.now - start) / (end - start)))
            pulses[axis] += count * progress
        dx, dy = self.jacobian.displacement_for(pulses["base"], pulses["radial"])
        return (-20.0 + dx, dy)

    def _send(self, axis, pulses, rpm, acceleration):
        if any(name == axis and end > self.clock.now + 1e-9
               for name, _start, end, _count, _rpm in self.moves):
            raise AssertionError("a new position command overwrote an unfinished move")
        duration = estimate_move_seconds(
            abs(pulses), rpm, self.pulses_per_revolution,
            acceleration, algorithms.FOLLOW_SETTLE_SECONDS,
        )
        self.moves.append(
            (axis, self.clock.now, self.clock.now + duration, pulses, rpm)
        )

    def rotate_cw(self, pulses, *, rpm, acceleration, **_kwargs):
        self._send("base", pulses, rpm, acceleration)

    def rotate_ccw(self, pulses, *, rpm, acceleration, **_kwargs):
        self._send("base", -pulses, rpm, acceleration)

    def move_forward(self, pulses, *, rpm, acceleration, **_kwargs):
        self._send("radial", pulses, rpm, acceleration)

    def move_backward(self, pulses, *, rpm, acceleration, **_kwargs):
        self._send("radial", -pulses, rpm, acceleration)


class TimedLink:
    """模拟 40 ms 的取帧和推理，保留采集时真实的机械臂位置。"""

    error = None

    def __init__(self, arm, clock, *, fail_after=None):
        self.arm = arm
        self.clock = clock
        self.frames_while_moving = 0
        self.reads = 0
        self.fail_after = fail_after

    def frame_fast(self, **_kwargs):
        self.reads += 1
        self.frames_while_moving += int(not self.arm.settled)
        if self.fail_after is not None and self.reads > self.fail_after:
            raise RuntimeError("camera disconnected")
        self.clock.sleep(0.04)
        x, y = self.arm.offset
        return (sighting(4, x, y, stamp=self.clock.now),)


class RadialFloorTests(unittest.TestCase):
    """伸缩回收不能顶到 0°：到了机械端位还发位置帧就是堵转（现场复现过）。"""

    class Arm:
        """只实现"读里程表 + 发伸缩帧"，够 :func:`_send_axis` 走完这一条路。"""

        def __init__(self, degrees: float) -> None:
            self.degrees = degrees
            self.pulses_per_revolution = 3200
            self.moves = []
            self.base_moves = []

        def read_abs_state(self, axis):
            self.last_axis = axis
            return SimpleNamespace(motor_degrees=self.degrees)

        def move_backward(self, pulses, **_kwargs):
            self.moves.append(-pulses)

        def move_forward(self, pulses, **_kwargs):
            self.moves.append(pulses)

        def rotate_cw(self, pulses, **_kwargs):
            self.base_moves.append(pulses)

        def rotate_ccw(self, pulses, **_kwargs):
            self.base_moves.append(-pulses)


    def test_a_long_retract_is_clamped_to_the_remaining_room(self) -> None:
        """90° 起步、下限 30°：−2000 脉冲（225°）只能走 60° = 533 脉冲。"""
        arm = self.Arm(degrees=90.0)
        self.assertEqual(clamp_radial_retract(arm, -2000), -533)

        _send_axis(arm, RADIAL_AXIS, -2000)
        self.assertEqual(arm.moves, [-533])
        self.assertEqual(arm.last_axis, "extension")

    def test_nothing_is_sent_once_the_extension_is_at_the_floor(self) -> None:
        arm = self.Arm(degrees=RADIAL_MIN_MOTOR_DEGREES)
        self.assertEqual(_send_axis(arm, RADIAL_AXIS, -2000), 0.0)
        self.assertEqual(arm.moves, [])

    def test_extending_is_never_clamped(self) -> None:
        arm = self.Arm(degrees=40.0)
        _send_axis(arm, RADIAL_AXIS, 500)
        self.assertEqual(arm.moves, [500])

    def test_the_blind_search_waits_for_what_it_really_sends(self) -> None:
        """回收被夹掉之后不能还按 2000 脉冲等：那是白丢一次取帧的机会。"""
        arm = self.Arm(degrees=RADIAL_MIN_MOTOR_DEGREES)
        clock = FollowClock()
        with patch.object(algorithms, "time", clock):
            waited = algorithms._blind_search_step(
                arm,
                base_pulses=20,
                radial_pulses=-2000,
                rpm=40,
                pulses_per_revolution=3200,
                acceleration=255,
                inter_frame_seconds=0.03,
            )
        self.assertEqual(arm.moves, [])            # 伸缩一动不动
        self.assertLess(waited, 0.2)               # 只按底座那 20 脉冲等
        self.assertLess(clock.now, 0.2)


class SmoothFollowTests(unittest.TestCase):
    JACOBIAN = VisualServoJacobian(
        base_dx=0.08, base_dy=0.0, radial_dx=0.0, radial_dy=0.02,
    )

    def follow(self, arm, link, **kwargs):
        return algorithms.follow_material_xy(
            arm, link, MaterialSelector(), VisualServoTarget(0.0, 0.0),
            self.JACOBIAN, fixtures.SingleStepTests.config(), **kwargs,
        )

    def test_default_step_approaches_without_crossing_a_stationary_target(self):
        arm = FakeArm(self.JACOBIAN)
        arm.offset = [5.0, 0.0]
        with patch.object(algorithms, "time", FollowClock()):
            algorithms.follow_x_y(
                arm, JumpyDepthLink(arm=arm), MaterialSelector(),
                VisualServoTarget(0.0, 0.0), self.JACOBIAN,
                fixtures.SingleStepTests.config(),
            )
        self.assertGreaterEqual(arm.offset[0], 0.0)
        self.assertLess(arm.offset[0], 5.0)

    def test_an_in_tolerance_target_does_not_trigger_corrections(self):
        arm = FakeArm(self.JACOBIAN)
        arm.offset = [2.0, 0.0]
        with patch.object(algorithms, "time", FollowClock()):
            result = self.follow(arm, JumpyDepthLink(arm=arm), timeout=5.0)
        self.assertTrue(result.aligned)
        self.assertEqual(arm.calls, [])

    def test_follow_uses_slow_short_moves_and_reads_during_motion(self):
        clock = FollowClock()
        arm = TimedArm(clock, self.JACOBIAN)
        link = TimedLink(arm, clock)
        with patch.object(algorithms, "time", clock):
            result = self.follow(arm, link, timeout=5.0)
        self.assertTrue(result.aligned)
        self.assertTrue(arm.settled)
        self.assertGreaterEqual(len(arm.moves), 3)
        self.assertGreater(link.frames_while_moving, 0)
        for _axis, start, end, _pulses, rpm in arm.moves:
            self.assertLessEqual(rpm, 40)
            self.assertLessEqual(end - start, 0.105)

    def test_camera_failure_waits_for_the_last_position_command(self):
        clock = FollowClock()
        arm = TimedArm(clock, self.JACOBIAN)
        link = TimedLink(arm, clock, fail_after=1)
        with patch.object(algorithms, "time", clock):
            with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
                self.follow(arm, link, timeout=3.0)
        self.assertTrue(arm.settled)

    def test_timeout_waits_for_the_last_position_command(self):
        clock = FollowClock()
        arm = TimedArm(clock, self.JACOBIAN)
        link = TimedLink(arm, clock)
        with patch.object(algorithms, "time", clock):
            with self.assertRaises(VisualServoError):
                self.follow(arm, link, timeout=0.05)
        self.assertTrue(arm.settled)

    def test_four_seconds_start_at_the_first_visible_material(self):
        clock = FollowClock()
        arm = FakeArm(self.JACOBIAN)
        arm.offset = [0.0, 0.0]

        class Link:
            first_seen_at = None

            def frame_fast(self, **_kwargs):
                clock.sleep(0.04)
                if clock.now < 1.0:
                    return ()
                if self.first_seen_at is None:
                    self.first_seen_at = clock.now
                return (sighting(4, 0.0, 0.0, stamp=clock.now),)

            def scene_stable(self):
                return True

        link = Link()
        with patch.object(algorithms, "time", clock):
            result = self.follow(
                arm, link, timeout=7.0, lost_base_pulses=0, lost_retract_pulses=0,
            )
        self.assertTrue(result.aligned)
        self.assertGreaterEqual(clock.now - link.first_seen_at, 4.0)

    def test_a_lost_material_is_not_chased_with_any_motion(self) -> None:
        """默认盲扫量都是 0：找不到料就**原地不动**，一个底座/伸缩帧都不发。"""

        class NeverSeenLink:
            error = None

            def frame(self, **_kwargs):
                return ()

        arm = FakeArm(self.JACOBIAN)
        with patch.object(algorithms, "time", FollowClock()), \
                self.assertRaisesRegex(VisualServoError, "跟了"):
            self.follow(arm, NeverSeenLink(), timeout=1.0)
        self.assertEqual(arm.calls, [])

    def test_a_moving_scene_cannot_finish_after_four_seconds(self):
        clock = FollowClock()
        arm = FakeArm(self.JACOBIAN)
        arm.offset = [0.0, 0.0]
        link = JumpyDepthLink(arm=arm)
        link.scene_stable = lambda: False
        with patch.object(algorithms, "time", clock):
            with self.assertRaises(VisualServoError):
                self.follow(arm, link, timeout=4.5)


if __name__ == "__main__":
    unittest.main()

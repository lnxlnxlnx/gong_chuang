"""Bounded planar speed sessions use measured motion and expire independently."""

import threading
import unittest
from unittest.mock import patch

from modules.arm.runtime import RobotArm
from modules.arm.test.unit.test_robot_arm import fixture_config
from modules.motor.drive.emm42 import Direction
from modules.motor.drive.emm42 import Emm42Driver
from modules.motor.test.unit.test_emm42 import FakeSerial


class EncoderMotor:
    def __init__(self):
        self.position = 0
        self.speeds = []
        self.stops = 0
        self.stopped = threading.Event()
        self.read_error = None
        self.stop_error = None
        self.moves = []

    def read_position(self, *, attempts=3):
        if self.read_error:
            raise self.read_error
        return self.position

    def set_speed(self, direction, rpm, acceleration):
        self.speeds.append((direction, rpm, acceleration))

    def stop(self):
        self.stops += 1
        self.stopped.set()
        if self.stop_error:
            raise self.stop_error

    def move_relative_pulses(self, pulses, rpm, acceleration, **kwargs):
        self.moves.append(pulses)


class PlanarVelocityTests(unittest.TestCase):
    def setUp(self):
        self.motors = {axis: EncoderMotor() for axis in ("base", "lift", "extension")}
        self.arm = RobotArm(fixture_config(), motors=self.motors, sleep=lambda _: None)
        self.session = None

    def tearDown(self):
        if self.session is not None:
            for motor in self.motors.values():
                motor.read_error = motor.stop_error = None
            self.session.close()

    def start(self, **overrides):
        settings = dict(
            max_rpm=8, motor_limits={"base": (-720, 720), "extension": (-720, 720)}
        )
        settings.update(overrides)
        self.assertTrue(
            hasattr(self.arm, "planar_velocity"),
            "RobotArm needs bounded planar_velocity",
        )
        self.session = self.arm.planar_velocity(**settings)
        return self.session

    def test_required_bounds_are_validated_before_reading_or_motion(self):
        for limits in (
            {"base": (-10, 10)},
            {"base": (10, -10), "extension": (0, 10)},
            {"base": (0, float("inf")), "extension": (0, 10)},
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.start(motor_limits=limits)
        self.assertFalse(self.motors["base"].speeds)

    def test_signed_speed_is_capped_and_unchanged_speed_does_not_restart(self):
        session = self.start()
        self.assertEqual(session.set_velocity(99, -99), (8, -8))
        self.assertEqual(session.set_velocity(99, -99), (8, -8))
        self.assertEqual(self.motors["base"].speeds, [(Direction.CW, 8, 200)])
        self.assertEqual(self.motors["extension"].speeds, [(Direction.CCW, 8, 200)])
        self.assertEqual(self.motors["lift"].stops, 0)

    def test_zero_stops_only_the_moving_axis(self):
        session = self.start()
        session.set_velocity(3, 4)
        session.set_velocity(0, 4)
        self.assertEqual(self.motors["base"].stops, 1)
        self.assertEqual(self.motors["extension"].stops, 0)

    def test_encoder_delta_reconciles_absolute_odometry_and_next_position_move(self):
        self.arm.turns_state.record("base", 400)
        self.motors["base"].position = 20000
        session = self.start()
        self.motors["base"].position += 16384
        snapshot = session.sample()
        self.assertEqual(snapshot.base_pulses, 1200)
        self.assertEqual(self.arm.turns_state.of("base").pulses, 1200)
        session.close()
        self.arm.move_axis_to_motor_pulses("base", 1250)
        self.assertEqual(self.motors["base"].moves, [50])

    def test_history_interpolates_measured_snapshots_and_rejects_extrapolation(self):
        with patch(
            "modules.arm.drive.planar_velocity.time.monotonic", return_value=10.0
        ) as clock:
            session = self.start()
            clock.return_value = 11.0
            before = session.sample()
            self.motors["base"].position += 32768
            clock.return_value = 13.0
            after = session.sample()
        midpoint = (before.stamp + after.stamp) / 2
        base, extension = session.position_at(midpoint)
        self.assertAlmostEqual(base, 800, delta=0.1)
        self.assertEqual(extension, 0)
        with self.assertRaises(ValueError):
            session.position_at(before.stamp - 10)

    def test_sequential_encoder_reads_keep_individual_measurement_times(self):
        with patch(
            "modules.arm.drive.planar_velocity.time.monotonic", return_value=10.0
        ) as clock:
            session = self.start()
            clock.return_value = 11.0
            session.sample()

            def delayed_extension():
                clock.return_value = 13.0
                return 0

            self.motors["extension"].read_position = delayed_extension
            self.motors["base"].position += 32768
            clock.return_value = 12.0
            session.sample()
            base, _ = session.position_at(12.0)
            self.assertEqual(base, 1600, "底座12s的测量不能错标为伸缩读完的13s")

    def test_reversal_brakes_before_restarting_in_the_opposite_direction(self):
        session = self.start(
            max_rpm=50,
            motor_limits={"base": (-900, 900), "extension": (-720, 720)},
        )
        session.set_velocity(50, 0)
        self.motors["base"].position = round(850 / 360 * 65536)
        session.sample()
        events = []
        motor = self.motors["base"]
        old_stop, old_speed = motor.stop, motor.set_speed

        def stop():
            events.append("brake")
            old_stop()

        def speed(*args):
            events.append("reverse")
            old_speed(*args)

        motor.stop, motor.set_speed = stop, speed
        session.set_velocity(-50, 0)
        self.assertEqual(events[:2], ["brake", "reverse"])

    def test_speed_is_reduced_near_absolute_limit_and_outward_motion_stops(self):
        session = self.start(motor_limits={"base": (-2, 2), "extension": (-720, 720)})
        actual, _ = session.set_velocity(8, 0)
        self.assertEqual(actual, 0)
        self.assertFalse(self.motors["base"].speeds)

    def test_limit_is_unwrapped_and_initial_position_outside_is_rejected(self):
        self.arm.turns_state.record("base", 6400)
        with self.assertRaises(ValueError):
            self.start(motor_limits={"base": (-360, 360), "extension": (-720, 720)})

    def test_watchdog_stops_both_axes_without_a_caller_and_rejects_refresh(self):
        session = self.start(command_timeout=0.05)
        session.set_velocity(3, 4)
        self.assertTrue(self.motors["base"].stopped.wait(0.5))
        self.assertTrue(self.motors["extension"].stopped.wait(0.5))
        with self.assertRaisesRegex(RuntimeError, "expir|timeout"):
            session.set_velocity(3, 4)

    def test_encoder_failure_stops_both_and_invalidates_absolute_position(self):
        session = self.start()
        session.set_velocity(3, 4)
        self.motors["base"].read_error = RuntimeError("encoder disconnected")
        with self.assertRaisesRegex(RuntimeError, "encoder"):
            session.sample()
        self.assertEqual(self.motors["base"].stops, 1)
        self.assertEqual(self.motors["extension"].stops, 1)
        session.close()
        with self.assertRaisesRegex(RuntimeError, "position|位置"):
            self.arm.move_axis_to_motor_pulses("base", 100)

    def test_transient_encoder_errors_recover_without_invalidating_position(self):
        serial = FakeSerial(read_replies={0x36: [
            bytes.fromhex("05 36 00 00 00 00 00 6B"),
            bytes.fromhex("05 9B 00 01 F7 0F 6B"), b"",
            bytes.fromhex("05 36 00 00 00 40 00 6B"),
        ]})
        self.motors["base"] = Emm42Driver(
            "test", 5, serial_port=serial, expect_ack=False, sleep=lambda _: None,
        )
        self.arm = RobotArm(fixture_config(), motors=self.motors, sleep=lambda _: None)
        session = self.start()
        with self.assertLogs("modules.motor.drive.emm42", level="WARNING"):
            snapshot = session.sample()
        self.assertEqual(snapshot.base_pulses, 800)
        self.assertEqual(self.arm.turns_state.of("base").pulses, 800)
        self.assertEqual(self.motors["extension"].stops, 0)
        session.close()
        self.arm.move_axis_to_motor_pulses("base", 850)
        self.assertEqual(serial.writes[-1][1], 0xFD)

    def test_persistent_encoder_errors_still_brake_both_and_block_position_moves(self):
        serial = FakeSerial(read_replies={0x36: [
            bytes.fromhex("05 36 00 00 00 00 00 6B"),
            bytes.fromhex("05 9B 00 01 F7 0F 6B"),
        ]})
        self.motors["base"] = Emm42Driver(
            "test", 5, serial_port=serial, expect_ack=False, sleep=lambda _: None,
        )
        self.arm = RobotArm(fixture_config(), motors=self.motors, sleep=lambda _: None)
        session = self.start()
        with self.assertLogs("modules.motor.drive.emm42", level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "3 consecutive"):
                session.sample()
        self.assertIn(bytes.fromhex("05 FE 98 00 6B"), serial.writes)
        self.assertEqual(self.motors["extension"].stops, 1)
        session.close()
        with self.assertRaisesRegex(RuntimeError, "position"):
            self.arm.move_axis_to_motor_pulses("base", 100)

    def test_first_stop_failure_does_not_skip_second_axis(self):
        session = self.start()
        session.set_velocity(3, 4)
        self.motors["base"].stop_error = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            session.stop()
        self.assertEqual(self.motors["extension"].stops, 1)
        with self.assertRaises(RuntimeError):
            session.set_velocity(3, 4)

    def test_position_motion_and_second_session_are_rejected_while_active(self):
        self.start()
        with self.assertRaises(RuntimeError):
            self.arm.move_up(10)
        with self.assertRaises(RuntimeError):
            self.arm.planar_velocity(
                max_rpm=8, motor_limits={"base": (-720, 720), "extension": (-720, 720)}
            )

    def test_context_stops_on_keyboard_interrupt(self):
        session = self.start()
        with self.assertRaises(KeyboardInterrupt):
            with session:
                session.set_velocity(3, 4)
                raise KeyboardInterrupt()
        self.assertEqual(self.motors["base"].stops, 1)
        self.assertEqual(self.motors["extension"].stops, 1)

    def test_encoder_crossing_limit_fails_and_stops_both(self):
        session = self.start()
        session.set_velocity(3, 4)
        self.motors["base"].position = 3 * 65536
        with self.assertRaisesRegex(RuntimeError, "limit"):
            session.sample()
        self.assertEqual(self.motors["base"].stops, 1)
        self.assertEqual(self.motors["extension"].stops, 1)

    def test_speed_exchange_past_lease_brakes_before_commanding_other_axis(self):
        with patch(
            "modules.arm.drive.planar_velocity.time.monotonic", return_value=10.0
        ) as clock:
            session = self.start()
            original = self.motors["base"].set_speed

            def delayed_speed(*args):
                original(*args)
                clock.return_value = 11.0

            self.motors["base"].set_speed = delayed_speed
            with self.assertRaisesRegex(RuntimeError, "expir"):
                session.set_velocity(3, 4)
            self.assertFalse(self.motors["extension"].speeds)
            self.assertEqual(self.motors["base"].stops, 1)
            self.assertEqual(self.motors["extension"].stops, 1)

    def test_unacknowledged_axes_reserve_one_failed_read_before_braking(self):
        self.arm.turns_state.record("extension", round(510 * 3200 / 360))
        for motor in self.motors.values():
            motor.expect_ack = False
            motor.timeout = 0.1
        session = self.start(
            max_rpm=50, motor_limits={"base": (-900, 900), "extension": (30, 990)}
        )
        # 运动中只读取一次；剩余行程足以覆盖 50 rpm 的命令租期和故障刹停。
        self.assertEqual(session.set_velocity(50, 50), (50, 50))

    def test_extension_can_catch_up_outward_without_reserving_read_retries(self):
        self.arm.turns_state.record("extension", round(800 * 3200 / 360))
        for motor in self.motors.values():
            motor.expect_ack = False
            motor.timeout = 0.1
        session = self.start(
            max_rpm=90, motor_limits={"base": (-900, 900), "extension": (30, 990)},
        )
        _, speed = session.set_velocity(0, 90)
        self.assertGreaterEqual(speed, 25)
        self.assertLessEqual(speed, 90)

    def test_an_encoder_failure_while_moving_brakes_without_retrying(self):
        serial = FakeSerial(read_replies={0x36: [
            bytes.fromhex("05 36 00 00 00 00 00 6B"),
            bytes.fromhex("05 9B 00 01 F7 0F 6B"),
            bytes.fromhex("05 36 00 00 00 40 00 6B"),
        ]})
        self.motors["base"] = Emm42Driver(
            "test", 5, serial_port=serial, expect_ack=False, sleep=lambda _: None,
        )
        self.arm = RobotArm(fixture_config(), motors=self.motors, sleep=lambda _: None)
        session = self.start()
        session.set_velocity(3, 4)
        with self.assertRaisesRegex(RuntimeError, "1 consecutive"):
            session.sample()
        reads = [frame for frame in serial.writes if frame[1] == 0x36]
        self.assertEqual(len(reads), 2)
        self.assertIn(bytes.fromhex("05 FE 98 00 6B"), serial.writes)
        self.assertEqual(self.motors["extension"].stops, 1)

    def test_reducing_speed_reserves_room_for_the_previous_moving_speed(self):
        session = self.start(
            max_rpm=50, motor_limits={"base": (-900, 900), "extension": (-720, 720)}
        )
        session.set_velocity(50, 0)
        self.motors["base"].position = round(850 / 360 * 65536)
        session.sample()
        actual, _ = session.set_velocity(50, 0)
        self.assertEqual(actual, 0)
        self.assertEqual(self.motors["base"].stops, 1)

    def test_watchdog_never_interleaves_a_stop_with_an_encoder_exchange(self):
        session = self.start(command_timeout=0.05)
        session.set_velocity(3, 4)
        entered, release = threading.Event(), threading.Event()
        errors = []

        def blocked_read(**_kwargs):
            entered.set()
            if not release.wait(0.5):
                raise RuntimeError("test read was not released")
            return 0

        def sample_in_thread():
            try:
                session.sample()
            except BaseException as error:
                errors.append(error)

        self.motors["base"].read_position = blocked_read
        worker = threading.Thread(target=sample_in_thread)
        worker.start()
        try:
            self.assertTrue(entered.wait(0.2))
            self.assertFalse(self.motors["base"].stopped.wait(0.08))
        finally:
            release.set()
            worker.join(0.5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertTrue(self.motors["extension"].stopped.wait(0.5))

    def test_real_driver_uses_only_planar_f6_frames_and_its_existing_silence(self):
        class EncoderSerial:
            is_open = True

            def __init__(self):
                self.buffer = bytearray()
                self.writes = []

            def write(self, frame):
                self.writes.append(bytes(frame))
                if frame[1] == 0x36:
                    self.buffer.extend(bytes((frame[0], 0x36, 0, 0, 0, 0, 0, 0x6B)))
                return len(frame)

            def read(self, size):
                data = bytes(self.buffer[:size])
                del self.buffer[:size]
                return data

            def flush(self):
                pass

            def reset_input_buffer(self):
                self.buffer.clear()

        serial = EncoderSerial()
        waits = []
        motors = {
            axis: Emm42Driver(
                "test",
                motor_id,
                serial_port=serial,
                expect_ack=False,
                sleep=waits.append,
            )
            for axis, motor_id in (("base", 5), ("lift", 6), ("extension", 7))
        }
        self.arm = RobotArm(fixture_config(), motors=motors, sleep=waits.append)
        session = self.start()
        serial.writes.clear()
        session.set_velocity(3, -4)
        session.set_velocity(3, -4)
        self.assertEqual(
            serial.writes,
            [
                bytes.fromhex("05 F6 00 00 03 C8 00 6B"),
                bytes.fromhex("07 F6 01 00 04 C8 00 6B"),
            ],
        )
        self.assertEqual(waits, [0.03, 0.03])


if __name__ == "__main__":
    unittest.main()

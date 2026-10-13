"""转盘应用层：工位分度、回零、读回到位判据与生命周期。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from modules.servo.config import TurntableConfig
from modules.servo.drive.bus_servo import BusServoConfig
from modules.servo.drive.turntable import (
    DEFAULT_POSITION_TOLERANCE,
    DEGREES_PER_UNIT,
    TurntableServo,
)

CONFIG = TurntableConfig(
    servo_id=2,
    duration_ms=1500,
    stations={"home": 0, "station_2": 500, "station_3": 1000},
)
BUS = BusServoConfig(port="FAKE", baudrate=115200, servo_id=1)


def make_turntable(positions, **overrides) -> tuple[TurntableServo, Mock, list]:
    servo = Mock()
    servo.read_position.side_effect = list(positions)
    slept: list = []
    settings = dict(
        config=CONFIG,
        bus_config=BUS,
        servo=servo,
        sleep=slept.append,
        settle_seconds=0.2,
    )
    settings.update(overrides)
    return TurntableServo("", **settings), servo, slept


class TurntableLifecycleTests(unittest.TestCase):
    def test_initialize_opens_the_bus_and_holds_torque(self) -> None:
        turntable, servo, _ = make_turntable([])

        turntable.initialize()

        servo.initialize.assert_called_once_with()
        servo.set_torque.assert_called_once_with(True)

    def test_close_only_closes_the_bus_and_leaves_the_disc_where_it_is(self) -> None:
        turntable, servo, _ = make_turntable([])

        turntable.close()

        servo.close.assert_called_once_with()
        servo.set_position.assert_not_called()

    def test_the_context_manager_initializes_and_closes(self) -> None:
        turntable, servo, _ = make_turntable([])

        with turntable:
            pass

        servo.initialize.assert_called_once_with()
        servo.close.assert_called_once_with()

    def test_injected_servo_means_no_serial_port_is_opened(self) -> None:
        turntable, servo, _ = make_turntable([])
        self.assertEqual(turntable.port, "FAKE")
        servo.initialize.assert_not_called()

    def test_the_station_table_comes_from_the_config(self) -> None:
        turntable, _, _ = make_turntable([])
        self.assertEqual(turntable.stations, CONFIG.stations)


class TurntableStationTests(unittest.TestCase):
    def test_go_home_uses_the_home_station(self) -> None:
        turntable, servo, slept = make_turntable([0])

        report = turntable.go_home()

        servo.set_position.assert_called_once_with(0, 1500)
        self.assertEqual(
            (report.command, report.target, report.position, report.reached),
            ("home", 0, 0, True),
        )
        self.assertEqual(slept, [1.7])

    def test_every_station_uses_its_configured_position(self) -> None:
        turntable, servo, _ = make_turntable([500, 1000])

        turntable.go_to_station("station_2")
        turntable.go_to_station("station_3")

        self.assertEqual(
            [call.args for call in servo.set_position.call_args_list],
            [(500, 1500), (1000, 1500)],
        )

    def test_an_unknown_station_is_rejected_before_any_frame_is_sent(self) -> None:
        turntable, servo, _ = make_turntable([])

        with self.assertRaisesRegex(ValueError, "station_9"):
            turntable.go_to_station("station_9")

        servo.set_position.assert_not_called()
        servo.read_position.assert_not_called()

    def test_go_to_drives_a_raw_position_for_calibration(self) -> None:
        """标定要试的正是"还不是工位"的位置，所以原始入口不查工位表。"""
        turntable, servo, _ = make_turntable([250])

        report = turntable.go_to(250)

        servo.set_position.assert_called_once_with(250, 1500)
        self.assertEqual((report.command, report.target, report.reached), ("set", 250, True))

    def test_the_duration_may_be_overridden_for_one_move(self) -> None:
        turntable, servo, slept = make_turntable([500])

        turntable.go_to_station("station_2", duration_ms=2500)

        servo.set_position.assert_called_once_with(500, 2500)
        self.assertEqual(slept, [2.7])

    def test_the_settle_margin_is_added_on_top_of_the_configured_duration(self) -> None:
        turntable, _, slept = make_turntable([0], settle_seconds=0.5)

        turntable.go_home()

        self.assertEqual(slept, [2.0])

    def test_wait_false_sends_the_command_without_sleeping_or_claiming_arrival(self) -> None:
        turntable, servo, slept = make_turntable([480])

        report = turntable.go_to_station("station_2", wait=False)

        servo.set_position.assert_called_once_with(500, 1500)
        self.assertEqual(slept, [])
        self.assertFalse(report.reached)
        # 读回的是"命令前"的位置，如实带回来而不是丢掉。
        self.assertEqual(report.position, 480)


class TurntableArrivalTests(unittest.TestCase):
    def test_a_readback_that_misses_the_target_is_not_reached(self) -> None:
        """转盘被挡住时读回值对不上目标，这是信号不是异常——工位之间差 120°，
        差 100 个单位说明它根本没转到位。"""
        turntable, _, _ = make_turntable([900])

        report = turntable.go_to_station("station_3")

        self.assertFalse(report.reached)
        self.assertEqual(report.error, -100)
        self.assertIn("未到位", report.summary())

    def test_the_tolerance_is_inclusive_at_its_boundary(self) -> None:
        turntable, _, _ = make_turntable([1000 - DEFAULT_POSITION_TOLERANCE])
        self.assertTrue(turntable.go_to_station("station_3").reached)

        turntable, _, _ = make_turntable([1000 - DEFAULT_POSITION_TOLERANCE - 1])
        self.assertFalse(turntable.go_to_station("station_3").reached)

    def test_at_station_is_a_readback_judgement_not_a_command_log(self) -> None:
        turntable, servo, _ = make_turntable([500, 460])

        self.assertTrue(turntable.at_station("station_2"))
        self.assertFalse(turntable.at_station("station_2"))

        self.assertEqual(servo.read_position.call_count, 2)
        servo.set_position.assert_not_called()

    def test_one_call_may_widen_the_tolerance(self) -> None:
        turntable, _, _ = make_turntable([560, 560])

        self.assertFalse(turntable.at_station("station_2"))
        self.assertTrue(turntable.at_station("station_2", tolerance=60))

    def test_is_at_compares_a_raw_position(self) -> None:
        turntable, _, _ = make_turntable([250])

        self.assertTrue(turntable.is_at(250))

    def test_a_negative_tolerance_is_rejected_at_the_call(self) -> None:
        turntable, _, _ = make_turntable([500])

        with self.assertRaises(ValueError):
            turntable.at_station("station_2", tolerance=-1)


class TurntableReadoutTests(unittest.TestCase):
    def test_the_report_converts_positions_to_nominal_degrees(self) -> None:
        turntable, _, _ = make_turntable([500])

        report = turntable.go_to_station("station_2")

        # 0..1000 对应 0..240°：三个工位正好相隔 120°。
        self.assertAlmostEqual(DEGREES_PER_UNIT, 0.24, places=9)
        self.assertAlmostEqual(report.degrees, 120.0, places=9)
        self.assertIn("120.0°", report.summary())

    def test_read_stop_and_torque_are_passed_through(self) -> None:
        turntable, servo, _ = make_turntable([377])

        self.assertEqual(turntable.read_position(), 377)
        turntable.stop()
        turntable.set_torque(False)

        servo.read_position.assert_called_once_with()
        servo.stop.assert_called_once_with()
        servo.set_torque.assert_called_once_with(False)


class TurntableValidationTests(unittest.TestCase):
    def test_negative_tolerance_and_settle_are_rejected(self) -> None:
        for overrides in ({"position_tolerance": -1}, {"settle_seconds": -0.1}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_turntable([], **overrides)


if __name__ == "__main__":
    unittest.main()

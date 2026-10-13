"""转盘真机脚本：菜单编号、工位顺序、逐项确认，以及串口接线。"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from modules.servo.config import TurntableConfig
from modules.servo.drive.bus_servo import BusServoConfig
from modules.servo.drive.turntable import TurntableServo
from modules.servo.test.manual import turntable as turntable_manual
from modules.servo.test.manual.turntable import run_manual, station_order

CONFIG = TurntableConfig(
    servo_id=2,
    duration_ms=1500,
    stations={"home": 0, "station_2": 500, "station_3": 1000},
)
BUS = BusServoConfig(port="COM14", baudrate=115200, servo_id=1)


def make_turntable(readings) -> tuple[TurntableServo, Mock]:
    servo = Mock()
    servo.read_position.side_effect = list(readings)
    turntable = TurntableServo(
        "", config=CONFIG, bus_config=BUS, servo=servo, sleep=lambda _seconds: None
    )
    return turntable, servo


class TurntableMenuTests(unittest.TestCase):
    def test_the_menu_number_does_not_collide_with_an_existing_entry(self) -> None:
        number, label = turntable_manual.MENU_ENTRY
        self.assertEqual((number, label), (20, "转盘测试"))


class StationOrderTests(unittest.TestCase):
    def test_home_comes_first_so_the_unknown_start_orientation_is_settled(self) -> None:
        self.assertEqual(station_order(CONFIG), ("home", "station_2", "station_3"))

    def test_home_leads_even_when_the_file_lists_it_last(self) -> None:
        config = TurntableConfig(
            servo_id=2,
            duration_ms=1500,
            stations={"station_3": 1000, "station_2": 500, "home": 0},
        )
        self.assertEqual(station_order(config), ("home", "station_3", "station_2"))


class ManualSequenceTests(unittest.TestCase):
    def test_every_station_needs_its_own_confirmation(self) -> None:
        turntable, servo = make_turntable([0, 0, 1000])
        answers = iter(["no", "", ""])

        run_manual(turntable, CONFIG, prompt=lambda _: next(answers))

        # 第一项被跳过，后两项按回车执行：第一次不会发送任何位置指令。
        self.assertEqual(
            [call.args for call in servo.set_position.call_args_list],
            [(500, 1500), (1000, 1500)],
        )

    def test_pressing_enter_executes_instead_of_skipping(self) -> None:
        """这次改动的核心：以前空输入（回车）等于跳过，现在等于执行。"""
        turntable, servo = make_turntable([0, 0])
        answers = iter(["", "q"])

        run_manual(turntable, CONFIG, prompt=lambda _: next(answers))

        self.assertEqual(
            [call.args for call in servo.set_position.call_args_list], [(0, 1500)]
        )

    def test_q_stops_the_remaining_stations(self) -> None:
        turntable, servo = make_turntable([0, 0])
        answers = iter(["", "q"])

        with redirect_stdout(io.StringIO()) as output:
            run_manual(turntable, CONFIG, prompt=lambda _: next(answers))

        self.assertEqual(
            [call.args for call in servo.set_position.call_args_list], [(0, 1500)]
        )
        self.assertIn("已结束后续转盘动作", output.getvalue())

    def test_the_stop_key_is_case_insensitive(self) -> None:
        turntable, servo = make_turntable([0])

        run_manual(turntable, CONFIG, prompt=lambda _: " Q ")

        servo.set_position.assert_not_called()

    def test_a_letter_skips_only_that_station(self) -> None:
        """跳过一项和结束全部是两件事，别把打错的字变成整段中止。"""
        turntable, servo = make_turntable([0, 0, 1000])
        answers = iter(["x", "", ""])

        run_manual(turntable, CONFIG, prompt=lambda _: next(answers))

        self.assertEqual(
            [call.args for call in servo.set_position.call_args_list],
            [(500, 1500), (1000, 1500)],
        )

    def test_confirmed_stations_are_walked_in_order(self) -> None:
        turntable, servo = make_turntable([1000, 0, 500, 1000])

        with redirect_stdout(io.StringIO()) as output:
            run_manual(turntable, CONFIG, prompt=lambda _: "")

        self.assertEqual(
            [call.args for call in servo.set_position.call_args_list],
            [(0, 1500), (500, 1500), (1000, 1500)],
        )
        self.assertIn("转盘home", output.getvalue())
        self.assertIn("转盘station_3", output.getvalue())

    def test_the_current_position_is_read_before_anything_moves(self) -> None:
        turntable, servo = make_turntable([250])

        run_manual(turntable, CONFIG, prompt=lambda _: "no")

        self.assertEqual(servo.read_position.call_count, 1)
        servo.set_position.assert_not_called()

    def test_a_station_that_did_not_arrive_is_reported_not_hidden(self) -> None:
        # 起始 250 → home 到位 0 → station_2 卡在 40（差 460）→ station_3 到位 1000。
        turntable, _ = make_turntable([250, 0, 40, 1000])

        with redirect_stdout(io.StringIO()) as output:
            run_manual(turntable, CONFIG, prompt=lambda _: "")

        self.assertIn("未到位", output.getvalue())

    def test_a_servo_that_does_not_answer_stops_the_script(self) -> None:
        turntable, servo = make_turntable([])
        servo.read_position.side_effect = TimeoutError("no response")

        with self.assertRaises(TimeoutError):
            run_manual(turntable, CONFIG, prompt=lambda _: "")

        servo.set_position.assert_not_called()


class TurntableWiringTests(unittest.TestCase):
    def run_main(self, *, port_override=None, initialize_error=None, sequence_error=None):
        """Run ``main()`` with every hardware touch point replaced by a mock."""
        turntable = Mock()
        if initialize_error is not None:
            turntable.initialize.side_effect = initialize_error
        run_sequence = Mock(side_effect=sequence_error)
        with patch.object(turntable_manual, "parse_port", return_value=port_override), \
                patch.object(turntable_manual, "load_turntable_config", return_value=CONFIG), \
                patch.object(turntable_manual, "load_bus_config", return_value=BUS), \
                patch.object(turntable_manual, "run_manual", run_sequence), \
                patch.object(turntable_manual, "BusServo") as bus_servo, \
                patch.object(
                    turntable_manual, "TurntableServo", return_value=turntable
                ) as wrapper:
            turntable_manual.main()
        return bus_servo, wrapper, turntable, run_sequence

    def test_main_opens_the_turntable_id_on_the_configured_bus_port(self) -> None:
        bus_servo, wrapper, turntable, run_sequence = self.run_main()

        bus_servo.assert_called_once_with("COM14", 2, baudrate=115200, timeout=0.5)
        self.assertEqual(wrapper.call_args.kwargs["config"], CONFIG)
        self.assertEqual(wrapper.call_args.kwargs["servo"], bus_servo.return_value)
        turntable.initialize.assert_called_once_with()
        run_sequence.assert_called_once_with(turntable, CONFIG)

    def test_the_uart_is_released_and_the_disc_is_stopped(self) -> None:
        _, _, turntable, _ = self.run_main()

        turntable.stop.assert_called_once_with()
        turntable.close.assert_called_once_with()

    def test_port_override_wins_over_the_configured_bus_port(self) -> None:
        bus_servo, _, _, _ = self.run_main(port_override="COM77")

        bus_servo.assert_called_once_with("COM77", 2, baudrate=115200, timeout=0.5)

    def test_an_unopenable_port_is_reported_without_running_the_sequence(self) -> None:
        with redirect_stdout(io.StringIO()) as output:
            _, _, turntable, run_sequence = self.run_main(
                initialize_error=OSError("port is busy")
            )

        self.assertIn("无法打开", output.getvalue())
        run_sequence.assert_not_called()
        turntable.close.assert_called_once_with()
        # 端口都没打开时不该再去发停止帧。
        turntable.stop.assert_not_called()

    def test_a_bus_error_during_the_sequence_is_reported_and_still_cleans_up(self) -> None:
        with redirect_stdout(io.StringIO()) as output:
            _, _, turntable, _ = self.run_main(
                sequence_error=TimeoutError("no response")
            )

        self.assertIn("转盘测试未通过", output.getvalue())
        turntable.stop.assert_called_once_with()
        turntable.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

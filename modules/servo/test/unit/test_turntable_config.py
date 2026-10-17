"""转盘的配置：项目里那份实际的值，以及每一种写错的方式都必须被拒绝。"""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from modules.servo.config import (
    HOME_STATION,
    load_gripper_config,
    load_turntable_config,
)
from tests.scratch import scratch_directory


@contextmanager
def config_file(document: object) -> Iterator[Path]:
    """Write one servo configuration document into a throwaway directory."""
    with scratch_directory() as root:
        path = root / "config.json"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        yield path


def turntable_document(**turntable_overrides) -> dict:
    """Return a valid project document with the turntable section overridden."""
    turntable = {
        "servo_id": 2,
        "duration_ms": 1500,
        "stations": {"home": 0, "station_2": 500, "station_3": 1000},
    }
    turntable.update(turntable_overrides)
    return {
        "windows": {"port": "COM14"},
        "bus": {"baudrate": 115200, "servo_id": 1},
        "gripper": {
            "servo_id": 1,
            "open_position": 300,
            "closed_position": 500,
            "duration_ms": 1000,
        },
        "turntable": turntable,
    }


class ShippedTurntableConfigTests(unittest.TestCase):
    def test_project_config_loads_the_three_material_stations(self) -> None:
        config = load_turntable_config()
        self.assertEqual(
            (config.servo_id, config.duration_ms),
            (2, 1500),
        )
        self.assertEqual(
            dict(config.stations),
            {"home": 0, "station_2": 500, "station_3": 1000},
        )

    def test_home_is_position_zero_and_the_other_two_are_120_degrees_apart(self) -> None:
        """三工位转盘：0/500/1000 对应 0°/120°/240°，间隔必须相等，否则放取会偏。"""
        stations = load_turntable_config().stations
        self.assertEqual(stations[HOME_STATION], 0)
        gaps = [
            stations["station_2"] - stations["home"],
            stations["station_3"] - stations["station_2"],
        ]
        self.assertEqual(gaps, [500, 500])

    def test_the_turntable_and_the_gripper_use_different_servo_ids(self) -> None:
        """同一条总线上的两个舵机只靠 ID 区分，撞号是静默错配。"""
        self.assertNotEqual(
            load_turntable_config().servo_id, load_gripper_config().servo_id
        )

    def test_station_lookup_names_the_configured_stations(self) -> None:
        config = load_turntable_config()

        self.assertEqual(config.station("station_3"), 1000)
        self.assertEqual(config.home_position, 0)
        with self.assertRaisesRegex(ValueError, "home"):
            config.station("station_9")


class TurntableConfigRejectionTests(unittest.TestCase):
    def assert_rejected(self, document: object, pattern: str) -> None:
        with config_file(document) as path:
            with self.assertRaisesRegex(ValueError, pattern):
                load_turntable_config(path)

    def test_a_missing_turntable_section_is_rejected(self) -> None:
        document = turntable_document()
        del document["turntable"]
        self.assert_rejected(document, "turntable")

    def test_a_non_object_turntable_section_is_rejected(self) -> None:
        document = turntable_document()
        document["turntable"] = [1, 2]
        self.assert_rejected(document, "turntable")

    def test_a_servo_id_outside_the_bus_range_is_rejected(self) -> None:
        for servo_id in (0, 254, "2"):
            with self.subTest(servo_id=servo_id):
                self.assert_rejected(
                    turntable_document(servo_id=servo_id), "turntable.servo_id"
                )

    def test_a_duration_outside_the_protocol_range_is_rejected(self) -> None:
        for duration_ms in (0, 30001):
            with self.subTest(duration_ms=duration_ms):
                self.assert_rejected(
                    turntable_document(duration_ms=duration_ms), "turntable.duration_ms"
                )

    def test_a_single_station_is_rejected(self) -> None:
        self.assert_rejected(
            turntable_document(stations={"home": 0}), "至少要有"
        )

    def test_a_blank_station_name_is_rejected(self) -> None:
        self.assert_rejected(
            turntable_document(stations={"home": 0, "  ": 500}), "工位名"
        )

    def test_a_position_outside_the_servo_range_is_rejected(self) -> None:
        for position in (-1, 1001):
            with self.subTest(position=position):
                self.assert_rejected(
                    turntable_document(stations={"home": 0, "station_2": position}),
                    "station_2",
                )

    def test_a_non_integer_position_is_rejected(self) -> None:
        """``True`` 是 int 的子类，位置写成布尔值同样要挡住。"""
        for position in (500.0, True, "500"):
            with self.subTest(position=position):
                self.assert_rejected(
                    turntable_document(stations={"home": 0, "station_2": position}),
                    "station_2",
                )

    def test_two_stations_on_the_same_position_are_rejected(self) -> None:
        """两个工位停在同一个位置通常是标定抄错了，分度盘上就是丢了一个工位。"""
        self.assert_rejected(
            turntable_document(stations={"home": 0, "station_2": 0}), "同一个位置"
        )

    def test_a_missing_home_station_is_rejected(self) -> None:
        """没有 home 工位的话 go_home() 就没有确定的目标。"""
        self.assert_rejected(
            turntable_document(stations={"station_1": 0, "station_2": 500}), "home"
        )

    def test_a_servo_id_shared_with_the_gripper_is_rejected(self) -> None:
        document = turntable_document(servo_id=1)
        self.assert_rejected(document, "ID 必须不同")

    def test_a_document_without_a_gripper_section_still_loads(self) -> None:
        document = turntable_document()
        del document["gripper"]
        with config_file(document) as path:
            self.assertEqual(load_turntable_config(path).servo_id, 2)

    def test_an_unreadable_file_is_reported_with_its_path(self) -> None:
        with scratch_directory() as root:
            missing = root / "config.json"
            with self.assertRaisesRegex(ValueError, "无法读取舵机配置"):
                load_turntable_config(missing)

    def test_broken_json_is_reported(self) -> None:
        with scratch_directory() as root:
            path = root / "config.json"
            path.write_text("{ not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "无法读取舵机配置"):
                load_turntable_config(path)


if __name__ == "__main__":
    unittest.main()

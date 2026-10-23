"""The manual bus-servo test must never move before explicit confirmation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules.servo.drive import bus_servo
from modules.servo.drive.bus_servo import AUTO_PORT, load_bus_config
from modules.servo.test.manual import servo


class BusServoConfigTests(unittest.TestCase):
    @staticmethod
    def _write(directory: str, document: dict) -> Path:
        path = Path(directory) / "servo.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_loads_port_baudrate_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                {
                    "windows": {"port": "COM14"},
                    "bus": {"baudrate": 115200, "timeout_seconds": 0.1},
                },
            )
            config = load_bus_config("windows", path)
        self.assertEqual(
            (config.port, config.baudrate, config.timeout_seconds), ("COM14", 115200, 0.1)
        )

    def test_port_must_be_configured_before_opening_serial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                {"jetson": {"port": ""}, "bus": {"baudrate": 115200, "timeout_seconds": 0.1}},
            )
            with self.assertRaisesRegex(ValueError, "jetson.port"):
                load_bus_config("jetson", path)

    def test_bus_section_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, {"windows": {"port": "COM14"}})
            with self.assertRaisesRegex(ValueError, "bus"):
                load_bus_config("windows", path)

    def test_serial_timeout_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                {"windows": {"port": "COM14"}, "bus": {"baudrate": 115200, "timeout_seconds": 0}},
            )
            with self.assertRaisesRegex(ValueError, "timeout_seconds"):
                load_bus_config("windows", path)

    def test_project_file_configures_the_verified_direct_bus(self) -> None:
        """The project file must supply a usable direct-bus bus config, not just a port."""
        config = load_bus_config("windows")
        self.assertEqual(config.baudrate, 115200)
        self.assertTrue(config.port.strip())
        self.assertGreater(config.timeout_seconds, 0)


def position_reply(servo_id: int, position: int = 500) -> bytes:
    """一条格式正确的读位置应答：``55 55 id 05 1C 位置(2B) 校验``。"""
    body = bytes((servo_id, 5, 0x1C)) + position.to_bytes(2, "little")
    return b"\x55\x55" + body + bytes(((~sum(body)) & 0xFF,))


class FakeProbedPort:
    """按"舵机 ID -> 回包"回答的假串口；没有条目的 ID 就当没人应答。"""

    def __init__(self, replies: dict[int, bytes] | None = None) -> None:
        self._replies = dict(replies or {})
        self._receive = b""
        self.writes: list[bytes] = []
        self.closed = False

    def reset_input_buffer(self) -> None:
        self._receive = b""

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        # 帧格式 55 55 id length command ...：第 3 个字节才是舵机 ID。
        self._receive = self._replies.get(frame[2], b"")
        return len(frame)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data, self._receive = self._receive[:size], b""
        return data

    def close(self) -> None:
        self.closed = True


class BusServoAutoPortTests(unittest.TestCase):
    """``port: "auto"`` 时按"哪个 ID 回了读位置帧"认总线，不认端口名。"""

    @staticmethod
    def _document(port: str) -> dict:
        return {
            "windows": {"port": "COM35"},
            "jetson": {"port": port},
            "bus": {"baudrate": 115200, "timeout_seconds": 0.1},
            "gripper": {"servo_id": 1},
            "turntable": {"servo_id": 2},
        }

    @staticmethod
    def _write(directory: str, document: dict) -> Path:
        path = Path(directory) / "servo.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_auto_picks_the_port_whose_servo_answers(self) -> None:
        def opener(port: str) -> FakeProbedPort:
            # 第二条口上只有 2 号转盘回话：夹爪 ID 探不到、转盘 ID 探得到。
            return FakeProbedPort({2: position_reply(2)} if port.endswith("L") else {})

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, self._document(AUTO_PORT))
            with patch.object(
                bus_servo,
                "usb_serial_candidates",
                return_value=("/dev/ttyPROBE_K", "/dev/ttyPROBE_L"),
            ), patch.object(bus_servo, "_serial_opener", return_value=opener):
                config = load_bus_config("jetson", path)

        self.assertEqual((config.port, config.baudrate), ("/dev/ttyPROBE_L", 115200))

    def test_auto_without_any_servo_id_says_what_is_missing(self) -> None:
        document = self._document(AUTO_PORT)
        del document["gripper"], document["turntable"]

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, document)
            with self.assertRaisesRegex(ValueError, "servo_id"):
                load_bus_config("jetson", path)

    def test_a_concrete_port_is_still_read_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, self._document("/dev/ttyUSB0"))
            config = load_bus_config("jetson", path)

        self.assertEqual(config.port, "/dev/ttyUSB0")


class ServoManualTests(unittest.TestCase):
    def test_read_only_and_rejected_confirmation_never_move(self) -> None:
        actuator = Mock()
        actuator.read_position.return_value = 500
        servo.run_manual(actuator, 1, prompt=lambda _: "q")
        actuator.set_position.assert_not_called()

        answers = iter(["520", "800", "no"])
        servo.run_manual(actuator, 1, prompt=lambda _: next(answers))
        actuator.set_position.assert_not_called()

    def test_confirmed_target_moves_once_and_checks_position(self) -> None:
        actuator = Mock()
        actuator.read_position.side_effect = [500, 520]
        answers = iter(["520", "800", "YES"])
        with patch.object(servo.time, "sleep"):
            servo.run_manual(actuator, 1, prompt=lambda _: next(answers))
        actuator.set_position.assert_called_once_with(520, 800)
        self.assertEqual(actuator.read_position.call_count, 2)

    def test_main_opens_configured_port_baudrate_and_id(self) -> None:
        config = Mock(port="COM14", baudrate=115200, servo_id=2)
        with patch.object(servo, "load_bus_config", return_value=config), \
                patch.object(servo, "run_manual"), \
                patch.object(servo, "BusServo") as bus_servo, \
                patch("builtins.input", side_effect=["q"]):
            servo.main()
        bus_servo.assert_called_once_with("COM14", 2, baudrate=115200, timeout=0.5)
        bus_servo.return_value.initialize.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

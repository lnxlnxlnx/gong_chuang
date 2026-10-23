"""``port: "auto"`` —— 按"谁回了读状态帧"认步进总线，而不是按端口名。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.motor import config
from modules.motor.config import (
    AUTO_PORT,
    PROBE_ADDRESSES,
    PROBE_FUNCTION,
    load_bus_config,
    resolve_bus_port,
)

#: 帧尾校验字节：Emm42 每个回包都以它结尾。
FRAME_TAIL = 0x6B


def status_reply(address: int) -> bytes:
    """一条格式正确的读状态应答：``地址 3A 状态 6B``。"""
    return bytes((address, PROBE_FUNCTION, 0x03, FRAME_TAIL))


class FakeProbedPort:
    """按"地址 -> 回包"回答的假串口；没有条目的地址就当没人应答。"""

    def __init__(self, replies: dict[int, bytes] | None = None) -> None:
        self._replies = dict(replies or {})
        self._receive = b""
        self.writes: list[bytes] = []
        self.drained = 0
        self.closed = False

    def reset_input_buffer(self) -> None:
        self.drained += 1
        self._receive = b""

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        self._receive = self._replies.get(frame[0], b"")
        return len(frame)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data, self._receive = self._receive[:size], b""
        return data

    def close(self) -> None:
        self.closed = True


def opener_for(answers: dict[str, dict[int, bytes]]):
    """返回 ``(opener, 打开过的端口)``；表里没有的端口就是"这条口上没人应答"。"""
    opened: list[str] = []

    def opener(port: str) -> FakeProbedPort:
        opened.append(port)
        return FakeProbedPort(answers.get(port))

    return opener, opened


class ResolveBusPortTests(unittest.TestCase):
    def test_concrete_port_is_returned_without_probing(self) -> None:
        def opener(port: str) -> FakeProbedPort:
            raise AssertionError("写死端口时不该探测")

        self.assertEqual(
            resolve_bus_port(
                "/dev/ttyUSB1", baudrate=115200, timeout_seconds=0.1, opener=opener
            ),
            "/dev/ttyUSB1",
        )

    def test_auto_picks_the_port_whose_motor_answers(self) -> None:
        opener, opened = opener_for({"/dev/ttyPROBE_B": {5: status_reply(5)}})

        port = resolve_bus_port(
            AUTO_PORT,
            baudrate=115200,
            timeout_seconds=0.05,
            candidates=("/dev/ttyPROBE_A", "/dev/ttyPROBE_B"),
            opener=opener,
        )

        self.assertEqual(port, "/dev/ttyPROBE_B")
        self.assertEqual(opened, ["/dev/ttyPROBE_A", "/dev/ttyPROBE_B"])

    def test_probe_sends_only_the_read_status_frame_and_closes(self) -> None:
        port = FakeProbedPort({9: status_reply(9)})

        resolve_bus_port(
            AUTO_PORT,
            baudrate=115200,
            timeout_seconds=0.05,
            candidates=("/dev/ttyPROBE_C",),
            opener=lambda _: port,
            addresses=(9,),
        )

        self.assertEqual(port.writes, [bytes((9, PROBE_FUNCTION, FRAME_TAIL))])
        self.assertTrue(port.closed)

    def test_auto_skips_ports_that_cannot_be_opened(self) -> None:
        def opener(port: str) -> FakeProbedPort:
            if port == "/dev/ttyPROBE_D":
                raise OSError("device is busy")
            return FakeProbedPort({1: status_reply(1)})

        port = resolve_bus_port(
            AUTO_PORT,
            baudrate=115200,
            timeout_seconds=0.05,
            candidates=("/dev/ttyPROBE_D", "/dev/ttyPROBE_E"),
            opener=opener,
        )

        self.assertEqual(port, "/dev/ttyPROBE_E")

    def test_auto_lists_every_candidate_when_nobody_answers(self) -> None:
        opener, _ = opener_for({})

        with self.assertRaises(RuntimeError) as caught:
            resolve_bus_port(
                AUTO_PORT,
                baudrate=115200,
                timeout_seconds=0.05,
                candidates=("/dev/ttyPROBE_F", "/dev/ttyPROBE_G"),
                opener=opener,
            )

        message = str(caught.exception)
        self.assertIn("/dev/ttyPROBE_F（无应答）", message)
        self.assertIn("/dev/ttyPROBE_G（无应答）", message)

    def test_a_frame_from_another_address_is_not_an_answer(self) -> None:
        """地址不对的回包不算应答——否则会认错总线。"""
        opener, _ = opener_for({"/dev/ttyPROBE_I": {1: status_reply(2)}})

        with self.assertRaises(RuntimeError):
            resolve_bus_port(
                AUTO_PORT,
                baudrate=115200,
                timeout_seconds=0.05,
                candidates=("/dev/ttyPROBE_I",),
                opener=opener,
                addresses=(1,),
            )

    def test_detection_is_cached_within_the_process(self) -> None:
        """arm / chassis / grasp 都调它，探一次就够了，也必须拿到同一条总线。"""
        opener, opened = opener_for({"/dev/ttyPROBE_H": {5: status_reply(5)}})

        resolved = [
            resolve_bus_port(
                AUTO_PORT,
                baudrate=115200,
                timeout_seconds=0.05,
                candidates=("/dev/ttyPROBE_H",),
                opener=opener,
            )
            for _ in range(3)
        ]

        self.assertEqual(resolved, ["/dev/ttyPROBE_H"] * 3)
        self.assertEqual(opened, ["/dev/ttyPROBE_H"])

    def test_probe_addresses_skip_the_axes_that_never_reply(self) -> None:
        """6/7 的驱动器 Response 是关的（``expect_ack=false``），不能当判据。"""
        self.assertEqual(PROBE_ADDRESSES, (1, 2, 3, 4, 5))


class LoadBusConfigAutoTests(unittest.TestCase):
    @staticmethod
    def _hardware(directory: str, port: str) -> Path:
        path = Path(directory) / "motor.json"
        path.write_text(
            json.dumps(
                {
                    "windows": {"port": "COM36"},
                    "jetson": {"port": port},
                    "bus": {
                        "baudrate": 115200,
                        "pulses_per_revolution": 3200,
                        "timeout_seconds": 0.1,
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_auto_goes_through_the_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._hardware(directory, AUTO_PORT)
            with patch.object(
                config, "usb_serial_candidates", return_value=("/dev/ttyPROBE_J",)
            ), patch.object(
                config,
                "_serial_opener",
                return_value=lambda port: FakeProbedPort({5: status_reply(5)}),
            ):
                settings = load_bus_config("jetson", path)

        self.assertEqual((settings.port, settings.baudrate), ("/dev/ttyPROBE_J", 115200))

    def test_a_concrete_port_is_still_read_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._hardware(directory, "/dev/ttyUSB7")
            settings = load_bus_config("jetson", path)

        self.assertEqual(settings.port, "/dev/ttyUSB7")


if __name__ == "__main__":
    unittest.main()

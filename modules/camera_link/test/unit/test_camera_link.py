"""接收链路：TCP 收行、丢旧取新、按帧分组、断线重连、端口占用、启动中途失败。

功能测试全程走**真 socket**（绑到 ``127.0.0.1`` 的临时端口），不 mock ``socket``：
这一层存在的意义就是"机械臂在动的时候接收线程还活着"，用假 socket 测不出这件事。

**启动失败的资源归属**是例外：那几条反过来必须注入假 socket / 假线程，因为要测的是
"setsockopt / listen / getsockname / 线程 start 分别失败时，谁负责关、关几次、未启动的
线程不许 join"——真 socket 走不到这些分支。
"""

from __future__ import annotations

import json
import socket
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from modules.camera_link.drive.link import (
    CameraLink,
    CameraLinkConfig,
    CameraLinkError,
    load_camera_link_config,
)

LINE_A = (
    "物料 4: @(566,376) 绿色物料（green），confidence=0.88，圆直径 268 px，"
    "内部σ=20.2，边界支撑=83%（36 个方向），xyz=(-8.3,1.8,113.2) mm，直线距离约 113.5 mm"
)


def local_config(**overrides) -> CameraLinkConfig:
    """一份只监听本机临时端口的配置。"""
    settings = {"host": "127.0.0.1", "port": 0, "queue_size": 8}
    settings.update(overrides)
    return CameraLinkConfig(**settings)


def wait_for(predicate, seconds: float = 5.0) -> bool:
    """轮询等待，避免对线程调度时序做假设。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class CameraLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.link = CameraLink(config=local_config()).start()
        self.addCleanup(self.link.stop)
        self.client = socket.create_connection(("127.0.0.1", self.link.port), timeout=5.0)
        self.addCleanup(self.client.close)
        self.assertTrue(wait_for(lambda: self.link.connected), "监听线程没接受连接")

    def send(self, *lines: str) -> None:
        payload = "".join(f"{line}\n" for line in lines)
        self.client.sendall(payload.encode("utf-8"))

    def test_port_is_assigned_when_zero(self) -> None:
        self.assertGreater(self.link.port, 0)
        self.assertEqual(self.link.host, "127.0.0.1")

    def test_receives_and_parses_one_line(self) -> None:
        self.send(LINE_A)
        self.assertTrue(wait_for(lambda: self.link.received == 1))
        sighting = self.link.latest(timeout=2.0)
        self.assertIsNotNone(sighting)
        assert sighting is not None  # 类型收窄
        self.assertEqual(sighting.material_id, 4)
        self.assertIsNotNone(sighting.received_at)

    def test_unparsable_lines_are_kept_for_diagnosis(self) -> None:
        """格式变了的时候，最想看到的就是"到底收到了什么"。"""
        self.send("[INFO] service up", LINE_A)
        self.assertTrue(wait_for(lambda: self.link.received == 1))
        self.assertEqual(self.link.rejected_count, 1)
        self.assertIn("[INFO]", self.link.rejected[0])

    def test_blank_lines_are_ignored_not_counted(self) -> None:
        self.send("", "   ", LINE_A)
        self.assertTrue(wait_for(lambda: self.link.received == 1))
        self.assertEqual(self.link.rejected_count, 0)

    def test_latest_drops_older_observations(self) -> None:
        """控制循环要的是"物料现在在哪"，攒着历史只会去追一个已经不在的位置。"""
        for index in range(6):
            self.send(f"物料 {index}: xyz=({index},0,100) mm")
        self.assertTrue(wait_for(lambda: self.link.received == 6))
        newest = self.link.latest(timeout=2.0)
        self.assertIsNotNone(newest)
        assert newest is not None
        self.assertEqual(newest.material_id, 5, "latest 必须给最后一条，而不是最老的")
        self.assertEqual(self.link.pending, 0)

    def test_get_is_first_in_first_out(self) -> None:
        for index in range(3):
            self.send(f"物料 {index}: xyz=({index},0,100) mm")
        self.assertTrue(wait_for(lambda: self.link.pending == 3))
        first = self.link.get(timeout=2.0)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.material_id, 0)
        self.assertEqual(self.link.pending, 2)

    def test_queue_is_bounded_and_drops_the_oldest(self) -> None:
        """阻塞比丢帧危险：队列满了必须丢旧的，不能让接收线程卡住。"""
        small = CameraLink(config=local_config(queue_size=2)).start()
        self.addCleanup(small.stop)
        with socket.create_connection(("127.0.0.1", small.port), timeout=5.0) as client:
            self.assertTrue(wait_for(lambda: small.connected))
            for index in range(5):
                client.sendall(f"物料 {index}: xyz=(0,0,100) mm\n".encode("utf-8"))
            self.assertTrue(wait_for(lambda: small.received == 5), small.received)
            self.assertLessEqual(small.pending, 2)
            # 保留的一定是最新的那几条。
            kept = []
            while (item := small.get(timeout=0.0)) is not None:
                kept.append(item.material_id)
            self.assertEqual(kept, sorted(kept), kept)
            self.assertEqual(kept[-1], 4, kept)

    def test_frame_groups_one_batch_and_keeps_the_newest_per_material(self) -> None:
        """靶标和物料是两条独立消息，按"最新一条"逐条取永远凑不齐一对。"""
        self.send(
            "物料 9: 红色物料（red），xyz=(100,-50,295) mm",  # 爪上靶标
            "物料 3: 绿色物料（green），xyz=(110,-30,285) mm",  # 物料
            "物料 3: 绿色物料（green），xyz=(111,-31,285) mm",  # 同一物料的更新
        )
        self.assertTrue(wait_for(lambda: self.link.pending >= 1))
        frame = self.link.frame(timeout=2.0, max_age_seconds=5.0)
        self.assertIsNotNone(frame)
        assert frame is not None
        # 按物料编号排序、去重后每个编号只留最新一条。
        self.assertEqual([item.material_id for item in frame], [3, 9])
        by_id = {item.material_id: item for item in frame}
        self.assertAlmostEqual(by_id[3].x_mm, 111.0)

    def test_frame_returns_none_when_nothing_arrives(self) -> None:
        self.assertIsNone(self.link.frame(timeout=0.1))

    def test_frame_drops_observations_older_than_max_age(self) -> None:
        """机械臂动过之后，几百毫秒前的坐标已经不成立了。

        ``max_age_seconds=0`` 表示"一点都不许旧"，所以手上这条必须被丢掉。
        丢掉之后队列是空的，所以下面要再发一条才能验证"给足余量时收得到"——
        否则第二次断言会因为没有数据而假通过。
        """
        self.send(LINE_A)
        self.assertTrue(wait_for(lambda: self.link.received == 1))
        self.assertIsNone(self.link.frame(timeout=0.0, max_age_seconds=0.0))

        self.send(LINE_A)
        self.assertTrue(wait_for(lambda: self.link.received == 2))
        kept = self.link.frame(timeout=0.0, max_age_seconds=60.0)
        self.assertIsNotNone(kept)
        assert kept is not None
        self.assertEqual(len(kept), 1)
        self.assertIsNotNone(kept[0].received_at, "接收线程必须给每条观测打时间戳")

    def test_reconnects_after_the_peer_disconnects(self) -> None:
        self.send(LINE_A)
        self.assertTrue(wait_for(lambda: self.link.received == 1))
        self.client.close()
        self.assertTrue(wait_for(lambda: not self.link.connected), "断开后 connected 应变假")
        self.assertIsNone(self.link.error)

        with socket.create_connection(("127.0.0.1", self.link.port), timeout=5.0) as second:
            self.assertTrue(wait_for(lambda: self.link.connected))
            second.sendall(f"{LINE_A}\n".encode("utf-8"))
            self.assertTrue(wait_for(lambda: self.link.received == 2))
        self.assertEqual(self.link.connections, 2)

    def test_clear_resets_queue_and_counters_but_keeps_the_connection(self) -> None:
        self.send("bad line", LINE_A)
        self.assertTrue(wait_for(lambda: self.link.received == 1))
        self.link.clear()
        self.assertEqual(self.link.pending, 0)
        self.assertEqual(self.link.received, 0)
        self.assertEqual(self.link.rejected_count, 0)
        self.assertEqual(self.link.rejected, ())
        self.assertTrue(self.link.connected, "clear 不该把镜头端踢下线")
        self.send(LINE_A)
        self.assertTrue(wait_for(lambda: self.link.received == 1))


class CameraLinkLifecycleTests(unittest.TestCase):
    def test_busy_port_fails_loudly(self) -> None:
        """端口被占用时必须报错，而不是静默地一直收不到东西。"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            taken = busy.getsockname()[1]
            link = CameraLink(config=local_config(port=taken))
            with self.assertRaises(CameraLinkError) as caught:
                link.start()
            self.assertIn(str(taken), str(caught.exception))

    def test_start_is_idempotent(self) -> None:
        link = CameraLink(config=local_config()).start()
        self.addCleanup(link.stop)
        port = link.port
        self.assertIs(link.start(), link)
        self.assertEqual(link.port, port)

    def test_stop_is_idempotent_and_releases_the_port(self) -> None:
        link = CameraLink(config=local_config()).start()
        port = link.port
        link.stop()
        link.stop()
        # 端口必须真的放开了：同一个端口还能再绑上。
        again = CameraLink(config=local_config(port=port)).start()
        self.addCleanup(again.stop)
        self.assertEqual(again.port, port)

    def test_context_manager_starts_and_stops(self) -> None:
        with CameraLink(config=local_config()) as link:
            self.assertTrue(link.port > 0)
            port = link.port
        replacement = CameraLink(config=local_config(port=port)).start()
        self.addCleanup(replacement.stop)

    def test_many_lines_arrive_in_one_recv_chunk(self) -> None:
        """一次 recv 里可能有多行（甚至半行），切分必须正确。"""
        with CameraLink(config=local_config()) as link:
            with socket.create_connection(("127.0.0.1", link.port), timeout=5.0) as client:
                self.assertTrue(wait_for(lambda: link.connected))
                payload = "".join(
                    f"物料 {index}: xyz=({index},0,100) mm\n" for index in range(20)
                )
                # 分两半发，故意在中间切断一行。
                raw = payload.encode("utf-8")
                client.sendall(raw[: len(raw) // 2])
                time.sleep(0.05)
                client.sendall(raw[len(raw) // 2 :])
                self.assertTrue(wait_for(lambda: link.received == 20), link.received)

    def test_oversized_garbage_does_not_grow_without_bound(self) -> None:
        """对面没有按"每行一条观测"发时，不能把半行无限攒在内存里。"""
        with CameraLink(config=local_config()) as link:
            with socket.create_connection(("127.0.0.1", link.port), timeout=5.0) as client:
                self.assertTrue(wait_for(lambda: link.connected))
                client.sendall(b"x" * (128 * 1024))
                self.assertTrue(wait_for(lambda: link.rejected_count >= 1), link.rejected_count)
                self.assertTrue(link.connected, "超长半行只该被丢掉，不该断开连接")


class CameraLinkStartFailureTests(unittest.TestCase):
    """启动中途失败的资源归属：**没交出去的资源由 CameraLink 自己收尾**。

    这里刻意注入假 socket / 假线程，不绑真实端口：要测的是"哪一步失败之后谁负责关、
    关闭恰好一次、未启动的线程不许 join"，真 socket 反而测不到这些分支。
    """

    def test_every_start_step_failure_closes_the_server_exactly_once(self) -> None:
        """setsockopt / listen / getsockname 分别在一步失败时，server 只关一次。"""
        for stage in ("setsockopt", "listen", "getsockname"):
            with self.subTest(stage=stage):
                fake = FakeSocket(fail_on=stage)
                link = CameraLink(config=local_config())
                with patch(
                    "modules.camera_link.drive.link.socket.socket", return_value=fake
                ):
                    with self.assertRaises(Exception):
                        link.start()
                self.assertEqual(fake.close_calls, 1, "server 必须恰好关一次")
                self.assertIsNone(link._server)
                self.assertIsNone(link._client)
                self.assertIsNone(link._thread)

    def test_bind_failure_still_reports_the_address_and_closes_once(self) -> None:
        """bind 失败保留原来的中文提示（含地址），close 仍由失败收尾统一负责。"""
        fake = FakeSocket(fail_on="bind")
        link = CameraLink(config=local_config(port=50007))
        with patch("modules.camera_link.drive.link.socket.socket", return_value=fake):
            with self.assertRaises(CameraLinkError) as caught:
                link.start()
        self.assertIn("127.0.0.1:50007", str(caught.exception))
        self.assertIn("端口被占用或没有权限", str(caught.exception))
        self.assertEqual(fake.close_calls, 1)

    def test_unstarted_thread_is_never_joined(self) -> None:
        """线程 start 本身失败时 ``ident`` 还是 ``None``：绝不能 join。

        ``join`` 一个没启动的线程会抛 ``cannot join thread before it is started``，
        那会把真正的启动错误盖掉——现场看到的就是一句莫名其妙的 join 报错。
        """
        fake_socket = FakeSocket()
        fake_thread = MockThread(start_fails=True)
        link = CameraLink(config=local_config())
        with patch("modules.camera_link.drive.link.socket.socket", return_value=fake_socket):
            with patch(
                "modules.camera_link.drive.link.threading.Thread",
                return_value=fake_thread,
            ):
                with self.assertRaises(RuntimeError) as caught:
                    link.start()
        self.assertIn("thread start failed", str(caught.exception))
        self.assertEqual(fake_thread.join_calls, 0, "没启动的线程不许 join")
        self.assertEqual(fake_socket.close_calls, 1)
        self.assertIsNone(link._thread)

    def test_close_failure_still_attempts_the_other_socket_and_the_join(self) -> None:
        """一个 socket close 失败不跳过另一个 socket，也不跳过已启动线程的 join。"""
        link = CameraLink(config=local_config())
        failing = FakeSocket(fail_on="close")
        healthy = FakeSocket()
        started = MockThread()
        link._server, link._client, link._thread = failing, healthy, started
        with self.assertRaises(CameraLinkError) as caught:
            link.stop()
        self.assertIn("server.close", str(caught.exception))
        self.assertIn("boom close", str(caught.exception))
        self.assertEqual(failing.close_calls, 1)
        self.assertEqual(healthy.close_calls, 1, "另一个 socket 仍要被关")
        self.assertEqual(started.join_calls, 1, "已启动线程仍要被 join")
        # 三项引用都已清空：再调一次 stop 不会有任何重复动作。
        self.assertIsNone(link._server)
        self.assertIsNone(link._client)
        self.assertIsNone(link._thread)
        link.stop()
        self.assertEqual(failing.close_calls, 1)
        self.assertEqual(healthy.close_calls, 1)
        self.assertEqual(started.join_calls, 1)

    def test_stop_joins_a_real_started_thread_without_blocking(self) -> None:
        """正常启动的链路：stop 必须真的把线程收掉（join 能返回）。"""
        link = CameraLink(config=local_config()).start()
        self.assertIsNotNone(link._thread)
        link.stop()
        self.assertIsNone(link._thread)


class FakeSocket:
    """一个假监听 socket：可以指定在哪一步失败，并记录 close 次数。"""

    def __init__(self, *, fail_on: str = "") -> None:
        self.fail_on = fail_on
        self.close_calls = 0

    def _maybe_fail(self, stage: str) -> None:
        if self.fail_on == stage:
            raise OSError(f"boom {stage}")

    def setsockopt(self, *_args) -> None:
        self._maybe_fail("setsockopt")

    def bind(self, _address) -> None:
        self._maybe_fail("bind")

    def listen(self, _backlog) -> None:
        self._maybe_fail("listen")

    def getsockname(self):
        self._maybe_fail("getsockname")
        return ("127.0.0.1", 50007)

    def shutdown(self, *_args) -> None:
        raise OSError("not connected")

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_on == "close":
            raise OSError("boom close")

    def settimeout(self, _seconds) -> None:
        pass

    def accept(self):
        raise OSError("stopped")


class MockThread:
    """假线程：可以模拟 ``start()`` 失败（那时 ``ident`` 仍是 ``None``）。"""

    def __init__(self, *, start_fails: bool = False) -> None:
        self.start_fails = start_fails
        self.join_calls = 0

    @property
    def ident(self):
        return None if self.start_fails else 4242

    def start(self) -> None:
        if self.start_fails:
            raise RuntimeError("thread start failed")

    def join(self, timeout=None) -> None:
        self.join_calls += 1


class CameraLinkConfigTests(unittest.TestCase):
    def test_project_config_loads(self) -> None:
        config = load_camera_link_config()
        self.assertEqual(config.port, 50007)
        self.assertEqual(config.encoding, "utf-8")
        self.assertGreaterEqual(config.queue_size, 1)

    def test_invalid_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for document in (
                {"host": ""},
                {"port": 0},
                {"port": 70000},
                {"queue_size": 0},
                {"accept_timeout_seconds": 0},
                {"read_timeout_seconds": -1},
                {"encoding": ""},
                "not an object",
            ):
                with self.subTest(document=document):
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_camera_link_config(path)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(ValueError):
            load_camera_link_config(Path("does-not-exist.json"))


if __name__ == "__main__":
    unittest.main()

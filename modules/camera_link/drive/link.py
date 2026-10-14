"""摄像头那边发观测、机械臂这边收观测：一条 TCP 长连接 + 一个线程 + 一个队列。

## 为什么是「线程 + 队列」而不是直接在主循环里 ``recv``

机械臂的动作是**阻塞**的：``move_base_degrees(wait=True)`` 会按估算耗时睡过去，
一条大角度指令能睡掉好几秒。如果接收写在主循环里，那几秒里镜头端发来的观测全部堵在
内核缓冲区，等机械臂停下来才被读到——读到的是**几秒前的旧位置**。

所以接收必须是独立的线程：它只管把行变成 :class:`~modules.vision.drive.sighting.MaterialSighting`
塞进队列，机械臂线程想取的时候取**最新的一条**。

## 两种取法，对应两种语义

- :meth:`CameraLink.latest` —— 拿队列里最新的一条，**把旧的丢掉**。
  控制循环用这个：机械臂要找的是"物料现在在哪"，不是"物料曾经在哪"。
  攒着一堆历史观测只会让机械臂去追一个已经不在的位置。
- :meth:`CameraLink.get` —— 取最老的一条，FIFO。
  想逐条回放、做统计、或者验证"一条都没丢"的时候用。

队列长度有上限（``queue_size``），满了就丢最旧的。这是刻意的：**阻塞比丢帧危险**——
接收线程一旦堵住，镜头端会看到连接不再被读取，而机械臂会拿着一份越来越旧的数据继续动。

## 协议

镜头端作为客户端连上来，每行一条观测，UTF-8，``\\n`` 结尾。就是
:func:`modules.vision.drive.sighting.parse_sighting` 认的那种文本。

解析不了的行（日志、心跳、空行）**不会被丢掉**，而是记进 :attr:`CameraLink.rejected`
并在 :meth:`CameraLink.latest` 里带出来——真机上"为什么一个观测都收不到"十有八九是
对面换了输出格式，这时候你需要看到那些原始行。
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from modules.vision.drive.sighting import MaterialSighting, try_parse_sighting

#: 配置文件在模块根目录（``modules/camera_link/config.json``），
#: 而本文件在 ``drive/`` 下，所以要从 ``drive`` 再往上走一层。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

#: 单行长度上限。超过它说明对面没在发"每行一条观测"——宁可断开也不要让内存无限涨。
MAX_LINE_BYTES = 64 * 1024

#: 最多记住多少条被拒绝的行（诊断用，避免无限增长）。
MAX_REJECTED_LINES = 50


@dataclass(frozen=True)
class CameraLinkConfig:
    """接收端设置。

    ``host`` 默认 ``0.0.0.0``：镜头端在**另一台机器**上，只监听回环地址会连不上。
    """

    host: str = "0.0.0.0"
    port: int = 50007
    queue_size: int = 8
    accept_timeout_seconds: float = 30.0
    read_timeout_seconds: float = 1.0
    encoding: str = "utf-8"


def load_camera_link_config(config_path: Path = DEFAULT_CONFIG_PATH) -> CameraLinkConfig:
    """读取 ``modules/camera_link/config.json``。"""
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取摄像头链路配置：{config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"摄像头链路配置不是合法 JSON：{config_path}") from error
    if not isinstance(document, dict):
        raise ValueError("摄像头链路配置必须是 JSON 对象")

    def integer(name: str, default: int, minimum: int, maximum: int) -> int:
        value = document.get(name, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} 必须是 {minimum}..{maximum} 的整数")
        return value

    def positive(name: str, default: float) -> float:
        value = document.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"{name} 必须是正数")
        return float(value)

    host = document.get("host", CameraLinkConfig.host)
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host 必须是非空字符串")
    encoding = document.get("encoding", CameraLinkConfig.encoding)
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding 必须是非空字符串")

    return CameraLinkConfig(
        host=host.strip(),
        port=integer("port", CameraLinkConfig.port, 1, 65535),
        queue_size=integer("queue_size", CameraLinkConfig.queue_size, 1, 4096),
        accept_timeout_seconds=positive(
            "accept_timeout_seconds", CameraLinkConfig.accept_timeout_seconds
        ),
        read_timeout_seconds=positive(
            "read_timeout_seconds", CameraLinkConfig.read_timeout_seconds
        ),
        encoding=encoding.strip(),
    )


class CameraLinkError(RuntimeError):
    """接收链路起不来，或者已经死掉。"""


class CameraLink:
    """后台线程收观测、队列里取观测。

    用法：

    ```python
    with CameraLink() as link:            # 起监听线程
        sighting = link.latest(timeout=5)  # 等一条观测
    ```

    :meth:`start` 之后 :attr:`port` 才是真正的监听端口（``port=0`` 时由系统分配，
    测试用得上）。
    """

    def __init__(
        self,
        *,
        config: Optional[CameraLinkConfig] = None,
        port: Optional[int] = None,
        host: Optional[str] = None,
    ) -> None:
        self.config = config if config is not None else load_camera_link_config()
        self.host = self.config.host if host is None else host
        self._requested_port = self.config.port if port is None else port
        self.port = self._requested_port

        self._queue: "queue.Queue[MaterialSighting]" = queue.Queue(
            maxsize=self.config.queue_size
        )
        self._rejected: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._error: Optional[str] = None
        #: 收到并解析成功的观测总条数（与队列长度无关，队列会丢旧的）。
        self.received = 0
        #: 解析失败的行总数。
        self.rejected_count = 0
        #: 已接受的连接数。
        self.connections = 0

    # ------------------------------------------------------------- 生命周期
    def start(self) -> "CameraLink":
        """绑定端口并起线程；重复调用是空操作。"""
        if self._thread is not None:
            return self
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server.bind((self.host, self._requested_port))
        except OSError as error:
            self._server.close()
            self._server = None
            raise CameraLinkError(
                f"无法监听 {self.host}:{self._requested_port}：{error}；"
                "端口被占用或没有权限"
            ) from error
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="camera-link", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """停线程、关连接；重复调用是空操作。"""
        self._stop.set()
        for sock in (self._client, self._server):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        thread, self._thread = self._thread, None
        self._client = None
        self._server = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def close(self) -> None:
        """``stop`` 的别名，让 ``with`` 和别的模块读起来一致。"""
        self.stop()

    def __enter__(self) -> "CameraLink":
        return self.start()

    def __exit__(self, *_exc_info) -> None:
        self.stop()

    @property
    def error(self) -> Optional[str]:
        """监听 / 收包过程中出现的致命错误描述；正常时为 ``None``。"""
        return self._error

    @property
    def connected(self) -> bool:
        """当前是否有镜头端连着。"""
        client = self._client
        return client is not None and client.fileno() != -1

    # ------------------------------------------------------------- 取观测
    def latest(self, *, timeout: Optional[float] = None) -> Optional[MaterialSighting]:
        """取**最新**的一条观测，把队列里更旧的都丢掉；超时返回 ``None``。

        控制循环用这个。丢掉旧观测是刻意的：机械臂要对着物料**现在**的位置动。
        """
        if timeout is None:
            if self._queue.empty():
                return None
            timeout = 0.0
        try:
            sighting = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        # 把这条之后堆积的旧观测全部丢弃。
        while True:
            try:
                sighting = self._queue.get_nowait()
            except queue.Empty:
                return sighting

    def get(self, *, timeout: Optional[float] = None) -> Optional[MaterialSighting]:
        """取最老的一条观测（FIFO）；超时返回 ``None``。

        回放和统计用这个，它**不丢中间观测**（只受 ``queue_size`` 上限约束）。
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def frame(
        self,
        *,
        max_age_seconds: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[MaterialSighting, ...]]:
        """把队列里当前压着的观测收成**一组**返回（同一批、去重后按物料编号排序）。

        为什么需要它：一帧画面里的爪上靶标和物料是**两条独立的消息**，分两次到达。
        如果按"最新一条"逐条取，第二次取到的可能又是靶标，永远凑不齐一对。

        两个参数：

        - ``timeout``：队列为空时最多等这么久（秒），等到第一条就把它同批的伙伴一起收走；
        - ``max_age_seconds``：只收"新到这么多秒以内"的观测；更旧的一条都不返回。
          机械臂动过之后再拿几秒前的坐标，算出来的位置早就不成立了。

        没有可用观测时返回 ``None``（而不是空元组：空元组会让调用方以为"收到了一帧，
        只是里面什么都没"，那是另一回事）。
        """
        first = self.get(timeout=timeout)
        if first is None:
            return None
        collected = [first]
        # 把已经到达的同批伙伴收走，不阻塞等待"还没来的"。
        while True:
            item = self.get(timeout=0.0)
            if item is None:
                break
            collected.append(item)

        if max_age_seconds is not None:
            # 严格小于：``max_age_seconds=0`` 的意思是"一点都不许旧"，
            # 用它做"必须丢掉"的判据时不能因为浮点上的 0.0000001 而通过。
            now = time.monotonic()
            collected = [
                item
                for item in collected
                if item.received_at is None or now - item.received_at < max_age_seconds
            ]
        if not collected:
            return None

        # 同一个物料可能被连报几条（镜头端每帧都发）：只留最新的一条。
        freshest: dict = {}
        for item in collected:
            freshest[item.material_id] = item
        return tuple(freshest[key] for key in sorted(freshest))

    @property
    def pending(self) -> int:
        """队列里还压着多少条。"""
        return self._queue.qsize()

    @property
    def rejected(self) -> Tuple[str, ...]:
        """最近被拒绝的原始行（最新在后），用来倒查格式变化。"""
        with self._lock:
            return tuple(self._rejected)

    def clear(self) -> None:
        """清空队列与统计，回到"从头开始"的状态。

        三个计数器一起归零（``received`` / ``rejected_count`` / ``connections``）：
        只清队列不清计数会让人看着一个对不上队列的累计值，反而更难判断链路状态。
        连接本身不断——清统计不该把镜头端踢下线。
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._rejected.clear()
        self.received = 0
        self.rejected_count = 0
        self.connections = 0

    # ------------------------------------------------------------- 接收线程
    def _serve(self) -> None:
        """接受连接并逐行读取，直到 :meth:`stop`。"""
        try:
            while not self._stop.is_set():
                if not self._accept_one():
                    continue
                self._read_lines()
        except OSError as error:  # 关 socket 时 recv/accept 会抛，正常退出不该记成错误
            if not self._stop.is_set():
                self._error = str(error)
        except Exception as error:  # noqa: BLE001 - 线程里的异常必须留住，否则静默死掉
            self._error = f"{type(error).__name__}: {error}"

    def _accept_one(self) -> bool:
        """等一个连接；超时或已停止时返回 ``False``。"""
        server = self._server
        if server is None:
            self._stop.wait(0.1)
            return False
        server.settimeout(self.config.accept_timeout_seconds)
        try:
            client, _address = server.accept()
        except socket.timeout:
            return False
        except OSError:
            if not self._stop.is_set():
                raise
            return False
        client.settimeout(self.config.read_timeout_seconds)
        # TCP_NODELAY：一条观测只有百来字节，攒包只会让延迟白白变大。
        try:
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self._client = client
        self.connections += 1
        return True

    def _read_lines(self) -> None:
        """把当前连接上的字节流切成行，解析成功就入队。"""
        client = self._client
        if client is None:
            return
        buffer = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break  # 对面主动断开
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    self._handle_line(raw)
                if len(buffer) > MAX_LINE_BYTES:
                    # 对面没有在发"每行一条观测"；丢掉这半行，别让它无限涨。
                    self._record_rejected(buffer.decode(self.config.encoding, "replace"))
                    buffer = b""
        finally:
            try:
                client.close()
            except OSError:
                pass
            self._client = None

    def _handle_line(self, raw: bytes) -> None:
        text = raw.decode(self.config.encoding, "replace").strip()
        if not text:
            return
        # 时间戳必须在这里打：下游（闭环对准）靠它判断"手上这帧是不是过期了"，
        # 少了它那条保护会静默失效——看起来在工作，其实一直在用旧坐标。
        sighting = try_parse_sighting(text, received_at=time.monotonic())
        if sighting is None:
            self._record_rejected(text)
            return
        self.received += 1
        self._offer(sighting)

    def _offer(self, sighting: MaterialSighting) -> None:
        """入队；满了就丢最旧的腾位子（阻塞比丢帧危险）。"""
        while True:
            try:
                self._queue.put_nowait(sighting)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def _record_rejected(self, text: str) -> None:
        self.rejected_count += 1
        with self._lock:
            self._rejected.append(text)
            if len(self._rejected) > MAX_REJECTED_LINES:
                del self._rejected[0]


def open_link(**kwargs) -> CameraLink:
    """起一条接收链路并返回它（``CameraLink(...).start()`` 的简写）。"""
    return CameraLink(**kwargs).start()


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """不需要真镜头：本机自己连自己，走一遍 TCP。"""
    import time

    from modules.vision.drive.sighting import parse_sighting

    sample = (
        "物料 4: @(566,376) 绿色物料（green），confidence=0.88，圆直径 268 px，"
        "内部σ=20.2，边界支撑=83%（36 个方向），xyz=(-8.3,1.8,113.2) mm，直线距离约 113.5 mm"
    )
    config = CameraLinkConfig(host="127.0.0.1", port=0, queue_size=3)
    with CameraLink(config=config) as link:
        assert link.port > 0, link.port

        def _wait_for(predicate, seconds: float = 5.0) -> bool:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if predicate():
                    return True
                time.sleep(0.02)
            return False

        with socket.create_connection(("127.0.0.1", link.port), timeout=5.0) as client:
            assert _wait_for(lambda: link.connected), "监听线程没接受连接"

            # 无关的日志行要被拒掉，而不是打断链路。
            client.sendall("[INFO] vision service up\n".encode("utf-8"))
            client.sendall((sample + "\n").encode("utf-8"))
            assert _wait_for(lambda: link.received == 1), (link.received, link.rejected)

            sighting = link.latest(timeout=2.0)
            assert sighting is not None and sighting.material_id == 4, sighting
            assert abs(sighting.x_mm + 8.3) < 1e-9, sighting
            assert link.rejected_count == 1, link.rejected_count
            assert "[INFO]" in link.rejected[0], link.rejected

            # 一次灌 5 条、队列上限 3：``latest`` 必须给出**最后**那条，而不是最老的。
            for index in range(5):
                client.sendall(f"物料 {index}: xyz=({index},0,100) mm\n".encode("utf-8"))
            assert _wait_for(lambda: link.received == 6), link.received
            newest = link.latest(timeout=2.0)
            assert newest is not None and newest.material_id == 4, newest
            assert newest.x_mm == 4.0, newest
            assert link.pending == 0, link.pending

            # FIFO 取法不跳条。先排空队列而不是 clear()：clear 会把计数器也归零，
            # 而后面还要用 received / connections 核对"断线重连"。
            while link.get(timeout=0.0) is not None:
                pass
            for index in range(3):
                client.sendall(f"物料 {index}: xyz=({index},0,100) mm\n".encode("utf-8"))
            assert _wait_for(lambda: link.pending == 3), link.pending
            first = link.get(timeout=2.0)
            assert first is not None and first.material_id == 0, first
            assert link.pending == 2, link.pending

            # clear() 连统计一起归零（它是"从头开始"，不是"只清队列"）。
            link.clear()
            assert link.pending == 0 and link.received == 0, (link.pending, link.received)
            assert link.rejected == () and link.rejected_count == 0, link.rejected
            assert link.connections == 0, link.connections

        # 对面断开之后，链路自己活着，新的连接还能进来。
        # connections 由接收线程在 accept 时自增，所以这里等它，而不是直接读。
        assert _wait_for(lambda: not link.connected), "对端断开后 connected 必须变假"
        with socket.create_connection(("127.0.0.1", link.port), timeout=5.0) as client:
            assert _wait_for(lambda: link.connections == 1), link.connections
            client.sendall((sample + "\n").encode("utf-8"))
            assert _wait_for(lambda: link.received == 1), link.received
        assert link.error is None, link.error

    # 端口被占用时必须响亮地失败，而不是静默地收不到东西。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        try:
            CameraLink(config=CameraLinkConfig(host="127.0.0.1", port=taken)).start()
        except CameraLinkError as error:
            assert str(taken) in str(error), error
        else:
            raise AssertionError("端口被占用必须报 CameraLinkError")

    print("摄像头链路自测通过：TCP 收行、拒日志行、丢旧取新、断线重连、端口占用保护都正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

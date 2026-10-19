"""动作票据与动作组：把"已经发出、还没走完"的动作收成一个能 join 的对象。

## 为什么要它

Emm42 的位置命令和总线舵机的 ``MOVE_TIME_WRITE`` 都是**发完就自己走**：驱动器收到帧就
开始运动，上位机不会自动知道它什么时候停。所以在本项目里"并行"不需要线程，真正的形状是：

```
各自发出（各模块自己发） → 一个 join 点 → 各自收尾
```

本模块只提供后半截：

- :class:`MotionTicket` —— 一张票据 = 一次**已发出**的动作 + 它什么时候算走完 +
  到点之后怎么收尾（读回位置、推进位姿、给出报告）；
- :class:`MotionBatch` —— 一组票据，一个 :meth:`~MotionBatch.wait`；
- :class:`MotionController` —— 会话级入口，``robot.motion.begin()`` 拿一个新的动作组。

## 票据 vs 报告

``wait=True`` 的动作直接返回**报告**（已经走完，结论已知）；``wait=False`` 返回**票据**
（只发了命令，结论要等）。票据的 :meth:`MotionTicket.wait` 返回的就是同一份报告，
所以"异步"不是另一套语义，只是把"等"挪到你要的位置：

```python
ticket = arm.move_axes({"lift": 800, "extension": -200}, wait=False)
...
reports = ticket.wait()
```

## 票据在等什么

等的是 :func:`modules.motor.drive.motion.estimate_move_seconds` 那一类**估算耗时**——
和现有 ``wait=True`` 走的是同一个模型，所以两条路不会给出互相矛盾的结果。按到位位轮询
（Emm42 ``0x3A`` / 舵机 ``POS_READ``）是另一条更准的路，开关在各自模块的 ``wait_mode``。

票据**不会**在构造时或等待中途偷偷读设备：唯一会碰硬件的时刻是收尾函数，
而且只跑一次。

## 超时与失败

- ``wait(timeout=...)`` 超时抛 :class:`MotionTimeout`，票据**仍是 pending**，可以再等；
- 收尾函数抛错时票据进 :data:`FAILED`，错误原样保存，**再 wait 还是抛同一个错误**
  （不吞掉，也不重跑收尾——收尾几乎都有副作用）;
- :meth:`MotionBatch.wait` 与仓库里 ``close`` 的规矩一致：**一项失败不阻断其它项**，
  全部试完才把错误连同报告一起抛出（:class:`MotionBatchError`）。

## 线程

本模块**自己不建线程**：动作在硬件上并行，join 在调用线程上串行。同一张票据的 ``wait``
请在同一个线程里调；跨线程分享票据属于调用方要自己同步的事——和项目里其余"一条会话
单线程"的约定一致。以后真要"边动边干别的"，后台 join 线程挂
:class:`MotionController`，调用点不用改。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

#: 票据状态：命令已发出，还没到预计完成时刻。
PENDING = "pending"
#: 收尾函数已经跑完。
DONE = "done"
#: 收尾函数抛了错（动作本身发出去过，只是结论拿不到）。
FAILED = "failed"
#: 被 :meth:`MotionTicket.cancel` 停掉了。
CANCELLED = "cancelled"

#: 所有合法状态，便于校验与打印。
STATES = (PENDING, DONE, FAILED, CANCELLED)

#: 两次检查"到点了没有"之间最长睡多久（秒）。它只影响"什么时候收尾"的精度，
#: 不影响动作本身——各模块的预计耗时通常以秒计，20 毫秒足够。
DEFAULT_POLL_INTERVAL = 0.02


class MotionTimeout(TimeoutError):
    """在 ``timeout`` 内没等到票据到期。票据仍然可以继续等。"""


class MotionBatchError(RuntimeError):
    """动作组里至少有一项收尾失败；报告仍然完整地带在异常上。

    与 ``Robot.close`` / ``MecanumChassis.close`` 同一套规矩：一项失败不阻断其它项，
    全部试完才报告，第一个错误作 ``__cause__``。
    """

    def __init__(self, report: "MotionBatchReport", errors: List[Tuple[str, BaseException]]) -> None:
        details = "; ".join(f"{subject}: {error}" for subject, error in errors)
        super().__init__(f"动作组收尾失败：{details}")
        self.report = report
        self.errors = tuple(errors)


class MotionTicket:
    """一次**已经发出**的动作，以及"它什么时候算走完"。

    构造票据 = 承认"命令已经出去了"：本类不负责发命令，也不负责校验动作是否合法，
    那些是各模块自己的事（它们必须先发帧成功、再拿报告来建票据）。

    :param subject: 这项动作是谁（例如 ``"arm:lift+extension"``、``"gripper:set"``），
        只用于日志与报告。
    :param estimated_seconds: 预计还要多久走完（秒）；动作已经发出，所以从**建票据那一刻**起算。
    :param finalize: 收尾函数，返回这一项的**报告**；``None`` 表示这项没有可收的尾
        （只等时间过去）。它最多被调用一次。
    :param cancel: 尽力停住这项动作；失败会记成 :data:`FAILED` 并抛出。
    :param clock: 单调时钟，测试时注入假实现。
    :param sleep: 睡函数，测试时注入假实现（配合假 ``clock`` 一起推进）。
    :param poll_interval: 两次检查之间最长睡多久（秒），必须为正。
    """

    def __init__(
        self,
        subject: str,
        estimated_seconds: float,
        *,
        finalize: Optional[Callable[[], object]] = None,
        cancel: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("subject 必须是非空字符串")
        if isinstance(estimated_seconds, bool) or not isinstance(estimated_seconds, (int, float)):
            raise ValueError("estimated_seconds 必须是数字")
        if float(estimated_seconds) < 0 or float(estimated_seconds) != float(estimated_seconds):
            raise ValueError("estimated_seconds 必须是非负有限数字")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        self.subject = subject
        self.estimated_seconds = float(estimated_seconds)
        self._finalize = finalize
        self._cancel_action = cancel
        self._clock = clock
        self._sleep = sleep
        self._poll_interval = poll_interval
        self._started = clock()
        self._deadline = self._started + self.estimated_seconds
        self._state = PENDING
        self._result: object = None
        self._error: Optional[BaseException] = None

    # ------------------------------------------------------------- 查询
    @property
    def state(self) -> str:
        """取 :data:`PENDING` / :data:`DONE` / :data:`FAILED` / :data:`CANCELLED`。"""
        return self._state

    @property
    def done(self) -> bool:
        """收尾是否已经跑过（成功或失败都算"跑过"）。"""
        return self._state in (DONE, FAILED, CANCELLED)

    @property
    def pending(self) -> bool:
        """是否还没到点、也还没被取消。"""
        return self._state == PENDING

    @property
    def deadline(self) -> float:
        """预计完成时刻（``clock`` 的读数，不是墙上时间）。"""
        return self._deadline

    @property
    def remaining(self) -> float:
        """从此刻到预计完成时刻还有多久（秒），到点后是 ``0.0``。"""
        return max(0.0, self._deadline - self._clock())

    @property
    def result(self) -> object:
        """收尾返回的报告；还没收尾时是 ``None``。"""
        return self._result

    @property
    def error(self) -> Optional[BaseException]:
        """收尾失败时的原始错误；其余情况是 ``None``。"""
        return self._error

    def summary(self) -> str:
        state = {
            PENDING: f"进行中（预计还剩 {self.remaining:.2f} s）",
            DONE: "已完成",
            FAILED: f"收尾失败（{self._error!r}）",
            CANCELLED: "已取消",
        }[self._state]
        return f"{self.subject}：{state}"

    # ------------------------------------------------------------- 等待
    def wait(self, timeout: Optional[float] = None) -> object:
        """等到预计完成时刻，跑一次收尾函数，返回它的结果。

        :param timeout: 本次最多等多久（秒）；``None`` 表示一直等到到点。
        :raises MotionTimeout: ``timeout`` 内没到点。票据**保持 pending**，可以再等。
        :raises BaseException: 收尾函数抛出的原始错误（再 wait 还是同一个错误）。

        已经收尾过的票据直接返回上次的结果（收尾只跑一次），所以"多等几次"是安全的。
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must not be negative")
        if self._state == PENDING:
            limit = None if timeout is None else self._clock() + float(timeout)
            while True:
                now = self._clock()
                if now >= self._deadline:
                    break
                if limit is not None and now >= limit:
                    raise MotionTimeout(
                        f"等待 {self.subject} 超时：预计完成时刻还差 "
                        f"{self._deadline - now:.3f} s"
                    )
                step = min(self._deadline - now, self._poll_interval)
                if limit is not None:
                    step = min(step, limit - now)
                self._sleep(step)
            self._finish()
        if self._state == FAILED and self._error is not None:
            raise self._error
        return self._result

    def cancel(self) -> bool:
        """尽力停住这项动作；返回是否真的停了（已经收尾过的返回 ``False``）。

        取消**不跑收尾函数**：动作没走完，收尾（读回位置、推进位姿）会给出错的结论。
        """
        if self._state != PENDING:
            return False
        if self._cancel_action is not None:
            try:
                self._cancel_action()
            except BaseException as error:
                self._state = FAILED
                self._error = error
                raise
        self._state = CANCELLED
        return True

    def _finish(self) -> None:
        """跑收尾函数；只跑一次，失败原样记下来。"""
        if self._state != PENDING:
            return
        try:
            self._result = None if self._finalize is None else self._finalize()
        except BaseException as error:
            # 收尾几乎都有副作用（读回、推进位姿、打印），所以失败就记下、不重试。
            self._state = FAILED
            self._error = error
            raise
        self._state = DONE


@dataclass(frozen=True)
class MotionBatchItem:
    """动作组里的一项：它的票据最后是什么结论。"""

    subject: str
    result: object = None
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        if self.error is not None:
            return f"{self.subject}：失败（{self.error!r}）"
        return f"{self.subject}：已完成"


@dataclass(frozen=True)
class MotionBatchReport:
    """一次 :meth:`MotionBatch.wait` 的完整结果（含失败项）。"""

    items: Tuple[MotionBatchItem, ...]

    @property
    def ok(self) -> bool:
        """所有项都收尾成功。"""
        return all(item.ok for item in self.items)

    @property
    def failures(self) -> Tuple[MotionBatchItem, ...]:
        return tuple(item for item in self.items if not item.ok)

    def summary(self) -> str:
        if not self.items:
            return "动作组是空的"
        return f"{len(self.items)} 项动作：" + "；".join(item.summary() for item in self.items)


class MotionBatch:
    """一组**已经发出**的动作，一个 join 点。

    ```python
    batch = robot.motion.begin()
    batch.add(robot.arm.move_axes({"lift": 800, "extension": -200}, wait=False))
    batch.add(robot.gripper.set_position(40, wait=False))
    batch.wait()
    ```

    各模块的动作在**硬件上**并行（各自的总线上自己走），这里只负责"一起等"。
    加进组里的顺序不影响结果：先等 A 再等 B，与先等 B 再等 A 得到同一份报告。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        # 票据自己计时，这里只用来判"整组超时"和包装没有票据的结果对象。
        self._clock = clock
        self._sleep = sleep
        self._poll_interval = poll_interval
        self._tickets: List[MotionTicket] = []

    # ------------------------------------------------------------- 收集
    def add(
        self,
        item: object,
        *,
        seconds: Optional[float] = None,
        subject: Optional[str] = None,
    ) -> MotionTicket:
        """把一项已发出的动作加进组里，返回它的票据。

        接受两种东西：

        - :class:`MotionTicket`（``wait=False`` 的动作返回的就是它）；
        - 任何带 ``estimated_seconds`` 的结果对象（包装成"只等时间、没有收尾"的票据）。

        都不是的时候，用 ``seconds=`` 显式声明这项动作预计走多久——底盘那类
        "``wait=False`` 只返回脉冲数"的动作走这条。

        :raises ValueError: 同一张票据被加两次，或者既给了票据又给了 ``seconds``。
        :raises TypeError: 看不出这项动作要等多久。
        """
        if isinstance(item, MotionTicket):
            if seconds is not None:
                raise ValueError("票据自己带着预计耗时，不要再传 seconds")
            ticket = item
        else:
            value = seconds
            if value is None:
                value = getattr(item, "estimated_seconds", None)
            if value is None:
                raise TypeError(
                    "动作组只收票据，或带 estimated_seconds 的结果对象；"
                    "不返回票据的动作（例如底盘）请用 add(结果, seconds=预计秒数) 声明耗时"
                )
            ticket = MotionTicket(
                subject if subject is not None else type(item).__name__,
                value,
                clock=self._clock,
                sleep=self._sleep,
                poll_interval=self._poll_interval,
            )
        if any(existing is ticket for existing in self._tickets):
            raise ValueError(f"{ticket.subject} 的票据已经在这个动作组里了")
        self._tickets.append(ticket)
        return ticket

    # ------------------------------------------------------------- 查询
    @property
    def tickets(self) -> Tuple[MotionTicket, ...]:
        """组里的全部票据，按加入顺序。"""
        return tuple(self._tickets)

    @property
    def subjects(self) -> Tuple[str, ...]:
        return tuple(ticket.subject for ticket in self._tickets)

    @property
    def pending(self) -> Tuple[MotionTicket, ...]:
        """还没收尾的票据。"""
        return tuple(ticket for ticket in self._tickets if ticket.pending)

    @property
    def estimated_seconds(self) -> float:
        """整组还要等多久：各票据剩余时间的最大值。"""
        return max((ticket.remaining for ticket in self._tickets), default=0.0)

    def __len__(self) -> int:
        return len(self._tickets)

    def __iter__(self) -> Iterator[MotionTicket]:
        return iter(self._tickets)

    # ------------------------------------------------------------- 收尾
    def wait(self, timeout: Optional[float] = None) -> MotionBatchReport:
        """等到组里所有票据都收尾，返回整组报告。

        **一项失败不阻断其它项**：先把所有票据都试一遍，再决定抛不抛。
        失败时抛 :class:`MotionBatchError`，完整的 :class:`MotionBatchReport`
        挂在异常的 ``report`` 上。

        :param timeout: 整组最多等多久（秒）；``None`` 表示等到最后一项到点。
        :raises MotionTimeout: 整组超时（会记进报告，并以 :class:`MotionBatchError` 抛出）。
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must not be negative")
        limit = None if timeout is None else self._clock() + float(timeout)
        errors: List[Tuple[str, BaseException]] = []
        for ticket in tuple(self._tickets):
            if not ticket.pending:
                continue
            remaining = None if limit is None else limit - self._clock()
            if remaining is not None and remaining <= 0:
                errors.append(
                    (ticket.subject, MotionTimeout(f"动作组超时：{ticket.subject} 还没到点"))
                )
                continue
            try:
                ticket.wait(timeout=remaining)
            except Exception as error:  # noqa: BLE001 - 逐项收集，最后一起报
                errors.append((ticket.subject, error))
        report = MotionBatchReport(
            tuple(
                MotionBatchItem(ticket.subject, ticket.result, ticket.error)
                for ticket in self._tickets
            )
        )
        if errors:
            raise MotionBatchError(report, errors)
        return report

    def cancel_all(self) -> Tuple[MotionTicket, ...]:
        """尽力停住组里所有还没收尾的动作，返回**没能停住**的那些。

        与 ``close`` 同一条规矩：一项失败不阻断其它项。取消成功的票据进
        :data:`CANCELLED`，不会跑收尾函数。
        """
        failed: List[MotionTicket] = []
        for ticket in tuple(self._tickets):
            if not ticket.pending:
                continue
            try:
                ticket.cancel()
            except Exception:  # noqa: BLE001 - 逐项尝试，最后把失败的票据交回去
                failed.append(ticket)
        return tuple(failed)


class MotionController:
    """``Robot.motion``：动作组的总入口。

    ```python
    batch = robot.motion.begin()
    ```

    :meth:`begin` 每次都返回一个**新的** :class:`MotionBatch`：一个组对应一段
    "同时发出、一起等"的动作，复用同一个组会让"这组里到底有什么"变得说不清。

    这个类现在只带着会话级的时钟与轮询间隔（离线测试可以整体注入假时钟）。
    以后要有后台 join 线程或"停住所有动作"，也挂在这里，调用点不用改。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        self._clock = clock
        self._sleep = sleep
        self._poll_interval = poll_interval

    def begin(self) -> MotionBatch:
        """开一个新的动作组。"""
        return MotionBatch(
            clock=self._clock, sleep=self._sleep, poll_interval=self._poll_interval
        )

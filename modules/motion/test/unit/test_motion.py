"""动作票据与动作组：全部离线，用假时钟驱动，不建线程、不开设备。"""

from __future__ import annotations

import unittest

from modules.motion import (
    CANCELLED,
    DONE,
    FAILED,
    PENDING,
    MotionBatch,
    MotionBatchError,
    MotionController,
    MotionTicket,
    MotionTimeoutError,
)


class FakeClock:
    """假时钟：``sleep`` 推进时间并记账，所以"等了多少"是可以断言的。"""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.slept: list = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise AssertionError(f"睡了一个负数：{seconds}")
        self.slept.append(seconds)
        self.now += seconds


def make_ticket(clock: FakeClock, seconds: float = 2.0, **kwargs) -> MotionTicket:
    kwargs.setdefault("clock", clock)
    kwargs.setdefault("sleep", clock.sleep)
    return MotionTicket("test:subject", seconds, **kwargs)


class MotionTicketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()

    def test_a_fresh_ticket_is_pending_and_does_not_touch_the_device(self) -> None:
        finalized = []
        ticket = make_ticket(self.clock, 1.5, finalize=lambda: finalized.append(True))

        self.assertEqual(ticket.state, PENDING)
        self.assertTrue(ticket.pending)
        self.assertFalse(ticket.done)
        self.assertAlmostEqual(ticket.remaining, 1.5, places=9)
        # 建票据 = 承认"命令已经发出去了"，所以这里**不**应该有收尾动作。
        self.assertEqual(finalized, [])
        self.assertEqual(self.clock.slept, [])

    def test_wait_sleeps_until_the_deadline_then_finalizes_once(self) -> None:
        ticket = make_ticket(self.clock, 1.5, finalize=lambda: "report")

        result = ticket.wait()

        self.assertEqual(result, "report")
        self.assertEqual(ticket.state, DONE)
        self.assertTrue(ticket.done)
        self.assertAlmostEqual(sum(self.clock.slept), 1.5, places=9)
        # 每次睡的都不超过轮询间隔，所以"到点"的精度不受估计时长影响。
        self.assertTrue(all(step <= 0.02 + 1e-9 for step in self.clock.slept))

    def test_waiting_again_returns_the_same_report_without_finalizing_again(self) -> None:
        calls = []
        ticket = make_ticket(self.clock, 1.0, finalize=lambda: calls.append(1) or "report")

        first = ticket.wait()
        second = ticket.wait()

        self.assertEqual((first, second), ("report", "report"))
        self.assertEqual(calls, [1])
        self.assertAlmostEqual(sum(self.clock.slept), 1.0, places=9)

    def test_a_zero_second_ticket_finalizes_without_sleeping(self) -> None:
        ticket = make_ticket(self.clock, 0.0, finalize=lambda: "done")

        self.assertEqual(ticket.wait(), "done")
        self.assertEqual(self.clock.slept, [])

    def test_a_ticket_without_a_finalizer_still_completes(self) -> None:
        ticket = make_ticket(self.clock, 0.5)

        self.assertIsNone(ticket.wait())
        self.assertEqual(ticket.state, DONE)

    def test_timeout_leaves_the_ticket_pending_and_waitable_again(self) -> None:
        ticket = make_ticket(self.clock, 3.0)

        with self.assertRaises(MotionTimeoutError):
            ticket.wait(timeout=1.0)
        self.assertEqual(ticket.state, PENDING)

        # 超时不是失败：剩下的时间还能接着等，只是不重复睡已经睡过的那一段。
        self.assertIsNone(ticket.wait())
        self.assertEqual(ticket.state, DONE)
        self.assertAlmostEqual(sum(self.clock.slept), 3.0, places=9)

    def test_a_failing_finalizer_is_kept_and_reraised_on_every_wait(self) -> None:
        calls = []

        def explode():
            calls.append(1)
            raise RuntimeError("读回位置失败")

        ticket = make_ticket(self.clock, 1.0, finalize=explode)

        with self.assertRaisesRegex(RuntimeError, "读回位置失败"):
            ticket.wait()
        self.assertEqual(ticket.state, FAILED)
        self.assertIsInstance(ticket.error, RuntimeError)

        # 收尾有副作用，所以**不重跑**；但错误也不吞掉，再等还是抛同一个。
        with self.assertRaisesRegex(RuntimeError, "读回位置失败"):
            ticket.wait()
        self.assertEqual(calls, [1])

    def test_cancel_stops_the_action_without_finalizing(self) -> None:
        stopped = []
        finalized = []
        ticket = make_ticket(
            self.clock,
            5.0,
            finalize=lambda: finalized.append(True),
            cancel=lambda: stopped.append(True),
        )

        self.assertTrue(ticket.cancel())

        self.assertEqual(ticket.state, CANCELLED)
        self.assertEqual(stopped, [True])
        # 取消之后不能收尾：动作没走完，收尾会给出错的结论。
        self.assertEqual(finalized, [])
        self.assertIsNone(ticket.wait())

    def test_cancelling_a_finished_ticket_does_nothing(self) -> None:
        stopped = []
        ticket = make_ticket(self.clock, 0.0, cancel=lambda: stopped.append(True))
        ticket.wait()

        self.assertFalse(ticket.cancel())
        self.assertEqual(stopped, [])

    def test_a_failing_cancel_action_is_reported(self) -> None:
        def explode():
            raise RuntimeError("串口断了")

        ticket = make_ticket(self.clock, 1.0, cancel=explode)

        with self.assertRaisesRegex(RuntimeError, "串口断了"):
            ticket.cancel()
        self.assertEqual(ticket.state, FAILED)

    def test_rejects_bad_construction_arguments(self) -> None:
        with self.assertRaises(ValueError):
            MotionTicket("", 1.0)
        with self.assertRaises(ValueError):
            MotionTicket("   ", 1.0)
        with self.assertRaises(ValueError):
            MotionTicket("ok", -1.0)
        with self.assertRaises(ValueError):
            MotionTicket("ok", float("nan"))
        with self.assertRaises(ValueError):
            MotionTicket("ok", 1.0, poll_interval=0)
        with self.assertRaises(ValueError):
            make_ticket(self.clock, 1.0).wait(timeout=-1)

    def test_summary_says_what_is_going_on(self) -> None:
        ticket = make_ticket(self.clock, 2.0)
        self.assertIn("进行中", ticket.summary())

        ticket.wait()
        self.assertIn("已完成", ticket.summary())


class MotionBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()

    def make_batch(self) -> MotionBatch:
        return MotionBatch(clock=self.clock, sleep=self.clock.sleep)

    def test_waits_every_ticket_regardless_of_the_order_they_were_added(self) -> None:
        slow = make_ticket(self.clock, 3.0, finalize=lambda: "slow")
        fast = make_ticket(self.clock, 1.0, finalize=lambda: "fast")
        batch = self.make_batch()
        batch.add(slow)
        batch.add(fast)

        report = batch.wait()

        self.assertTrue(report.ok)
        self.assertEqual([item.subject for item in report.items], ["test:subject"] * 2)
        self.assertEqual([item.result for item in report.items], ["slow", "fast"])
        self.assertEqual((slow.state, fast.state), (DONE, DONE))
        self.assertAlmostEqual(sum(self.clock.slept), 3.0, places=9)

    def test_accepts_a_plain_result_object_that_carries_its_own_estimate(self) -> None:
        class Report:
            estimated_seconds = 2.0

        batch = self.make_batch()
        ticket = batch.add(Report())

        self.assertEqual(ticket.subject, "Report")
        self.assertAlmostEqual(ticket.estimated_seconds, 2.0, places=9)
        self.assertTrue(batch.wait().ok)

    def test_seconds_can_declare_the_deadline_for_things_without_a_ticket(self) -> None:
        """底盘那类只返回脉冲数的动作走这条路：等的是调用方声明的耗时。"""
        batch = self.make_batch()
        ticket = batch.add([1600, 1600, 1600, 1600], seconds=1.25, subject="chassis:forward")

        self.assertEqual(ticket.subject, "chassis:forward")
        self.assertAlmostEqual(ticket.estimated_seconds, 1.25, places=9)

        self.assertTrue(batch.wait().ok)
        self.assertAlmostEqual(sum(self.clock.slept), 1.25, places=9)

    def test_rejects_things_it_cannot_time(self) -> None:
        batch = self.make_batch()
        with self.assertRaises(TypeError):
            batch.add([1600, 1600, 1600, 1600])

    def test_add_all_moves_a_whole_group_in_so_its_tickets_really_run(self) -> None:
        """``batch.add(动作组)`` 曾经只等时长、内层收尾全丢；add_all 才把票据搬进来。"""
        ran: list = []
        inner = self.make_batch()
        inner.add(make_ticket(self.clock, 1.0, finalize=lambda: ran.append("lift") or "ok"))
        inner.add(make_ticket(self.clock, 3.0, finalize=lambda: ran.append("base") or "ok"))

        outer = self.make_batch()
        added = outer.add_all(inner)

        self.assertEqual(len(added), 2)
        self.assertAlmostEqual(outer.estimated_seconds, 3.0, places=9)
        self.assertTrue(outer.wait().ok)
        self.assertEqual(ran, ["lift", "base"])
        # 只等最慢的那一条，不是内层时长再加一遍
        self.assertAlmostEqual(sum(self.clock.slept), 3.0, places=9)

    def test_a_whole_group_is_not_silently_swallowed(self) -> None:
        inner = self.make_batch()
        inner.add(make_ticket(self.clock, 1.0))
        with self.assertRaises(TypeError):
            self.make_batch().add(inner)

    def test_rejects_the_same_ticket_twice(self) -> None:
        batch = self.make_batch()
        ticket = make_ticket(self.clock, 1.0)
        batch.add(ticket)

        with self.assertRaises(ValueError):
            batch.add(ticket)

    def test_rejects_seconds_alongside_a_ticket(self) -> None:
        batch = self.make_batch()
        with self.assertRaises(ValueError):
            batch.add(make_ticket(self.clock, 1.0), seconds=1.0)

    def test_one_failure_does_not_stop_the_others(self) -> None:
        def explode():
            raise RuntimeError("转盘读回失败")

        batch = self.make_batch()
        batch.add(make_ticket(self.clock, 1.0, finalize=explode))
        batch.add(make_ticket(self.clock, 2.0, finalize=lambda: "ok"))

        with self.assertRaises(MotionBatchError) as caught:
            batch.wait()

        # 第二个票据照样收尾了，报告完整地带在异常上——与 close 的规矩一致。
        report = caught.exception.report
        self.assertEqual([item.ok for item in report.items], [False, True])
        self.assertEqual(report.items[1].result, "ok")
        self.assertEqual(len(report.failures), 1)
        self.assertIn("转盘读回失败", str(caught.exception))

    def test_a_group_timeout_is_reported_per_ticket_and_leaves_them_pending(self) -> None:
        batch = self.make_batch()
        first = batch.add(make_ticket(self.clock, 5.0))
        second = batch.add(make_ticket(self.clock, 6.0))

        with self.assertRaises(MotionBatchError) as caught:
            batch.wait(timeout=1.0)

        self.assertEqual(len(caught.exception.errors), 2)
        self.assertTrue(all(isinstance(error, MotionTimeoutError) for _, error in caught.exception.errors))
        self.assertTrue(first.pending and second.pending)

    def test_reports_what_is_still_pending_and_how_long_the_group_needs(self) -> None:
        batch = self.make_batch()
        self.assertEqual(len(batch), 0)
        self.assertEqual(batch.estimated_seconds, 0.0)
        self.assertIn("空", batch.wait().summary())

        first = batch.add(make_ticket(self.clock, 1.0))
        batch.add(make_ticket(self.clock, 4.0))
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch.subjects, ("test:subject", "test:subject"))
        self.assertAlmostEqual(batch.estimated_seconds, 4.0, places=9)

        first.wait()
        self.assertEqual(batch.pending, (batch.tickets[1],))
        self.assertAlmostEqual(batch.estimated_seconds, 3.0, places=9)

    def test_cancel_all_stops_the_ones_still_running(self) -> None:
        stopped = []
        batch = self.make_batch()
        batch.add(make_ticket(self.clock, 3.0, cancel=lambda: stopped.append("a")))
        done = batch.add(make_ticket(self.clock, 0.0, cancel=lambda: stopped.append("b")))
        done.wait()

        self.assertEqual(batch.cancel_all(), ())
        self.assertEqual(stopped, ["a"])
        self.assertEqual(batch.tickets[0].state, CANCELLED)

    def test_cancel_all_hands_back_the_ones_it_could_not_stop(self) -> None:
        def explode():
            raise RuntimeError("串口断了")

        batch = self.make_batch()
        stuck = batch.add(make_ticket(self.clock, 3.0, cancel=explode))

        self.assertEqual(batch.cancel_all(), (stuck,))
        self.assertEqual(stuck.state, FAILED)


class MotionControllerTests(unittest.TestCase):
    def test_begin_hands_out_a_fresh_batch_sharing_the_session_clock(self) -> None:
        clock = FakeClock()
        controller = MotionController(clock=clock, sleep=clock.sleep)

        first = controller.begin()
        second = controller.begin()

        self.assertIsNot(first, second)
        first.add(MotionTicket("a", 1.0, clock=clock, sleep=clock.sleep))
        second.add(MotionTicket("b", 2.0, clock=clock, sleep=clock.sleep))
        self.assertEqual(first.subjects, ("a",))
        self.assertEqual(second.subjects, ("b",))

    def test_rejects_a_non_positive_poll_interval(self) -> None:
        with self.assertRaises(ValueError):
            MotionController(poll_interval=0)


if __name__ == "__main__":
    unittest.main()

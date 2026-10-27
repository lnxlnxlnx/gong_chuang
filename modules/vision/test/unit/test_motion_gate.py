"""转动/停稳门控：**静止不等于停稳**，报告一个周期只发一次。

这一组是**回归测试**：改动之前，判断"停稳了没有"只看"连续几帧读数一致"，
而画面本来就静止时读数当然一致——于是转盘还没开始转，上一轮的残留状态就被
当成这一轮的结果上报了。这个错误只在"刚好赶上静止的那一瞬间"复现，最难查。

所以下面第一条用例就是"一直静止绝不上报"，它是这一组存在的理由。
"""

from __future__ import annotations

import unittest

from modules.vision.drive.motion_gate import (
    STATE_IDLE,
    STATE_MOVING,
    STATE_REPORTED,
    STATE_SETTLED,
    MotionGate,
    feed,
)


def moving_samples(
    count: int,
    *,
    step_px: float = 10.0,
    interval_s: float = 0.1,
    start_stamp: float = 0.0,
    start_x: float = 0.0,
    y: float = 0.0,
):
    """合成一串匀速直线运动：每帧走 ``step_px``、时刻走 ``interval_s``。"""
    return [
        ((start_x + index * step_px, y), start_stamp + index * interval_s)
        for index in range(count)
    ]


def stopping_samples(
    count: int,
    *,
    step_px: float = 0.5,
    interval_s: float = 0.1,
    start_stamp: float = 0.0,
    start_x: float = 80.0,
    y: float = 0.0,
):
    """合成一串"几乎不动"的观测（速度远低于停稳阈值）。"""
    return [
        ((start_x + index * step_px, y), start_stamp + index * interval_s)
        for index in range(count)
    ]


class StillIsNotSettledTests(unittest.TestCase):
    def test_a_frame_that_never_moves_is_never_reported(self) -> None:
        """**这一组存在的理由**：一直静止时，连着十帧一样也不许上报。"""
        gate = MotionGate()
        verdict = feed(gate, [((10.0, 10.0), index * 0.1) for index in range(10)])
        self.assertFalse(verdict.ever_moved, "画面一直静止，不该被认为动过")
        self.assertFalse(verdict.settled, "静止 ≠ 停稳，不该判为停稳")
        self.assertFalse(verdict.report_now, "静止时绝不能上报")
        self.assertEqual(verdict.state, STATE_IDLE)

    def test_motion_must_be_confirmed_before_settling(self) -> None:
        """只动了一帧就停：确认帧数不够，仍然不许上报。"""
        gate = MotionGate(required_moving_frames=2)
        samples = moving_samples(2, step_px=10.0) + stopping_samples(6, start_stamp=0.2)
        verdict = feed(gate, samples)
        # 第 2 帧才算"连续 2 帧在动"，所以 ever_moved 应当成立；
        # 但把 required_moving_frames 调到 3 就该不成立。
        self.assertTrue(verdict.ever_moved)
        strict = MotionGate(required_moving_frames=3)
        strict_verdict = feed(strict, samples)
        self.assertFalse(strict_verdict.ever_moved, "只连续动了 2 帧，3 帧的要求不该满足")


class SettleCycleTests(unittest.TestCase):
    def test_a_move_then_stop_cycle_reports_exactly_once(self) -> None:
        gate = MotionGate()
        samples = moving_samples(8) + stopping_samples(6, start_stamp=0.8)
        verdicts = [gate.observe(position, stamp) for position, stamp in samples]
        reports = [index for index, item in enumerate(verdicts) if item.report_now]
        self.assertEqual(len(reports), 1, "一个周期应当恰好上报一次")
        self.assertTrue(verdicts[-1].ever_moved)
        self.assertTrue(verdicts[-1].settled)
        self.assertEqual(verdicts[-1].state, STATE_REPORTED)

    def test_the_latch_resets_on_the_next_cycle(self) -> None:
        """第二次转动之后必须能再报一次——锁存要能复位。"""
        gate = MotionGate()
        feed(gate, moving_samples(8) + stopping_samples(6, start_stamp=0.8))
        self.assertTrue(gate.reported)
        second = moving_samples(8, start_stamp=1.5) + stopping_samples(6, start_stamp=2.3)
        verdicts = [gate.observe(position, stamp) for position, stamp in second]
        self.assertEqual(
            len([item for item in verdicts if item.report_now]), 1,
            "第二轮应当恰好再上报一次",
        )
        self.assertTrue(verdicts[-1].settled)

    def test_a_miss_does_not_undo_the_motion_fact(self) -> None:
        """漏检只重建测速基准，**不清** ``ever_moved``：转过是已发生的事实。"""
        gate = MotionGate()
        feed(gate, moving_samples(8))
        self.assertTrue(gate.ever_moved)
        feed(gate, [(None, 1.0), (None, 1.1)])
        self.assertTrue(gate.ever_moved, "漏检不该清掉 ever_moved")
        self.assertFalse(gate.reported, "还没停稳就不该处于已上报状态")

    def test_only_a_reset_clears_the_motion_fact(self) -> None:
        gate = MotionGate()
        feed(gate, moving_samples(8))
        gate.reset()
        self.assertFalse(gate.ever_moved)
        self.assertFalse(gate.reported)
        self.assertIsNone(gate.observe((0.0, 0.0), 0.0).speed_px_s)

    def test_the_summary_explains_the_idle_case(self) -> None:
        """给人看的一行必须点明"静止不等于停稳"，否则现场会以为工具坏了。"""
        verdict = feed(MotionGate(), [((0.0, 0.0), index * 0.1) for index in range(5)])
        self.assertIn("静止不等于停稳", verdict.summary())


class HysteresisTests(unittest.TestCase):
    def test_speeds_inside_the_band_do_not_flip_the_state(self) -> None:
        """阈值之间（25–30 px/s）来回抖，不该把状态翻来翻去。"""
        gate = MotionGate()
        feed(gate, moving_samples(4, step_px=5.0))            # 50 px/s，确认动过
        # 27 px/s：落在迟滞带以内。
        jitter = [((20.0 + (index % 2) * 2.7, 0.0), 0.4 + index * 0.1) for index in range(8)]
        verdict = feed(gate, jitter)
        self.assertNotEqual(verdict.state, STATE_REPORTED, "迟滞带以内不该判停稳")
        self.assertTrue(verdict.ever_moved)

    def test_a_speed_at_the_moving_threshold_counts_as_moving(self) -> None:
        """边界是闭区间：正好等于阈值算"在动"（否则现场永远差一点点）。"""
        gate = MotionGate(required_moving_frames=1, moving_speed_px_s=30.0)
        verdict = feed(gate, [((0.0, 0.0), 0.0), ((3.0, 0.0), 0.1)])   # 30 px/s
        self.assertTrue(verdict.ever_moved)


class IntervalTests(unittest.TestCase):
    def test_an_over_long_interval_yields_no_speed(self) -> None:
        """跨了掉帧的间隔算不出速度，也不该被当成"动过"。"""
        gate = MotionGate()
        verdict = feed(gate, [((0.0, 0.0), 0.0), ((100.0, 0.0), 1.0)])
        self.assertIsNone(verdict.speed_px_s)
        self.assertFalse(verdict.ever_moved)

    def test_an_over_short_interval_yields_no_speed(self) -> None:
        """同一帧被处理两次（间隔 0）时算不出速度，而不是算出无穷大。"""
        gate = MotionGate()
        verdict = feed(gate, [((0.0, 0.0), 0.0), ((100.0, 0.0), 0.0)])
        self.assertIsNone(verdict.speed_px_s)

    def test_the_position_still_becomes_the_next_baseline(self) -> None:
        """间隔不合理时位置仍然成为下一次的基准——宁可少算一次速度，也别丢基准。"""
        gate = MotionGate()
        feed(gate, [((0.0, 0.0), 0.0), ((500.0, 0.0), 1.0)])
        # 上一次的位置是 (500, 0)；从这里再走 10 px/0.1 s = 100 px/s。
        verdict = gate.observe((510.0, 0.0), 1.1)
        self.assertAlmostEqual(verdict.speed_px_s, 100.0, delta=1e-6)

    def test_a_non_finite_stamp_is_rejected(self) -> None:
        gate = MotionGate()
        for bad in (float("nan"), float("inf")):
            with self.subTest(stamp=bad):
                with self.assertRaises(ValueError):
                    gate.observe((0.0, 0.0), bad)


class TargetChangeTests(unittest.TestCase):
    def test_changing_the_key_restarts_the_cycle(self) -> None:
        """换了目标就重新开始：动过的是**上一个**目标。"""
        gate = MotionGate(key="red")
        feed(gate, moving_samples(8))
        self.assertTrue(gate.ever_moved)
        verdict = gate.observe((0.0, 0.0), 1.0, key="blue")
        self.assertFalse(verdict.ever_moved, "换目标之后 ever_moved 必须清掉")
        self.assertEqual(gate.key, "blue")

    def test_the_same_key_keeps_the_state(self) -> None:
        gate = MotionGate(key="red")
        feed(gate, moving_samples(8))
        verdict = gate.observe((100.0, 0.0), 1.0, key="red")
        self.assertTrue(verdict.ever_moved)


class ConstructorTests(unittest.TestCase):
    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs, name in (
            ({"moving_speed_px_s": 20.0, "near_stop_speed_px_s": 30.0}, "停稳阈值高于转动阈值"),
            ({"required_moving_frames": 0}, "转动帧数为 0"),
            ({"required_low_speed_frames": 0}, "停稳帧数为 0"),
            ({"required_moving_frames": True}, "转动帧数是布尔值"),
            ({"min_interval_s": 0.3, "max_interval_s": 0.2}, "间隔上下限颠倒"),
            ({"min_interval_s": 0.1, "max_interval_s": 0.1}, "间隔上下限相等"),
            ({"moving_speed_px_s": -1.0}, "转动阈值为负"),
            ({"near_stop_speed_px_s": -1.0}, "停稳阈值为负"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    MotionGate(**kwargs)

    def test_equal_thresholds_are_allowed(self) -> None:
        """两个阈值相等是允许的（没有迟滞），只是现场会更爱抖——不该报错。"""
        gate = MotionGate(moving_speed_px_s=30.0, near_stop_speed_px_s=30.0)
        self.assertEqual(gate.moving_speed_px_s, 30.0)

    def test_the_verdict_carries_the_counters(self) -> None:
        """连续计数要如实报出来，否则现场没法判断"还差几帧"。"""
        gate = MotionGate(required_moving_frames=5)
        verdict = feed(gate, moving_samples(3))
        self.assertEqual(verdict.moving_frames, 2)   # 第一帧立基准，算不出速度
        self.assertFalse(verdict.ever_moved)


if __name__ == "__main__":
    unittest.main()

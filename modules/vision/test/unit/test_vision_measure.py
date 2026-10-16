"""多帧融合的会话接口：``Vision.measure`` / ``Vision.measure_settled``。

这一组守住的是**"静止"与"运动"两种坏法不能被同一套平均糊过去"**：

- 静止时读数在真值附近抖，要剔野值、取稳健中心；
- 运动时读数在时间上单调走，取平均会给出一个"哪一时刻都不是"的位置，
  而按"离中心太远"剔野值会把所有帧都剔掉。

所以断言分两头：静止的一串要拿到接近真值的数，运动的一串要**如实报出在动**
并且报出速度，而不是悄悄给个平均数。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from modules.vision import Vision
from modules.vision.config import TargetSettings, VisionConfig
from modules.vision.drive.hough_lock import RingLock
from modules.vision.reports import MaterialReport, RegionReport


def _settings() -> VisionConfig:
    return VisionConfig(camera_ids={"windows": 7, "jetson": 7}, targets=TargetSettings())


def _report(x_mm: float, y_mm: float, z_mm: float, *, name="red", radius=146.0):
    """造一帧"看到一块料"的报告：几何量与 :func:`_timed_point_from_report` 一一对应。"""
    region = SimpleNamespace(center_x=500.0, center_y=360.0, radius=radius)
    classification = SimpleNamespace(label="红色物料", name=name, score=0.9, found=True)
    face = SimpleNamespace(
        interior_std=5.0, edge_support=0.8, sampled_rays=36, summary=lambda: "面"
    )
    return MaterialReport(
        lock=RingLock(()),
        hits=(),
        regions=(
            RegionReport(
                region=region,
                classification=classification,
                face=face,
                accepted=True,
                x_mm=x_mm,
                y_mm=y_mm,
                distance_mm=z_mm,
            ),
        ),
        focal_px=1012.0,
    )


EMPTY = MaterialReport(lock=RingLock(()), hits=(), regions=(), focal_px=1012.0)


def _session(*, reports, times=None, drain=0):
    """一个只差设备的会话：相机永远读到帧，定位器按 ``reports`` 依次返回。

    ``times`` 给定时钟按它依次返回（每个时钟调用取一个），用来造出"帧与帧之间
    过了多久"；用完就一直返回最后一个值，免得测试被调用次数绑死。
    """
    vision = Vision(_settings())
    camera = Mock()
    camera.read.return_value = object()
    locator = Mock()
    locator.locate.side_effect = list(reports)
    vision._camera, vision._locator = camera, locator
    vision._drain_frames = drain
    if times is None:
        vision._clock = lambda: 1000.0
    else:
        remaining = list(times)

        def clock():
            if len(remaining) > 1:
                return remaining.pop(0)
            return remaining[0]

        vision._clock = clock
    return vision, camera, locator


class MeasureTests(unittest.TestCase):
    def test_a_still_target_gives_one_value_and_is_not_moving(self) -> None:
        # 抖动刻意取成**对称**的：拟合值取在最后一帧的时刻，而一串不对称的抖动
        # 本身带一点点趋势，会被如实外推出去——那是对的行为，但会让"中心值等于
        # 真值"这条断言测到别的东西。
        jitter = (0.0, 0.1, -0.1, 0.1, -0.1, 0.0, 0.0)
        vision, _camera, _locator = _session(
            reports=[_report(10.0 + d, -2.0 + d, 103.0) for d in jitter],
            times=[1000.0 + 0.2 * i for i in range(len(jitter))],
        )
        fused = vision.measure()

        self.assertIsNotNone(fused)
        self.assertEqual(fused.samples, len(jitter))
        self.assertEqual(fused.rejected, 0)
        self.assertFalse(fused.moving)
        self.assertTrue(fused.confident)
        self.assertAlmostEqual(fused.point.x_mm, 10.0, delta=0.05)
        self.assertAlmostEqual(fused.point.y_mm, -2.0, delta=0.05)
        self.assertAlmostEqual(fused.point.z_mm, 103.0, delta=0.05)
        self.assertLess(fused.spread_mm, 0.2)

    def test_a_moving_target_is_reported_as_moving_with_its_velocity(self) -> None:
        """转盘还没停稳：不能给平均数，要给"还在动"和速度。"""
        steps = range(7)
        vision, _camera, _locator = _session(
            reports=[_report(3.0 * i * 0.2, 0.0, 103.0) for i in steps],   # 3 mm/s
            times=[1000.0 + 0.2 * i for i in steps],
        )
        fused = vision.measure()

        self.assertIsNotNone(fused)
        self.assertTrue(fused.moving)
        self.assertFalse(fused.confident, "还在动就不该是 confident")
        self.assertAlmostEqual(fused.velocity_mm_s[0], 3.0, delta=0.05)
        # 参考时刻是最后一帧：值应当是**外推到现在**的位置，不是整段的平均。
        self.assertAlmostEqual(fused.point.x_mm, 3.0 * 6 * 0.2, delta=0.1)

    def test_a_blurred_outlier_frame_is_dropped(self) -> None:
        reports = [_report(10.0, 0.0, 103.0) for _ in range(6)]
        reports.insert(3, _report(80.0, 40.0, 60.0))       # 这一帧框错了圆
        vision, _camera, _locator = _session(
            reports=reports, times=[1000.0 + 0.2 * i for i in range(7)]
        )
        fused = vision.measure()

        self.assertEqual(fused.rejected, 1)
        self.assertEqual(fused.samples, 6)
        self.assertAlmostEqual(fused.point.x_mm, 10.0, delta=0.05)
        self.assertFalse(fused.moving)

    def test_frames_without_a_material_count_as_misses(self) -> None:
        vision, _camera, _locator = _session(
            reports=[_report(10.0, 0.0, 103.0), EMPTY, _report(10.0, 0.0, 103.0),
                     EMPTY, _report(10.0, 0.0, 103.0), EMPTY, _report(10.0, 0.0, 103.0)],
            times=[1000.0 + 0.2 * i for i in range(7)],
        )
        fused = vision.measure()

        self.assertEqual(fused.misses, 3)
        self.assertEqual(fused.samples, 4)
        self.assertEqual(fused.frames, 7, "frames 要覆盖整批，不只是内点")

    def test_too_few_readings_give_none_rather_than_a_single_frame(self) -> None:
        """有效读数不够时必须返回 ``None``——退回单帧值正是要挡掉的东西。"""
        vision, _camera, _locator = _session(
            reports=[_report(10.0, 0.0, 103.0), EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            times=[1000.0 + 0.2 * i for i in range(7)],
        )
        self.assertIsNone(vision.measure())

    def test_a_camera_read_failure_is_a_miss_not_an_exception(self) -> None:
        """相机掉线的那几帧算 miss，不该把整次测量炸掉。

        ``_read_report`` 每次会先丢 ``max(1, drain_frames)`` 帧缓冲再读真正的那一帧
        （丢掉的帧**不看返回值**），所以这里按 ``drain=1`` 排序列：每个定位周期
        消耗两次 ``read()``，偶数下标是丢帧、奇数下标才是真正读的那一帧。
        """
        vision, camera, _locator = _session(
            reports=[_report(10.0, 0.0, 103.0)] * 5,
            times=[1000.0 + 0.2 * i for i in range(5)],
            drain=1,
        )
        frame = object()
        camera.read.side_effect = [
            frame, None,     # 第 1 帧：读到的是 None -> miss
            frame, None,     # 第 2 帧：同上
            frame, frame,    # 第 3–5 帧：正常
            frame, frame,
            frame, frame,
        ]
        fused = vision.measure(frames=5)

        self.assertIsNotNone(fused)
        self.assertEqual(fused.misses, 2)
        self.assertEqual(fused.samples, 3)

    def test_settle_mm_stops_the_burst_early(self) -> None:
        """稳定就早收工：省下的是几帧 × 约 178 ms 的定位时间。"""
        vision, _camera, locator = _session(
            reports=[_report(10.0, 0.0, 103.0)] * 7,
            times=[1000.0 + 0.2 * i for i in range(7)],
        )
        fused = vision.measure(frames=7, settle_mm=1.0)

        self.assertIsNotNone(fused)
        self.assertEqual(locator.locate.call_count, 3, "攒够 min_samples 就该收工")
        self.assertEqual(fused.samples, 3)

    def test_without_settle_mm_the_whole_burst_is_taken(self) -> None:
        """默认不早停：行为可预测，代价是固定的帧数。"""
        vision, _camera, locator = _session(
            reports=[_report(10.0, 0.0, 103.0)] * 7,
            times=[1000.0 + 0.2 * i for i in range(7)],
        )
        vision.measure(frames=7)
        self.assertEqual(locator.locate.call_count, 7)

    def test_a_frame_count_of_zero_is_rejected(self) -> None:
        vision, _camera, _locator = _session(reports=[_report(1.0, 0.0, 100.0)])
        with self.assertRaises(ValueError):
            vision.measure(frames=0)

    def test_a_closed_session_still_refuses(self) -> None:
        vision, _camera, _locator = _session(reports=[_report(1.0, 0.0, 100.0)])
        vision.close()
        with self.assertRaises(RuntimeError):
            vision.measure(frames=1)


class MeasureSettledTests(unittest.TestCase):
    def test_it_keeps_sampling_until_the_target_stops_moving(self) -> None:
        """前两批还在动，第三批停了 —— 要一直拍到停稳那一批。"""
        moving = [_report(3.0 * i * 0.2, 0.0, 103.0) for i in range(7)]
        still = [_report(4.2, 0.0, 103.0) for _ in range(7)]
        vision, _camera, _locator = _session(
            reports=moving + moving + still,
            times=[1000.0 + 0.05 * i for i in range(40)],
        )
        fused = vision.measure_settled(timeout=100.0, frames=7)

        self.assertIsNotNone(fused)
        self.assertTrue(fused.confident)
        self.assertFalse(fused.moving)
        self.assertAlmostEqual(fused.point.x_mm, 4.2, delta=0.05)

    def test_on_timeout_it_returns_the_last_burst_instead_of_nothing(self) -> None:
        """超时也要把**最后一批**交出去，并且带着"还在动"这个事实。"""
        moving = [_report(3.0 * i * 0.2, 0.0, 103.0) for i in range(7)]
        ticks = [1000.0 + 0.5 * i for i in range(60)]
        vision, _camera, _locator = _session(
            reports=moving * 6, times=ticks
        )
        fused = vision.measure_settled(timeout=1.0, frames=7)

        self.assertIsNotNone(fused)
        self.assertTrue(fused.moving)
        self.assertFalse(fused.confident)

    def test_nothing_visible_at_all_gives_none(self) -> None:
        vision, _camera, _locator = _session(
            reports=[EMPTY] * 40, times=[1000.0 + 0.5 * i for i in range(60)]
        )
        self.assertIsNone(vision.measure_settled(timeout=1.0, frames=7))


if __name__ == "__main__":
    unittest.main()

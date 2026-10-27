"""视觉会话的新入口：两点测量、工位识别与"动过才算停稳"。

这一组用**假相机 + 假定位器**替身驱动 ``Vision`` 会话，不开真设备、不加载模型。
它测的是**接线**：配置里的三个 ``station`` 块有没有真的传到检测器、检测器有没有被
缓存、以及"读不到画面"与"画面里没有工位"这两件事有没有被分开表达。

（检测器本身的几何正确性在 ``test_ring_station`` / ``test_stacked_ring`` 里测；
这里只保证会话把它们按对的参数装了起来。）
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from modules.vision.config import (
    StationDetectionSettings,
    StationGeometrySettings,
    StationReferenceSettings,
    StackedRingSettings,
    VisionConfig,
)
from modules.vision.drive.ring_station import synthetic_station
from modules.vision.drive.stacked_ring import synthetic_stacked
from modules.vision.runtime import Vision

CENTERS = ((380.0, 320.0), (633.0, 320.0), (886.0, 320.0))


class FakeCamera:
    """假相机：**重复**发同一帧，发够 ``limit`` 次之后返回 ``None``（模拟掉线）。

    为什么是"重复"而不是"一帧一帧弹"：``Vision._read_frame`` 会先丢掉几帧缓冲再取
    真正的那一帧（那是为了避开机械臂动完时压在缓冲里的旧画面），所以一次
    ``detect_*`` 调用要读 3 帧。弹队列的假相机会在第 3 帧就断线，测的就不是检测器了。
    """

    def __init__(self, frame, *, limit: int = 100, width=1280, height=720) -> None:
        self.width = width
        self.height = height
        self._frame = frame
        self._limit = limit
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        if self.reads > self._limit:
            return None
        return self._frame

    def release(self) -> None:
        self.released = True


class FakeLocator:
    """只提供会话收尾所需的接口——这一组不测物料定位。"""

    def __init__(self) -> None:
        self.closed = False

    def locate(self, _frame, **_kwargs):
        raise AssertionError("工位识别不该调用物料定位器")

    def draw(self, frame, _report):
        return frame

    def close(self) -> None:
        self.closed = True


def session_with(frame, *, limit: int = 100) -> Vision:
    """造一个**已就绪**的会话：相机与定位器都是替身，配置用合成帧对应的那一份。"""
    settings = VisionConfig(
        camera_ids={"windows": 0, "jetson": 0},
        station_geometry=StationGeometrySettings(mm_per_px=0.5),
        station_detection=StationDetectionSettings(),
        station_reference=StationReferenceSettings(
            ring_centers_px=CENTERS,
            stable_frame_count=3,
            minimum_confidence=0.0,
        ),
        stacked_ring=StackedRingSettings(
            middle_reference_px=(640.0, 360.0), middle_search_radius_px=260.0
        ),
    )
    session = Vision(settings)
    session._camera = FakeCamera(frame, limit=limit)
    session._locator = FakeLocator()
    return session


class RingStationSessionTests(unittest.TestCase):
    def test_a_station_is_detected_through_the_session(self) -> None:
        session = session_with(synthetic_station(centers=CENTERS))
        detection = session.detect_ring_station()
        self.assertTrue(detection.detected, detection.detail)
        self.assertEqual(len(detection.centers_px), 3)

    def test_the_configured_scale_reaches_the_detector(self) -> None:
        """配置里的 ``mm_per_px`` 必须真的传到检测器，否则毫米字段会是 ``None``。"""
        session = session_with(synthetic_station(centers=CENTERS))
        detection = session.detect_ring_station()
        self.assertTrue(detection.scale_known, "配置里的比例尺没有生效")
        self.assertIsNotNone(detection.position_error_mm)

    def test_the_detector_is_cached_across_calls(self) -> None:
        """稳定窗口要跨帧连续，所以检测器必须是同一个实例。"""
        session = session_with(synthetic_station(centers=CENTERS))
        first = session._ring_station_detector()
        session.detect_ring_station()
        self.assertIs(session._ring_station_detector(), first)

    def test_a_frame_without_a_station_is_a_normal_result(self) -> None:
        """**语义差异**：画面里没有工位是正常结果，不该抛异常。"""
        session = session_with(np.zeros((720, 1280, 3), dtype=np.uint8))
        detection = session.detect_ring_station()
        self.assertFalse(detection.detected)
        self.assertTrue(detection.detail, "没识别到也要说明原因")

    def test_an_unreadable_frame_raises(self) -> None:
        """读不到画面才是失败——与"画面里没有工位"分开表达。"""
        session = session_with(None, limit=0)      # 一次都读不到
        with self.assertRaises(RuntimeError):
            session.detect_ring_station()

    def test_the_locator_is_never_touched(self) -> None:
        """工位识别是经典 CV，不该为了它把 YOLO 也叫起来（FakeLocator 会抛）。"""
        session = session_with(synthetic_station(centers=CENTERS))
        session.detect_ring_station()              # 不抛就说明没调用 locate

    def test_a_closed_session_refuses_to_work(self) -> None:
        session = session_with(synthetic_station(centers=CENTERS))
        session.close()
        with self.assertRaises(RuntimeError):
            session.detect_ring_station()


class StackedRingSessionTests(unittest.TestCase):
    def test_a_stacked_station_is_detected_through_the_session(self) -> None:
        session = session_with(synthetic_stacked(center=(640.0, 360.0)))
        detection = session.detect_stacked_ring()
        self.assertTrue(detection.detected, detection.detail)
        self.assertEqual(detection.line_count, 2)

    def test_the_middle_reference_comes_from_the_station_block(self) -> None:
        """中环参考圆心取 ``station.ring_centers_px`` 的中间那个，不另配一份。"""
        detector = session_with(None, limit=0)._stacked_ring_detector()
        self.assertEqual(detector.middle_reference_px, CENTERS[1])

    def test_the_detector_is_cached_across_calls(self) -> None:
        session = session_with(None, limit=0)
        first = session._stacked_ring_detector()
        self.assertIs(session._stacked_ring_detector(), first)

    def test_an_unreadable_frame_raises(self) -> None:
        session = session_with(None, limit=0)
        with self.assertRaises(RuntimeError):
            session.detect_stacked_ring()


class PixelDistanceSessionTests(unittest.TestCase):
    def test_a_measurement_needs_no_camera_at_all(self) -> None:
        """两点测量是纯算术：会话没开相机也能用。"""
        session = Vision(VisionConfig())
        measured = session.pixel_distance((100.0, 200.0), (400.0, 600.0))
        self.assertEqual(measured.dx_px, 300.0)
        self.assertEqual(measured.dy_px, 400.0)
        self.assertAlmostEqual(measured.length_px, 500.0, delta=1e-9)

    def test_the_frame_size_comes_from_the_camera(self) -> None:
        session = Vision(VisionConfig())
        session._camera = FakeCamera([], width=1920, height=1080)
        measured = session.pixel_distance((0.0, 0.0), (2000.0, 500.0))
        self.assertEqual(measured.image_size, (1920, 1080))
        self.assertIs(measured.end_inside, False)

    def test_the_frame_size_falls_back_to_720p_without_a_camera(self) -> None:
        """相机没开时也能答画幅（配置里就写着 1280×720），所以 inside 不是 None。"""
        measured = Vision(VisionConfig()).pixel_distance((0.0, 0.0), (100.0, 100.0))
        self.assertEqual(measured.image_size, (1280, 720))
        self.assertIs(measured.start_inside, True)

    def test_millimetres_need_a_working_distance(self) -> None:
        """没标定台面深度时报错，而不是猜一个数——比例尺错一个倍数，读数看着却很正常。"""
        session = Vision(VisionConfig())
        with self.assertRaises(ValueError):
            session.pixel_distance_mm((0.0, 0.0), (100.0, 0.0))

    def test_millimetres_are_computed_when_told_the_distance(self) -> None:
        session = Vision(VisionConfig())
        measured = session.pixel_distance_mm(
            (0.0, 0.0), (500.0, 0.0), working_distance_mm=100.0, focal_px=500.0
        )
        self.assertAlmostEqual(measured.length_mm, 100.0, delta=1e-9)
        self.assertAlmostEqual(measured.scale_mm_per_px, 0.2, delta=1e-9)

    def test_a_non_positive_focal_length_is_rejected(self) -> None:
        session = Vision(VisionConfig())
        with self.assertRaises(ValueError):
            session.pixel_distance_mm(
                (0.0, 0.0), (100.0, 0.0), working_distance_mm=100.0, focal_px=0.0
            )


class MotionGatedSettleTests(unittest.TestCase):
    """``require_motion=True`` 的门控接线。

    这里**替掉 ``Vision.measure``**：真正的融合要连拍多帧、还要有相机，
    而这一组要验的是"门控有没有被接上、以及它在什么条件下放行"。

    时钟也替掉：``measure_settled`` 用 ``self._clock()`` 判断超时，而
    ``timeout=0.0`` 会让它在**第一批**没放行时就立刻返回，根本走不到后面那几批。
    假时钟每次调用走 0.01 秒，于是循环能真的迭代、而测试不会真的等。
    """

    def _session(self) -> Vision:
        session = Vision(VisionConfig())
        ticks = [0.0]

        def clock() -> float:
            ticks[0] += 0.01
            return ticks[0]

        session._clock = clock
        return session

    def _fused(self, x_mm: float, stamp: float, *, confident: bool = True):
        from modules.vision.drive.sequence import Fusion, TimedPoint

        return Fusion(
            point=TimedPoint(stamp=stamp, x_mm=x_mm, y_mm=0.0, z_mm=100.0, color="red"),
            spread_mm=0.1,
            samples=5,
            rejected=0,
            misses=0,
            moving=not confident,
            velocity_mm_s=(0.0, 0.0, 0.0),
            span_seconds=0.7,
            colors=("red",),
        )

    def test_without_require_motion_the_first_confident_batch_wins(self) -> None:
        session = self._session()
        with patch.object(Vision, "measure", return_value=self._fused(10.0, 0.0)):
            self.assertIsNotNone(session.measure_settled(timeout=1.0))

    def test_a_stationary_target_never_passes_the_motion_gate(self) -> None:
        """**核心回归**：位置一直不变时，门控永远不放行——静止不等于停稳。

        断言的是**门控本身**（``ever_moved`` 一直是假），不是返回值：
        ``measure_settled`` 超时后会返回最后一批（那是它既有的契约，
        调用方自己决定要不要用），所以"没放行"这件事只能从门控上看出来。
        """
        from modules.vision.drive.motion_gate import MotionGate

        session = self._session()
        gate = MotionGate(moving_speed_px_s=30.0, near_stop_speed_px_s=25.0)
        with patch.object(Vision, "measure", return_value=self._fused(10.0, 0.0)):
            session.measure_settled(timeout=0.2, require_motion=True, motion=gate)
        self.assertFalse(gate.ever_moved, "位置一直不变，门控不该认为它动过")
        self.assertFalse(gate.reported, "没动过就绝不该进入已上报状态")

    def test_motion_then_stop_is_released(self) -> None:
        """先动后停：门控应当在停稳的那一批放行，而且放行的是**停稳之后**那一批。"""
        session = self._session()
        batches = [
            self._fused(0.0, 0.0),
            self._fused(40.0, 0.1),     # 400 mm/s，确认在动
            self._fused(80.0, 0.2),
            self._fused(80.5, 0.3),     # 5 mm/s，低速
            self._fused(80.6, 0.4),
            self._fused(80.7, 0.5),     # 连续 3 帧低速 -> 放行
        ]
        with patch.object(Vision, "measure", side_effect=batches):
            result = session.measure_settled(timeout=5.0, require_motion=True)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.point.x_mm, 80.7, delta=1e-9)

    def test_an_external_gate_keeps_its_state_across_calls(self) -> None:
        """传进来的门控要能跨调用累积状态（转盘一个完整周期跨多次调用时用得上）。"""
        from modules.vision.drive.motion_gate import MotionGate

        session = self._session()
        gate = MotionGate(moving_speed_px_s=30.0, near_stop_speed_px_s=25.0)
        # 第一次调用：位置一直不变，门控不该被置位。
        with patch.object(Vision, "measure", return_value=self._fused(0.0, 0.0)):
            session.measure_settled(timeout=0.2, require_motion=True, motion=gate)
        self.assertFalse(gate.ever_moved)

        # 第二次调用：同一个门控，位置开始移动，累积到"动过"。
        moving = [self._fused(0.0, 1.0), self._fused(40.0, 1.1), self._fused(80.0, 1.2)]
        with patch.object(Vision, "measure", side_effect=moving * 20):
            session.measure_settled(timeout=0.3, require_motion=True, motion=gate)
        self.assertTrue(gate.ever_moved, "同一个门控应当累积到动过这个事实")


if __name__ == "__main__":
    unittest.main()

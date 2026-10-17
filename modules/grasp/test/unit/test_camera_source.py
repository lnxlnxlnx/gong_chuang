"""视觉会话（``modules.vision.runtime.Vision``）：观测转换、取帧语义与资源所有权。

``modules/grasp/drive/camera_source.py`` 已经删除，它的两件事搬到了这里：

1. ``MaterialReport`` → ``MaterialSighting`` 的映射，现在是私有函数
   :func:`~modules.vision.runtime._sighting_from_region` /
   :func:`~modules.vision.runtime._sightings_from_report`；
2. "丢缓冲 → 读帧 → 定位"的取帧流程，现在是 :meth:`Vision._read_report`。

最容易错的一件事还是**编号**：终端里 ``MaterialReport.summary()`` 打的是
``物料 1:``、``物料 2:``……，而 ``ObjectSelector(material_id=N)`` 要能选中同一条。
所以这里把两边的编号对上当作硬判据——错开了，用户看到的编号就指挥不动程序。

另一件是**所有权**：``Vision`` 是会话，它**总是**拥有自己取得的相机与定位器，
不像原来的 ``CameraFrameSource`` 那样按 ``owns_*`` 参数区分。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from modules.vision.drive.hough_lock import LockedRegion
from modules.vision.drive.material_class import ClassResult
from modules.vision.drive.material_face import FaceQuality
from modules.vision.reports import MaterialReport, RegionReport
from modules.vision.runtime import (
    Vision,
    _sighting_from_region,
    _sightings_from_report,
)


def make_region(
    *,
    center_x: float = 543.0,
    center_y: float = 382.0,
    radius: float = 136.0,
    x_mm: float = -10.8,
    y_mm: float = 2.4,
    distance_mm: float = 111.8,
    accepted: bool = True,
    name: "str | None" = "green",
    label: "str | None" = "绿色",
    score: float = 0.85,
    found: bool = True,
    interior_std: float = 12.7,
    edge_support: float = 0.61,
    sampled_rays: int = 36,
) -> RegionReport:
    return RegionReport(
        region=LockedRegion(center_x, center_y, radius),
        classification=ClassResult(found=found, name=name, label=label, score=score),
        face=FaceQuality(
            interior_std=interior_std, edge_support=edge_support, sampled_rays=sampled_rays
        ),
        accepted=accepted,
        x_mm=x_mm,
        y_mm=y_mm,
        distance_mm=distance_mm,
    )


def make_report(regions) -> MaterialReport:
    return MaterialReport(lock=(), hits=(), regions=tuple(regions))


class SightingFromRegionTests(unittest.TestCase):
    def test_maps_every_field(self) -> None:
        sighting = _sighting_from_region(make_region(), 1, received_at=123.5)
        self.assertEqual(sighting.material_id, 1)
        self.assertAlmostEqual(sighting.x_mm, -10.8)
        self.assertAlmostEqual(sighting.y_mm, 2.4)
        self.assertAlmostEqual(sighting.z_mm, 111.8)
        self.assertAlmostEqual(sighting.pixel_u, 543.0)
        self.assertAlmostEqual(sighting.pixel_v, 382.0)
        self.assertEqual(sighting.color, "green")
        self.assertAlmostEqual(sighting.confidence, 0.85)
        # 直径是半径的两倍，不是半径。
        self.assertAlmostEqual(sighting.diameter_px, 272.0)
        self.assertAlmostEqual(sighting.inner_sigma, 12.7)
        # 边界支撑在报告里是 0–1 的比例，观测里是百分数。
        self.assertAlmostEqual(sighting.edge_support, 61.0)
        self.assertEqual(sighting.edge_directions, 36)
        self.assertEqual(sighting.received_at, 123.5)

    def test_unclassified_region_has_no_colour_or_confidence(self) -> None:
        """YOLO 没认出来时不能编一个颜色或置信度——那会让 ObjectSelector 选错。"""
        sighting = _sighting_from_region(
            make_region(found=False, name=None, label=None, score=0.0), 2
        )
        self.assertIsNone(sighting.color)
        self.assertIsNone(sighting.confidence)

    def test_records_the_region_summary_as_raw(self) -> None:
        sighting = _sighting_from_region(make_region(), 1)
        self.assertIn("xyz=", sighting.raw)
        self.assertIn("绿色", sighting.raw)


class SightingsFromReportTests(unittest.TestCase):
    def test_numbers_accepted_faces_from_one(self) -> None:
        report = make_report([make_region(), make_region(), make_region()])
        sightings = _sightings_from_report(report)
        self.assertEqual([item.material_id for item in sightings], [1, 2, 3])

    def test_rejected_candidates_are_not_numbered(self) -> None:
        """被丢掉的候选（夹爪圆弧、背景纹理）不是物料，不该占编号。"""
        report = make_report(
            [
                make_region(accepted=True, x_mm=1.0),
                make_region(accepted=False, x_mm=99.0),
                make_region(accepted=True, x_mm=2.0),
            ]
        )
        sightings = _sightings_from_report(report)
        self.assertEqual([item.material_id for item in sightings], [1, 2])
        self.assertEqual([round(item.x_mm) for item in sightings], [1, 2])

    def test_numbering_matches_what_the_terminal_prints(self) -> None:
        """硬判据：``summary()`` 里的「物料 N:」必须和 ``material_id=N`` 是同一条。

        两边错开的话，用户在终端看到的编号就指挥不动程序——这是最难倒查的一类错误。
        """
        report = make_report(
            [make_region(x_mm=-10.8), make_region(accepted=False, x_mm=50.0), make_region(x_mm=7.5)]
        )
        printed = [
            line.strip()
            for line in report.summary().splitlines()
            if line.strip().startswith("物料 ")
        ]
        self.assertEqual(len(printed), 2, printed)
        for index, line in enumerate(printed, start=1):
            self.assertTrue(line.startswith(f"物料 {index}:"), line)
            # 打印出来的那一条里带的位置，必须等于编号相同的观测。
            marker = f"xyz=({_sightings_from_report(report)[index - 1].x_mm:.1f},"
            self.assertIn(marker, line)

    def test_empty_frame_gives_no_sightings(self) -> None:
        self.assertEqual(_sightings_from_report(make_report([])), ())


class FakeCamera:
    """按脚本给出帧；脚本用完之后一直返回 ``None``。"""

    def __init__(self, frames) -> None:
        self._frames = list(frames)
        self.reads = 0
        self.released = 0
        self.initialized = False

    def initialize(self):
        self.initialized = True
        return self

    def read(self):
        self.reads += 1
        if not self._frames:
            return None
        return self._frames.pop(0)

    def release(self):
        self.released += 1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()


class FakeLocator:
    def __init__(self, reports) -> None:
        self._reports = list(reports)
        self.calls = 0
        self.closed = 0

    def locate(self, _frame):
        self.calls += 1
        return self._reports.pop(0)

    def close(self):
        self.closed += 1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class VisionSessionTests(unittest.TestCase):
    def build(self, frames, reports, **overrides):
        """按私有装配缝隙造一个会话：不读配置、不加载模型、不开相机。"""
        vision = Vision(None)
        camera = FakeCamera(frames)
        locator = FakeLocator(reports)
        vision._camera = camera
        vision._locator = locator
        for name, value in overrides.items():
            setattr(vision, name, value)
        return vision, camera, locator

    # ------------------------------------------------------------ 取帧与定位
    def test_returns_the_converted_sightings(self) -> None:
        vision, camera, locator = self.build(
            # 默认 drain_frames=2：先丢 2 帧，第 3 帧才是要用的那一帧。
            ["drained1", "drained2", "used"],
            [make_report([make_region()])],
        )
        frame = vision.frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame[0].material_id, 1)
        self.assertAlmostEqual(frame[0].x_mm, -10.8)
        self.assertIsNotNone(frame[0].received_at, "必须打时间戳，下游靠它判断新鲜度")
        self.assertEqual(camera.reads, 3, "2 帧丢弃 + 1 帧使用")
        self.assertEqual(locator.calls, 1)
        self.assertEqual(vision.frames, 1)

    def test_the_timestamp_is_taken_after_the_read_and_before_the_inference(self) -> None:
        """``received_at`` 是"手上这帧有多旧"的依据，不能用推理结束时间冒充。

        给了超时的时候时钟被读两次：一次算 deadline、一次**读完帧之后**打时间戳。
        时间戳必须是第二次（读完之后），不是推理结束之后。
        """
        vision, _camera, _locator = self.build(
            ["d1", "d2", "used"], [make_report([make_region()])]
        )
        vision._clock = Mock(side_effect=[10.0, 30.0])

        frame = vision.frame(timeout=0.0)

        assert frame is not None
        self.assertEqual(frame[0].received_at, 30.0)
        self.assertEqual(vision._clock.call_count, 2, "算 deadline + 打时间戳")

    def test_every_read_drains_buffered_frames_first(self) -> None:
        """**每次**读都要先丢缓冲：第一次读同样可能拿到驱动里的旧画面。

        实测第一次读数会偏一两毫米，随后才稳定；而伺服容差只有几毫米——
        那种偏差足以让闭环算错方向，所以不能只对"动作之后"那次读做丢弃。
        """
        vision, camera, _locator = self.build(
            ["old1", "old2", "fresh", "old3", "fresh2"],
            [make_report([make_region()]), make_report([make_region()])],
            _drain_frames=2,
        )
        vision.frame()  # 丢 2 帧 + 取 1 帧
        self.assertEqual(camera.reads, 3)
        vision.frame(min_received_at=1000.0)  # 同样丢 2 帧 + 取 1 帧
        self.assertEqual(camera.reads, 6)

    def test_no_frame_returns_none_without_calling_the_locator(self) -> None:
        vision, _camera, locator = self.build([], [make_report([make_region()])])
        self.assertIsNone(vision.frame(timeout=0.0))
        self.assertEqual(locator.calls, 0, "没画面就不该去定位")
        self.assertGreaterEqual(vision.misses, 1)

    def test_locate_reports_a_missing_frame_instead_of_returning_none(self) -> None:
        """``locate`` 是面向业务的：读不到画面是失败，不是"空的报告"。"""
        vision, _camera, locator = self.build([], [])
        with self.assertRaisesRegex(RuntimeError, "读不到摄像头画面"):
            vision.locate()
        self.assertEqual(locator.calls, 0)

    def test_locate_returns_the_report_itself(self) -> None:
        report = make_report([make_region()])
        vision, _camera, _locator = self.build(["d1", "d2", "used"], [report])
        self.assertIs(vision.locate(), report)

    def test_empty_report_is_still_a_frame(self) -> None:
        """画面里没有物料时返回**空元组**而不是 None——那是"看到了，但没有物料"。"""
        vision, _camera, _locator = self.build(
            ["drained1", "drained2", "used"], [make_report([])]
        )
        self.assertEqual(vision.frame(), ())

    def test_a_read_failure_is_counted_and_stops_at_the_deadline(self) -> None:
        """掉线时不能无限重试：到点就如实返回"没读到"，也不去推理。"""
        vision, camera, locator = self.build([], [make_report([])])
        # 时钟：deadline = 0 + 1.0；第一次检查 0.0（还早，睡一觉重试），第二次 1.5（收工）。
        ticks = iter([0.0, 0.0, 1.5])
        vision._clock = lambda: next(ticks)

        with patch("modules.vision.runtime.time.sleep") as sleep:
            self.assertIsNone(vision.frame(timeout=1.0))

        sleep.assert_called_once()
        self.assertGreaterEqual(vision.misses, 2)
        self.assertGreaterEqual(camera.reads, 3, "丢缓冲 2 帧 + 至少一次真实读")
        self.assertEqual(locator.calls, 0, "读不到帧就不该推理")

    # ------------------------------------------------------------ 资源所有权
    def test_close_always_releases_the_acquired_resources(self) -> None:
        """会话**总是**拥有自己取得的资源：没有 ``owns_*`` 开关这回事。"""
        vision, camera, locator = self.build(["f1"], [make_report([])])
        vision.close()
        self.assertEqual(camera.released, 1)
        self.assertEqual(locator.closed, 1)
        # 重复 close 不该重复关。
        vision.close()
        self.assertEqual(camera.released, 1)
        self.assertEqual(locator.closed, 1)

    def test_context_manager_closes(self) -> None:
        vision, camera, locator = self.build(["f1"], [make_report([])])
        with vision:
            pass
        self.assertEqual((camera.released, locator.closed), (1, 1))

    def test_close_reports_every_failing_stage(self) -> None:
        """相机、定位器分别尝试：一个失败不阻断另一个。"""
        vision, camera, locator = self.build(["f1"], [make_report([])])
        camera.release = Mock(side_effect=RuntimeError("相机没释放"))
        locator.close = Mock(side_effect=RuntimeError("定位器没关"))

        with self.assertRaises(RuntimeError) as caught:
            vision.close()

        message = str(caught.exception)
        self.assertIn("camera.release", message)
        self.assertIn("locator.close", message)
        self.assertIn("Vision 清理失败", message)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception.__cause__), "相机没释放")

    def test_actions_after_close_raise(self) -> None:
        vision, _camera, _locator = self.build(["f1"], [make_report([])])
        vision.close()
        for name in ("frame", "locate"):
            with self.subTest(method=name):
                with self.assertRaisesRegex(RuntimeError, "已关闭"):
                    getattr(vision, name)()

    def test_open_config_assembles_the_locator_and_the_camera_in_order(self) -> None:
        """真正的装配路径：先加载模型，再构造相机，最后 ``initialize``。"""
        order = []
        settings = Mock()
        settings.camera_id.return_value = 7
        camera = FakeCamera([])
        with patch(
            "modules.vision.runtime._build_locator",
            side_effect=lambda *_a, **_k: order.append("locator") or Mock(),
        ), patch(
            "modules.vision.runtime._new_camera",
            side_effect=lambda camera_id: order.append(("camera", camera_id)) or camera,
        ):
            vision = Vision._open_config(settings)

        self.assertEqual(order, ["locator", ("camera", 7)])
        self.assertEqual(camera.initialized, True)

    def test_satisfies_the_frame_source_protocol_shape(self) -> None:
        """``servo_observation`` 只要求一个 ``frame(**kwargs)``；签名对不上就会在真机上炸。"""
        import inspect

        vision, _camera, _locator = self.build(["f1"], [make_report([])])
        parameters = inspect.signature(vision.frame).parameters
        for name in ("timeout", "max_age_seconds", "min_received_at"):
            self.assertIn(name, parameters, name)
            self.assertIsNone(parameters[name].default)


if __name__ == "__main__":
    unittest.main()

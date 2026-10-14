"""本机摄像头观测源：``MaterialReport`` → ``MaterialSighting`` 的映射。

最容易错的一件事是**编号**：终端里 ``MaterialReport.summary()`` 打的是
``物料 1:``、``物料 2:``……，而 ``ObjectSelector(material_id=N)`` 要能选中同一条。
所以这里把两边的编号对上当作硬判据——错开了，用户看到的编号就指挥不动程序。
"""

from __future__ import annotations

import unittest

from modules.grasp.drive.camera_source import (
    CameraFrameSource,
    sighting_from_region,
    sightings_from_report,
)
from modules.vision.drive.hough_lock import LockedRegion
from modules.vision.drive.material_class import ClassResult
from modules.vision.drive.material_face import FaceQuality
from modules.vision.drive.material_locate import MaterialReport, RegionReport


def make_region(
    *,
    center_x: float = 543.0,
    center_y: float = 382.0,
    radius: float = 136.0,
    x_mm: float = -10.8,
    y_mm: float = 2.4,
    distance_mm: float = 111.8,
    accepted: bool = True,
    name: str | None = "green",
    label: str | None = "绿色",
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
        sighting = sighting_from_region(make_region(), 1, received_at=123.5)
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
        sighting = sighting_from_region(
            make_region(found=False, name=None, label=None, score=0.0), 2
        )
        self.assertIsNone(sighting.color)
        self.assertIsNone(sighting.confidence)

    def test_records_the_region_summary_as_raw(self) -> None:
        sighting = sighting_from_region(make_region(), 1)
        self.assertIn("xyz=", sighting.raw)
        self.assertIn("绿色", sighting.raw)


class SightingsFromReportTests(unittest.TestCase):
    def test_numbers_accepted_faces_from_one(self) -> None:
        report = make_report([make_region(), make_region(), make_region()])
        sightings = sightings_from_report(report)
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
        sightings = sightings_from_report(report)
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
            marker = f"xyz=({sightings_from_report(report)[index - 1].x_mm:.1f},"
            self.assertIn(marker, line)

    def test_empty_frame_gives_no_sightings(self) -> None:
        self.assertEqual(sightings_from_report(make_report([])), ())


class FakeCamera:
    """按脚本给出帧；``read()`` 返回 ``None`` 之后一直返回 ``None``。"""

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
        value = self._frames.pop(0)
        return value

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
        self.exited = 0

    def locate(self, _frame):
        self.calls += 1
        return self._reports.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.exited += 1


class CameraFrameSourceTests(unittest.TestCase):
    def build(self, frames, reports, **kwargs):
        camera = FakeCamera(frames)
        locator = FakeLocator(reports)
        return CameraFrameSource(camera, locator, **kwargs), camera, locator

    def test_returns_the_converted_sightings(self) -> None:
        source, camera, locator = self.build(
            # 默认 drain_frames=2：先丢 2 帧，第 3 帧才是要用的那一帧。
            ["drained1", "drained2", "used"],
            [make_report([make_region()])],
        )
        frame = source.frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame[0].material_id, 1)
        self.assertAlmostEqual(frame[0].x_mm, -10.8)
        self.assertIsNotNone(frame[0].received_at, "必须打时间戳，下游靠它判断新鲜度")
        self.assertEqual(camera.reads, 3, "2 帧丢弃 + 1 帧使用")
        self.assertEqual(locator.calls, 1)
        self.assertEqual(source.frames, 1)

    def test_every_read_drains_buffered_frames_first(self) -> None:
        """**每次**读都要先丢缓冲：第一次读同样可能拿到驱动里的旧画面。

        实测第一次读数会偏一两毫米，随后才稳定；而伺服容差只有几毫米——
        那种偏差足以让闭环算错方向，所以不能只对"动作之后"那次读做丢弃。
        """
        source, camera, _locator = self.build(
            ["old1", "old2", "fresh", "old3", "fresh2"],
            [make_report([make_region()]), make_report([make_region()])],
            drain_frames=2,
        )
        source.frame()  # 丢 2 帧 + 取 1 帧
        self.assertEqual(camera.reads, 3)
        source.frame(min_received_at=1000.0)  # 同样丢 2 帧 + 取 1 帧
        self.assertEqual(camera.reads, 6)

    def test_no_frame_returns_none_without_calling_the_locator(self) -> None:
        source, camera, locator = self.build([], [make_report([make_region()])])
        self.assertIsNone(source.frame(timeout=0.0))
        self.assertEqual(locator.calls, 0, "没画面就不该去定位")
        self.assertGreaterEqual(source.misses, 1)

    def test_empty_report_is_still_a_frame(self) -> None:
        """画面里没有物料时返回**空元组**而不是 None——那是"看到了，但没有物料"。"""
        source, _camera, _locator = self.build(
            ["drained1", "drained2", "used"], [make_report([])]
        )
        frame = source.frame()
        self.assertEqual(frame, ())

    def test_close_only_releases_owned_resources(self) -> None:
        source, camera, locator = self.build(["f1"], [make_report([])])
        source.close()
        self.assertEqual((camera.released, locator.exited), (0, 0),
                         "注入进来的资源由调用方负责，不该被 here 关掉")

    def test_close_releases_owned_resources(self) -> None:
        camera = FakeCamera(["f1"])
        locator = FakeLocator([make_report([])])
        source = CameraFrameSource(
            camera, locator, owns_camera=True, owns_locator=True
        )
        source.close()
        self.assertEqual(camera.released, 1)
        self.assertEqual(locator.exited, 1)
        # 重复 close 不该重复关。
        source.close()
        self.assertEqual((camera.released, locator.exited), (1, 1))

    def test_context_manager_closes(self) -> None:
        camera = FakeCamera(["f1"])
        locator = FakeLocator([make_report([])])
        with CameraFrameSource(camera, locator, owns_camera=True, owns_locator=True):
            pass
        self.assertEqual((camera.released, locator.exited), (1, 1))

    def test_negative_drain_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CameraFrameSource(FakeCamera([]), FakeLocator([]), drain_frames=-1)

    def test_satisfies_the_frame_source_protocol_shape(self) -> None:
        """``eye_in_hand`` 只要求一个 ``frame(**kwargs)``；签名对不上就会在真机上炸。"""
        import inspect

        source, _camera, _locator = self.build(["f1"], [make_report([])])
        parameters = inspect.signature(source.frame).parameters
        for name in ("timeout", "max_age_seconds", "min_received_at"):
            self.assertIn(name, parameters, name)
            self.assertIsNone(parameters[name].default)


if __name__ == "__main__":
    unittest.main()

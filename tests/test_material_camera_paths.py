"""Exercise the camera/CLI branches of the material modules without hardware.

These branches only run when a USB camera is plugged in, which is exactly how a
wrong import (``looks_blank`` lives in ``vision.camera``, not ``vision.preview``)
once slipped through: nothing in the offline suite ever executed them. Here the
camera, the preview window and the GUI calls are all faked, so imports, the
frame loop, cleanup and return codes are covered by the normal test run.
"""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vision import hough_lock
from vision.hough_lock import HoughRingLocator, LockedRegion, RingLock
from vision.yolov8 import Detection


class FakeCamera:
    """Same lifecycle as ``USBCamera``, always returns one flat frame."""

    instances: list["FakeCamera"] = []

    def __init__(self, device_id: int = 0, **kwargs) -> None:
        self.device_id = device_id
        self.width = kwargs.get("width")
        self.height = kwargs.get("height")
        self.frames_read = 0
        FakeCamera.instances.append(self)

    def initialize(self) -> "FakeCamera":
        return self

    def __enter__(self) -> "FakeCamera":
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def read(self):
        self.frames_read += 1
        return np.full((480, 640, 3), 100, dtype=np.uint8)


def stop_after_one_frame():
    """A ``cv2.waitKey`` replacement that reports q on its first call."""
    state = {"calls": 0}

    def fake_wait_key(_delay: int = 0) -> int:
        state["calls"] += 1
        return ord("q") if state["calls"] >= 1 else -1

    return fake_wait_key


def detection(name: str = "obj-red", **kwargs) -> Detection:
    settings = dict(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.9, class_id=0)
    settings.update(kwargs)
    return Detection(name=name, **settings)


class PackageImportGuardTests(unittest.TestCase):
    def test_camera_helpers_are_importable_from_the_package(self) -> None:
        # 摄像头分支统一用 from vision import ...，这些名字必须在包上可见。
        import vision

        for name in (
            "BLANK_FRAME_WARNING",
            "USBCamera",
            "close_previews",
            "looks_blank",
            "open_preview",
            "warn_blank_preview",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(vision, name), f"vision.{name} 不存在")

    def test_the_camera_branches_do_not_import_helpers_from_the_wrong_module(self) -> None:
        # looks_blank 定义在 vision.camera；从 vision.preview 导入它是曾经的 bug。
        from vision import camera, preview

        self.assertTrue(hasattr(camera, "looks_blank"))
        self.assertFalse(hasattr(preview, "looks_blank"))


class HoughLockCameraTests(unittest.TestCase):
    def test_camera_branch_runs_one_frame_and_closes_the_window(self) -> None:
        with patch("vision.USBCamera", FakeCamera), \
                patch("vision.open_preview", return_value="vision-preview") as open_preview, \
                patch("vision.close_previews") as close_previews, \
                patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = hough_lock._run_camera(HoughRingLocator(), 1)

        self.assertEqual(result, 0)
        open_preview.assert_called_once()
        close_previews.assert_called_once()
        self.assertEqual(imshow.call_count, 1)
        self.assertEqual(FakeCamera.instances[-1].device_id, 1)


class MaterialYoloCameraTests(unittest.TestCase):
    class FakeYOLO:
        def __init__(self, *_args, **_kwargs) -> None:
            self.materials = ("obj-red",)
            self.detector = self

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def detect(self, _frame):
            return [detection()]

        def draw(self, frame, _hits):
            return frame.copy()

    def test_camera_branch_runs_one_frame_and_closes_the_window(self) -> None:
        from vision import material_yolo

        with patch.object(material_yolo, "MaterialYOLO", self.FakeYOLO), \
                patch("vision.USBCamera", FakeCamera), \
                patch("vision.open_preview", return_value="vision-preview"), \
                patch("vision.close_previews") as close_previews, \
                patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = material_yolo._run_camera(None, 1, 0.25)

        self.assertEqual(result, 0)
        close_previews.assert_called_once()
        self.assertEqual(imshow.call_count, 1)


class MaterialLocateCameraTests(unittest.TestCase):
    class FakeReport:
        def __init__(self, best=None) -> None:
            self.lock = RingLock(regions=(LockedRegion(320.0, 240.0, 40.0),), expected=3)
            self.best = best

        def summary(self) -> str:
            return "fake report"

    class FakeLocator:
        report = None

        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def locate(self, _frame):
            return type(self).report

        def draw(self, frame, _report):
            return frame.copy()

    def build_args(self, **overrides) -> argparse.Namespace:
        settings = dict(
            model=None, camera=1, conf=0.25, focal=None
        )
        settings.update(overrides)
        return argparse.Namespace(**settings)

    def run_camera(self, report) -> int:
        from vision import material_locate

        self.FakeLocator.report = report
        with patch.object(material_locate, "MaterialLocator", self.FakeLocator), \
                patch("vision.USBCamera", FakeCamera), \
                patch("vision.open_preview", return_value="vision-preview"), \
                patch("vision.close_previews") as close_previews, \
                patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = material_locate._run_camera(self.build_args())
        self.assertTrue(close_previews.called)
        self.assertEqual(imshow.call_count, 1)
        return result

    def test_camera_branch_without_a_material(self) -> None:
        self.assertEqual(self.run_camera(self.FakeReport(best=None)), 0)

    def test_camera_branch_with_a_material(self) -> None:
        best = SimpleNamespace(name="obj-red", classification=SimpleNamespace(score=0.91))
        self.assertEqual(self.run_camera(self.FakeReport(best=best)), 0)

    def test_camera_requests_720p(self) -> None:
        self.run_camera(self.FakeReport(best=None))
        self.assertEqual((FakeCamera.instances[-1].width, FakeCamera.instances[-1].height), (1280, 720))


class MainDemoMaterialTests(unittest.TestCase):
    """main.demo_material is the documented entry point; cover its wiring too."""

    class FakeReport:
        best = None

        @property
        def regions(self):
            return () if self.best is None else (self.best,)

        @property
        def present(self):
            return bool(self.regions)

        def summary(self) -> str:
            return "fake"

    class FakeLocator:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def locate(self, _frame):
            return MainDemoMaterialTests.FakeReport()

        def draw(self, frame, _report):
            return frame.copy()

    def test_demo_runs_one_frame_and_closes_the_window(self) -> None:
        import main

        with patch("vision.MaterialLocator", self.FakeLocator), \
                patch("vision.USBCamera", FakeCamera), \
                patch("vision.open_preview", return_value="vision-preview"), \
                patch("vision.close_previews") as close_previews, \
                patch("cv2.imshow"), \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            main.demo_material(model_path=Path(__file__), camera_id=1)

        self.assertTrue(close_previews.called)
        self.assertEqual(FakeCamera.instances[-1].device_id, 1)

    def test_demo_shows_coordinates_and_prints_all_results_on_p(self) -> None:
        import main

        item = SimpleNamespace(
            name="obj-blue", confidence=0.81,
            x_mm=20.0, y_mm=-10.0, z_mm=75.0, range_mm=78.3,
        )
        with patch.object(self.FakeReport, "best", item), \
                patch("vision.MaterialLocator", self.FakeLocator), \
                patch("vision.USBCamera", FakeCamera), \
                patch("vision.open_preview", return_value="vision-preview"), \
                patch("vision.close_previews"), \
                patch("cv2.imshow"), \
                patch("cv2.putText") as put_text, \
                patch("cv2.waitKey", side_effect=[ord("p"), ord("q")]), \
                patch("builtins.print") as printed:
            main.demo_material(model_path=Path(__file__), camera_id=1)

        overlay = " ".join(call.args[1] for call in put_text.call_args_list)
        self.assertIn("obj-blue", overlay)
        self.assertIn("0.81", overlay)
        self.assertIn("20", overlay)
        self.assertIn("-10", overlay)
        self.assertIn("75", overlay)
        self.assertIn("78", overlay)
        self.assertTrue(any(call.args == ("fake",) for call in printed.call_args_list))


class ManualMenuEntryTests(unittest.TestCase):
    """菜单 18 的真机入口（tests/manual/material_locate.py）也要能离线跑通。"""

    class FakeReport:
        class _Lock:
            expected = 3

            def __len__(self) -> int:
                return 3

        lock = _Lock()
        best = None

        def summary(self) -> str:
            return "fake"

    class FakeLocator:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def locate(self, _frame):
            return ManualMenuEntryTests.FakeReport()

        def draw(self, frame, _report):
            return frame.copy()

    def test_run_returns_zero_after_q(self) -> None:
        from tests.manual import material_locate as menu

        # 手动脚本把名字直接 import 到了自己的命名空间，所以 patch 它的模块属性。
        with patch.object(menu, "MaterialLocator", self.FakeLocator), \
                patch.object(menu, "USBCamera", FakeCamera), \
                patch.object(menu, "open_preview", return_value="vision-preview") as open_preview, \
                patch.object(menu, "close_previews") as close_previews, \
                patch("cv2.imshow"), \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = menu.run(1, Path(__file__))

        self.assertEqual(result, 0)
        open_preview.assert_called_once()
        self.assertTrue(close_previews.called)

    def test_missing_model_is_reported_without_opening_a_window(self) -> None:
        from tests.manual import material_locate as menu

        with patch.object(menu, "open_preview") as open_preview:
            result = menu.run(1, Path("no-such-model.onnx"))

        self.assertEqual(result, 1)
        open_preview.assert_not_called()

    def test_menu_entry_number_and_label(self) -> None:
        from tests.manual import material_locate as menu

        self.assertEqual(menu.MENU_ENTRY, (18, "物料定位"))
        self.assertTrue(callable(menu.main))


if __name__ == "__main__":
    unittest.main()

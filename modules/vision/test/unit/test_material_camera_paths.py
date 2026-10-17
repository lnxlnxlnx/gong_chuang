"""Exercise the camera/CLI branches of the material modules without hardware.

These branches only run when a USB camera is plugged in, which is exactly how a
wrong import (``looks_blank`` lives in ``modules.vision.drive.camera``, not ``modules.vision.drive.preview``)
once slipped through: nothing in the offline suite ever executed them. Here the
camera, the preview window and the GUI calls are all faked, so imports, the
frame loop, cleanup and return codes are covered by the normal test run.

Patch targets follow the code, not the old package surface: tools imported at
module top level (menu 18) are patched on that module, tools imported inside a
function are patched where they are defined, and session creation is patched on
``modules.vision.runtime`` (``_load_settings`` / ``_build_locator``) or on
``modules.vision.drive.camera.USBCamera`` for the real ``_new_camera``.
"""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from modules.vision.drive import hough_lock
from modules.vision.drive.hough_lock import HoughRingLocator, LockedRegion, RingLock
from modules.vision.drive.yolov8 import Detection


class FakeCamera:
    """Same lifecycle as ``USBCamera``, always returns one flat frame."""

    instances: list["FakeCamera"] = []

    def __init__(self, device_id: int = 0, **kwargs) -> None:
        self.device_id = device_id
        self.width = kwargs.get("width")
        self.height = kwargs.get("height")
        self.frames_read = 0
        self.releases = 0
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

    def release(self) -> None:
        """``Vision.close`` 归还设备走的就是它；假摄像头只要记下这次调用。"""
        self.releases += 1


class SilentCamera(FakeCamera):
    """A camera that opens but never delivers a frame (unplugged / occupied)."""

    def read(self):
        self.frames_read += 1
        return None


def stop_after_one_frame():
    """A ``cv2.waitKey`` replacement that reports q on its first call."""
    state = {"calls": 0}

    def fake_wait_key(_delay: int = 0) -> int:
        state["calls"] += 1
        return ord("q") if state["calls"] >= 1 else -1

    return fake_wait_key


def detection(name: str = "red", **kwargs) -> Detection:
    settings = dict(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.9, class_id=0)
    settings.update(kwargs)
    return Detection(name=name, **settings)


def vision_settings(camera_id: int = 1):
    """A hand-built session snapshot: this platform's camera id, no targets.

    会话入口现在都先读一份快照再走 ``Vision._open_config``。这里不读真的
    ``config.json``：快照就是 ``main()`` / ``demo_material`` 会拿到的那份形状，
    装配步骤由各用例显式 mock 掉。
    """
    from modules.vision.config import TargetSettings, VisionConfig, current_platform

    return VisionConfig(
        camera_ids={current_platform(): camera_id},
        targets=TargetSettings(),
    )


class CameraTestCase(unittest.TestCase):
    """Base for the camera branches: every test starts with an empty fake list."""

    def setUp(self) -> None:
        FakeCamera.instances.clear()


class PackageImportGuardTests(unittest.TestCase):
    def test_the_package_exports_only_the_session_and_the_two_reports(self) -> None:
        """收敛后的契约：包出口恰好三个符号，而且是定义处的同一个对象。"""
        from modules import vision
        from modules.vision.reports import MaterialReport, RegionReport
        from modules.vision.runtime import Vision

        self.assertEqual(vision.__all__, ["MaterialReport", "RegionReport", "Vision"])
        self.assertIs(vision.MaterialReport, MaterialReport)
        self.assertIs(vision.RegionReport, RegionReport)
        self.assertIs(vision.Vision, Vision)

    def test_camera_helpers_live_in_their_defining_modules_only(self) -> None:
        """相机/预览/取流工具各有真实归属，包顶层不再转发它们。"""
        from modules import vision
        from modules.vision.config import default_model_path
        from modules.vision.detector_factory import create_detector
        from modules.vision.drive import camera, preview, stream

        self.assertTrue(callable(camera.USBCamera))
        self.assertTrue(callable(camera.looks_blank))
        self.assertTrue(callable(camera.load_camera_id))
        for name in (
            "BLANK_FRAME_WARNING",
            "close_previews",
            "open_preview",
            "warn_blank_preview",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(preview, name), f"preview.{name} 不存在")
        self.assertTrue(callable(stream.run_preview))
        self.assertTrue(callable(default_model_path))
        self.assertTrue(callable(create_detector))

        # 包出口不再有这些名字：旧调用方必须改成从定义处导入，而不是靠包转发。
        for name in (
            "BLANK_FRAME_WARNING",
            "USBCamera",
            "close_previews",
            "create_detector",
            "default_model_path",
            "load_camera_id",
            "looks_blank",
            "open_preview",
            "run_preview",
            "warn_blank_preview",
            "MaterialLocator",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(vision, name), f"vision.{name} 不该再存在")

    def test_the_camera_branches_do_not_import_helpers_from_the_wrong_module(self) -> None:
        # looks_blank 定义在 modules.vision.drive.camera；从 modules.vision.drive.preview 导入它是曾经的 bug。
        from modules.vision.drive import camera, preview

        self.assertTrue(hasattr(camera, "looks_blank"))
        self.assertFalse(hasattr(preview, "looks_blank"))


class HoughLockCameraTests(CameraTestCase):
    def test_camera_branch_runs_one_frame_and_closes_the_window(self) -> None:
        # _run_camera 在函数内 from ...drive.camera import USBCamera，所以 patch 定义处。
        with patch("modules.vision.drive.camera.USBCamera", FakeCamera), \
                patch("modules.vision.drive.preview.open_preview",
                      return_value="vision-preview") as open_preview, \
                patch("modules.vision.drive.preview.close_previews") as close_previews, \
                patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = hough_lock._run_camera(HoughRingLocator(), 1)

        self.assertEqual(result, 0)
        open_preview.assert_called_once()
        close_previews.assert_called_once()
        self.assertEqual(imshow.call_count, 1)
        self.assertEqual(FakeCamera.instances[-1].device_id, 1)


class MaterialYoloCameraTests(CameraTestCase):
    class FakeYOLO:
        def __init__(self, *_args, **_kwargs) -> None:
            self.materials = ("red",)
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
        from modules.vision.drive import material_yolo

        with patch.object(material_yolo, "MaterialYOLO", self.FakeYOLO), \
                patch("modules.vision.drive.camera.USBCamera", FakeCamera), \
                patch("modules.vision.drive.preview.open_preview", return_value="vision-preview"), \
                patch("modules.vision.drive.preview.close_previews") as close_previews, \
                patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = material_yolo._run_camera(None, 1, 0.25)

        self.assertEqual(result, 0)
        close_previews.assert_called_once()
        self.assertEqual(imshow.call_count, 1)
        self.assertEqual(FakeCamera.instances[-1].device_id, 1)
        self.assertEqual((FakeCamera.instances[-1].width,
                          FakeCamera.instances[-1].height), (1280, 720))


class MaterialLocateCameraTests(CameraTestCase):
    class FakeReport:
        def __init__(self, best=None) -> None:
            self.lock = RingLock(regions=(LockedRegion(320.0, 240.0, 40.0),), expected=3)
            self.best = best

        def summary(self) -> str:
            return "fake report"

    class FakeLocator:
        report = None

        def __init__(self, **_kwargs) -> None:
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def locate(self, _frame):
            return type(self).report

        def draw(self, frame, _report):
            return frame.copy()

        def close(self) -> None:
            """``Vision.close`` 会调用它；假定位器只要记下这次调用。"""
            self.closed = True

    def build_args(self, **overrides) -> argparse.Namespace:
        settings = dict(
            model=None, camera=1, conf=0.25, focal=None, depth_range=None,
            material_bgr=None, material_palette=None, tolerance=60.0, diameter=None,
        )
        settings.update(overrides)
        return argparse.Namespace(**settings)

    def run_camera(self, report) -> int:
        from modules.vision.drive import material_locate

        self.FakeLocator.report = report
        locator = self.FakeLocator()
        # --camera 走 _overridden_settings（内部 from runtime import 快照与覆盖）
        # 和 Vision._open_config（内部 _build_locator + _new_camera）：
        # 读配置/建定位器 patch 到 runtime，相机走真实的 _new_camera（见下面的 720p 断言）。
        with patch("modules.vision.runtime._load_settings",
                   return_value=vision_settings()), \
                patch("modules.vision.runtime._build_locator", return_value=locator), \
                patch("modules.vision.drive.camera.USBCamera", FakeCamera), \
                patch("modules.vision.drive.preview.open_preview", return_value="vision-preview"), \
                patch("modules.vision.drive.preview.close_previews") as close_previews, \
                patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = material_locate._run_camera(self.build_args())
        self.assertTrue(close_previews.called)
        self.assertEqual(imshow.call_count, 1)
        self.assertTrue(FakeCamera.instances[-1].releases)
        self.assertTrue(locator.closed)
        return result

    def test_camera_branch_without_a_material(self) -> None:
        self.assertEqual(self.run_camera(self.FakeReport(best=None)), 0)

    def test_camera_branch_with_a_material(self) -> None:
        # 预览会把"最好的那个物料"的距离写在画面上，所以假报告要带上真实
        # RegionReport 的字段（name / distance_mm），不能只给个空壳。
        best = SimpleNamespace(name="red", distance_mm=75.0,
                               classification=SimpleNamespace(score=0.91))
        self.assertEqual(self.run_camera(self.FakeReport(best=best)), 0)

    def test_camera_requests_720p(self) -> None:
        self.run_camera(self.FakeReport(best=None))
        camera = FakeCamera.instances[-1]
        self.assertEqual((camera.width, camera.height), (1280, 720))
        # 镜头序号来自快照，不在工具里另读一遍命令行/配置。
        self.assertEqual(camera.device_id, 1)


class MainDemoMaterialTests(CameraTestCase):
    """modules.vision.app.demo.demo_material is the entry point; cover its wiring too."""

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
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def locate(self, _frame):
            return MainDemoMaterialTests.FakeReport()

        def draw(self, frame, _report):
            return frame.copy()

        def close(self) -> None:
            self.closed = True

    def test_demo_runs_one_frame_and_closes_the_window(self) -> None:
        from modules.vision.app import demo as vision_demo

        locator = self.FakeLocator()
        # demo_material 在函数内 from runtime import _load_settings / Vision，
        # 以及 from ...drive.preview / ...drive.stream import 工具：
        # 读配置 patch runtime，窗口/取流 patch 定义模块，相机构造 patch drive.camera。
        with patch("modules.vision.runtime._load_settings",
                   return_value=vision_settings()), \
                patch("modules.vision.runtime._build_locator", return_value=locator), \
                patch("modules.vision.drive.camera.USBCamera", FakeCamera), \
                patch("modules.vision.drive.preview.open_preview",
                      return_value="vision-preview") as open_preview, \
                patch("modules.vision.drive.preview.close_previews") as close_previews, \
                patch("cv2.imshow"), \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            vision_demo.demo_material(
                model_path=Path(__file__).with_suffix(".onnx"), camera_id=1
            )

        self.assertTrue(close_previews.called)
        open_preview.assert_called_once()
        camera = FakeCamera.instances[-1]
        self.assertEqual(camera.device_id, 1)
        self.assertTrue(camera.releases)
        self.assertTrue(locator.closed)

    def test_demo_shows_coordinates_and_prints_all_results_on_p(self) -> None:
        from modules.vision.app import demo as vision_demo

        item = SimpleNamespace(
            name="blue", confidence=0.81,
            x_mm=20.0, y_mm=-10.0, z_mm=75.0, range_mm=78.3,
        )
        with patch.object(self.FakeReport, "best", item), \
                patch("modules.vision.runtime._load_settings",
                      return_value=vision_settings()), \
                patch("modules.vision.runtime._build_locator",
                      return_value=self.FakeLocator()), \
                patch("modules.vision.drive.camera.USBCamera", FakeCamera), \
                patch("modules.vision.drive.preview.open_preview", return_value="vision-preview"), \
                patch("modules.vision.drive.preview.close_previews"), \
                patch("cv2.imshow"), \
                patch("cv2.putText") as put_text, \
                patch("cv2.waitKey", side_effect=[ord("p"), ord("q")]), \
                patch("builtins.print") as printed:
            vision_demo.demo_material(
                model_path=Path(__file__).with_suffix(".onnx"), camera_id=1
            )

        overlay = " ".join(call.args[1] for call in put_text.call_args_list)
        self.assertIn("blue", overlay)
        self.assertIn("0.81", overlay)
        self.assertIn("20", overlay)
        self.assertIn("-10", overlay)
        self.assertIn("75", overlay)
        self.assertIn("78", overlay)
        self.assertTrue(any(call.args == ("fake",) for call in printed.call_args_list))


class ManualMenuEntryTests(CameraTestCase):
    """菜单 18 的真机入口（modules/vision/test/manual/material_locate.py）也要能离线跑通。"""

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
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def locate(self, _frame):
            return ManualMenuEntryTests.FakeReport()

        def draw(self, frame, _report):
            return frame.copy()

        def close(self) -> None:
            self.closed = True

    def test_run_returns_zero_after_q(self) -> None:
        from modules.vision.test.manual import material_locate as menu

        # 菜单 18 把工具 import 到了自己的命名空间，所以 patch 它的模块属性；
        # 会话装配走 runtime._build_locator / _new_camera。
        locator = self.FakeLocator()
        camera = FakeCamera()
        with patch("modules.vision.runtime._build_locator", return_value=locator), \
                patch("modules.vision.runtime._new_camera",
                      return_value=camera) as new_camera, \
                patch.object(menu, "open_preview",
                             return_value="vision-preview") as open_preview, \
                patch.object(menu, "close_previews") as close_previews, \
                patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = menu.run(vision_settings())

        self.assertEqual(result, 0)
        open_preview.assert_called_once()
        self.assertTrue(close_previews.called)
        self.assertEqual(imshow.call_count, 1)
        new_camera.assert_called_once_with(1)
        self.assertTrue(camera.releases)
        self.assertTrue(locator.closed)

    def test_run_returns_one_when_the_camera_never_delivers_a_frame(self) -> None:
        """摄像头掉线/被占用时退出码是 1，与 run_preview 的 reason 一致。"""
        from modules.vision.test.manual import material_locate as menu

        with patch("modules.vision.runtime._build_locator",
                   return_value=self.FakeLocator()), \
                patch("modules.vision.runtime._new_camera",
                      return_value=SilentCamera()), \
                patch.object(menu, "open_preview", return_value="vision-preview"), \
                patch.object(menu, "close_previews"), \
                patch("cv2.imshow") as imshow:
            result = menu.run(vision_settings())

        self.assertEqual(result, 1)
        imshow.assert_not_called()

    def test_blank_frames_warn_once_with_the_closed_camera_id(self) -> None:
        """连续纯色画面提示一次；检查与提示都来自真实定义模块。"""
        from modules.vision.test.manual import material_locate as menu

        with patch("modules.vision.runtime._build_locator",
                   return_value=self.FakeLocator()), \
                patch("modules.vision.runtime._new_camera", return_value=FakeCamera()), \
                patch.object(menu, "looks_blank", return_value=True), \
                patch.object(menu, "BLANK_FRAME_WARNING", 1), \
                patch.object(menu, "warn_blank_preview") as warned, \
                patch.object(menu, "open_preview", return_value="vision-preview"), \
                patch.object(menu, "close_previews"), \
                patch("cv2.imshow"), \
                patch("cv2.waitKey", side_effect=stop_after_one_frame()):
            result = menu.run(vision_settings())

        self.assertEqual(result, 0)
        warned.assert_called_once_with(1)

    def test_missing_model_is_reported_without_constructing_a_camera(self) -> None:
        """模型缺失时 ``require_model_path`` 的 FileNotFoundError 直接抛出。

        新的 ``run()`` 先开预览窗口、再建会话（生产代码的顺序），所以旧断言
        "open_preview 没被调用" 已经不成立；成立的是更重要的那条：相机一个都没
        构造、一帧都没显示，已经开的窗口由 ``finally`` 关掉。
        """
        from modules.vision.runtime import _override_settings
        from modules.vision.test.manual import material_locate as menu

        settings = _override_settings(
            vision_settings(), model_path=Path("no-such-model.onnx")
        )
        with patch.object(menu, "open_preview"), \
                patch.object(menu, "close_previews") as close_previews, \
                patch("modules.vision.drive.camera.USBCamera") as camera_type, \
                patch("cv2.imshow") as imshow:
            with self.assertRaises(FileNotFoundError) as raised:
                menu.run(settings)

        self.assertIn("no-such-model.onnx", str(raised.exception))
        camera_type.assert_not_called()
        imshow.assert_not_called()
        close_previews.assert_called_once()

    def test_menu_entry_number_and_label(self) -> None:
        from modules.vision.test.manual import material_locate as menu

        self.assertEqual(menu.MENU_ENTRY, (18, "物料定位"))
        self.assertTrue(callable(menu.main))


if __name__ == "__main__":
    unittest.main()

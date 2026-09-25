"""Camera selection shared by the menu and application entry points."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vision.camera import load_camera_id


class VisionCameraConfigTests(unittest.TestCase):
    def test_project_defaults_select_external_camera_on_windows(self) -> None:
        self.assertEqual(load_camera_id("windows"), 1)
        self.assertEqual(load_camera_id("jetson"), 0)

    def test_selects_camera_for_each_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision.json"
            path.write_text(json.dumps({"windows": {"camera_id": 2},
                                        "jetson": {"camera_id": 0}}), encoding="utf-8")
            self.assertEqual(load_camera_id("windows", path), 2)
            self.assertEqual(load_camera_id("jetson", path), 0)

    def test_rejects_missing_or_invalid_camera_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision.json"
            for settings in ({"windows": {}}, {"windows": {"camera_id": -1}},
                             {"windows": {"camera_id": True}}):
                path.write_text(json.dumps(settings), encoding="utf-8")
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    load_camera_id("windows", path)

    def test_menu_demo_reads_configured_camera_index(self) -> None:
        from tests.manual import vision as vision_demo

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            model_path.touch()
            with patch.object(sys, "argv", ["vision.py"]), \
                    patch.object(vision_demo, "default_model_path", return_value=model_path), \
                    patch.object(vision_demo, "load_camera_id", return_value=3), \
                    patch.object(vision_demo, "create_detector"), \
                    patch.object(vision_demo, "USBCamera") as camera_type:
                camera_type.return_value.initialize.return_value.__enter__.return_value.read.return_value = None
                vision_demo.main()
                camera_type.assert_called_once_with(3, width=640, height=480, fps=30)


if __name__ == "__main__":
    unittest.main()

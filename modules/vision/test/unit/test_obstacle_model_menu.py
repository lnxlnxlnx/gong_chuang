"""The obstacle-model menu test must ask before it opens a model or a camera."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.vision.test.manual import obstacle_model


def _answers(*values):
    """Return a prompt callable that replays ``values``, then empty answers."""
    remaining = list(values)

    def prompt(_question: str) -> str:
        return remaining.pop(0) if remaining else ""

    return prompt


def _fake_cv2():
    """Stand-in for OpenCV whose ``imread`` reports an unreadable frame."""
    fake = MagicMock()
    fake.imread.return_value = None
    return fake


def _silent_camera(camera_type):
    """Wire a mocked ``USBCamera`` that opens but never delivers a frame."""
    camera = camera_type.return_value.initialize.return_value
    camera.read.return_value = None
    camera_type.return_value.initialize.return_value.__enter__.return_value = camera
    return camera


class ObstacleModelMenuTests(unittest.TestCase):
    def test_menu_entry_and_discovery(self) -> None:
        self.assertEqual(obstacle_model.MENU_ENTRY, (15, "障碍物模型测试"))
        self.assertEqual(obstacle_model.WINDOW_TITLE, "obstacle model test")
        import test as launcher

        self.assertIn(
            ("15", "障碍物模型测试", "modules.vision.test.manual.obstacle_model"),
            launcher.discover_manual_tests(),
        )

    def test_back_at_the_first_question_opens_nothing(self) -> None:
        with patch.object(obstacle_model, "create_detector") as create, \
                patch.object(obstacle_model, "USBCamera") as camera:
            obstacle_model.run_manual(prompt=_answers("0"))
        create.assert_not_called()
        camera.assert_not_called()

    def test_invalid_mode_is_rejected_without_opening_a_model(self) -> None:
        with patch.object(obstacle_model, "create_detector") as create:
            obstacle_model.run_manual(prompt=_answers("9"))
        create.assert_not_called()

    def test_invalid_plan_is_rejected_without_opening_a_model(self) -> None:
        with patch.object(obstacle_model, "create_detector") as create:
            obstacle_model.run_manual(prompt=_answers("2", "9"))
        create.assert_not_called()

    def test_missing_new_model_is_reported_without_opening_a_model(self) -> None:
        with patch.object(obstacle_model, "NEW_MODEL",
                          Path("missing") / "best_obstacle.onnx"), \
                patch.object(obstacle_model, "create_detector") as create:
            obstacle_model.run_manual(prompt=_answers("2", "1"))
        create.assert_not_called()

    def test_competition_plan_keeps_the_builtin_class_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "images"
            folder.mkdir()
            with patch.dict(sys.modules, {"cv2": _fake_cv2()}), \
                    patch.object(obstacle_model, "default_model_path",
                                 return_value=Path(__file__)), \
                    patch.object(obstacle_model, "DEFAULT_SAVE",
                                 Path(directory) / "out"), \
                    patch.object(obstacle_model, "create_detector") as create:
                obstacle_model.run_manual(prompt=_answers("3", "2", str(folder), ""))
        self.assertEqual(create.call_count, 1)
        self.assertIsNone(create.call_args.kwargs["class_names"])

    def test_new_model_plan_passes_the_single_class_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "best_obstacle.onnx"
            model.touch()
            folder = Path(directory) / "images"
            folder.mkdir()
            with patch.dict(sys.modules, {"cv2": _fake_cv2()}), \
                    patch.object(obstacle_model, "NEW_MODEL", model), \
                    patch.object(obstacle_model, "DEFAULT_SAVE",
                                 Path(directory) / "out"), \
                    patch.object(obstacle_model, "create_detector") as create:
                obstacle_model.run_manual(prompt=_answers("3", "1", str(folder), ""))
        self.assertEqual(create.call_args.kwargs["class_names"], ["obstacle"])
        self.assertEqual(create.call_args.args[0], model)

    def test_camera_mode_reads_the_configured_camera_index(self) -> None:
        with patch.dict(sys.modules, {"cv2": _fake_cv2()}), \
                patch.object(obstacle_model, "default_model_path",
                             return_value=Path(__file__)), \
                patch.object(obstacle_model, "load_camera_id", return_value=3), \
                patch.object(obstacle_model, "create_detector"), \
                patch.object(obstacle_model, "USBCamera") as camera_type:
            _silent_camera(camera_type)
            obstacle_model.run_manual(prompt=_answers("1", "2", "", ""))
        camera_type.assert_called_once_with(3, width=640, height=480, fps=30)

    def test_camera_that_never_delivers_a_frame_gives_up(self) -> None:
        with patch.dict(sys.modules, {"cv2": _fake_cv2()}), \
                patch.object(obstacle_model, "default_model_path",
                             return_value=Path(__file__)), \
                patch.object(obstacle_model, "load_camera_id", return_value=3), \
                patch.object(obstacle_model, "create_detector"), \
                patch.object(obstacle_model, "USBCamera") as camera_type:
            camera = _silent_camera(camera_type)
            obstacle_model.run_manual(prompt=_answers("1", "2", "", ""))
        self.assertEqual(camera.read.call_count, obstacle_model.READ_ATTEMPTS)

    def test_selfcheck_needs_no_model(self) -> None:
        with patch.dict(sys.modules, {"cv2": _fake_cv2()}), \
                patch.object(obstacle_model, "scan_cameras") as scan, \
                patch.object(obstacle_model, "create_detector") as create:
            obstacle_model.run_manual(prompt=_answers("5"))
        scan.assert_called_once()
        create.assert_not_called()

    def test_selfcheck_reports_when_no_camera_delivers(self) -> None:
        with patch.dict(sys.modules, {"cv2": _fake_cv2()}), \
                patch.object(obstacle_model, "create_detector") as create, \
                patch.object(obstacle_model, "USBCamera") as camera_type:
            _silent_camera(camera_type)
            obstacle_model.run_manual(prompt=_answers("5"))
        self.assertEqual(camera_type.call_count, 4)
        create.assert_not_called()

    def test_missing_source_path_is_reported_without_opening_a_model(self) -> None:
        with patch.object(obstacle_model, "default_model_path",
                          return_value=Path(__file__)), \
                patch.object(obstacle_model, "create_detector") as create:
            obstacle_model.run_manual(prompt=_answers("4", "2", "no/such/file.mp4"))
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Smoke checks for the two project entry points without opening hardware."""

import subprocess
import sys
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import test as test_launcher
from vision import default_model_path


ROOT = Path(__file__).resolve().parents[1]


class EntryPointTests(unittest.TestCase):
    def test_vision_demo_selects_the_local_model_format(self) -> None:
        model_path = default_model_path()
        expected = ".onnx" if os.name == "nt" else ".engine"
        self.assertEqual(model_path.suffix, expected)
        self.assertEqual(model_path.name, f"best{expected}")

    def test_menu_dispatches_arm_and_returns_to_menu(self) -> None:
        with patch("builtins.input", side_effect=["2", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                test_launcher.main()
        import_module.assert_called_once_with("tests.manual.arm")
        import_module.return_value.main.assert_called_once_with()

    def test_test_menu_can_exit_without_opening_hardware(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "test.py")],
            input="0\n",
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("机械臂", result.stdout)
        self.assertIn("离线", result.stdout)

    def test_main_can_be_imported_without_opening_hardware(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import main"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

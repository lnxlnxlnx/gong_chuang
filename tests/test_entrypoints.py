"""Smoke checks for the two project entry points without opening hardware."""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import test as test_launcher
from vision import default_model_path


ROOT = Path(__file__).resolve().parents[1]


class EntryPointTests(unittest.TestCase):
    def test_manual_test_is_discovered_without_importing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "future_scan.py"
            script.write_text(
                'MENU_ENTRY = (11, "新测试")\nraise RuntimeError("imported")\n',
                encoding="utf-8",
            )
            entries = test_launcher.discover_manual_tests(Path(directory))
        self.assertEqual(entries, (("11", "新测试", "tests.manual.future_scan"),))

    def test_duplicate_manual_menu_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("first", "second"):
                (Path(directory) / f"{name}.py").write_text(
                    'MENU_ENTRY = (10, "重复")\n', encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "10"):
                test_launcher.discover_manual_tests(Path(directory))

    def test_menu_dispatches_laser_scan(self) -> None:
        with patch("builtins.input", side_effect=["10", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("10. 激光障碍物扫描", output.getvalue())
        import_module.assert_called_once_with("tests.manual.laser_scan")

    def test_menu_dispatches_fixed_laser_distance(self) -> None:
        with patch("builtins.input", side_effect=["11", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("11. 激光固定测距", output.getvalue())
        import_module.assert_called_once_with("tests.manual.laser_fixed")

    def test_menu_dispatches_bus_servo_test(self) -> None:
        with patch("builtins.input", side_effect=["12", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("12. 总线舵机测试", output.getvalue())
        import_module.assert_called_once_with("tests.manual.servo")

    def test_menu_dispatches_coin_d6_point_cloud(self) -> None:
        with patch("builtins.input", side_effect=["16", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("16. D6 激光雷达点云", output.getvalue())
        import_module.assert_called_once_with("tests.manual.coin_d6")

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

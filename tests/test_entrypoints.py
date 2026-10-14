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
from project_paths import PROJECT_ROOT
from modules.vision import default_model_path, load_model_settings


ROOT = PROJECT_ROOT


class EntryPointTests(unittest.TestCase):
    def test_manual_test_is_discovered_without_importing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "future_scan.py"
            script.write_text(
                'MENU_ENTRY = (11, "新测试")\nraise RuntimeError("imported")\n',
                encoding="utf-8",
            )
            entries = test_launcher.discover_manual_tests((Path(directory),), root=Path(directory))
        self.assertEqual(entries, (("11", "新测试", "future_scan"),))

    def test_duplicate_manual_menu_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("first", "second"):
                (Path(directory) / f"{name}.py").write_text(
                    'MENU_ENTRY = (10, "重复")\n', encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "10"):
                test_launcher.discover_manual_tests((Path(directory),), root=Path(directory))

    def test_menu_dispatches_laser_scan(self) -> None:
        with patch("builtins.input", side_effect=["10", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("10. 激光障碍物扫描", output.getvalue())
        import_module.assert_called_once_with("modules.laser_scan.test.manual.laser_scan")

    def test_menu_dispatches_fixed_laser_distance(self) -> None:
        with patch("builtins.input", side_effect=["11", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("11. 激光固定测距", output.getvalue())
        import_module.assert_called_once_with("modules.laser_scan.test.manual.laser_fixed")

    def test_menu_dispatches_bus_servo_test(self) -> None:
        with patch("builtins.input", side_effect=["12", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("12. 总线舵机测试", output.getvalue())
        import_module.assert_called_once_with("modules.servo.test.manual.servo")

    def test_menu_dispatches_coin_d6_point_cloud(self) -> None:
        with patch("builtins.input", side_effect=["16", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("16. D6 激光雷达点云", output.getvalue())
        import_module.assert_called_once_with("modules.lidar.test.manual.coin_d6")

    def test_menu_dispatches_coin_d6_live_window(self) -> None:
        with patch("builtins.input", side_effect=["17", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                with redirect_stdout(io.StringIO()) as output:
                    test_launcher.main()
        self.assertIn("17. D6 点云实时显示", output.getvalue())
        import_module.assert_called_once_with("modules.lidar.test.manual.coin_d6_live")

    def test_vision_demo_selects_the_local_model_format(self) -> None:
        """模型路径来自 config.json 的 model 块，不是代码里的 os.name 分支。"""
        model_path = default_model_path()
        expected = ".onnx" if os.name == "nt" else ".engine"
        self.assertEqual(model_path.suffix, expected)
        self.assertEqual(model_path.name, f"best{expected}")
        # 两平台各自的文件都在配置里指明，换模型时改的是这里。
        settings = load_model_settings()
        self.assertEqual(settings.onnx_path.name, "best.onnx")
        self.assertEqual(settings.engine_path.name, "best.engine")

    def test_menu_dispatches_arm_and_returns_to_menu(self) -> None:
        with patch("builtins.input", side_effect=["2", "0"]):
            with patch.object(test_launcher.importlib, "import_module") as import_module:
                test_launcher.main()
        import_module.assert_called_once_with("modules.arm.test.manual.arm")
        import_module.return_value.main.assert_called_once_with()

    def test_test_menu_can_exit_without_opening_hardware(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "test.py")],
            input="0\n",
            capture_output=True,
            text=True,
            # 子进程用 UTF-8 写菜单（代码里有中文），而 text=True 默认按
            # Windows 本地编码（GBK）解码，会抛 UnicodeDecodeError 并让
            # stdout 变成 None。这里把两端的编码都钉死。
            encoding="utf-8",
            errors="replace",
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
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_module_self_test_entries_stay_warning_free(self) -> None:
        """``python -m`` self-tests must not trip runpy's "found in sys.modules".

        ``modules.vision`` resolves the material-locator symbols lazily on
        purpose; a package ``__init__`` that imported its own submodules eagerly
        would make these entry points warn (and fail under ``-W error``).
        """
        for module in (
            "modules.vision.drive.hough_lock",
            "modules.vision.drive.material_locate",
        ):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-W", "error::RuntimeWarning", "-m", module],
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                    timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

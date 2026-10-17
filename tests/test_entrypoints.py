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
from modules.vision.config import default_model_path, load_model_settings


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
            # ``-X utf8`` 是**必须的**，不是保险：子进程的 stdout 在 Windows 上默认按
            # 控制台代码页（本机 cp936）编码，而父进程按 UTF-8 解码，
            # 于是中文菜单变成乱码、下面的断言永远找不到「机械臂」。
            # 把两端的编码都钉死在 UTF-8，这条测试才测的是"菜单能跑"而不是"本机代码页"。
            [sys.executable, "-X", "utf8", str(ROOT / "test.py")],
            input="0\n",
            capture_output=True,
            text=True,
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

        包出口只显式导出公开类，并且不预加载“既能 import、又能 ``python -m``”的流水线模块；
        一旦有人在 ``__init__`` 里急切导入它们，这些入口就会警告（``-W error`` 下直接失败）。
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

    def test_known_runpy_preloads_are_only_the_documented_ones(self) -> None:
        """``python -m`` 的 runpy 预加载警告只允许出现在**已记录**的模块上。

        ``modules/arm/__init__.py`` 必须导出 ``RobotArm``，而它的真实定义在
        ``modules/arm/runtime.py``，那条链会经由 ``modules/arm/config.py`` 预加载
        ``modules.arm.drive.kinematics``。所以 ``python -m modules.arm.drive.kinematics``
        会警告——这是"包根导出上层类"的既定代价，不是漏改。

        这条用例把它**钉住**：以后新增的预加载（也就是新出现的警告）会让它失败，
        必须在这里显式记录，否则没人会注意到又一个自测入口被包导入吃掉了。
        """
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::RuntimeWarning",
                "-c",
                "import modules.arm.drive.kinematics",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=60,
        )
        # 纯 import 不该有警告：只有 `python -m` 才有 runpy 的"先被导入过"问题。
        self.assertEqual(result.returncode, 0, result.stderr)
        # 已记录的例外：这个入口确实会警告，而且是刻意的。
        warned = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::RuntimeWarning",
                "-m",
                "modules.arm.drive.kinematics",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=180,
        )
        self.assertNotEqual(
            warned.returncode,
            0,
            "如果这里不再警告，说明包导入形态变了——请同时更新本用例与那段说明",
        )


if __name__ == "__main__":
    unittest.main()

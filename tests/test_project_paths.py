"""Every project path resolves the same way, wherever it is asked from.

Two rules from :mod:`project_paths` are pinned here: project-wide resources
(``models/``, ``output/``) hang off the repository root and ignore the current
working directory, while a module's own ``config.json`` sits next to the module
that reads it.  A loader that guessed the root with ``parents[N]`` would keep
working until the file moved, so the rule is checked statically too.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from project_paths import PROJECT_ROOT, models_path, output_path


#: (模块, 路径常量名, 该常量应指向的文件, 加载器源文件)。文件必须与模块同目录且存在。
#: 硬件参数独立成 ``hardware.json`` 之后，同一个加载器可能有两个默认路径，所以
#: 常量名也写进表里，别让"哪个常量"靠猜。
LOADER_CONFIGS = (
    ("modules.chassis.config", "DEFAULT_CONFIG_PATH", "modules/chassis/config.json", "modules/chassis/config.py"),
    ("modules.chassis.config", "DEFAULT_HARDWARE_PATH", "modules/chassis/hardware.json", "modules/chassis/config.py"),
    ("modules.arm.config", "DEFAULT_CONFIG_PATH", "modules/arm/config.json", "modules/arm/config.py"),
    ("modules.arm.config", "DEFAULT_HARDWARE_PATH", "modules/arm/hardware.json", "modules/arm/config.py"),
    ("modules.motor.config", "DEFAULT_HARDWARE_PATH", "modules/motor/hardware.json", "modules/motor/config.py"),
    ("modules.lidar.config", "DEFAULT_CONFIG_PATH", "modules/lidar/config.json", "modules/lidar/config.py"),
    (
        "modules.laser_scan.drive.laser_scanner",
        "DEFAULT_CONFIG_PATH",
        "modules/laser_scan/config.json",
        "modules/laser_scan/drive/laser_scanner.py",
    ),
    (
        "modules.servo.config",
        "DEFAULT_CONFIG_PATH",
        "modules/servo/config.json",
        "modules/servo/config.py",
    ),
    (
        "modules.servo.config",
        "DEFAULT_HARDWARE_PATH",
        "modules/servo/hardware.json",
        "modules/servo/config.py",
    ),
    (
        "modules.servo.drive.bus_servo",
        "DEFAULT_HARDWARE_PATH",
        "modules/servo/hardware.json",
        "modules/servo/drive/bus_servo.py",
    ),
    (
        "modules.vision.drive.camera",
        "DEFAULT_CONFIG_PATH",
        "modules/vision/config.json",
        "modules/vision/drive/camera.py",
    ),
)

#: ``Path(__file__).resolve().parents[N]`` 里 N 必须为 0 或 1：模块自己的
#: 配置就在旁边，而从 drive/ 回到模块根也只需要一层。再往上就是在猜仓库根。
ROOT_GUESS = re.compile(r"parents\[(\d+)\]")


class ProjectPathTests(unittest.TestCase):
    def test_project_root_survives_a_foreign_working_directory(self) -> None:
        """A helper anchored on ``__file__`` must not follow the caller's cwd."""
        environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import project_paths; print(project_paths.PROJECT_ROOT)",
            ],
            capture_output=True,
            text=True,
            cwd=Path(sys.executable).parent,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), PROJECT_ROOT)

    def test_each_module_reads_the_config_next_to_it(self) -> None:
        for module_name, constant, expected, _ in LOADER_CONFIGS:
            with self.subTest(module=module_name, constant=constant):
                module = importlib.import_module(module_name)
                resolved = Path(getattr(module, constant)).resolve()
                self.assertEqual(resolved, (PROJECT_ROOT / expected).resolve())
                self.assertTrue(resolved.is_file(), f"配置缺失：{expected}")

    def test_module_configuration_is_not_reached_through_the_repository_root(self) -> None:
        """A module's own config must not be named via project_paths."""
        offenders = [
            relative
            for _, _, _, relative in LOADER_CONFIGS
            if "from project_paths import" in (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [], "模块自己的配置不应通过仓库根定位")

    def test_loaders_do_not_guess_the_repository_root(self) -> None:
        """``parents[2]`` or deeper is a root guess that only breaks after a move."""
        offenders = []
        for _, _, _, relative in LOADER_CONFIGS:
            source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            for match in ROOT_GUESS.finditer(source):
                if int(match.group(1)) >= 2:
                    offenders.append(f"{relative}: parents[{match.group(1)}]")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_ignored_directories_are_reachable_without_existing(self) -> None:
        """models/ and output/ are git-ignored; the helper only names paths."""
        self.assertEqual(models_path("best.onnx"), PROJECT_ROOT / "models" / "best.onnx")
        self.assertEqual(
            output_path("lidar_cloud", "scan.png"),
            PROJECT_ROOT / "output" / "lidar_cloud" / "scan.png",
        )


if __name__ == "__main__":
    unittest.main()

"""Every project path resolves the same way, wherever it is asked from.

The ``modules/`` layout moves files several levels deep, so a loader that still
computes the repository root with ``Path(__file__).resolve().parents[N]`` would
start pointing at the wrong directory without any visible error.  These checks
pin the two rules from :mod:`project_paths`: project-wide resources hang off the
repository root, and they do not depend on the current working directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from project_paths import PROJECT_ROOT, config_path, models_path, output_path


#: 受版本管理的共享配置：这些文件必须存在（models/ 与 output/ 则可以不存在）。
TRACKED_CONFIG_NAMES = (
    "coin_d6.json",
    "laser_scan.json",
    "mecanum_hardware.json",
    "robot_arm.json",
    "servo.json",
    "vision.json",
)

#: 加载共享配置的模块：它们必须从 helper 取路径，而不是自己推算仓库根。
#: 在旧布局下 ``parents[N]`` 恰好等于仓库根，所以只有静态检查能提前发现它。
LOADER_MODULES = (
    "modules/servo/drive/gripper.py",
    "modules/laser_scan/drive/laser_scanner.py",
    "modules/lidar/config.py",
    "modules/chassis/config.py",
    "modules/servo/drive/bus_servo.py",
    "modules/arm/config.py",
    "modules/vision/drive/camera.py",
)


def loader_default_paths() -> dict[str, Path]:
    """Return each module's default configuration path, imported lazily."""
    from modules.servo.drive.gripper import DEFAULT_CONFIG_PATH as gripper_config
    from modules.laser_scan.drive.laser_scanner import DEFAULT_CONFIG_PATH as laser_scan_config
    from modules.lidar.config import DEFAULT_CONFIG_PATH as lidar_config
    from modules.chassis.config import DEFAULT_CONFIG_PATH as chassis_config
    from modules.servo.drive.bus_servo import DEFAULT_CONFIG_PATH as servo_bus_config
    from modules.arm.config import DEFAULT_CONFIG_PATH as arm_config
    from modules.vision.drive.camera import DEFAULT_CONFIG_PATH as vision_config

    return {
        "gripper": gripper_config,
        "laser_scanner": laser_scan_config,
        "lidar.config": lidar_config,
        "mecanum_config": chassis_config,
        "motor_drivers.bus_servo": servo_bus_config,
        "robot_arm_config": arm_config,
        "vision.camera": vision_config,
    }


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

    def test_every_loader_uses_the_shared_helper(self) -> None:
        expected = {
            "gripper": config_path("servo.json"),
            "laser_scanner": config_path("laser_scan.json"),
            "lidar.config": config_path("coin_d6.json"),
            "mecanum_config": config_path("mecanum_hardware.json"),
            "motor_drivers.bus_servo": config_path("servo.json"),
            "robot_arm_config": config_path("robot_arm.json"),
            "vision.camera": config_path("vision.json"),
        }
        self.assertEqual(loader_default_paths(), expected)

    def test_tracked_configuration_exists(self) -> None:
        missing = [
            name for name in TRACKED_CONFIG_NAMES if not config_path(name).is_file()
        ]
        self.assertEqual(missing, [], "受版本管理的配置缺失")

    def test_loaders_never_compute_their_own_root(self) -> None:
        """Catch a loader that would only break once the file moves deeper."""
        offenders = []
        for relative in LOADER_MODULES:
            source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            if "from project_paths import" not in source:
                offenders.append(f"{relative}: 没有使用 project_paths")
            if "Path(__file__)" in source:
                offenders.append(f"{relative}: 仍然用 Path(__file__) 推算路径")
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

"""One interactive entry point for offline checks and manual hardware tests."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path


MENU = (
    ("1", "离线单测", None),
    ("2", "机械臂三轴", "tests.manual.arm"),
    ("3", "麦轮逐轮", "tests.manual.wheels"),
    ("4", "麦轮底盘", "tests.manual.chassis"),
    ("5", "单电机基础测试", "tests.manual.motor"),
    ("6", "单电机诊断", "tests.manual.motor_diag"),
    ("7", "单电机停机诊断", "tests.manual.motor_stop_diag"),
    ("8", "底盘演示", "tests.manual.mecanum_demo"),
    ("9", "视觉演示", "tests.manual.vision"),
)


def run_offline_tests() -> None:
    """Discover only test_*.py files; manual scripts never run here."""
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(root / "tests"), pattern="test_*.py", top_level_dir=str(root)
    )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    while True:
        print("\n项目测试菜单：")
        for key, label, _ in MENU:
            print(f"  {key}. {label}")
        print("  0. 退出")
        choice = input("请选择：").strip()
        if choice == "0":
            return
        selected = next((item for item in MENU if item[0] == choice), None)
        if selected is None:
            print("无效选项，请重新选择。")
            continue
        _, label, module_name = selected
        print(f"开始：{label}")
        if module_name is None:
            run_offline_tests()
        else:
            importlib.import_module(module_name).main()


if __name__ == "__main__":
    main()

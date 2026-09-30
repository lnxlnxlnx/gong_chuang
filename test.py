"""One interactive entry point for offline checks and manual hardware tests."""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path


MANUAL_TEST_DIR = Path(__file__).resolve().parent / "tests" / "manual"


def discover_manual_tests(manual_dir: Path = MANUAL_TEST_DIR) -> tuple[tuple[str, str, str], ...]:
    """Read MENU_ENTRY declarations without importing hardware test modules."""
    entries = []
    seen = {0, 1}
    for path in manual_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "MENU_ENTRY"
                for target in node.targets
            ):
                continue
            try:
                number, label = ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}: MENU_ENTRY must be (number, label)") from error
            if (
                type(number) is not int
                or number < 2
                or not isinstance(label, str)
                or not label.strip()
            ):
                raise ValueError(f"{path}: invalid MENU_ENTRY {number!r}, {label!r}")
            if number in seen:
                raise ValueError(f"{path}: duplicate menu number {number}")
            seen.add(number)
            entries.append((str(number), label, f"tests.manual.{path.stem}"))
            break
    return tuple(sorted(entries, key=lambda entry: int(entry[0])))


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
    menu = (("1", "离线单测", None),) + discover_manual_tests()
    while True:
        print("\n项目测试菜单：")
        for key, label, _ in menu:
            print(f"  {key}. {label}")
        print("  0. 退出")
        choice = input("请选择：").strip()
        if choice == "0":
            return
        selected = next((item for item in menu if item[0] == choice), None)
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

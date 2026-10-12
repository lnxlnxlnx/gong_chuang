"""One interactive entry point for offline checks and manual hardware tests."""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path

from project_paths import PROJECT_ROOT


MANUAL_TEST_DIRS: tuple[Path, ...] = (
    PROJECT_ROOT / "modules" / "arm" / "test" / "manual",
    PROJECT_ROOT / "modules" / "chassis" / "test" / "manual",
    PROJECT_ROOT / "modules" / "grasp" / "test" / "manual",
    PROJECT_ROOT / "modules" / "laser_scan" / "test" / "manual",
    PROJECT_ROOT / "modules" / "lidar" / "test" / "manual",
    PROJECT_ROOT / "modules" / "motor" / "test" / "manual",
    PROJECT_ROOT / "modules" / "servo" / "test" / "manual",
    PROJECT_ROOT / "modules" / "vision" / "test" / "manual",
)

#: 离线测试的发现根目录。测试包搬到哪里，就把根目录加到这里；
#: ``tests/test_discovery_coverage.py`` 会在有 test_*.py 落在所有根目录之外时失败，
#: 所以指向错位置的扫描不可能安静地少跑一个目录。
OFFLINE_TEST_DIRS: tuple[Path, ...] = (PROJECT_ROOT / "tests", PROJECT_ROOT / "modules")


def _module_name(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Return the importable dotted name of one manual script.

    Scripts are addressed as ``modules.<module>.test.manual.<script>``, derived
    from the path so moving a script keeps the menu working.  A script outside
    ``root`` (a test fixture in a temporary directory) falls back to its stem.
    """
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.stem
    return ".".join(relative.with_suffix("").parts)


def discover_manual_tests(
    manual_dirs: tuple[Path, ...] = MANUAL_TEST_DIRS,
    root: Path = PROJECT_ROOT,
) -> tuple[tuple[str, str, str], ...]:
    """Read MENU_ENTRY declarations from every module without importing them."""
    entries = []
    seen = {0, 1}
    for manual_dir in manual_dirs:
        for path in sorted(manual_dir.glob("*.py")):
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
                entries.append((str(number), label, _module_name(path, root)))
                break
    return tuple(sorted(entries, key=lambda entry: int(entry[0])))


def offline_suite(roots: tuple[Path, ...] = OFFLINE_TEST_DIRS) -> unittest.TestSuite:
    """Build the offline suite without running it; manual scripts are never imported."""
    root = PROJECT_ROOT
    loader = unittest.defaultTestLoader
    return unittest.TestSuite(
        loader.discover(start_dir=str(scan_root), pattern="test_*.py", top_level_dir=str(root))
        for scan_root in roots
    )


def run_offline_tests(roots: tuple[Path, ...] = OFFLINE_TEST_DIRS) -> None:
    """Run every discovered test; discovery tests never open hardware."""
    result = unittest.TextTestRunner(verbosity=1).run(offline_suite(roots))
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

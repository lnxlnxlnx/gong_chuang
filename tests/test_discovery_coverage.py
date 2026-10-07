"""Guards for the two test entry points: offline discovery and the manual menu.

Both entry points fail silently by omission.  ``unittest`` reports success for
whatever it happens to find, so a mis-pointed discovery root looks green while
whole directories never run; the menu skips any manual script that forgets its
``MENU_ENTRY``.  These checks turn both into failures, which is what makes it
safe to move test files between packages later.

The offline roots are read from :mod:`test`, so `test.py` stays the single
source of truth and these guards follow it automatically.
"""

from __future__ import annotations

import ast
import inspect
import os
import unittest
from pathlib import Path

import test as test_launcher
from project_paths import PROJECT_ROOT
from tests.scratch import scratch_directory


ROOT = PROJECT_ROOT

#: 数据、生成产物和随仓库附带的厂商例程都不是源码，扫描必须跳过它们。
SKIPPED_DIRECTORIES = frozenset({"__pycache__", "docs", "models", "output"})


def test_files_on_disk(root: Path = ROOT) -> tuple[Path, ...]:
    """Return every ``test_*.py`` outside hidden, data and vendor directories."""
    found = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name
            for name in subdirectories
            if not name.startswith(".") and name not in SKIPPED_DIRECTORIES
        ]
        found.extend(
            Path(directory) / name
            for name in filenames
            if name.startswith("test_") and name.endswith(".py")
        )
    return tuple(sorted(found))


def _walk_tests(suite: unittest.TestSuite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _walk_tests(test)
        else:
            yield test


def collected_test_files(suite: unittest.TestSuite) -> set[Path]:
    """Return the files backing every test the offline runner would run."""
    return {
        Path(inspect.getfile(type(test))).resolve() for test in _walk_tests(suite)
    }


class OfflineDiscoveryTests(unittest.TestCase):
    def test_scanned_roots_exist(self) -> None:
        for path in test_launcher.OFFLINE_TEST_DIRS:
            self.assertTrue(path.is_dir(), f"离线发现根目录不存在：{path}")

    def test_every_test_file_on_disk_is_collected(self) -> None:
        """An undiscovered file keeps the suite green while silently never running.

        Discovery itself is reused from :mod:`test`, so this catches both a
        mis-pointed root and a directory that is not importable as a package.
        """
        collected = collected_test_files(test_launcher.offline_suite())
        orphaned = [
            path.relative_to(ROOT).as_posix()
            for path in test_files_on_disk()
            if path.resolve() not in collected
        ]
        self.assertEqual(orphaned, [], "这些测试文件不会被离线跑器发现")

    def test_data_and_vendor_directories_are_skipped(self) -> None:
        """Generated data and bundled examples must not be treated as source."""
        with scratch_directory() as root:
            (root / "tests").mkdir()
            (root / "tests" / "test_kept.py").write_text("", encoding="utf-8")
            (root / "models").mkdir()
            (root / "models" / "test_ignored.py").write_text("", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "test_hidden.py").write_text("", encoding="utf-8")

            found = [
                path.relative_to(root).as_posix() for path in test_files_on_disk(root)
            ]

        self.assertEqual(found, ["tests/test_kept.py"])


def manual_script_paths() -> tuple[Path, ...]:
    """Return every manual script under the declared manual directories."""
    return tuple(
        sorted(
            path
            for directory in test_launcher.MANUAL_TEST_DIRS
            for path in directory.glob("*.py")
            if path.name != "__init__.py"
        )
    )


def expected_module_name(path: Path) -> str:
    """Independently derive the module name the menu is supposed to import."""
    return ".".join(path.resolve().relative_to(ROOT).with_suffix("").parts)


def is_source(path: Path) -> bool:
    """Return ``True`` for project source, skipping hidden, data and vendor trees."""
    parts = path.relative_to(ROOT).parts
    return not any(part.startswith(".") or part in SKIPPED_DIRECTORIES for part in parts)


class ManualMenuTests(unittest.TestCase):
    def test_manual_directories_exist(self) -> None:
        for directory in test_launcher.MANUAL_TEST_DIRS:
            self.assertTrue(directory.is_dir(), f"真机脚本目录不存在：{directory}")

    def test_every_manual_script_is_reachable_from_the_menu(self) -> None:
        declared = {module for _, _, module in test_launcher.discover_manual_tests()}
        on_disk = {expected_module_name(path) for path in manual_script_paths()}
        self.assertEqual(
            sorted(on_disk - declared),
            [],
            "这些真机脚本没有 MENU_ENTRY（或模块名不符），菜单看不到它们",
        )
        self.assertEqual(sorted(declared - on_disk), [], "菜单声明了磁盘上不存在的脚本")

    def test_every_manual_script_exposes_main(self) -> None:
        missing = []
        for path in manual_script_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if not any(
                isinstance(node, ast.FunctionDef) and node.name == "main"
                for node in tree.body
            ):
                missing.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(missing, [], "菜单脚本缺少模块级 main()")

    def test_no_manual_script_lives_outside_the_declared_directories(self) -> None:
        """A manual script moved elsewhere would silently drop out of the menu."""
        declared = {path.resolve() for path in test_launcher.MANUAL_TEST_DIRS}
        strays = [
            path.relative_to(ROOT).as_posix()
            for path in sorted(ROOT.rglob("*.py"))
            if is_source(path)
            and path.parent.name == "manual"
            and path.resolve().parent not in declared
            and path.name != "__init__.py"
        ]
        self.assertEqual(strays, [], "这些真机脚本不在声明的目录里，菜单不会发现它们")


if __name__ == "__main__":
    unittest.main()

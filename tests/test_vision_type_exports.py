"""Keep the lazy exports and the static re-exports in step.

``vision/__init__.py`` resolves the material-locator symbols through PEP 562
``__getattr__`` at runtime, and re-exports them under ``if TYPE_CHECKING`` so
static analysis sees real classes instead of ``Any``.  The two lists can drift
in silence: a symbol missing from the type-checking block becomes ``Any`` again,
and one missing from the lazy map raises ``AttributeError`` only when somebody
happens to use it.
"""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path

import vision


SOURCE = Path(vision.__file__).resolve()


def _module() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _assignment_value(name: str) -> ast.expr:
    """Return the value assigned to a module-level ``name`` (annotated or not)."""
    for node in _module().body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return node.value
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return node.value
    raise AssertionError(f"{name} not found in {SOURCE}")


def lazy_attributes() -> dict[str, str]:
    """Return the symbol -> defining module map that ``__getattr__`` uses."""
    return dict(ast.literal_eval(_assignment_value("_LAZY_ATTRIBUTES")))


def static_attributes() -> dict[str, str]:
    """Return the symbol -> defining module map declared under TYPE_CHECKING."""
    declared: dict[str, str] = {}
    for node in _module().body:
        if not isinstance(node, ast.If):
            continue
        if not isinstance(node.test, ast.Name) or node.test.id != "TYPE_CHECKING":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.ImportFrom) or statement.module is None:
                continue
            module_name = statement.module.rsplit(".", 1)[-1]
            for alias in statement.names:
                declared[alias.asname or alias.name] = module_name
    return declared


class VisionTypeExportTests(unittest.TestCase):
    def test_static_exports_cover_every_lazy_attribute(self) -> None:
        lazy = lazy_attributes()
        static = static_attributes()
        self.assertEqual(
            sorted(set(lazy) - set(static)),
            [],
            "这些符号在 TYPE_CHECKING 块里缺失，静态分析仍会当成 Any",
        )
        self.assertEqual(
            sorted(set(static) - set(lazy)),
            [],
            "TYPE_CHECKING 块里多出了运行时取不到的符号",
        )
        for name, module_name in lazy.items():
            self.assertEqual(
                static[name], module_name, f"{name} 的静态来源与懒加载来源不一致"
            )

    def test_lazy_attributes_resolve_to_the_declared_module(self) -> None:
        for name, module_name in lazy_attributes().items():
            with self.subTest(symbol=name):
                module = importlib.import_module(f".{module_name}", "vision")
                self.assertIs(getattr(vision, name), getattr(module, name))
                self.assertIn(name, vision.__all__)


if __name__ == "__main__":
    unittest.main()

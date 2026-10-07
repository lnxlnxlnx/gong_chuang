"""Keep the sources runnable on the Jetson's Python 3.8.

The board runs JetPack 5.1 (L4T 35.6.1), whose Python is 3.8; the training
environment is a separate Python 3.9 conda env.  A file that only parses or
imports on 3.9+ therefore breaks the deployed path while staying green on a
3.12 development machine, which is what these checks catch:

  - ``with (a as x, b as y):`` is 3.9+ syntax (SyntaxError on 3.8).
  - ``collections.abc.Callable[[int], object]`` used as a runtime value is
    evaluated at import time and raises TypeError on 3.8 (PEP 585 is 3.9).
  - ``attr.removeprefix`` / ``functools.cache`` / ``itertools.pairwise`` and
    friends simply do not exist before 3.9/3.10.

``ast.parse(feature_version=(3, 8))`` is deliberately *not* used as the only
check: it accepts the parenthesized ``with`` form, and it cannot see runtime
subscripting or missing APIs at all.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tokenize
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

#: 数据、生成产物与随仓库附带的厂商例程不是本项目的源码。
SKIPPED_PARTS = frozenset({"__pycache__", ".git", ".worktrees", "docs", "models", "output"})

#: 3.9+ 才有的方法名。
POST38_METHODS = frozenset({"removeprefix", "removesuffix", "bit_count"})

#: (模块, 属性) 形式的 3.9+/3.10+ API。
POST38_ATTRIBUTES = frozenset({
    ("math", "lcm"),
    ("functools", "cache"),
    ("itertools", "pairwise"),
    ("graphlib", "TopologicalSorter"),
})

#: 3.11+ 才有的内置名。
POST38_NAMES = frozenset({"ExceptionGroup", "BaseExceptionGroup"})

#: 只在 3.9+ 存在的模块。
POST38_MODULES = frozenset({"zoneinfo", "tomllib", "graphlib"})

#: `@dataclass(...)` 里 3.10+ 才支持的参数。
POST38_DATACLASS_ARGUMENTS = frozenset({"slots", "kw_only"})

#: 注解位置用到的 3.9+ 泛型写法需要 `from __future__ import annotations` 才不会在
#: 导入时求值；没有它，3.8 会在导入这个模块时直接 TypeError。
ABC_GENERICS = frozenset({
    "Awaitable", "Callable", "Collection", "Container", "Coroutine", "Hashable",
    "Iterable", "Iterator", "Mapping", "MutableMapping", "MutableSet", "Sequence",
    "Set", "Sized",
})
BUILTIN_GENERICS = frozenset({"dict", "frozenset", "list", "set", "tuple", "type"})

#: 交给真实 3.8 解释器执行的编译检查。
COMPILE_SCRIPT = """
import pathlib, sys, tokenize
root = pathlib.Path(sys.argv[1])
skipped = {"__pycache__", ".git", ".worktrees", "docs", "models", "output"}
bad = []
for path in sorted(root.rglob("*.py")):
    if any(part in skipped for part in path.parts):
        continue
    try:
        with tokenize.open(path) as handle:
            compile(handle.read(), str(path), "exec")
    except SyntaxError as error:
        bad.append("%s:%s: %s" % (path, error.lineno, error.msg))
print("\\n".join(bad))
sys.exit(1 if bad else 0)
"""


def maintained_sources() -> tuple[Path, ...]:
    """Return every project source file, excluding data and vendor examples."""
    return tuple(sorted(
        path
        for path in ROOT.rglob("*.py")
        if not any(part in SKIPPED_PARTS for part in path.relative_to(ROOT).parts)
    ))


def parse(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def annotation_node_ids(tree: ast.Module) -> set[int]:
    """Collect the ids of every expression node sitting in an annotation."""
    found: set[int] = set()
    for node in ast.walk(tree):
        annotations: list[ast.expr] = []
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
            arguments = [*node.args.args, *node.args.posonlyargs, *node.args.kwonlyargs]
            for extra in (node.args.vararg, node.args.kwarg):
                if extra is not None:
                    arguments.append(extra)
            annotations.extend(
                argument.annotation
                for argument in arguments
                if argument.annotation is not None
            )
        for annotation in annotations:
            found.update(id(inner) for inner in ast.walk(annotation))
    return found


def parenthesized_multi_manager_with(path: Path) -> list[int]:
    """Return the lines of ``with (a as x, b as y):`` — 3.9+ syntax."""
    lines: list[int] = []
    with tokenize.open(path) as handle:
        tokens = list(tokenize.generate_tokens(handle.readline))
    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string != "with":
            continue
        previous = next(
            (item for item in reversed(tokens[:index])
             if item.type not in (tokenize.NL, tokenize.COMMENT)),
            None,
        )
        if previous is not None and previous.type not in (
            tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.OP,
        ):
            continue
        opening = next(
            (item for item in tokens[index + 1:]
             if item.type not in (tokenize.NL, tokenize.COMMENT)),
            None,
        )
        if opening is None or opening.string != "(":
            continue
        depth, commas, closing_index = 0, 0, None
        for position in range(tokens.index(opening), len(tokens)):
            item = tokens[position]
            if item.string == "(":
                depth += 1
            elif item.string == ")":
                depth -= 1
                if depth == 0:
                    closing_index = position
                    break
            elif item.string == "," and depth == 1:
                commas += 1
        if closing_index is None or not commas:
            continue
        after = next(
            (item for item in tokens[closing_index + 1:]
             if item.type not in (tokenize.NL, tokenize.COMMENT)),
            None,
        )
        if after is not None and after.string == ":":
            lines.append(token.start[0])
    return lines


class Python38SyntaxTests(unittest.TestCase):
    def test_sources_avoid_parenthesized_multi_manager_with(self) -> None:
        offenders = [
            f"{path.relative_to(ROOT)}:{line}"
            for path in maintained_sources()
            for line in parenthesized_multi_manager_with(path)
        ]
        self.assertEqual(
            offenders,
            [],
            "括号化多管理器 with 是 3.9+ 语法，Jetson 的 3.8 会 SyntaxError",
        )


class Python38RuntimeTests(unittest.TestCase):
    def test_generic_subscripts_are_38_safe(self) -> None:
        """Runtime subscripting must not reach 3.9-only generics."""
        offenders = []
        for path in maintained_sources():
            source, tree = parse(path)
            has_future = "from __future__ import annotations" in source
            abc_names = {
                alias.asname or alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "collections.abc"
                for alias in node.names
            }
            annotations = annotation_node_ids(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
                    continue
                name = node.value.id
                in_annotation = id(node) in annotations
                if name in abc_names and not in_annotation:
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} 运行时下标 "
                        f"collections.abc.{name}（3.8 上 TypeError）"
                    )
                elif not has_future and not in_annotation and (
                    name in abc_names or name in BUILTIN_GENERICS
                ):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} 运行时下标 {name}[...]"
                    )
                elif not has_future and in_annotation and (
                    name in abc_names or name in BUILTIN_GENERICS
                ):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} 注解里的 {name}[...] "
                        "需要 from __future__ import annotations"
                    )
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_sources_avoid_post38_runtime_apis(self) -> None:
        offenders = []
        for path in maintained_sources():
            _, tree = parse(path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else [(node.module or "").split(".")[0]]
                    )
                    offenders.extend(
                        f"{path.relative_to(ROOT)}:{node.lineno} 模块 {name} 需要更高版本"
                        for name in names
                        if name.split(".")[0] in POST38_MODULES
                    )
                if isinstance(node, ast.Call):
                    function = node.func
                    if isinstance(function, ast.Attribute):
                        if function.attr in POST38_METHODS:
                            offenders.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} "
                                f".{function.attr}() 需要 3.9+"
                            )
                        if isinstance(function.value, ast.Name) and (
                            function.value.id, function.attr
                        ) in POST38_ATTRIBUTES:
                            offenders.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} "
                                f"{function.value.id}.{function.attr} 需要更高版本"
                            )
                    if isinstance(function, ast.Name):
                        if function.id in POST38_NAMES:
                            offenders.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} "
                                f"{function.id} 需要 3.11+"
                            )
                        if function.id == "dataclass":
                            offenders.extend(
                                f"{path.relative_to(ROOT)}:{node.lineno} dataclass "
                                f"参数 {keyword.arg} 需要 3.10+"
                                for keyword in node.keywords
                                if keyword.arg in POST38_DATACLASS_ARGUMENTS
                            )
                        if function.id == "isinstance":
                            offenders.extend(
                                f"{path.relative_to(ROOT)}:{node.lineno} "
                                "isinstance 的联合类型需要 3.10+"
                                for argument in node.args
                                if isinstance(argument, ast.BinOp)
                                and isinstance(argument.op, ast.BitOr)
                            )
        self.assertEqual(offenders, [], "\n".join(offenders))


class Python38InterpreterTests(unittest.TestCase):
    def test_sources_compile_under_a_real_python38(self) -> None:
        """A real 3.8 interpreter, when available, is the only complete check."""
        interpreter = find_python38()
        if interpreter is None:
            self.skipTest(
                "未找到 Python 3.8 解释器；设置 GC_PYTHON38 或安装 uv 后重跑"
            )
        result = subprocess.run(
            [interpreter, "-c", COMPILE_SCRIPT, str(ROOT)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_manual_entry_points_import_without_hardware(self) -> None:
        """The arm menu entry must import on the interpreter running this suite."""
        result = subprocess.run(
            [sys.executable, "-c", "import tests.manual.arm"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


def find_python38() -> str | None:
    """Locate a Python 3.8 interpreter: explicit env var, PATH, then uv."""
    explicit = os.environ.get("GC_PYTHON38")
    if explicit and Path(explicit).exists():
        return explicit
    for name in ("python3.8", "python38"):
        found = shutil.which(name)
        if found:
            return found
    uv = shutil.which("uv")
    if uv is None:
        return None
    result = subprocess.run(
        [uv, "python", "find", "3.8"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    unittest.main()

"""Offline checks for the Jetson engine builder (no TensorRT, no board).

``build_engine`` 里真正碰 GPU 的只有两个 ``build_with_*`` 函数；其余部分——路径从
配置里怎么读、命令怎么拼、什么时候该跳过重建——都能在 PC 上钉住。这里只测后者：
在开发机上跑的单测不应该因为没有 TensorRT 就整片跳过。
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.vision.training import build_engine


def _touch(path: Path, *, age_seconds: float = 0.0) -> Path:
    """Create ``path`` (and its parents) with an optional mtime in the past."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))
    return path


class ConfigPathTests(unittest.TestCase):
    """``config.json`` 的 ``model`` 块是 model 路径的唯一来源。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = self.root / "config.json"

    def _write_config(self, model) -> None:
        self.config.write_text(
            json.dumps({"model": model}, ensure_ascii=False), encoding="utf-8"
        )

    def test_missing_config_falls_back_to_models_directory(self) -> None:
        self.assertEqual(build_engine.default_onnx_path(self.config), build_engine.DEFAULT_ONNX)
        self.assertEqual(
            build_engine.default_engine_path(self.config), build_engine.DEFAULT_ENGINE
        )

    def test_relative_paths_resolve_against_the_project_root(self) -> None:
        self._write_config({"onnx": "models/new.onnx", "engine": "models/new.engine"})
        # 关键点：相对路径按项目根解析，不按当前工作目录——否则在 tests/ 里跑就找不到模型。
        self.assertEqual(
            build_engine.default_onnx_path(self.config), build_engine.PROJECT_ROOT / "models/new.onnx"
        )
        self.assertEqual(
            build_engine.default_engine_path(self.config),
            build_engine.PROJECT_ROOT / "models/new.engine",
        )

    def test_absolute_paths_are_used_verbatim(self) -> None:
        absolute = self.root / "elsewhere" / "model.onnx"
        self._write_config({"onnx": str(absolute)})
        self.assertEqual(build_engine.default_onnx_path(self.config), absolute)

    def test_broken_or_unrelated_config_is_ignored(self) -> None:
        self.config.write_text("{not json", encoding="utf-8")
        self.assertEqual(build_engine.default_onnx_path(self.config), build_engine.DEFAULT_ONNX)

        self._write_config("not an object")
        self.assertEqual(build_engine.default_onnx_path(self.config), build_engine.DEFAULT_ONNX)

        self._write_config({"onnx": "   "})
        self.assertEqual(build_engine.default_onnx_path(self.config), build_engine.DEFAULT_ONNX)

    def test_class_names_read_from_the_same_block(self) -> None:
        self._write_config({"class_names": ["red", "yellow", "blue", "green"]})
        self.assertEqual(
            build_engine.configured_class_names(self.config), ["red", "yellow", "blue", "green"]
        )

    def test_class_names_absent_means_use_the_builtin_table(self) -> None:
        self._write_config({"onnx": "models/best.onnx"})
        self.assertIsNone(build_engine.configured_class_names(self.config))


class TrtexecCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trtexec = Path("/usr/src/tensorrt/bin/trtexec")
        self.onnx = Path("/tmp/best.onnx")
        self.engine = Path("/tmp/best.engine")
        # 帮助文本里带 --memPoolSize 时用新写法（TensorRT 8.6+），否则用 --workspace。
        self._help = ""

    def _flag_for(self, help_text: str) -> str:
        with patch.object(
            build_engine.subprocess, "run", return_value=SimpleNamespace(stdout=help_text)
        ):
            return build_engine._workspace_flag(self.trtexec)

    def test_uses_workspace_flag_on_tensorrt_85(self) -> None:
        flag = self._flag_for("--workspace=N  Set workspace size")
        self.assertEqual(flag.format(2048), "--workspace=2048")

    def test_uses_mempool_flag_when_available(self) -> None:
        flag = self._flag_for("--memPoolSize=SPEC  Set memory pool size")
        self.assertEqual(flag.format(2048), "--memPoolSize=workspace:2048MiB")

    def test_falls_back_when_the_help_call_itself_fails(self) -> None:
        with patch.object(build_engine.subprocess, "run", side_effect=OSError("nope")):
            self.assertEqual(
                build_engine._workspace_flag(self.trtexec).format(1024), "--workspace=1024"
            )

    def test_command_carries_paths_workspace_and_fp16(self) -> None:
        with patch.object(
            build_engine.subprocess, "run", return_value=SimpleNamespace(stdout="")
        ):
            command = build_engine.trtexec_command(
                self.trtexec,
                self.onnx,
                self.engine,
                fp16=True,
                workspace_mb=1024,
                verbose=False,
            )
        self.assertEqual(command[0], str(self.trtexec))
        self.assertIn(f"--onnx={self.onnx}", command)
        self.assertIn(f"--saveEngine={self.engine}", command)
        self.assertIn("--workspace=1024", command)
        self.assertIn("--fp16", command)
        self.assertNotIn("--verbose", command)

    def test_no_fp16_omits_the_flag(self) -> None:
        with patch.object(
            build_engine.subprocess, "run", return_value=SimpleNamespace(stdout="")
        ):
            command = build_engine.trtexec_command(
                self.trtexec,
                self.onnx,
                self.engine,
                fp16=False,
                workspace_mb=512,
                verbose=True,
            )
        self.assertNotIn("--fp16", command)
        self.assertIn("--verbose", command)


class FindTrtexecTests(unittest.TestCase):
    def test_explicit_path_that_exists_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = _touch(Path(directory) / "trtexec")
            self.assertEqual(build_engine.find_trtexec(fake), fake)

    def test_explicit_path_that_is_missing_reports_nothing(self) -> None:
        self.assertIsNone(build_engine.find_trtexec(Path("/no/such/trtexec")))


class StalenessTests(unittest.TestCase):
    def test_newer_onnx_means_the_engine_must_be_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = _touch(root / "best.engine", age_seconds=600)
            onnx = _touch(root / "best.onnx", age_seconds=10)
            self.assertTrue(build_engine._is_stale(engine, onnx))

    def test_engine_newer_than_onnx_is_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = _touch(root / "best.engine", age_seconds=10)
            onnx = _touch(root / "best.onnx", age_seconds=600)
            self.assertFalse(build_engine._is_stale(engine, onnx))


class MainFlowTests(unittest.TestCase):
    """``main`` 在碰 GPU 之前的每一步都要能在 PC 上跑完。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _run(self, argv) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = build_engine.main(argv)
        return status, out.getvalue(), err.getvalue()

    def test_missing_onnx_is_refused_before_any_build(self) -> None:
        status, _out, err = self._run(
            ["--onnx", str(self.root / "nope.onnx"), "--engine", str(self.root / "nope.engine")]
        )
        self.assertEqual(status, 2)
        self.assertIn("找不到 ONNX", err)

    def test_current_engine_is_left_alone(self) -> None:
        onnx = _touch(self.root / "best.onnx", age_seconds=600)
        engine = _touch(self.root / "best.engine", age_seconds=10)
        status, out, _err = self._run(["--onnx", str(onnx), "--engine", str(engine)])
        self.assertEqual(status, 0)
        self.assertIn("看起来还是新的", out)

    def test_stale_engine_needs_an_explicit_force(self) -> None:
        onnx = _touch(self.root / "best.onnx", age_seconds=10)
        engine = _touch(self.root / "best.engine", age_seconds=600)
        status, out, _err = self._run(["--onnx", str(onnx), "--engine", str(engine)])
        # 没有 --force 就只提示、不重建：重编一次要几分钟，不该被一次误跑触发。
        self.assertEqual(status, 0)
        self.assertIn("建议重建", out)
        self.assertIn("--force", out)
        self.assertEqual(engine.read_bytes(), b"x")

    def test_check_reports_a_missing_engine_instead_of_building(self) -> None:
        status, _out, err = self._run(
            ["--check", "--engine", str(self.root / "absent.engine")]
        )
        self.assertEqual(status, 2)
        self.assertIn("找不到 engine", err)

    def test_parser_defaults_come_from_the_project_config(self) -> None:
        args = build_engine.build_parser().parse_args([])
        self.assertIsNone(args.onnx)
        self.assertIsNone(args.engine)
        self.assertTrue(args.fp16, "FP16 默认应当是开着的")
        self.assertFalse(args.force)
        self.assertFalse(args.check)


if __name__ == "__main__":
    unittest.main()

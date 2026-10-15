"""模型从哪来：``config.json`` 的 ``model`` 块解析、路径解析与后端选择。

这是"换模型只改配置"这条承诺的看门测试：字段怎么读、相对路径按谁解析、Jetson 上
到底会去加载哪个文件、配错了会不会当场报错，都在这里钉住。全部离线，不碰摄像头/GPU。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.vision.config import (
    BACKEND_AUTO,
    BACKEND_ONNX,
    BACKEND_TENSORRT,
    DEFAULT_ENGINE_MODEL,
    DEFAULT_ONNX_MODEL,
    ModelSettings,
    current_platform,
    default_model_path,
    load_model_settings,
    require_model_path,
)
from modules.vision.drive.yolov8 import DEFAULT_CLASS_NAMES


class TempConfigTestCase(unittest.TestCase):
    """把 ``config.json`` 放到临时目录里，单测不读也依赖真实工程配置。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = Path(self._tmp.name) / "config.json"

    def write(self, model=None, **root_extra) -> Path:
        document = dict(root_extra)
        if model is not None:
            document["model"] = model
        self.config.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return self.config


class DefaultsTests(TempConfigTestCase):
    def test_missing_file_yields_builtin_defaults(self) -> None:
        settings = load_model_settings(self.config)
        self.assertEqual(settings.backend, BACKEND_AUTO)
        self.assertEqual(settings.onnx_path.name, Path(DEFAULT_ONNX_MODEL).name)
        self.assertEqual(settings.engine_path.name, Path(DEFAULT_ENGINE_MODEL).name)
        self.assertEqual(settings.class_names, tuple(DEFAULT_CLASS_NAMES))
        self.assertEqual(settings.materials, settings.class_names)
        self.assertIsNone(settings.input_size)
        self.assertEqual(settings.conf_thresh, 0.25)
        self.assertEqual(settings.iou_thresh, 0.45)

    def test_config_without_a_model_block_is_the_same_as_no_config(self) -> None:
        self.write(None, windows={"camera_id": 1})
        settings = load_model_settings(self.config)
        self.assertEqual(settings.class_names, tuple(DEFAULT_CLASS_NAMES))
        self.assertEqual(settings.backend, BACKEND_AUTO)


class ParsingTests(TempConfigTestCase):
    def test_full_block_is_read(self) -> None:
        self.write({
            "backend": "tensorrt",
            "onnx": "models/other.onnx",
            "engine": "models/other.engine",
            "class_names": ["red", "yellow"],
            "materials": ["red"],
            "input_size": 640,
            "conf_thresh": 0.4,
            "iou_thresh": 0.6,
        })
        settings = load_model_settings(self.config)
        self.assertEqual(settings.backend, BACKEND_TENSORRT)
        self.assertEqual(settings.onnx_path.name, "other.onnx")
        self.assertEqual(settings.engine_path.name, "other.engine")
        self.assertEqual(settings.class_names, ("red", "yellow"))
        self.assertEqual(settings.materials, ("red",))
        self.assertEqual(settings.input_size, 640)
        self.assertAlmostEqual(settings.conf_thresh, 0.4)
        self.assertAlmostEqual(settings.iou_thresh, 0.6)

    def test_relative_paths_follow_the_project_root_not_the_cwd(self) -> None:
        # 关键点：在 tests/ 或任何子目录里跑，也得解析到同一个文件。
        self.write({"onnx": "models/new.onnx", "engine": "models/new.engine"})
        settings = load_model_settings(self.config)
        self.assertTrue(settings.onnx_path.is_absolute())
        self.assertEqual(settings.onnx_path.parts[-2:], ("models", "new.onnx"))
        self.assertEqual(settings.onnx_path.parent.parent, Path.cwd().resolve())

    def test_absolute_paths_are_used_verbatim(self) -> None:
        absolute = Path(self._tmp.name) / "elsewhere" / "m.onnx"
        self.write({"onnx": str(absolute)})
        self.assertEqual(load_model_settings(self.config).onnx_path, absolute)

    def test_materials_default_to_every_class(self) -> None:
        self.write({"class_names": ["red", "blue"]})
        settings = load_model_settings(self.config)
        self.assertEqual(settings.materials, ("red", "blue"))

    def test_null_fields_mean_use_the_builtin_value(self) -> None:
        self.write({"class_names": None, "materials": None, "input_size": None,
                    "onnx": None, "conf_thresh": None})
        settings = load_model_settings(self.config)
        self.assertEqual(settings.class_names, tuple(DEFAULT_CLASS_NAMES))
        self.assertEqual(settings.materials, tuple(DEFAULT_CLASS_NAMES))
        self.assertIsNone(settings.input_size)
        self.assertEqual(settings.conf_thresh, 0.25)


class RejectionTests(TempConfigTestCase):
    """配错了要在启动时就说清楚，而不是等到第一帧推理崩在别处。"""

    def assert_rejected(self, model, message: str) -> None:
        self.write(model)
        with self.assertRaisesRegex(ValueError, message):
            load_model_settings(self.config)

    def test_unknown_backend(self) -> None:
        self.assert_rejected({"backend": "tensor"}, "backend")

    def test_empty_or_duplicated_class_table(self) -> None:
        self.assert_rejected({"class_names": []}, "class_names")
        self.assert_rejected({"class_names": ["red", "red"]}, "重复")

    def test_material_not_in_the_class_table(self) -> None:
        # 写错一个名字会让那一类永远被判成非物料、静默漏检，所以直接拒绝。
        self.assert_rejected(
            {"class_names": ["red", "blue"], "materials": ["green"]}, "materials"
        )

    def test_input_size_must_be_a_positive_int(self) -> None:
        for bad in (0, -320, 320.5, True, "320"):
            with self.subTest(bad=bad):
                self.assert_rejected({"input_size": bad}, "input_size")

    def test_thresholds_must_be_probabilities(self) -> None:
        self.assert_rejected({"conf_thresh": 1.5}, "conf_thresh")
        self.assert_rejected({"iou_thresh": -0.1}, "iou_thresh")

    def test_empty_path_string(self) -> None:
        self.assert_rejected({"onnx": "   "}, "model.onnx")


class BackendTests(TempConfigTestCase):
    def test_auto_picks_opencv_on_windows_and_tensorrt_on_jetson(self) -> None:
        settings = ModelSettings()
        self.assertEqual(settings.resolve_backend("windows"), BACKEND_ONNX)
        self.assertEqual(settings.resolve_backend("jetson"), BACKEND_TENSORRT)

    def test_explicit_backend_overrides_the_platform(self) -> None:
        settings = ModelSettings(backend=BACKEND_ONNX)
        self.assertEqual(settings.resolve_backend("jetson"), BACKEND_ONNX)
        forced = ModelSettings(backend=BACKEND_TENSORRT)
        self.assertEqual(forced.resolve_backend("windows"), BACKEND_TENSORRT)

    def test_model_path_follows_the_resolved_backend(self) -> None:
        settings = ModelSettings(
            onnx_path=Path("/a/best.onnx"), engine_path=Path("/a/best.engine")
        )
        self.assertEqual(settings.model_path("windows"), Path("/a/best.onnx"))
        self.assertEqual(settings.model_path("jetson"), Path("/a/best.engine"))

    def test_current_platform_names_the_two_supported_ones(self) -> None:
        self.assertIn(current_platform(), ("windows", "jetson"))


class ProjectConfigTests(unittest.TestCase):
    """工程里那份 ``config.json`` 本身也要对得上当前模型。"""

    def test_deployed_config_describes_the_four_class_material_model(self) -> None:
        settings = load_model_settings()
        self.assertEqual(settings.class_names, ("red", "yellow", "blue", "green"))
        self.assertEqual(settings.input_size, 320)
        self.assertEqual(settings.onnx_path.name, "best.onnx")
        self.assertEqual(settings.engine_path.name, "best.engine")

    def test_default_model_path_follows_the_configured_backend(self) -> None:
        with patch("modules.vision.config.current_platform", return_value="windows"):
            self.assertEqual(default_model_path().suffix, ".onnx")
        with patch("modules.vision.config.current_platform", return_value="jetson"):
            self.assertEqual(default_model_path().suffix, ".engine")


class MissingModelTests(TempConfigTestCase):
    def test_jetson_without_an_engine_points_at_the_builder(self) -> None:
        settings = ModelSettings(backend=BACKEND_TENSORRT,
                                 engine_path=Path(self._tmp.name) / "best.engine")
        with self.assertRaises(FileNotFoundError) as caught:
            require_model_path(settings)
        message = str(caught.exception)
        # 这条提示就是本项目最容易踩的坑：engine 只能在板上生成，不能从 PC 拷。
        self.assertIn("build_engine", message)
        self.assertIn("best.engine", message)

    def test_missing_onnx_points_at_the_config_key(self) -> None:
        settings = ModelSettings(backend=BACKEND_ONNX,
                                 onnx_path=Path(self._tmp.name) / "best.onnx")
        with self.assertRaises(FileNotFoundError) as caught:
            require_model_path(settings)
        self.assertIn("model.onnx", str(caught.exception))

    def test_existing_file_is_returned_as_is(self) -> None:
        onnx = Path(self._tmp.name) / "best.onnx"
        onnx.touch()
        settings = ModelSettings(backend=BACKEND_ONNX, onnx_path=onnx)
        self.assertEqual(require_model_path(settings), onnx)


if __name__ == "__main__":
    unittest.main()

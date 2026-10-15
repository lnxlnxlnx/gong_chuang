"""视觉包出口：只三个显式公开符号，而且导入包不产生副作用。

包出口原先维护一张"符号 → 模块"的惰性映射表（PEP 562 ``__getattr__``）加上一份
``TYPE_CHECKING`` 再导出，两张表会各自漂移，而且真正的公开契约被淹没在几十个内部
符号里。现在契约只有三个符号，于是这里改测契约本身：

1. ``__all__`` 恰好是那三个；
2. 每个符号就是它定义处的**同一个对象**（不是复制、不是别名、不是转发壳）；
3. 单独 ``import modules.vision`` 不读配置、不开设备、不构造相机或模型，
   也不预加载作为 ``python -m`` 入口的 CLI 模块。
"""

from __future__ import annotations

import subprocess
import sys
import unittest

from modules import vision
from modules.vision.config import VisionConfig
from modules.vision.reports import MaterialReport, RegionReport
from modules.vision.runtime import Vision


class VisionPublicExportsTests(unittest.TestCase):
    def test_all_is_exactly_the_three_public_symbols(self) -> None:
        self.assertEqual(sorted(vision.__all__), ["MaterialReport", "RegionReport", "Vision"])

    def test_public_symbols_are_the_single_definitions(self) -> None:
        """导出的对象必须就是定义处的对象；多一份副本就说明有人在复制类型。"""
        self.assertIs(vision.Vision, Vision)
        self.assertIs(vision.MaterialReport, MaterialReport)
        self.assertIs(vision.RegionReport, RegionReport)

    def test_report_types_have_exactly_one_definition(self) -> None:
        """报告类型只有一份定义（在 reports），旧位置不再自己定义一份。"""
        from modules.vision.drive import material_locate

        self.assertIs(material_locate.MaterialReport, MaterialReport)
        self.assertIs(material_locate.RegionReport, RegionReport)
        self.assertEqual(MaterialReport.__module__, "modules.vision.reports")
        self.assertEqual(RegionReport.__module__, "modules.vision.reports")

    def test_vision_config_carries_the_model_snapshot(self) -> None:
        """``VisionConfig.model`` 在字段尾部：已有的位置参数顺序不变。"""
        from modules.vision.config import ModelSettings

        config = VisionConfig()
        self.assertIsInstance(config.model, ModelSettings)
        self.assertEqual(
            [field for field in VisionConfig.__dataclass_fields__][-1], "model"
        )

    def test_importing_the_package_preloads_no_cli_or_backend(self) -> None:
        """子进程里单独导入本包，检查 sys.modules 与"没开设备"。

        这条用子进程而不是本进程：本进程早就因为别的测试导入过一堆模块了，
        在这里断言等于什么都没测。
        """
        program = (
            "import sys, json\n"
            "import modules.vision\n"
            "loaded = sorted(m for m in sys.modules if m.startswith('modules.vision'))\n"
            "print(json.dumps(loaded))\n"
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", program],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = set(__import__("json").loads(result.stdout.strip().splitlines()[-1]))
        for forbidden in (
            "modules.vision.drive.hough_lock",
            "modules.vision.drive.material_locate",
            "modules.vision.drive.material_yolo",
            "modules.vision.drive.stream",
            "modules.vision.detector_factory",
            "modules.vision.app.demo",
        ):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, loaded)
        # 三个公开符号所在的模块本身当然要在。
        self.assertIn("modules.vision.runtime", loaded)
        self.assertIn("modules.vision.reports", loaded)

    def test_opening_a_session_is_what_reads_config_and_touches_devices(self) -> None:
        """导入不读配置；``Vision.open`` 才读。用假配置路径证明这一点。"""
        from unittest.mock import patch

        with patch("modules.vision.runtime._load_settings") as load:
            load.return_value = VisionConfig()
            with self.assertRaises(ValueError):
                # 空 camera_ids：配置读完了、编号缺了，于是**在开设备之前**就失败。
                Vision.open(config_path="does-not-exist.json")
        load.assert_called_once()


if __name__ == "__main__":
    unittest.main()

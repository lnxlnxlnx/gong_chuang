"""可调阈值：新配置项、形状判据的按需阈值，以及颜色块的"放松档"。

为什么单独一组：这几项原来是**写死在** ``material_blob`` / ``material_face`` 里的常量，
现场光照一变就得改代码。搬进 ``config.json`` 的 ``targets`` 块之后，必须钉住三件事：

1. 默认值与那些常量**逐字相同**——不配就与搬之前完全一致；
2. 写错的值会被拒绝，而不是安静地按默认值跑；
3. 放松档只在"严格容差一个色块都没找到"时才生效，而且**只放大 BGR 容差**
   （色相与亮度下限一放，台面就会整片进掩膜——那两个常量是拿台面实测出来的）。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.vision.config import TargetSettings, load_vision_config
from modules.vision.drive.material_blob import (
    DEFAULT_HUE_TOLERANCE,
    DEFAULT_MAX_BOUNDARY_VARIATION,
    DEFAULT_MIN_AREA_PX,
    DEFAULT_MIN_FILL_RATIO,
    DEFAULT_VALUE_RATIO,
)
from modules.vision.drive.material_face import (
    DEFAULT_EDGE_SUPPORT_MIN,
    DEFAULT_INTERIOR_STD_MAX,
    FaceQuality,
)
from modules.vision.drive.material_locate import MaterialLocator
from modules.vision.runtime import _override_settings

#: 参考色（绿料，config.json 里的调色板项）。
GREEN = (165.0, 252.0, 114.0)

#: 与参考色**色相差 22、BGR 距离 72.1** 的一块料：严格容差 60 找不到它，
#: 放大到 90 才找得到——正是"换了灯、颜色整片漂移"要救回来的那类目标。
SHIFTED = (114.0, 252.0, 165.0)


class FakeYOLO:
    """不出框的模型：这一组测的是颜色与形状门槛，不该被模型结果干扰。"""

    def detect(self, _frame, *, materials_only=False):
        return []

    def close(self):
        pass


def disc_frame(colour, *, radius: int = 40, width: int = 640, height: int = 360) -> np.ndarray:
    """深色背景上的一块纯色圆面（面积、填充率、边界起伏都过得了形状判据）。"""
    import cv2

    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (width // 2, height // 2), radius, tuple(int(v) for v in colour), -1)
    return frame


def locator(*, relax: float, tolerance: float = 60.0) -> MaterialLocator:
    return MaterialLocator(
        yolo=FakeYOLO(),
        focal_px=300.0,
        material_palette={"green": GREEN},
        colour_tolerance=tolerance,
        colour_relax_factor=relax,
    )


class DefaultThresholdTests(unittest.TestCase):
    def test_defaults_are_the_module_constants(self) -> None:
        """不配 config 时必须与写死常量逐字相同，否则搬动本身就改变了行为。"""
        targets = TargetSettings()
        self.assertEqual(targets.hue_tolerance, DEFAULT_HUE_TOLERANCE)
        self.assertEqual(targets.value_ratio, DEFAULT_VALUE_RATIO)
        self.assertEqual(targets.min_area_px, DEFAULT_MIN_AREA_PX)
        self.assertEqual(targets.max_boundary_variation, DEFAULT_MAX_BOUNDARY_VARIATION)
        self.assertEqual(targets.min_fill_ratio, DEFAULT_MIN_FILL_RATIO)
        self.assertEqual(targets.interior_std_max, DEFAULT_INTERIOR_STD_MAX)
        self.assertEqual(targets.edge_support_min, DEFAULT_EDGE_SUPPORT_MIN)


class ThresholdConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.config = self.root / "config.json"
        self.calibration = self.root / "calibration.json"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write(self, targets: dict) -> None:
        self.config.write_text(
            json.dumps({"windows": {"camera_id": 1}, "targets": targets}), encoding="utf-8"
        )

    def load(self) -> TargetSettings:
        return load_vision_config(self.config, self.calibration).targets

    def test_the_new_keys_are_read(self) -> None:
        self.write(
            {
                "hue_tolerance": 12.0,
                "value_ratio": 0.5,
                "min_area_px": 300.0,
                "max_boundary_variation": 0.5,
                "min_fill_ratio": 0.4,
                "interior_std_max": 20.0,
                "edge_support_min": 0.35,
                "colour_relax_factor": 2.0,
            }
        )

        targets = self.load()

        self.assertEqual(targets.hue_tolerance, 12.0)
        self.assertEqual(targets.value_ratio, 0.5)
        self.assertEqual(targets.min_area_px, 300.0)
        self.assertEqual(targets.max_boundary_variation, 0.5)
        self.assertEqual(targets.min_fill_ratio, 0.4)
        self.assertEqual(targets.interior_std_max, 20.0)
        self.assertEqual(targets.edge_support_min, 0.35)
        self.assertEqual(targets.colour_relax_factor, 2.0)

    def test_missing_keys_fall_back_to_the_constants(self) -> None:
        self.write({"color_tolerance": 60.0})

        targets = self.load()

        self.assertEqual(targets.interior_std_max, DEFAULT_INTERIOR_STD_MAX)
        self.assertEqual(targets.colour_relax_factor, TargetSettings().colour_relax_factor)

    def test_null_disables_the_two_optional_shape_checks(self) -> None:
        """这两项写 null 表示"不查这一项"，与"没写"（用默认值）是两件事。"""
        self.write({"max_boundary_variation": None, "min_fill_ratio": None})

        targets = self.load()

        self.assertIsNone(targets.max_boundary_variation)
        self.assertIsNone(targets.min_fill_ratio)

    def test_a_non_positive_value_is_rejected(self) -> None:
        for key in ("hue_tolerance", "value_ratio", "min_area_px", "interior_std_max",
                    "edge_support_min", "colour_relax_factor"):
            with self.subTest(key=key):
                self.write({key: 0})
                with self.assertRaisesRegex(ValueError, key):
                    self.load()


class FaceThresholdTests(unittest.TestCase):
    def test_passes_uses_the_given_thresholds(self) -> None:
        noisy = FaceQuality(interior_std=18.0, edge_support=0.9, sampled_rays=36)

        self.assertFalse(noisy.passes())
        self.assertTrue(noisy.passes(interior_std_max=25.0))

    def test_the_default_property_keeps_its_meaning(self) -> None:
        clean = FaceQuality(interior_std=4.0, edge_support=0.8, sampled_rays=36)
        self.assertTrue(clean.looks_like_face)
        self.assertTrue(clean.passes())


class RelaxedColourTests(unittest.TestCase):
    def test_without_relaxing_a_hue_shifted_blob_is_not_found_by_colour(self) -> None:
        report = locator(relax=1.0).locate(disc_frame(SHIFTED))

        self.assertNotEqual(report.colour_source, "palette-relaxed")

    def test_relaxing_finds_it_and_marks_the_report(self) -> None:
        report = locator(relax=1.5).locate(disc_frame(SHIFTED))

        self.assertEqual(report.colour_source, "palette-relaxed")
        self.assertTrue(report.regions, report.summary())
        self.assertIsNotNone(report.best)

    def test_the_relaxed_pass_only_widens_the_bgr_tolerance(self) -> None:
        """色相与亮度下限必须原样带过去：那两个是拿台面实测定的，一放台面就整片进掩膜。"""
        session = locator(relax=1.5)
        relaxed = session._relaxed_locator()

        self.assertIsNotNone(relaxed)
        self.assertAlmostEqual(relaxed.tolerance, session.locator.tolerance * 1.5)
        self.assertEqual(relaxed.hue_tolerance, session.locator.hue_tolerance)
        self.assertEqual(relaxed.value_ratio, session.locator.value_ratio)
        self.assertEqual(relaxed.min_area_px, session.locator.min_area_px)

    def test_there_is_no_relaxed_pass_when_disabled_or_on_the_hough_path(self) -> None:
        self.assertIsNone(locator(relax=1.0)._relaxed_locator())
        # 没配颜色 → 走霍夫，霍夫那条路有自己的 relaxed，不归这里管。
        self.assertIsNone(MaterialLocator(yolo=FakeYOLO(), focal_px=300.0)._relaxed_locator())


class OverrideTests(unittest.TestCase):
    def test_the_new_knobs_can_be_overridden_for_one_run(self) -> None:
        from modules.vision.runtime import _load_settings

        settings = _override_settings(
            _load_settings(),
            interior_std_max=18.0,
            edge_support_min=0.4,
            colour_relax_factor=1.0,
            min_area_px=250.0,
        )

        self.assertEqual(settings.targets.interior_std_max, 18.0)
        self.assertEqual(settings.targets.edge_support_min, 0.4)
        self.assertEqual(settings.targets.colour_relax_factor, 1.0)
        self.assertEqual(settings.targets.min_area_px, 250.0)
        # 没给的那些保持原样（覆盖不是"重置成默认值"）。
        self.assertEqual(settings.targets.color_tolerance, _load_settings().targets.color_tolerance)


if __name__ == "__main__":
    unittest.main()

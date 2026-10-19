"""工位识别与叠放位的配置块：读取、默认值与非法值拒绝。

这两个块（``station`` / ``stacked_ring``）的字段多，写错一个名字不会崩，只会让那一项
悄悄走默认值——现场表现是"改了配置没反应"。所以这一组对每个块都同时测三件事：

1. **整块缺失时全部走默认值**（不能因为多了一个块就让老配置文件读不出来）；
2. **显式写进去的值真的生效**；
3. **写错的值当场报错**，不静默当成默认值。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.vision.config import (
    StationDetectionSettings,
    StationGeometrySettings,
    StationReferenceSettings,
    StackedRingSettings,
    VisionConfig,
    load_vision_config,
)


class ConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        self.config_path = self.directory / "config.json"
        # 标定文件**故意不存在**：这两个块不该依赖它。
        self.calibration_path = self.directory / "absent-calibration.json"

    def write(self, document) -> None:
        self.config_path.write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8"
        )

    def load(self) -> VisionConfig:
        return load_vision_config(self.config_path, self.calibration_path)

    def assert_rejected(self, block: str, value, expected: str) -> None:
        """把 ``value`` 写进 ``block``，断言读取时抛错且消息里提到 ``expected``。"""
        self.write({"windows": {"camera_id": 0}, block: value})
        with self.assertRaisesRegex(ValueError, expected):
            self.load()


class DefaultsTests(ConfigTestCase):
    def test_both_blocks_are_optional(self) -> None:
        """老配置文件里没有这两个块，必须照旧能读出来（全部走默认值）。"""
        self.write({"windows": {"camera_id": 3}})
        config = self.load()
        self.assertEqual(config.station_geometry, StationGeometrySettings())
        self.assertEqual(config.station_detection, StationDetectionSettings())
        self.assertEqual(config.station_reference, StationReferenceSettings())
        self.assertEqual(config.stacked_ring, StackedRingSettings())

    def test_an_empty_station_block_is_all_defaults(self) -> None:
        self.write({"windows": {"camera_id": 0}, "station": {}})
        config = self.load()
        self.assertEqual(config.station_geometry, StationGeometrySettings())
        self.assertEqual(config.station_detection, StationDetectionSettings())
        self.assertEqual(config.station_reference, StationReferenceSettings())

    def test_an_empty_stacked_block_is_all_defaults(self) -> None:
        self.write({"windows": {"camera_id": 0}, "stacked_ring": {}})
        self.assertEqual(self.load().stacked_ring, StackedRingSettings())

    def test_the_model_field_stays_last(self) -> None:
        """``VisionConfig.model`` 必须在字段尾部——新块只能插在它前面。"""
        self.assertEqual([name for name in VisionConfig.__dataclass_fields__][-1], "model")

    def test_the_default_scale_is_configured_but_the_affine_is_not(self) -> None:
        """默认配了各向同性比例尺、没配仿射矩阵；两者都没有才算"没标定"。"""
        geometry = StationGeometrySettings()
        self.assertIsNotNone(geometry.mm_per_px)
        self.assertIsNone(geometry.image_to_station_affine)
        self.assertTrue(geometry.has_scale)


class StationGeometryTests(ConfigTestCase):
    def test_explicit_values_are_used(self) -> None:
        self.write({
            "windows": {"camera_id": 0},
            "station": {
                "center_spacing_mm": 120.0,
                "outer_diameter_mm": 80.0,
                "ring1_side": "right",
                "mm_per_px": 0.4,
                "image_to_station_affine": [[0.5, 0.01], [-0.02, 0.6]],
            },
        })
        geometry = self.load().station_geometry
        self.assertEqual(geometry.center_spacing_mm, 120.0)
        self.assertEqual(geometry.outer_diameter_mm, 80.0)
        self.assertEqual(geometry.ring1_side, "right")
        self.assertEqual(geometry.mm_per_px, 0.4)
        self.assertEqual(geometry.image_to_station_affine, ((0.5, 0.01), (-0.02, 0.6)))

    def test_no_scale_at_all_is_representable(self) -> None:
        """两个尺度都写 null 表示"没标定"——那时只报像素，不编毫米。"""
        self.write({
            "windows": {"camera_id": 0},
            "station": {"mm_per_px": None, "image_to_station_affine": None},
        })
        geometry = self.load().station_geometry
        self.assertFalse(geometry.has_scale)

    def test_an_unknown_ring1_side_is_rejected(self) -> None:
        self.assert_rejected("station", {"ring1_side": "middle"}, "ring1_side")

    def test_a_malformed_affine_matrix_is_rejected(self) -> None:
        for value, name in (
            ([[1.0, 0.0]], "只有一行"),
            ([[1.0, 0.0], [0.0]], "第二行只有一个数"),
            ([[1.0, 0.0], [0.0, "x"]], "有非数字"),
            ("not-a-matrix", "根本不是数组"),
        ):
            with self.subTest(name=name):
                self.assert_rejected("station", {"image_to_station_affine": value},
                                     "image_to_station_affine")

    def test_non_positive_dimensions_are_rejected(self) -> None:
        for name in ("center_spacing_mm", "outer_diameter_mm"):
            with self.subTest(field=name):
                self.assert_rejected("station", {name: 0.0}, name)
                self.assert_rejected("station", {name: -1.0}, name)


class StationDetectionTests(ConfigTestCase):
    def test_explicit_values_are_used(self) -> None:
        self.write({
            "windows": {"camera_id": 0},
            "station": {
                "canny_low": 40.0,
                "canny_high": 120.0,
                "min_contour_points": 20,
                "min_cluster_support": 3,
                "min_diameter_bands": 1,
                "center_cluster_radius_px": 15.0,
                "max_collinearity_error_ratio": 0.1,
            },
        })
        detection = self.load().station_detection
        self.assertEqual(detection.canny_low, 40.0)
        self.assertEqual(detection.canny_high, 120.0)
        self.assertEqual(detection.min_contour_points, 20)
        self.assertEqual(detection.min_cluster_support, 3)
        self.assertEqual(detection.min_diameter_bands, 1)
        self.assertEqual(detection.center_cluster_radius_px, 15.0)
        self.assertEqual(detection.max_collinearity_error_ratio, 0.1)

    def test_probability_fields_reject_values_over_one(self) -> None:
        for name in ("min_ellipse_aspect", "min_station_span_ratio"):
            with self.subTest(field=name):
                self.assert_rejected("station", {name: 1.5}, name)

    def test_count_fields_reject_zero_and_floats(self) -> None:
        for name in ("min_contour_points", "min_cluster_support", "min_diameter_bands"):
            with self.subTest(field=name):
                self.assert_rejected("station", {name: 0}, name)
                self.assert_rejected("station", {name: 1.5}, name)

    def test_a_boolean_is_not_an_integer(self) -> None:
        """``True`` 是 ``int`` 的实例，不显式排除就会被当成 1 悄悄收下。"""
        self.assert_rejected("station", {"min_cluster_support": True}, "min_cluster_support")


class StationReferenceTests(ConfigTestCase):
    def test_explicit_centres_are_used(self) -> None:
        self.write({
            "windows": {"camera_id": 0},
            "station": {
                "station_yaw_deg": 1.5,
                "ring_centers_px": [[100.0, 200.0], [300.0, 200.0], [500.0, 200.0]],
                "perspective_ratio_range": [0.9, 1.1],
                "maximum_position_error_mm": 1.0,
                "maximum_angle_error_deg": 0.5,
            },
        })
        reference = self.load().station_reference
        self.assertEqual(reference.station_yaw_deg, 1.5)
        self.assertEqual(
            reference.ring_centers_px, ((100.0, 200.0), (300.0, 200.0), (500.0, 200.0))
        )
        self.assertEqual(reference.perspective_ratio_range, (0.9, 1.1))
        self.assertEqual(reference.maximum_position_error_mm, 1.0)
        self.assertEqual(reference.maximum_angle_error_deg, 0.5)

    def test_a_negative_reference_yaw_is_allowed(self) -> None:
        """参考偏角可以是负的（工位本来就斜着装）——这一项不该要求正数。"""
        self.write({
            "windows": {"camera_id": 0},
            "station": {"station_yaw_deg": -12.5},
        })
        self.assertEqual(self.load().station_reference.station_yaw_deg, -12.5)

    def test_the_wrong_number_of_reference_centres_is_rejected(self) -> None:
        """少给一个参考圆心会拿默认坐标去算偏差——那种错误现场最难查。"""
        for value, name in (
            ([[100.0, 200.0], [300.0, 200.0]], "只有两个"),
            ([[100.0, 200.0], [300.0, 200.0], [500.0, 200.0], [700.0, 200.0]], "有四个"),
            ([], "空的"),
        ):
            with self.subTest(name=name):
                self.assert_rejected("station", {"ring_centers_px": value}, "ring_centers_px")

    def test_a_reference_centre_with_the_wrong_shape_is_rejected(self) -> None:
        self.assert_rejected(
            "station",
            {"ring_centers_px": [[100.0, 200.0], [300.0], [500.0, 200.0]]},
            "ring_centers_px",
        )

    def test_an_inverted_ratio_range_is_rejected(self) -> None:
        """低值不小于高值的话，这个区间永远选不中东西——必须报错。"""
        self.assert_rejected("station", {"perspective_ratio_range": [1.2, 0.8]},
                             "perspective_ratio_range")
        self.assert_rejected("station", {"perspective_ratio_range": [1.0, 1.0]},
                             "perspective_ratio_range")

    def test_a_malformed_ratio_range_is_rejected(self) -> None:
        self.assert_rejected("station", {"perspective_ratio_range": [1.0]},
                             "perspective_ratio_range")
        self.assert_rejected("station", {"perspective_ratio_range": 1.0},
                             "perspective_ratio_range")


class StackedRingConfigTests(ConfigTestCase):
    def test_explicit_values_are_used(self) -> None:
        self.write({
            "windows": {"camera_id": 0},
            "stacked_ring": {
                "middle_min_radius_px": 50.0,
                "middle_max_radius_px": 70.0,
                "middle_edge_support": 0.8,
                "middle_search_radius_px": 180.0,
                "middle_reference_px": [600.0, 300.0],
                "line_offset_mm": 60.0,
                "max_line_angle_deg": 5.0,
                "min_line_points": 40,
                "minimum_confidence": 0.6,
            },
        })
        stacked = self.load().stacked_ring
        self.assertEqual(stacked.middle_min_radius_px, 50.0)
        self.assertEqual(stacked.middle_max_radius_px, 70.0)
        self.assertEqual(stacked.middle_edge_support, 0.8)
        self.assertEqual(stacked.middle_search_radius_px, 180.0)
        self.assertEqual(stacked.middle_reference_px, (600.0, 300.0))
        self.assertEqual(stacked.line_offset_mm, 60.0)
        self.assertEqual(stacked.max_line_angle_deg, 5.0)
        self.assertEqual(stacked.min_line_points, 40)
        self.assertEqual(stacked.minimum_confidence, 0.6)

    def test_the_search_radius_and_reference_can_be_disabled(self) -> None:
        """两个都写 null 表示全图搜、且不按"离参考太远"过滤。"""
        self.write({
            "windows": {"camera_id": 0},
            "stacked_ring": {"middle_search_radius_px": None, "middle_reference_px": None},
        })
        stacked = self.load().stacked_ring
        self.assertIsNone(stacked.middle_search_radius_px)
        self.assertIsNone(stacked.middle_reference_px)

    def test_an_unknown_key_is_ignored_rather_than_fatal(self) -> None:
        """多余字段不该让配置读不出来——写错名字的问题由"值没生效"暴露，不是崩掉。"""
        self.write({
            "windows": {"camera_id": 0},
            "stacked_ring": {"line_offset_mm": 60.0, "no_such_field": 1},
        })
        self.assertEqual(self.load().stacked_ring.line_offset_mm, 60.0)

    def test_invalid_values_are_rejected(self) -> None:
        for value, expected in (
            ({"line_offset_mm": 0.0}, "line_offset_mm"),
            ({"middle_edge_support": 2.0}, "middle_edge_support"),
            ({"min_line_points": 0}, "min_line_points"),
            ({"middle_reference_px": [1.0]}, "middle_reference_px"),
            ({"middle_search_radius_px": -5.0}, "middle_search_radius_px"),
            ({"stable_frame_count": 0}, "stable_frame_count"),
        ):
            with self.subTest(value=value):
                self.assert_rejected("stacked_ring", value, expected)

    def test_a_station_block_does_not_disturb_the_stacked_block(self) -> None:
        """两个块互相独立：只写一个时另一个走默认值。"""
        self.write({"windows": {"camera_id": 0}, "station": {"mm_per_px": 0.4}})
        config = self.load()
        self.assertEqual(config.station_geometry.mm_per_px, 0.4)
        self.assertEqual(config.stacked_ring, StackedRingSettings())


class ShippedConfigTests(unittest.TestCase):
    """仓库里那份 ``config.json`` 本身必须能读出来——它是现场在用的那一份。"""

    def test_the_shipped_config_loads(self) -> None:
        config = load_vision_config()
        self.assertGreater(config.station_geometry.center_spacing_mm, 0.0)
        self.assertGreater(config.stacked_ring.line_offset_mm, 0.0)

    def test_the_shipped_config_has_six_palette_colours(self) -> None:
        """调色板配了六色（含黑、浅蓝），比模型的两类多——辅助路径靠它认更多料。"""
        palette = load_vision_config().targets.material_palette
        self.assertEqual(
            set(palette), {"red", "yellow", "blue", "green", "black", "light_blue"}
        )

    def test_the_shipped_reference_centres_are_three(self) -> None:
        self.assertEqual(len(load_vision_config().station_reference.ring_centers_px), 3)


if __name__ == "__main__":
    unittest.main()

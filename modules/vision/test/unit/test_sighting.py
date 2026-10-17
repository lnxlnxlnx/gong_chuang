"""镜头端文本协议的解析。

判据分两类：

- **必须认出来的**：现场真实发出的那种行（含全角标点、可选字段齐全）；
- **必须拒绝的**：没有 ``xyz`` 或没有编号的行。这两类失败方式完全相反——
  前者漏了会让抓取停摆，后者漏了会让机械臂照着一个编出来的位置动。
"""

from __future__ import annotations

import unittest

from modules.vision.drive.sighting import (
    MaterialSighting,
    SightingParseError,
    parse_sighting,
    try_parse_sighting,
)

#: 现场真实的完整样例。
FULL_LINE = (
    "物料 4: @(566,376) 绿色物料（green），confidence=0.88，圆直径 268 px，"
    "内部σ=20.2，边界支撑=83%（36 个方向），xyz=(-8.3,1.8,113.2) mm，直线距离约 113.5 mm"
)


class ParseSightingTests(unittest.TestCase):
    def test_full_sample_line(self) -> None:
        sighting = parse_sighting(FULL_LINE)
        self.assertEqual(sighting.material_id, 4)
        self.assertAlmostEqual(sighting.x_mm, -8.3)
        self.assertAlmostEqual(sighting.y_mm, 1.8)
        self.assertAlmostEqual(sighting.z_mm, 113.2)
        self.assertEqual((sighting.pixel_u, sighting.pixel_v), (566.0, 376.0))
        self.assertEqual(sighting.color, "green")
        self.assertAlmostEqual(sighting.confidence, 0.88)
        self.assertAlmostEqual(sighting.diameter_px, 268.0)
        self.assertAlmostEqual(sighting.inner_sigma, 20.2)
        self.assertAlmostEqual(sighting.edge_support, 83.0)
        self.assertEqual(sighting.edge_directions, 36)
        self.assertAlmostEqual(sighting.straight_distance_mm, 113.5)
        self.assertEqual(sighting.raw, FULL_LINE)

    def test_redundant_distance_agrees_with_xyz(self) -> None:
        """``直线距离`` 是冗余字段：它和 ``xyz`` 对不上就说明镜头端拿错了深度。"""
        sighting = parse_sighting(FULL_LINE)
        self.assertAlmostEqual(sighting.distance_mm, sighting.straight_distance_mm, delta=1.0)

    def test_planar_distance_ignores_z(self) -> None:
        sighting = parse_sighting("物料 1: xyz=(3,4,100) mm")
        self.assertAlmostEqual(sighting.planar_distance_mm, 5.0)
        self.assertAlmostEqual(sighting.distance_mm, (9 + 16 + 10000) ** 0.5)

    def test_optional_fields_may_be_missing(self) -> None:
        """镜头端调参时会加减字段，少几个可选字段不该炸掉整条链路。"""
        sighting = parse_sighting("物料12:@(1,2) xyz=(3.5, -4, 5) mm")
        self.assertEqual(sighting.material_id, 12)
        self.assertEqual((sighting.x_mm, sighting.y_mm, sighting.z_mm), (3.5, -4.0, 5.0))
        self.assertEqual((sighting.pixel_u, sighting.pixel_v), (1.0, 2.0))
        for field in (
            "color",
            "confidence",
            "diameter_px",
            "inner_sigma",
            "edge_support",
            "edge_directions",
            "straight_distance_mm",
        ):
            with self.subTest(field=field):
                self.assertIsNone(getattr(sighting, field))

    def test_full_width_punctuation_and_spacing(self) -> None:
        """全角括号、全角逗号、全角等号、大写 XYZ 都要认。"""
        sighting = parse_sighting("物料 1 @（3，4） XYZ＝（1，2，3）mm")
        self.assertEqual(sighting.material_id, 1)
        self.assertEqual((sighting.x_mm, sighting.y_mm, sighting.z_mm), (1.0, 2.0, 3.0))

    def test_material_id_without_colon_but_with_space(self) -> None:
        sighting = parse_sighting("物料 7 xyz=(1,2,3) mm")
        self.assertEqual(sighting.material_id, 7)

    def test_negative_and_decimal_coordinates(self) -> None:
        sighting = parse_sighting("物料 2: xyz=(-123.456,+7.5,-0.25) mm")
        self.assertEqual((sighting.x_mm, sighting.y_mm, sighting.z_mm), (-123.456, 7.5, -0.25))

    def test_english_only_colour_falls_back(self) -> None:
        self.assertEqual(parse_sighting("物料 1: green xyz=(1,2,3) mm").color, "green")

    def test_chinese_colour_maps_to_the_project_palette_keys(self) -> None:
        """中文色名要归一成 ``modules/vision/config.json`` 里那几个键。"""
        for chinese, expected in (
            ("绿色", "green"),
            ("蓝色", "blue"),
            ("红色", "red"),
            ("黄色", "yellow"),
        ):
            with self.subTest(chinese=chinese):
                line = f"物料 1: {chinese}物料，xyz=(1,2,3) mm"
                self.assertEqual(parse_sighting(line).color, expected)

    def test_custom_english_colour_in_parentheses_is_kept(self) -> None:
        sighting = parse_sighting("物料 1: 青色物料（cyan），xyz=(1,2,3) mm")
        self.assertEqual(sighting.color, "cyan")

    def test_received_at_is_carried_through(self) -> None:
        self.assertIsNone(parse_sighting(FULL_LINE).received_at)
        self.assertEqual(parse_sighting(FULL_LINE, received_at=12.5).received_at, 12.5)

    def test_summary_mentions_the_target_identity(self) -> None:
        text = parse_sighting(FULL_LINE).summary()
        self.assertIn("物料 4", text)
        self.assertIn("green", text)


class ParseSightingRejectionTests(unittest.TestCase):
    def test_missing_xyz_is_rejected(self) -> None:
        with self.assertRaisesRegex(SightingParseError, "xyz"):
            parse_sighting("物料 4: @(566,376) 绿色物料（green），confidence=0.88")

    def test_missing_material_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(SightingParseError, "物料"):
            parse_sighting("xyz=(1,2,3) mm")

    def test_two_dimensional_xyz_is_rejected(self) -> None:
        """只有 x/y 而没有 z 的行不能被当成 3D 坐标收下。"""
        with self.assertRaises(SightingParseError):
            parse_sighting("物料 1: xyz=(1,2) mm")

    def test_empty_and_blank_lines_are_rejected(self) -> None:
        for line in ("", "   ", "\t"):
            with self.subTest(line=line):
                with self.assertRaises(SightingParseError):
                    parse_sighting(line)

    def test_non_string_input_is_rejected(self) -> None:
        with self.assertRaises(SightingParseError):
            parse_sighting(None)  # type: ignore[arg-type]

    def test_log_lines_are_not_sightings(self) -> None:
        for line in ("[INFO] 视觉服务已启动", "warmup done", "frame 1234"):
            with self.subTest(line=line):
                self.assertIsNone(try_parse_sighting(line))

    def test_try_parse_returns_none_instead_of_raising(self) -> None:
        self.assertIsNone(try_parse_sighting("物料 4: 没有坐标"))
        self.assertIsInstance(try_parse_sighting(FULL_LINE), MaterialSighting)


if __name__ == "__main__":
    unittest.main()

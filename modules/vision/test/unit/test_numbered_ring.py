"""环心与环号：模型给框，这里只负责"挑哪个、怎么变读数"。

这个模块的价值不在"能算"，而在**不猜**：

1. 环号只来自 ``ring_target.classes``（类别名 → 环号），代码不解析类别名、也不按
   左右顺序硬排；名字写错在构造时就报错，而不是现场一直"没找到环"。
2. 框被画面裁掉时**不给**可对准读数——那时框心不是环心，误差可以到十几毫米，而
   读数看起来完全正常。
3. 没有比例尺就不报毫米，不拿焦距或框宽凑一个像样的数。

所以用例围绕这三件事，外加"毫米换算与 ``RegionReport`` 同一套坐标"。
不需要模型、不需要摄像头：``FakeDetector`` 直接按给定框返回 ``Detection``。
"""

from __future__ import annotations

import unittest

import numpy as np

from modules.vision.config import (
    RingTargetSettings,
    StationGeometrySettings,
    _ring_target,
    load_vision_config,
)
from modules.vision.drive.numbered_ring import RingTargetLocator
from modules.vision.drive.yolov8 import Detection

SIZE = (1280, 720)

#: 假检测器的类别表：与随包模型一致（四色 + 三个靶标）。
CLASS_NAMES = ("red", "yellow", "blue", "green", "tg1", "tg2", "tg3")


def detection(name: str, score: float, box) -> Detection:
    """一个 ``Detection``；``class_id`` 按类别表反查，便于断言。"""
    x1, y1, x2, y2 = box
    return Detection(
        x1=x1, y1=y1, x2=x2, y2=y2, score=score,
        class_id=CLASS_NAMES.index(name), name=name,
    )


def centred(name: str = "tg2", *, score: float = 0.95, width: float = 520.0,
            height: float = 516.0, centre=(640.0, 360.0)) -> Detection:
    """以 ``centre`` 为中心的一个框（默认整框可见、在光轴上）。"""
    cx, cy = centre
    return detection(name, score, (cx - width / 2, cy - height / 2,
                                   cx + width / 2, cy + height / 2))


class FakeDetector:
    """只回放给定框的检测器；``class_names`` 与真检测器同名字段。"""

    def __init__(self, hits, *, class_names=CLASS_NAMES) -> None:
        self.hits = list(hits)
        # 真检测器总有类别表；假检测器允许给空/None（那就跳过类别表校验）。
        self.class_names = list(class_names) if class_names else []
        self.frames = 0

    def detect(self, _frame):
        self.frames += 1
        return list(self.hits)


def frame() -> np.ndarray:
    return np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)


def make_locator(hits, *, mm_per_px: float | None = 0.5, **kwargs) -> RingTargetLocator:
    return RingTargetLocator(
        FakeDetector(hits),
        geometry=StationGeometrySettings(mm_per_px=mm_per_px),
        **kwargs,
    )


class CentreTests(unittest.TestCase):
    def test_a_fully_visible_box_gives_its_centre(self) -> None:
        target = make_locator([centred("tg2")]).detect(frame())
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.center_px[0], 640.0, delta=0.5)
        self.assertAlmostEqual(target.center_px[1], 360.0, delta=0.5)
        self.assertFalse(target.clipped)
        reading = target.reading()
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading[0], 0.0, delta=0.5)
        self.assertAlmostEqual(reading[1], 0.0, delta=0.5)

    def test_camera_millimetres_use_the_image_centre_as_optical_axis(self) -> None:
        """偏光轴 200 px、0.5 mm/px ⇒ x = -100 mm（与 RegionReport 同一套坐标）。"""
        target = make_locator(
            [centred(centre=(440.0, 360.0))], settings=RingTargetSettings(outer_diameter_mm=None)
        ).detect(frame())
        self.assertAlmostEqual(target.reading()[0], -100.0, delta=0.5)

    def test_the_known_outer_diameter_beats_the_fixed_scale(self) -> None:
        """有已知外径时用"外径 / 模型框"：与距离无关，且压过给固定相机的 mm_per_px。"""
        target = make_locator([centred(centre=(440.0, 360.0), width=520.0)]).detect(frame())
        self.assertEqual(target.scale_source, "ring box")
        # 520 px 框、95 mm 外径 ⇒ 0.1827 mm/px；偏 200 px ⇒ -36.5 mm
        self.assertAlmostEqual(target.scale_mm_per_px, 95.0 / 520.0, places=6)
        self.assertAlmostEqual(target.reading()[0], -200.0 * 95.0 / 520.0, delta=0.5)

    def test_radius_comes_from_the_box_so_it_can_be_drawn(self) -> None:
        target = make_locator([centred(width=520.0, height=400.0)]).detect(frame())
        self.assertAlmostEqual(target.radius_px, 260.0, delta=0.5)


class SelectionTests(unittest.TestCase):
    def test_the_ring_class_wins_over_material_boxes(self) -> None:
        hits = [detection("red", 0.99, (10.0, 10.0, 60.0, 60.0)), centred("tg3", score=0.4)]
        target = make_locator(hits).detect(frame())
        self.assertIsNotNone(target)
        self.assertEqual(target.digit, 3)

    def test_a_fully_visible_box_wins_over_a_higher_scoring_clipped_one(self) -> None:
        hits = [
            detection("tg1", 0.99, (-10.0, 60.0, 240.0, 620.0)),   # 贴左边，分更高
            centred("tg3", score=0.40, width=400.0, height=400.0,
                    centre=(750.0, 350.0)),                        # 整框可见
        ]
        target = make_locator(hits).detect(frame())
        self.assertEqual(target.digit, 3)

    def test_no_ring_box_is_none_with_a_diagnosis(self) -> None:
        locator = make_locator([detection("blue", 0.9, (0.0, 0.0, 80.0, 80.0))])
        self.assertIsNone(locator.detect(frame()))
        self.assertIn("没有环", locator.last_diagnosis)

    def test_a_wanted_digit_ignores_the_higher_scoring_other_ring(self) -> None:
        """两个环都看得见时，要 2 号就拿 2 号——不按分数挑到 3 号。"""
        hits = [
            centred("tg3", score=0.99, width=360.0, height=390.0, centre=(1000.0, 270.0)),
            centred("tg2", score=0.97, width=386.0, height=405.0, centre=(390.0, 284.0)),
        ]
        locator = make_locator(hits)

        self.assertEqual(locator.detect(frame()).digit, 3)          # 不指定：分高的赢
        self.assertEqual(locator.detect(frame(), digit=2).digit, 2)

    def test_the_wanted_digit_missing_is_none_and_says_which(self) -> None:
        locator = make_locator([centred("tg3", score=0.99)])
        self.assertIsNone(locator.detect(frame(), digit=2))
        self.assertIn("2 号环", locator.last_diagnosis)

    def test_none_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_locator([centred()]).detect(None)

    def test_one_digit_on_two_rings_is_refused(self) -> None:
        """模型把相邻环读成同一号（实测会这样）：这一帧作废，绝不按分数挑一个。

        真机上的样子：三个环读出 ``tg1 / tg1 / tg2``，于是"要 2 号"落到了 3 号身上，
        车被正对着开到隔壁环前面，而残差打印看起来完全正常。
        """
        hits = [
            centred("tg1", score=0.96, centre=(200.0, 300.0)),
            centred("tg1", score=0.73, centre=(640.0, 300.0)),
            centred("tg2", score=0.92, centre=(1050.0, 300.0)),
        ]
        locator = make_locator(hits)

        self.assertIsNone(locator.detect(frame()))
        self.assertIn("1 号环出现了 2 次", locator.last_diagnosis)
        # 指定 2 号也一样作废：这一帧的编号整体不可信，2 号也可能就是隔壁那个。
        self.assertIsNone(locator.detect(frame(), digit=2))

    def test_two_boxes_of_the_same_ring_are_not_a_clash(self) -> None:
        """同一个环漏过 NMS 出两个框（中心几乎重合）不算矛盾，照常按分数挑。"""
        hits = [
            centred("tg2", score=0.95, centre=(640.0, 360.0)),
            centred("tg2", score=0.71, centre=(648.0, 362.0)),
        ]
        target = make_locator(hits).detect(frame(), digit=2)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.score, 0.95, places=2)


class ClippedTests(unittest.TestCase):
    """框被画面裁掉：照样返回（说明"看到了"），但不给可对准读数。"""

    def test_clipped_box_has_no_readable_centre(self) -> None:
        target = make_locator([detection("tg1", 0.9, (-40.0, 60.0, 220.0, 620.0))]).detect(frame())
        self.assertIsNotNone(target)
        self.assertTrue(target.clipped)
        self.assertIsNone(target.reading())
        self.assertIn("裁掉", target.summary())

    def test_every_border_counts(self) -> None:
        boxes = {
            "left": (-40.0, 60.0, 220.0, 620.0),
            "top": (400.0, -40.0, 900.0, 300.0),
            "right": (1100.0, 60.0, 1300.0, 620.0),
            "bottom": (400.0, 500.0, 900.0, 780.0),
        }
        for where, box in boxes.items():
            with self.subTest(border=where):
                target = make_locator([detection("tg2", 0.9, box)]).detect(frame())
                self.assertTrue(target.clipped, where)


class ScaleTests(unittest.TestCase):
    def test_pinhole_calibration_wins_over_the_configured_scale(self) -> None:
        """针孔标定在时用 depth/focal：环心读数必须与物料的毫米同一套单位。"""
        locator = make_locator(
            [centred(centre=(440.0, 360.0))],
            mm_per_px=0.4,
            focal_px=800.0,
            table_depth_mm=400.0,   # 0.5 mm/px，而不是配置里的 0.4
        )
        target = locator.detect(frame())
        self.assertEqual(target.scale_source, "depth/focal")
        self.assertAlmostEqual(target.reading()[0], -100.0, delta=0.5)

    def test_no_calibration_falls_back_to_the_configured_scale(self) -> None:
        """针孔标定与已知外径都没有时，才退回 station.mm_per_px。"""
        target = make_locator(
            [centred(centre=(440.0, 360.0))],
            mm_per_px=0.5,
            settings=RingTargetSettings(outer_diameter_mm=None),
        ).detect(frame())
        self.assertEqual(target.scale_source, "station.mm_per_px")
        self.assertAlmostEqual(target.reading()[0], -100.0, delta=0.5)

    def test_no_scale_means_no_readable_centre(self) -> None:
        target = make_locator(
            [centred()],
            mm_per_px=None,
            settings=RingTargetSettings(outer_diameter_mm=None),
        ).detect(frame())
        self.assertIsNotNone(target, "环还是认得的，只是没有毫米")
        self.assertIsNone(target.reading())
        self.assertIn("没有比例尺", target.detail)

    def test_the_shipped_config_carries_the_measured_outer_diameter(self) -> None:
        """随包配置里必须填上靶标外径，否则相机装臂上时读数会退回固定比例尺。"""
        settings = load_vision_config()
        self.assertIsNotNone(settings.ring_target.outer_diameter_mm)
        self.assertGreater(settings.ring_target.outer_diameter_mm, 0)


class RefineTests(unittest.TestCase):
    """框心 ≠ 环心：框内同心圆精修把环心拉回来（实测框心能偏到 80 px）。"""

    TRUE = (640.0, 360.0)

    def concentric(self, *, centre=None, radii=(120.0, 168.0, 228.0)) -> np.ndarray:
        """白底黑同心圆：一圈圆给出好几条椭圆观测，圆心都该是同一个点。"""
        import cv2

        centre = self.TRUE if centre is None else centre
        frame = np.full((SIZE[1], SIZE[0], 3), 255, dtype=np.uint8)
        for radius in radii:
            cv2.circle(frame, (int(centre[0]), int(centre[1])), int(radius),
                       (0, 0, 0), 4)
        return frame

    def test_an_offset_box_is_corrected_to_the_ring_centre(self) -> None:
        frame = self.concentric()
        # 框偏 40 px / -25 px，半径比最外圈略大（真机上模型框就是这样，比图案大一点）
        hit = centred("tg2", width=520.0, height=520.0,
                      centre=(self.TRUE[0] + 40.0, self.TRUE[1] - 25.0))
        target = make_locator([hit]).detect(frame)
        self.assertTrue(target.refined)
        self.assertAlmostEqual(target.center_px[0], self.TRUE[0], delta=2.0)
        self.assertAlmostEqual(target.center_px[1], self.TRUE[1], delta=2.0)
        # 原始框心留着，现场能看出"修了多少"
        self.assertAlmostEqual(target.box_center_px[0], self.TRUE[0] + 40.0, delta=0.1)

    def test_the_summary_says_the_centre_was_refined(self) -> None:
        target = make_locator(
            [centred("tg2", width=520.0, height=520.0, centre=(680.0, 335.0))]
        ).detect(self.concentric())
        self.assertIn("精修", target.summary())

    def test_a_blank_frame_falls_back_to_the_box_centre(self) -> None:
        frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        target = make_locator([centred("tg2")]).detect(frame)
        self.assertFalse(target.refined, "什么也没画，精修应当退回去")
        self.assertAlmostEqual(target.center_px[0], 640.0, delta=0.5)

    def test_refinement_can_be_switched_off(self) -> None:
        hit = centred("tg2", width=520.0, height=520.0, centre=(660.0, 350.0))
        target = RingTargetLocator(
            FakeDetector([hit]), settings=RingTargetSettings(refine_center=False)
        ).detect(self.concentric())
        self.assertFalse(target.refined)
        self.assertAlmostEqual(target.center_px[0], 660.0, delta=0.5)

    def test_clipped_boxes_are_not_refined(self) -> None:
        """贴边的框只有一段弧，精修会把圆心拽偏——干脆不修。"""
        hit = detection("tg1", 0.9, (-30.0, 100.0, 300.0, 620.0))
        target = make_locator([hit]).detect(self.concentric(centre=(120.0, 360.0)))
        self.assertFalse(target.refined)


class ClassTableTests(unittest.TestCase):
    def test_ring_numbers_come_from_config_not_from_the_code(self) -> None:
        """换一遍编号只改配置：tg2 变成 5 号。"""
        locator = make_locator(
            [centred("tg2")],
            settings=RingTargetSettings(classes={"tg1": 3, "tg2": 5, "tg3": 7}),
        )
        self.assertEqual(locator.detect(frame()).digit, 5)

    def test_unknown_class_name_fails_at_construction(self) -> None:
        """名字写错（tg_1）不能变成"一直没找到环"。"""
        with self.assertRaises(ValueError) as caught:
            RingTargetLocator(
                FakeDetector([]),
                settings=RingTargetSettings(classes={"tg_1": 1}),
            )
        self.assertIn("tg_1", str(caught.exception))

    def test_a_detector_without_a_class_table_is_accepted(self) -> None:
        """假检测器/自造检测器不该因为"没有 class_names"就构造不出来。"""
        RingTargetLocator(FakeDetector([], class_names=None))
        locator = RingTargetLocator(
            FakeDetector([], class_names=()), settings=RingTargetSettings(),
        )
        self.assertIsNone(locator.detect(frame()))


class ConfigTests(unittest.TestCase):
    def test_shipped_config_lists_the_three_target_classes(self) -> None:
        settings = load_vision_config()
        self.assertEqual(dict(settings.ring_target.classes),
                         {"tg1": 1, "tg2": 2, "tg3": 3})

    def test_shipped_config_keeps_targets_out_of_the_materials(self) -> None:
        """7 类模型里 tg* 不是物料：否则抓取链路会去抓靶标。"""
        settings = load_vision_config()
        self.assertNotIn("tg1", settings.model.materials)
        self.assertIn("tg1", settings.model.class_names)

    def test_missing_block_falls_back_to_the_builtin_table(self) -> None:
        self.assertEqual(_ring_target(None), RingTargetSettings())

    def test_duplicate_ring_numbers_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ring_target({"classes": {"tg1": 1, "tg2": 1}})

    def test_empty_or_wrongly_typed_classes_are_rejected(self) -> None:
        for bad in ({}, [], "tg1", {"tg1": 0}, {"tg1": True}, {"": 1}):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    _ring_target({"classes": bad})


class SessionTests(unittest.TestCase):
    """会话入口：定位器借用物料定位器那一个检测器，不加载第二份模型。"""

    def make_session(self, frame_image: np.ndarray | None = None):
        from modules.vision.config import VisionConfig
        from modules.vision.runtime import Vision

        vision = Vision(VisionConfig(camera_ids={"windows": 0, "jetson": 0},
                                     station_geometry=StationGeometrySettings(mm_per_px=0.5)))
        # 就绪性由 ``_require_resources`` 一处判定：相机与定位器都得在。
        vision._camera = type("FakeCamera", (), {"release": lambda self: None})()
        vision._read_frame = lambda timeout=None: (
            frame() if frame_image is None else frame_image
        )
        fake = FakeDetector([centred("tg2")])
        vision._locator = type(
            "FakeLocator",
            (),
            {
                "yolo": type("FakeYOLO", (), {"detector": fake})(),
                "close": lambda self: None,
            },
        )()
        return vision, fake

    def test_locate_ring_finds_the_single_ring(self) -> None:
        vision, fake = self.make_session()
        target = vision.locate_ring()
        self.assertIsNotNone(target)
        self.assertEqual(target.digit, 2)
        self.assertEqual(fake.frames, 1, "一帧只能推理一次")

    def test_ring_center_returns_the_align_xy_reading(self) -> None:
        vision, _ = self.make_session()
        centre = vision.ring_center()
        self.assertIsNotNone(centre)
        self.assertAlmostEqual(centre[0], 0.0, delta=0.5)

    def test_the_locator_is_cached_and_shares_the_session_detector(self) -> None:
        vision, fake = self.make_session()
        first = vision._ring_target_locator()
        self.assertIs(vision._ring_target_locator(), first)
        self.assertIs(first.detector, fake)

    def test_a_closed_session_cannot_hand_out_a_locator(self) -> None:
        vision, _ = self.make_session()
        vision.close()
        with self.assertRaises(RuntimeError):
            vision._ring_target_locator()


if __name__ == "__main__":
    unittest.main()

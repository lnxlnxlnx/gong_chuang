"""Verify the material detection trio offline: YOLO wrapper, presence, class."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from modules.vision.config import ModelSettings
from modules.vision.drive.hough_lock import LockedRegion
from modules.vision.drive.material_class import (
    DEFAULT_MATERIAL_LABELS,
    UNKNOWN_MATERIAL_LABEL,
    MaterialClassifier,
)
from modules.vision.drive.material_presence import (
    REASON_LOW_SCORE,
    REASON_NO_HIT,
    REASON_NON_MATERIAL,
    REASON_OK,
    REASON_OUTSIDE,
    REASON_TOO_SMALL,
    MaterialPresence,
)
from modules.vision.drive.material_yolo import MATERIAL_CLASS_NAMES, MaterialYOLO, is_inside_region, is_material
from modules.vision.drive.yolov8 import Detection


def detection(name="red", *, score=0.9, x1=280.0, y1=200.0, x2=360.0, y2=280.0, class_id=0):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id, name=name)


class FakeDetector:
    """Returns a fixed detection list, whatever frame it is given."""

    def __init__(self, hits) -> None:
        self.hits = list(hits)
        self.shapes: list[tuple[int, ...]] = []
        self.closed = False

    def detect(self, frame):
        self.shapes.append(tuple(frame.shape))
        return list(self.hits)

    def draw(self, frame, detections, fps=None):
        """Stand in for the preview helper: return the frame unchanged."""
        return frame.copy()

    def close(self) -> None:
        self.closed = True


class MaterialClassTableTests(unittest.TestCase):
    def test_material_class_names_match_the_deployed_model(self) -> None:
        # 四分类物料模型：四类都是物料，中文标签一一对应。
        self.assertEqual(MATERIAL_CLASS_NAMES, ("red", "yellow", "blue", "green"))
        self.assertEqual(set(DEFAULT_MATERIAL_LABELS), set(MATERIAL_CLASS_NAMES))

    def test_is_material(self) -> None:
        self.assertTrue(is_material("red"))
        self.assertTrue(is_material("green"))
        self.assertFalse(is_material("target-1"))
        self.assertFalse(is_material("zhangai"))

    def test_is_inside_region_uses_the_box_centre(self) -> None:
        region = LockedRegion(center_x=320.0, center_y=240.0, radius=50.0)
        self.assertTrue(is_inside_region(detection(x1=300, y1=220, x2=340, y2=260), region))
        self.assertFalse(is_inside_region(detection(x1=0, y1=0, x2=40, y2=40), region))


class MaterialYOLOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.region = LockedRegion(center_x=200.0, center_y=150.0, radius=60.0)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_detect_returns_everything_and_can_filter_materials(self) -> None:
        fake = FakeDetector([detection("red"), detection("target-1", class_id=6)])
        # materials 显式传死：这条测的是"筛选"本身，不该随 config.json 变。
        yolo = MaterialYOLO(detector=fake, materials=("red",))

        self.assertEqual(len(yolo.detect(self.frame)), 2)
        self.assertEqual([hit.name for hit in yolo.detect(self.frame, materials_only=True)], ["red"])
        self.assertEqual(fake.shapes, [(480, 640, 3), (480, 640, 3)])

    def test_material_set_comes_from_the_injected_snapshot(self) -> None:
        """没显式给 materials 时取构造时那份 ``ModelSettings`` 快照的 materials。

        刻意**不**回读默认配置文件：会话装配时定的类别就是这一会话的类别。
        """
        fake = FakeDetector([detection("red"), detection("blue", class_id=2)])
        snapshot = ModelSettings(class_names=("red", "blue", "green"), materials=("red",))
        yolo = MaterialYOLO(detector=fake, settings=snapshot)
        self.assertEqual(yolo.effective_materials, ("red",))
        self.assertEqual(
            [hit.name for hit in yolo.detect(self.frame, materials_only=True)], ["red"]
        )

    def test_snapshot_is_not_reread_after_construction(self) -> None:
        """实例建好之后改配置文件不影响它——快照在构造时就定死了。"""
        fake = FakeDetector([detection("red")])
        snapshot = ModelSettings(class_names=("red",), materials=("red",))
        yolo = MaterialYOLO(detector=fake, settings=snapshot)
        with patch(
            "modules.vision.config.load_model_settings",
            side_effect=AssertionError("不该在构造之后回读配置"),
        ):
            self.assertEqual(yolo.effective_materials, ("red",))
            self.assertEqual(len(yolo.detect(self.frame, materials_only=True)), 1)

    def test_explicit_materials_win_over_the_snapshot(self) -> None:
        fake = FakeDetector([detection("red")])
        snapshot = ModelSettings(class_names=("red", "blue"), materials=("blue",))
        yolo = MaterialYOLO(detector=fake, materials=("red",), settings=snapshot)
        self.assertEqual(yolo.effective_materials, ("red",))

    def test_detect_in_region_maps_boxes_back_to_the_frame(self) -> None:
        fake = FakeDetector([detection(x1=4.0, y1=6.0, x2=40.0, y2=42.0)])
        hits = MaterialYOLO(detector=fake).detect_in_region(self.frame, self.region, margin=10.0)

        self.assertEqual(len(hits), 1)
        self.assertEqual((hits[0].x1, hits[0].y1, hits[0].x2, hits[0].y2), (134.0, 86.0, 170.0, 122.0))

    def test_detect_in_region_drops_boxes_outside_the_circle(self) -> None:
        corner = FakeDetector([detection(x1=1.0, y1=1.0, x2=9.0, y2=9.0)])
        far_region = LockedRegion(center_x=500.0, center_y=400.0, radius=40.0)

        self.assertEqual(MaterialYOLO(detector=corner).detect_in_region(self.frame, far_region), [])
        kept = MaterialYOLO(detector=corner).detect_in_region(self.frame, far_region, keep_outside=True)
        self.assertEqual(len(kept), 1)

    def test_region_outside_the_frame_yields_nothing(self) -> None:
        fake = FakeDetector([detection()])
        outside = LockedRegion(center_x=-500.0, center_y=-500.0, radius=20.0)
        self.assertEqual(MaterialYOLO(detector=fake).detect_in_region(self.frame, outside), [])
        self.assertEqual(fake.shapes, [])

    def test_crop_region_clamps_to_the_frame(self) -> None:
        box, offset = MaterialYOLO.crop_region(self.frame, LockedRegion(10.0, 10.0, 50.0))
        self.assertEqual(box, (0, 0, 60, 60))
        self.assertEqual(offset, (0, 0))

    def test_context_manager_closes_the_injected_detector(self) -> None:
        fake = FakeDetector([detection()])
        with MaterialYOLO(detector=fake) as yolo:
            self.assertIs(yolo.detector, fake)
        self.assertTrue(fake.closed)

    def test_missing_model_file_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            MaterialYOLO("no-such-model.onnx").detector


class MaterialPresenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.region = LockedRegion(center_x=320.0, center_y=240.0, radius=80.0)
        self.presence = MaterialPresence()

    def test_accepts_a_material_inside_the_region(self) -> None:
        result = self.presence.check([detection()], self.region)
        self.assertTrue(result.present)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.reason, REASON_OK)
        self.assertEqual(result.best.name, "red")

    def test_reports_every_rejection_reason(self) -> None:
        cases = [
            ([], REASON_NO_HIT),
            ([detection("target-1", class_id=6)], REASON_NON_MATERIAL),
            ([detection(score=0.1)], REASON_LOW_SCORE),
            ([detection(x1=300, y1=220, x2=306, y2=226)], REASON_TOO_SMALL),
            ([detection(x1=0, y1=0, x2=80, y2=80)], REASON_OUTSIDE),
        ]
        for hits, expected in cases:
            with self.subTest(reason=expected):
                result = self.presence.check(hits, self.region)
                self.assertFalse(result.present)
                self.assertEqual(result.reason, expected)
                self.assertIsNone(result.best)

    def test_dominant_reason_is_reported_when_nothing_passes(self) -> None:
        weak = [detection(score=0.1), detection("green", score=0.2, class_id=1)]
        self.assertEqual(self.presence.check(weak, self.region).reason, REASON_LOW_SCORE)

    def test_any_class_can_count_as_material(self) -> None:
        presence = MaterialPresence(require_material_class=False)
        self.assertTrue(presence.check([detection("target-1", class_id=6)], self.region).present)

    def test_region_mode_intersect_accepts_a_straddling_box(self) -> None:
        # 框中心在圆外，但框与圆的外接矩形相交
        straddling = detection(x1=400.0, y1=230.0, x2=460.0, y2=250.0)
        self.assertFalse(self.presence.check([straddling], self.region).present)
        intersecting = MaterialPresence(region_mode="intersect")
        self.assertTrue(intersecting.check([straddling], self.region).present)

    def test_region_mode_any_ignores_the_region(self) -> None:
        far = detection(x1=0.0, y1=0.0, x2=80.0, y2=80.0)
        presence = MaterialPresence(region_mode="any")
        self.assertTrue(presence.check([far], self.region).present)

    def test_pick_nearest_prefers_the_closest_box(self) -> None:
        # 两个框都在圆内：near 更靠圆心，far 置信度更高
        near = detection("red", x1=300.0, y1=220.0, x2=340.0, y2=260.0)
        far = detection("blue", score=0.95, class_id=2, x1=350.0, y1=220.0, x2=390.0, y2=260.0)

        by_score = self.presence.check([near, far], self.region)
        self.assertIs(by_score.best, far)
        by_distance = MaterialPresence(pick="nearest").check([near, far], self.region)
        self.assertIs(by_distance.best, near)

    def test_region_can_be_omitted(self) -> None:
        presence = MaterialPresence()
        self.assertTrue(presence.check([detection(x1=0, y1=0, x2=80, y2=80)], None).present)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            MaterialPresence(region_mode="nope")
        with self.assertRaises(ValueError):
            MaterialPresence(pick="nope")
        with self.assertRaises(ValueError):
            MaterialPresence(min_score=-1)

    def test_result_is_truthy_when_present(self) -> None:
        self.assertTrue(self.presence.check([detection()], self.region))
        self.assertFalse(self.presence.check([], self.region))


class MaterialClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = MaterialClassifier()

    def test_picks_the_most_confident_material_and_labels_it(self) -> None:
        hits = [
            detection("blue", score=0.6, class_id=2),
            detection("red", score=0.9),
            detection("zhangai", score=0.99, class_id=9),
        ]
        result = self.classifier.classify(hits)

        self.assertTrue(result.found)
        self.assertEqual(result.name, "red")
        self.assertEqual(result.label, "红色物料")
        self.assertAlmostEqual(result.score, 0.9)
        self.assertEqual(result.names, ("red", "blue"))
        self.assertEqual(len(result.candidates), 2)

    def test_empty_result_when_nothing_qualifies(self) -> None:
        result = self.classifier.classify([detection("target-2", score=0.9, class_id=7)])
        self.assertFalse(result.found)
        self.assertIsNone(result.name)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.names, ())

    def test_low_score_and_tiny_boxes_are_filtered(self) -> None:
        self.assertFalse(self.classifier.classify([detection(score=0.1)]).found)
        self.assertFalse(self.classifier.classify([detection(x1=300, y1=220, x2=306, y2=226)]).found)

    def test_unknown_class_falls_back_to_a_neutral_label(self) -> None:
        classifier = MaterialClassifier(materials=("obj-purple",))
        result = classifier.classify([detection("obj-purple", class_id=3)])
        self.assertTrue(result.found)
        self.assertEqual(result.label, UNKNOWN_MATERIAL_LABEL)

    def test_describe_does_not_filter(self) -> None:
        candidate = self.classifier.describe(detection("blue", score=0.05, class_id=2))
        self.assertEqual(candidate.name, "blue")
        self.assertEqual(candidate.label, "蓝色物料")
        self.assertAlmostEqual(candidate.score, 0.05)

    def test_label_for_known_and_unknown(self) -> None:
        self.assertEqual(self.classifier.label_for("green"), "绿色物料")
        self.assertEqual(self.classifier.label_for("whatever"), UNKNOWN_MATERIAL_LABEL)

    def test_legacy_obj_prefixed_names_still_get_a_label(self) -> None:
        """老模型（``obj-red`` 那一版）的类别名不该因为换表就丢掉中文标签。"""
        classifier = MaterialClassifier(labels={"obj-red": "红色物料"})
        self.assertEqual(classifier.label_for("red"), "红色物料")


if __name__ == "__main__":
    unittest.main()

"""Offline unit tests for the vision peripheral (no TensorRT, CUDA or camera).

These tests import vision package modules and exercise the pure math helpers
(letterbox preprocessing, prediction decoding, NMS, class naming and distance
estimation).  They run anywhere Python + NumPy are installed and are meant to
be executed inside the stdlib unittest runner.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from vision.distance import DistanceEstimator, focal_length_from_fov
from vision.yolov8 import (
    DEFAULT_CLASS_NAMES,
    Detection,
    _nms,
    class_name,
    decode,
    YOLOv8ONNXDetector,
)


class TestClassNames(unittest.TestCase):
    def test_default_is_ten_competition_classes(self):
        expected = [
            "obj-red", "obj-green", "obj-blue", "obj-yellow",
            "obj-light blue", "obj-black",
            "target-1", "target-2", "target-3", "zhangai",
        ]
        self.assertEqual(DEFAULT_CLASS_NAMES, expected)

    def test_class_name_in_range(self):
        self.assertEqual(class_name(0, DEFAULT_CLASS_NAMES), "obj-red")
        self.assertEqual(class_name(9, DEFAULT_CLASS_NAMES), "zhangai")

    def test_class_name_out_of_range_falls_back_to_number(self):
        self.assertEqual(class_name(17, DEFAULT_CLASS_NAMES), "17")


class TestNMS(unittest.TestCase):
    def test_two_overlapping_boxes_keeps_only_high_score(self):
        boxes = np.array([
            [10.0, 10.0, 110.0, 110.0],   # score 0.9
            [20.0, 20.0, 100.0, 100.0],   # score 0.8 (inside the first, IoU high)
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        keep = _nms(boxes, scores, 0.45)
        self.assertEqual(keep.tolist(), [0])

    def test_distant_boxes_are_both_kept(self):
        boxes = np.array([
            [10.0, 10.0, 110.0, 110.0],
            [300.0, 10.0, 500.0, 110.0],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        keep = _nms(boxes, scores, 0.45)
        self.assertEqual(set(keep.tolist()), {0, 1})

    def test_empty_input(self):
        keep = _nms(np.empty((0, 4)), np.empty(0), 0.45)
        self.assertEqual(len(keep), 0)


class TestDecode(unittest.TestCase):
    def _output(self, num_classes=10, conf=0.9, cx=320.0, cy=320.0, w=192.0, h=192.0):
        pred = np.zeros((1, num_classes + 4, 1), dtype=np.float32)
        pred[0, 0, 0] = cx
        pred[0, 1, 0] = cy
        pred[0, 2, 0] = w
        pred[0, 3, 0] = h
        pred[0, 4 + 0, 0] = conf  # class 0 wins
        return pred

    def test_decode_maps_model_coords_back_to_original_pixels(self):
        # YOLOv8 outputs absolute pixel coords in the letterboxed 640x640 model
        # image.  A box centered at model (320, 320) with 192x192 px on a
        # 640x360 original (scale 1.0, top/bottom padding 140) maps back to
        # original center (320, 180).
        pred = self._output()
        boxes, scores, cls_ids = decode(
            pred,
            orig_shape=(640, 360),
            pad=(0, 140),
            scale=1.0,
            conf_thresh=0.25,
            iou_thresh=0.45,
        )
        self.assertEqual(len(boxes), 1)
        x1, y1, x2, y2 = boxes[0]
        self.assertAlmostEqual((x1 + x2) / 2, 320, places=4)
        self.assertAlmostEqual((y1 + y2) / 2, 180, places=4)
        self.assertAlmostEqual(x2 - x1, 192, places=3)
        self.assertAlmostEqual(scores[0], 0.9)
        self.assertEqual(cls_ids[0], 0)

    def test_decode_drops_low_confidence(self):
        pred = self._output(conf=0.1)
        boxes, scores, cls_ids = decode(
            pred, (640, 480), (0, 0), 1.0, conf_thresh=0.25, iou_thresh=0.45
        )
        self.assertEqual(len(boxes), 0)


class TestDetection(unittest.TestCase):
    def test_props(self):
        det = Detection(10, 20, 210, 220, 0.9, 1, "target-2")
        self.assertEqual(det.width, 200)
        self.assertEqual(det.height, 200)
        self.assertEqual(det.center, (110.0, 120.0))


class TestDistance(unittest.TestCase):
    def test_known_class_returns_distance(self):
        est = DistanceEstimator(focal_px=560.0, real_widths={"target-1": 100.0})
        det = Detection(100, 100, 200, 180, 0.9, 6, "target-1")
        self.assertAlmostEqual(est.estimate(det), 560.0 * 100.0 / 100.0)

    def test_unknown_class_returns_none(self):
        est = DistanceEstimator(focal_px=560.0, real_widths={"target-1": 100.0})
        det = Detection(100, 100, 200, 180, 0.9, 9, "zhangai")
        self.assertIsNone(est.estimate(det))

    def test_focal_from_fov(self):
        # 90-degree FOV at 640 wide -> focal 320 px.
        self.assertAlmostEqual(focal_length_from_fov(90.0, 640), 320.0, places=4)


class TestONNXDetector(unittest.TestCase):
    def test_onnx_backend_returns_the_same_detection_interface(self):
        raw = np.zeros((1, 14, 1), dtype=np.float32)
        raw[0, :4, 0] = [320, 320, 100, 100]
        raw[0, 4, 0] = 0.95
        with patch("cv2.dnn.readNetFromONNX") as load_model:
            load_model.return_value.forward.return_value = raw
            with YOLOv8ONNXDetector("best.onnx") as detector:
                hits = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        load_model.assert_called_once_with("best.onnx")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].name, "obj-red")
        self.assertEqual(hits[0].center, (320.0, 320.0))


if __name__ == "__main__":
    unittest.main()

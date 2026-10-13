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

from modules.vision.drive.distance import DistanceEstimator, focal_length_from_fov
from modules.vision.drive.yolov8 import (
    DEFAULT_CLASS_NAMES,
    Detection,
    _nms,
    class_name,
    decode,
    YOLOv8ONNXDetector,
)


class TestClassNames(unittest.TestCase):
    def test_default_is_the_four_deployed_material_classes(self):
        # 索引顺序 = 训练顺序 = ONNX 元数据里的 {0:red, 1:yellow, 2:blue, 3:green}
        # （也就是"1 红 2 黄 3 蓝 4 绿"）。表错了框还是准的、类名会整体错位。
        expected = ["red", "yellow", "blue", "green"]
        self.assertEqual(DEFAULT_CLASS_NAMES, expected)

    def test_class_name_in_range(self):
        self.assertEqual(class_name(0, DEFAULT_CLASS_NAMES), "red")
        self.assertEqual(class_name(3, DEFAULT_CLASS_NAMES), "green")

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
        # 四分类模型的输出是 (1, 4 + 4, 锚框数)；这里只放一个锚框。
        # 坐标是 320×320 模型图里的绝对像素：640×640 的原图 letterbox 进 320 是
        # 缩放 0.5、无填充，所以模型里的中心 (160,160) 映射回原图就是 (320,320)。
        raw = np.zeros((1, 8, 1), dtype=np.float32)
        raw[0, :4, 0] = [160, 160, 100, 100]
        raw[0, 4, 0] = 0.95
        with patch("cv2.dnn.readNetFromONNX") as load_model, \
                patch("modules.vision.drive.yolov8.resolve_onnx_metadata",
                      return_value=(320, None)):
            load_model.return_value.forward.return_value = raw
            with YOLOv8ONNXDetector("best.onnx") as detector:
                hits = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        load_model.assert_called_once_with("best.onnx")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].name, "red")
        self.assertEqual(hits[0].center, (320.0, 320.0))

    def test_onnx_input_size_comes_from_the_file_when_it_can_be_read(self):
        with patch("cv2.dnn.readNetFromONNX"), \
                patch("modules.vision.drive.yolov8.resolve_onnx_metadata",
                      return_value=(320, ["red", "yellow", "blue", "green"])):
            detector = YOLOv8ONNXDetector("best.onnx")
        self.assertEqual(detector.input_size, 320)
        self.assertEqual(detector.class_names, ["red", "yellow", "blue", "green"])

    def test_configured_input_size_that_disagrees_with_the_file_is_refused(self):
        # 写死 640 去喂 320 的模型，OpenCV 会在 Reshape 层抛一个和模型无关的断言；
        # 在这里就把话说清楚。
        with patch("cv2.dnn.readNetFromONNX"), \
                patch("modules.vision.drive.yolov8.resolve_onnx_metadata",
                      return_value=(320, None)):
            with self.assertRaisesRegex(ValueError, "320"):
                YOLOv8ONNXDetector("best.onnx", input_size=640)


if __name__ == "__main__":
    unittest.main()

"""TensorRT-accelerated YOLOv8 detection as a reusable Jetson peripheral.

This module wraps a serialized TensorRT engine (``*.engine``) exported from
an ultralytics YOLOv8 model.  It is intentionally dependency-light at import
time: the heavy ``tensorrt`` / ``pycuda`` packages are imported lazily inside
:meth:`YOLOv8Detector.__init__`, so the pure helpers (letterbox math,
prediction decoding, NMS, class naming) stay importable and unit-testable on
machines without a GPU or TensorRT.

Typical usage::

    detector = YOLOv8Detector("models/best.engine")
    detections = detector.detect(bgr_frame)     # list[Detection]
    annotated = detector.draw(bgr_frame, detections, fps=25.0)
    detector.close()
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

#: 内置类别表，索引即训练时的类别顺序。
#:
#: 当前物料模型（``models/best.onnx``，四类）在 ONNX 元数据里写的是
#: ``{0: 'red', 1: 'yellow', 2: 'blue', 3: 'green'}``，也就是"1 红 2 黄 3 蓝 4 绿"，
#: 这张表必须一字不差地对上：**框的坐标由模型给出，类名由这张表按索引查**，
#: 表错了框还是准的、类名会整体错位。
#:
#: 换模型优先改 ``modules/vision/config.json`` 的 ``model.class_names``；
#: 这里只是没有配置（以及 ONNX 里也没有元数据）时的兜底。
DEFAULT_CLASS_NAMES = ["red", "yellow", "blue", "green"]

#: Decode thresholds used when not overridden by the caller.
DEFAULT_CONF_THRESH = 0.25
DEFAULT_IOU_THRESH = 0.45

#: 方形输入的默认边长。ONNX 通道优先问模型自己（见 :func:`resolve_onnx_metadata`），
#: TensorRT 通道优先问引擎自己；这个常量只是两者都问不出来时的兜底。
#: 当前模型是 320×320（ONNX 元数据 ``imgsz``）。
DEFAULT_INPUT_SIZE = 320

#: Fill color used for letterbox padding when resizing into the square input.
_LETTERBOX_FILL = 114

#: Built-in BGR palette, reused cyclically when no per-class colors are given.
_BGR_PALETTE = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 255, 0),    # light blue (cyan)
    (0, 0, 0),        # black
    (128, 0, 128),    # purple
    (255, 165, 0),    # orange
    (255, 192, 203),  # pink
    (120, 120, 120),  # grey
]


def _cv2():
    """Import and return ``cv2`` lazily (keeps module import free of OpenCV)."""
    import cv2
    return cv2


def class_name(class_id: int, class_names) -> str:
    """Look up a class name, falling back to the numeric index as a string.

    A detector may be loaded with an engine whose class count differs from the
    configured table; this guarantee prevents out-of-range label lookups.
    """
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)


def resolve_onnx_metadata(model_path) -> tuple[int | None, list[str] | None]:
    """读 ONNX 文件自身的输入边长与类别表；读不出来时返回 ``(None, None)``。

    模型换了就得跟着改的两件事——输入尺寸和类别表——都写在 ONNX 里，能从文件本身
    读出来就不该再靠代码里的常量猜。装了 ``onnx`` 才读（Windows 上用 OpenCV 跑
    推理并不需要它），没装就退回默认值：输入 :data:`DEFAULT_INPUT_SIZE`、
    类别表 :data:`DEFAULT_CLASS_NAMES`。

    **TensorRT 引擎没有这条退路**：``.engine`` 里只有张量名和形状，ONNX 的
    ``names`` 元数据不会跟过去。板上要正确的类名，只能靠
    ``modules/vision/config.json`` 的 ``model.class_names`` 或内置表。

    实测教训：新导出的 320×320 四分类模型配上写死的 640 输入，OpenCV 会在
    Reshape 层直接抛 ``computeShapeByReshapeMask`` 断言失败。
    """
    try:
        import onnx
    except ImportError:
        return None, None
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception:  # onnx 对损坏/异构文件抛的异常类型不定，探测失败就当读不出来
        return None, None

    size: int | None = None
    for value in model.graph.input:
        dimensions = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        if len(dimensions) == 4 and dimensions[2] > 0 and dimensions[3] > 0:
            size = int(dimensions[2])
            break

    names: list[str] | None = None
    for prop in model.metadata_props:
        if prop.key != "names":
            continue
        try:
            table = ast.literal_eval(prop.value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(table, dict) and table:
            names = [str(table[key]) for key in sorted(table)]
    return size, names


def _class_color(class_id: int, colors) -> tuple[int, int, int]:
    """Pick a BGR color for ``class_id``, reusing the palette cyclically."""
    if colors is not None and len(colors) > 0:
        return colors[class_id % len(colors)]
    return _BGR_PALETTE[class_id % len(_BGR_PALETTE)]


@dataclass(frozen=True)
class Detection:
    """A single predicted object box in original-image pixel coordinates.

    Attributes:
        x1, y1: top-left corner (float, clipped to the original image bounds).
        x2, y2: bottom-right corner.
        score: class-agnostic confidence ``[0, 1]`` from the model.
        class_id: predicted class index (position in the class table).
        name: human-readable class name (falls back to ``str(class_id)``).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    name: str

    @property
    def width(self) -> float:
        """Bounding box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Bounding box height in pixels."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """``(cx, cy)`` center of the box in pixels."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class Detector(Protocol):
    """The detector interface applications may inject.

    :class:`YOLOv8Detector` implements it; :class:`modules.vision.MaterialYOLO` accepts
    any other object with the same three methods (its own preview helper draws
    through this interface).
    """

    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        fps: float | None = None,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Class-agnostic NMS implemented with NumPy only.

    Mirrors the behaviour of ``cv2.dnn.NMSBoxes`` when boxes from all classes
    are passed together (which is how the source robot code invoked it).
    Returns the flat indices (into ``boxes``) kept after suppression.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # highest confidence first
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-9
        ious = inter / union
        order = rest[ious <= iou_thresh]

    return np.asarray(keep, dtype=int)


def decode(raw_output: np.ndarray, orig_shape, pad, scale,
           conf_thresh: float, iou_thresh: float):
    """Convert raw model output into filtered NMS boxes (pure NumPy).

    Parameters:
        raw_output: engine output shaped ``(1, 4 + num_classes, anchors)``.
        orig_shape: ``(orig_w, orig_h)`` of the input frame.
        pad: letterbox pad ``(dw, dh)`` produced by :meth:`YOLOv8Detector.preprocess`.
        scale: letterbox scale factor produced by preprocessing.
        conf_thresh: minimum class score to keep.
        iou_thresh: IoU threshold for NMS suppression.

    Returns:
        ``(boxes_xyxy, scores, class_ids)`` in original-image pixel
        coordinates (all empty when nothing passes the confidence filter).
    """
    preds = np.transpose(raw_output[0], (1, 0))  # (anchors, 4 + num_classes)
    boxes = preds[:, :4]        # cx, cy, w, h (model coordinates)
    cls_scores = preds[:, 4:]

    max_scores = np.max(cls_scores, axis=1)
    cls_ids = np.argmax(cls_scores, axis=1)

    mask = max_scores > conf_thresh
    boxes, scores, cls_ids = boxes[mask], max_scores[mask], cls_ids[mask]
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    # Remove letterbox padding from centers, then scale back to the original size.
    dw, dh = pad
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, :4] /= scale

    orig_w, orig_h = orig_shape
    x1 = np.clip(boxes[:, 0] - boxes[:, 2] / 2.0, 0, orig_w)
    y1 = np.clip(boxes[:, 1] - boxes[:, 3] / 2.0, 0, orig_h)
    x2 = np.clip(boxes[:, 0] + boxes[:, 2] / 2.0, 0, orig_w)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3] / 2.0, 0, orig_h)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms(boxes_xyxy, scores, iou_thresh)
    return boxes_xyxy[keep], scores[keep], cls_ids[keep]


class YOLOv8Detector:
    """Load a YOLOv8 TensorRT engine and run end-to-end detection on BGR frames.

    The engine's input/output layouts are read from the engine itself, so the
    same class works with engines exported at any input size (e.g. 320x320 or
    640x640) and any number of classes.

    Parameters:
        engine_path: path to the serialized TensorRT ``*.engine`` file.
        class_names: optional list of class names in training-index order;
            defaults to :data:`DEFAULT_CLASS_NAMES`.
        conf_thresh: minimum class score kept after decoding.
        iou_thresh: IoU threshold for NMS suppression.
        colors: optional list of BGR color tuples used by :meth:`draw`,
            one per class; falls back to a built-in palette.
    """

    def __init__(self, engine_path: str, *, class_names=None,
                 conf_thresh: float = DEFAULT_CONF_THRESH,
                 iou_thresh: float = DEFAULT_IOU_THRESH,
                 colors=None, input_size: int | None = None):
        # Lazy imports: tensorrt/pycuda must not be required just to import
        # this module.  pycuda.autoinit is imported here (not at module level)
        # to avoid conflicts when the vision code runs in child processes.
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)

        self._trt = trt
        self._cuda = cuda
        self.class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.colors = colors

        self.engine = None
        self.context = None
        self.input_shape = None
        self.output_shape = None
        self.input_size = DEFAULT_INPUT_SIZE
        self.h_input = None
        self.h_output = None
        self.d_input = None
        self.d_output = None
        self.stream = None

        self._load_engine(engine_path)
        if input_size is not None and int(input_size) != self.input_size:
            # 引擎的输入尺寸写在引擎里；配置里写的不一致就直接说清楚，
            # 别等到第一帧推理时才崩在 reshape 上。
            raise ValueError(
                f"配置的模型输入边长 {int(input_size)} 与引擎的 {self.input_size} 不一致：{engine_path}"
            )
        self._alloc_buffers()
        self._warmup()

    # ------------------------------------------------------------------ setup

    def _load_engine(self, engine_path: str) -> None:
        """Deserialize the engine and cache input/output tensor shapes."""
        trt = self._trt
        with open(engine_path, "rb") as handle:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise ValueError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Prefer explicit names when present (robust on newer TensorRT).
        in_shape = out_shape = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if name == "images":
                in_shape = shape
            elif name == "output0":
                out_shape = shape

        # Fall back to binding order for older exports.
        if in_shape is None:
            in_shape = tuple(self.engine.get_binding_shape(0))
        if out_shape is None:
            out_shape = tuple(self.engine.get_binding_shape(1))

        if len(in_shape) != 4:
            raise ValueError(f"unexpected engine input shape: {in_shape}")
        self.input_shape = in_shape
        self.output_shape = out_shape
        _, _, self.input_height, self.input_width = in_shape
        self.input_size = self.input_height

    def _alloc_buffers(self) -> None:
        """Allocate page-locked host buffers, GPU memory and a CUDA stream."""
        cuda = self._cuda
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

    def _warmup(self) -> None:
        """Run one dummy inference to let cuDNN/TensorRT settle on their plans."""
        self.infer(np.random.rand(*self.input_shape).astype(np.float32))

    # --------------------------------------------------------------- pipeline

    @staticmethod
    def preprocess(frame: np.ndarray, input_size: int):
        """Letterbox a BGR frame into the model's square input blob.

        Returns ``(blob, orig_shape, pad, scale)`` where ``blob`` is
        ``(1, 3, input_size, input_size)`` float32 in CHW order.
        """
        cv2 = _cv2()
        orig_h, orig_w = frame.shape[:2]
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((input_size, input_size, 3), _LETTERBOX_FILL, dtype=np.uint8)
        dw = (input_size - new_w) // 2
        dh = (input_size - new_h) // 2
        padded[dh:dh + new_h, dw:dw + new_w] = resized

        # BGR -> RGB, normalize to [0, 1], HWC -> CHW, add batch dimension.
        rgb = padded[..., ::-1]
        norm = rgb.astype(np.float32) / 255.0
        chw = np.transpose(norm, (2, 0, 1))
        blob = np.expand_dims(chw, axis=0)
        return np.ascontiguousarray(blob), (orig_w, orig_h), (dw, dh), scale

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Copy to GPU, run async inference, copy back and return the raw output."""
        if blob.shape != self.input_shape:
            raise ValueError(f"blob shape {blob.shape} != engine input {self.input_shape}")
        cuda = self._cuda
        np.copyto(self.h_input, blob)

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle,
        )
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run full detection on one BGR frame and return ``list[Detection]``."""
        blob, orig_shape, pad, scale = self.preprocess(frame, self.input_size)
        raw_output = self.infer(blob)
        boxes, scores, cls_ids = decode(
            raw_output, orig_shape, pad, scale, self.conf_thresh, self.iou_thresh
        )
        return [
            Detection(
                x1=float(box[0]), y1=float(box[1]),
                x2=float(box[2]), y2=float(box[3]),
                score=float(score), class_id=int(cid),
                name=class_name(int(cid), self.class_names),
            )
            for box, score, cid in zip(boxes, scores, cls_ids)
        ]

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        fps: float | None = None,
    ) -> np.ndarray:
        """Draw boxes, labels and an optional FPS readout on a copy of ``frame``."""
        cv2 = _cv2()
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
            color = _class_color(det.class_id, self.colors)
            label = f"{det.name} {det.score:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        if fps is not None:
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )
        return annotated

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Release GPU buffers, context and engine (call once when done)."""
        if self.stream is not None:
            self.stream.synchronize()
        for attr in ("d_input", "d_output", "h_input", "h_output",
                     "context", "engine", "stream"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    del obj
                except (TypeError, AttributeError):
                    pass
            setattr(self, attr, None)

    def __enter__(self) -> "YOLOv8Detector":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class YOLOv8ONNXDetector(YOLOv8Detector):
    """Run the same YOLOv8 preprocessing and decoding with OpenCV on Windows.

    输入边长与类别表**优先从 ONNX 文件自身读**（``resolve_onnx_metadata``）：换模型
    （例如新导出的 320×320 四分类）时不必再改代码。装了 ``onnx`` 才读得到，
    读不出来就退回 :data:`DEFAULT_INPUT_SIZE` 输入和 :data:`DEFAULT_CLASS_NAMES`。
    这个适配器只负责模型加载与推理，检测结果与画图跟 TensorRT 实现完全一致。
    """

    def __init__(self, model_path: str, *, class_names=None,
                 conf_thresh: float = DEFAULT_CONF_THRESH,
                 iou_thresh: float = DEFAULT_IOU_THRESH, colors=None,
                 input_size: int | None = None):
        detected_size, detected_names = resolve_onnx_metadata(model_path)
        if input_size is not None and detected_size and int(input_size) != int(detected_size):
            raise ValueError(
                f"配置的模型输入边长 {int(input_size)} 与 {model_path} 里的 {detected_size} 不一致"
            )
        self.model_path = str(model_path)
        self.detected_class_names = detected_names
        self.class_names = (
            list(class_names) if class_names
            else (list(detected_names) if detected_names else list(DEFAULT_CLASS_NAMES))
        )
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.colors = colors
        self.input_size = int(detected_size or input_size or DEFAULT_INPUT_SIZE)
        self.net = _cv2().dnn.readNetFromONNX(model_path)

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Evaluate a preprocessed frame using OpenCV's ONNX reader."""
        if blob.shape != (1, 3, self.input_size, self.input_size):
            raise ValueError(f"unexpected ONNX input shape: {blob.shape}")
        self.net.setInput(blob)
        return self.net.forward()

    def close(self) -> None:
        """Release the OpenCV network reference."""
        self.net = None


if __name__ == "__main__":  # pragma: no cover - exercised manually on the board
    import time

    import cv2

    from modules.vision.drive.preview import close_previews, open_preview

    detector = YOLOv8Detector("models/best.engine")
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        raise SystemExit("camera failed to open: check ls /dev/video*")

    fps = 0.0
    previous = time.time()
    print("preview running; press q to quit")
    window = open_preview()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            detections = detector.detect(frame)
            now = time.time()
            fps = fps * 0.9 + (1.0 / (now - previous + 1e-6)) * 0.1
            previous = now
            annotated = detector.draw(frame, detections, fps=fps)
            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        close_previews()
        detector.close()

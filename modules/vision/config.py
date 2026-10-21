"""Vision settings: hand-edited targets plus machine-calibrated numbers.

两个文件、两种性质，刻意分开：

- ``config.json`` —— **手改**。摄像头编号、**用哪个模型**，以及抓取用的目标尺寸与颜色。
- ``calibration.json`` —— **标定产物**。像素焦距、镜头到台面的深度、平面标定结果，
  由标定脚本写入（见 ``modules/grasp/docs/抓取坐标链路.md``）。

分开的理由是它们的生命周期不同：换摄像头要改 ``config.json``，重新标定只动
``calibration.json``，两者不会互相覆盖，git 里也一眼能看出"这台机器标定过了"。

``config.json`` 里的 ``targets`` 块整体可选。缺失时物料按默认 30 mm 圆面处理，
靶标视为未配置——此时抓取流程必须在打开摄像头之前拒绝启动，而不是拿一个假直径去算距离。

``config.json`` 里的 ``model`` 块决定"加载哪个模型文件"（见 :class:`ModelSettings`）。
它是**唯一**换模型的地方：文件路径、类别表、输入边长、阈值都在这里，代码里不再写死
``models/best.onnx`` / ``best.engine``。相对路径按项目根解析（不是当前工作目录），
绝对路径原样使用。整块缺失时全部走内置默认值，行为与加这个块之前一致。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

from project_paths import PROJECT_ROOT
from modules.vision.drive.calibration import PlaneCalibration
from modules.vision.drive.pinhole import normalize_depth_range
from modules.vision.drive.yolov8 import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_CONF_THRESH,
    DEFAULT_INPUT_SIZE,
    DEFAULT_IOU_THRESH,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"

#: 物料顶部内侧圆面的实际直径；与 ``modules/vision/drive/material_locate`` 的默认值一致。
DEFAULT_MATERIAL_DIAMETER_MM = 30.0

#: 默认颜色匹配容差（BGR 欧氏距离）。同一盏灯下同一颜色的实测散布通常远小于这个值。
DEFAULT_COLOR_TOLERANCE = 60.0

#: 标定脚本支持的平台名，与 ``config.json`` 顶层键一致。
PLATFORMS = ("windows", "jetson")

#: ``station.ring1_side`` 的合法取值：哪一端算环 1。工位翻面装时改这一个字段。
RING1_SIDES = ("left", "right")

#: 没配 ``model.onnx`` 时的 ONNX 路径（相对项目根）。
DEFAULT_ONNX_MODEL = "models/best.onnx"

#: 没配 ``model.engine`` 时的 TensorRT 引擎路径（相对项目根）。
DEFAULT_ENGINE_MODEL = "models/best.engine"

# ---------------------------------------------------------------- 工位识别默认值
# 这些默认值原先只写在 ``ring_station`` / ``stacked_ring`` 的模块常量里。搬到这里是
# 为了让 ``config.json`` 能覆盖它们——现场换工位尺寸时改配置，不用改代码。
# 默认值与那两个模块的常量**逐字相同**，不配就与搬之前完全一致。

#: 三个工位环的圆心间距（毫米）。
DEFAULT_STATION_CENTER_SPACING_MM = 150.0

#: 工位环的外径（毫米）。
DEFAULT_STATION_OUTER_DIAMETER_MM = 95.0

#: 没有仿射矩阵时的各向同性比例尺（毫米/像素）。``None`` 表示没标定过。
#: 1280×720 画幅下 0.495 是 300 mm 转盘外圆实测出来的（606.09 px 直径）。
DEFAULT_STATION_MM_PER_PX = 0.495

#: 验收门槛：三个环里最大的位移（毫米）。
DEFAULT_STATION_MAX_POSITION_ERROR_MM = 2.0

#: 验收门槛：偏角误差（度）。
DEFAULT_STATION_MAX_ANGLE_ERROR_DEG = 1.0

#: 判定"稳定"要连续看几帧。
DEFAULT_STATION_STABLE_FRAME_COUNT = 3

#: 稳定窗口内的圆心抖动上限（像素）。
DEFAULT_STATION_MAX_CENTER_JITTER_PX = 1.5

#: 叠放位：中环圆心到上下基准线的距离（毫米）。
DEFAULT_STACKED_LINE_OFFSET_MM = 75.0

#: 叠放位：中环的像素半径窗口。半径落在窗口外的圆不是中环。
DEFAULT_STACKED_MIDDLE_MIN_RADIUS_PX = 60.0
DEFAULT_STACKED_MIDDLE_MAX_RADIUS_PX = 82.0

#: 叠放位：中环圆周上的边缘支撑率下限（0–1）。中环被物料挡住一角时支撑率会掉，
#: 取 0.72 是"缺一小段还认、缺一大块不认"的分界。
DEFAULT_STACKED_MIDDLE_EDGE_SUPPORT = 0.72

#: 叠放位：基准线方向相对水平的允许倾角（度）。上下基准线在画面里基本是横的，
#: 限在 ±8° 以内可以挡掉台面纹理拼出来的斜线。
DEFAULT_STACKED_MAX_LINE_ANGLE_DEG = 8.0

#: 叠放位：两条基准线的夹角超过它就不配对（度）。它们本该平行。
DEFAULT_STACKED_MAX_LINE_DISAGREEMENT_DEG = 2.0

#: 模型里哪些类别是"带数字的环"，以及各自算几号。
#:
#: 环心与"环里是几号"都由模型一次给出（``tg1`` / ``tg2`` / ``tg3``），所以这里只剩
#: 一张**类别名 → 环号**的表：它是环号的唯一来源，代码既不解析类别名，也不按左右
#: 顺序硬排。换一套靶标只改这张表（见 ``config.json`` 的 ``ring_target`` 块）。
DEFAULT_RING_TARGET_CLASSES = {"tg1": 1, "tg2": 2, "tg3": 3}

#: 靶标图案的**外径**（毫米）：模型框圈住的那一圈虚线外圆。现场 2026-10-20 实测 95 mm
#: （与 ``station.outer_diameter_mm`` 同一个工件）。它当比例尺的好处是与镜头距离无关，
#: 代价是精度取决于模型框的松紧（实测同一条靶标 449~580 px）。
DEFAULT_RING_OUTER_DIAMETER_MM = 95.0

#: 推理后端名。``auto`` 按平台选：Windows 走 OpenCV ONNX，Jetson 走 TensorRT。
BACKEND_AUTO = "auto"
BACKEND_TENSORRT = "tensorrt"
BACKEND_ONNX = "onnx"
BACKENDS = (BACKEND_AUTO, BACKEND_TENSORRT, BACKEND_ONNX)


def current_platform() -> str:
    """本机对应的配置平台名：Windows 是 ``"windows"``，Jetson/Linux 是 ``"jetson"``。"""
    return "windows" if os.name == "nt" else "jetson"


def _project_file(value: str) -> Path:
    """配置里的模型路径：相对路径按项目根解析（不按当前工作目录）。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class StationGeometrySettings:
    """三圆环工位的**真实尺寸**与像素→毫米的换算。

    尺寸是"工件实际上多大"（毫米），换算回答"一个像素算几毫米"。两者都不是检测阈值，
    所以和 :class:`StationDetectionSettings` 分开：调检测阈值不该动到物理尺寸。
    """

    #: 相邻两个环的圆心间距（毫米）。用于校验"选中的三个环间距是否合理"。
    center_spacing_mm: float = DEFAULT_STATION_CENTER_SPACING_MM
    #: 工位环外径（毫米）。
    outer_diameter_mm: float = DEFAULT_STATION_OUTER_DIAMETER_MM
    #: 哪一端是环 1：``"left"``（x 小的一端）或 ``"right"``。工位翻面装时改这一个字段。
    ring1_side: str = "left"
    #: 各向同性比例尺（毫米/像素）。配了仿射矩阵时**以矩阵为准**，这一项被忽略。
    #: ``None`` 表示没标定过——那时只报像素偏差，毫米字段是 ``None``。
    mm_per_px: Optional[float] = DEFAULT_STATION_MM_PER_PX
    #: 像素误差 → 工位系毫米的 2×2 矩阵 ``[[a, b], [c, d]]``（行优先）。
    #: 工位被斜着看时两个方向的尺度不一样，只有它能如实换算。``None`` 表示没有。
    image_to_station_affine: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    @property
    def has_scale(self) -> bool:
        """有没有任何可用的尺度（决定毫米字段是数还是 ``None``）。"""
        return self.image_to_station_affine is not None or self.mm_per_px is not None


@dataclass(frozen=True)
class StationDetectionSettings:
    """三圆环的**检测阈值**：轮廓怎么提取、观测怎么并簇、三个环怎么选。"""

    #: Canny 双阈值。60/160 这一对是从实拍图调出来的：低到 60 才接得上被光照压暗的
    #: 那段环边，高到 160 才不让台面纹理整片变成边缘。
    canny_low: float = 60.0
    canny_high: float = 160.0
    #: 轮廓点数下限。低于它的轮廓多半是台面纹理，不是环的边。
    min_contour_points: int = 40
    #: ``fitEllipse`` 结果的短轴下限（像素）；太小的椭圆是噪点。
    min_minor_axis_px: float = 25.0
    #: 长轴上限（像素）；比它大的椭圆是转盘外圆那一类，不是工位环。
    max_major_axis_px: float = 320.0
    #: 长短轴之比下限。太扁的椭圆是斜视下的边，不是正对镜头的环。
    min_ellipse_aspect: float = 0.25
    #: 圆心聚类半径（像素）：两个观测的圆心离得比它近就当成同一个环。
    center_cluster_radius_px: float = 22.0
    #: 直径带的分档间隔（像素）：相邻观测的直径差超过它就算新的一带。
    diameter_band_gap_px: float = 5.0
    #: 一个候选环至少要几条观测支撑。
    min_cluster_support: int = 4
    #: 一个候选环至少要几个不同的直径带。**这一条才是真正管用的**：同一条边缘被 Canny
    #: 断成几截会产生一堆直径相同的碎片，条数够了但直径带只有 1 个；环的内外边缘
    #: 才会给出多个带。调低到 1 会让画面里随便一段弧都变成"候选环"。
    min_diameter_bands: int = 2
    #: 两端环的跨度至少占画面宽的多少。三个环挤在一角不是工位。
    min_station_span_ratio: float = 0.14
    #: 中间环到两端连线的垂距 / 跨度上限。共线度判据。
    max_collinearity_error_ratio: float = 0.07


@dataclass(frozen=True)
class StationReferenceSettings:
    """参考工位与验收门槛：**实测圆心相对谁**、**偏多少算合格**。"""

    #: 参考偏角（度）：工位摆正时环 1→环 3 的方向角。
    station_yaw_deg: float = 0.0
    #: 三个参考圆心（像素），顺序是 环1、环2（中间）、环3。
    ring_centers_px: Tuple[Tuple[float, float], ...] = (
        (380.0, 320.0),
        (633.0, 320.0),
        (886.0, 320.0),
    )
    #: 综合置信度下限。
    minimum_confidence: float = 0.9
    #: 判定"稳定"要连续看几帧。
    stable_frame_count: int = DEFAULT_STATION_STABLE_FRAME_COUNT
    #: 稳定窗口内允许的最大圆心抖动（像素）。
    maximum_center_jitter_px: float = DEFAULT_STATION_MAX_CENTER_JITTER_PX
    #: 三环共线度残差上限。
    maximum_affine_residual: float = 0.35
    #: 间距比 ``d12/d23`` 的允许区间；三个环等距时是 1。
    perspective_ratio_range: Tuple[float, float] = (0.85, 1.15)
    #: 验收门槛：三个环里最大的位移（毫米）。
    maximum_position_error_mm: float = DEFAULT_STATION_MAX_POSITION_ERROR_MM
    #: 验收门槛：偏角误差（度）。
    maximum_angle_error_deg: float = DEFAULT_STATION_MAX_ANGLE_ERROR_DEG


@dataclass(frozen=True)
class StackedRingSettings:
    """叠放位（中环 + 上下基准线）的检测参数与参考。

    和 :class:`StationDetectionSettings` 分开：叠放位找的是**一个环 + 两条线**，
    判据与三圆环完全不同，共用一个块只会让两边都放不下自己的字段。
    """

    #: 中环的像素半径窗口。
    middle_min_radius_px: float = DEFAULT_STACKED_MIDDLE_MIN_RADIUS_PX
    middle_max_radius_px: float = DEFAULT_STACKED_MIDDLE_MAX_RADIUS_PX
    #: 中环圆周上的边缘支撑率下限。
    middle_edge_support: float = DEFAULT_STACKED_MIDDLE_EDGE_SUPPORT
    #: 找中环的搜索半径（像素）：以参考圆心为中心开一个方窗。``None`` 表示全图搜。
    middle_search_radius_px: Optional[float] = 220.0
    #: 中环的参考圆心（像素）；``None`` 表示不做"离参考太远就丢"的过滤。
    middle_reference_px: Optional[Tuple[float, float]] = (633.5, 321.6)
    #: 中环候选离参考圆心的距离上限（像素）。
    middle_max_reference_distance_px: float = 180.0
    #: 基准线到中环圆心的距离（毫米）。
    line_offset_mm: float = DEFAULT_STACKED_LINE_OFFSET_MM
    #: 基准线的倾角上限（度）。
    max_line_angle_deg: float = DEFAULT_STACKED_MAX_LINE_ANGLE_DEG
    #: 两条基准线的夹角超过它就不配对（度）。
    max_line_disagreement_deg: float = DEFAULT_STACKED_MAX_LINE_DISAGREEMENT_DEG
    #: 一条基准线至少要几个边缘点、x 方向跨度至少多少像素。
    min_line_points: int = 60
    min_line_x_span_px: float = 220.0
    #: 综合置信度下限。
    minimum_confidence: float = 0.5
    #: 判定"稳定"要连续看几帧。
    stable_frame_count: int = DEFAULT_STATION_STABLE_FRAME_COUNT
    #: 稳定窗口内允许的最大圆心抖动（像素）。
    maximum_center_jitter_px: float = DEFAULT_STATION_MAX_CENTER_JITTER_PX


@dataclass(frozen=True)
class RingTargetSettings:
    """``config.json`` 的 ``ring_target`` 块：模型里哪几个类别是环、各自算几号。

    这笔配置以前分两块：``ring_digits`` 管"左中右各算几号 + 环里有没有墨迹"，
    ``ring_target`` 管"最内圈直径，用来当比例尺"。现在两件事都归模型了——类别名就是
    环号，框心就是环心；比例尺退回 ``depth/focal`` 或 ``station.mm_per_px``
    （那个比例尺只决定伺服每一步走多远，见 ``grasp.drive.eye_in_hand``）。

    :attr:`classes` 是**唯一**的环号来源：名字写错不会静默降级，而是在
    :class:`~modules.vision.drive.numbered_ring.RingTargetLocator` 构造时直接报错。

    :attr:`outer_diameter_mm` 是"这个靶标图案实际多大"——有了它，环心读数就能用
    ``毫米/像素 = 外径 / 模型框像素`` 换算，与镜头到纸的距离无关（相机装在臂上时，
    固定比例尺 ``station.mm_per_px`` 会差好几倍）。代价是精度受模型框松紧影响。
    """

    #: 类别名 → 环号（从 1 开始，且不能重复）。
    classes: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_RING_TARGET_CLASSES)
    )
    #: 靶标图案的**外径**（毫米），也就是模型框圈住的那一圈。配了它，环心读数就能用
    #: ``毫米/像素 = 外径 / 框像素`` 当比例尺（与镜头距离无关）；``None`` 表示不这么用。
    #: 注意它的精度取决于模型框的松紧（实测 ±15%），不是测量级的数。
    outer_diameter_mm: Optional[float] = DEFAULT_RING_OUTER_DIAMETER_MM

    # ------------------------------------------------------------ 环心精修
    # 模型框是回归出来的，框心只在框正好套住图案时才等于环心（实测有一帧偏 82 px）。
    # 框内一圈圈同心圆的圆心是同一个点，所以在框内做一次椭圆拟合、取圆心中位数，
    # 就能把精度拉回 1–2 px。下面这几项就是那一步的开关与验收。

    #: 要不要做框内精修；``False`` 表示直接用模型框心。
    refine_center: bool = True
    #: 至少找到几条椭圆观测才敢精修。
    refine_min_observations: int = 3
    #: 参与"同一圈"的圆心距离（像素）：以最大圆为种子，只收离它这么近的观测。
    refine_cluster_radius_px: float = 4.0
    #: 各观测圆心到中位数的距离中位数上限（像素）。同心圆本该重合，超了说明框里混了别的东西。
    refine_max_spread_px: float = 6.0
    #: 裁图比模型框大多少（相对框长边）：精修要把整圈环包进来。
    refine_margin_ratio: float = 0.08
    #: 只接受直径在 ``[min, max] × 框长边`` 的椭圆观测：滤掉数字笔画和隔壁环。
    refine_min_diameter_ratio: float = 0.30
    refine_max_diameter_ratio: float = 1.00
    #: 精修结果离框心太远就说明框套错了东西，不要它（相对框长边）。
    refine_max_offset_ratio: float = 0.25

    @property
    def names(self) -> Tuple[str, ...]:
        """模型里算"环"的类别名。"""
        return tuple(self.classes)

    def digit_for(self, name: str) -> Optional[int]:
        """类别名 → 环号；不是环的类别返回 ``None``。"""
        return self.classes.get(name)


@dataclass(frozen=True)
class ModelSettings:
    """用哪个模型文件、它有哪些类别、怎么解码——全部来自 ``config.json``。

    换模型的完整流程就是改这一个块，然后（Jetson 上）用
    :mod:`modules.vision.training.build_engine` 重新生成引擎。
    """

    #: ``"auto"`` / ``"tensorrt"`` / ``"onnx"``。
    backend: str = BACKEND_AUTO
    #: OpenCV ONNX 通道要加载的文件。
    onnx_path: Path = field(default_factory=lambda: _project_file(DEFAULT_ONNX_MODEL))
    #: TensorRT 通道要加载的文件；引擎与板子/GPU/TensorRT 版本绑定，只能在 Jetson 上生成。
    engine_path: Path = field(default_factory=lambda: _project_file(DEFAULT_ENGINE_MODEL))
    #: 类别名，顺序必须与训练时的类别索引一致（ONNX 的 ``names`` 元数据）。
    class_names: Tuple[str, ...] = tuple(DEFAULT_CLASS_NAMES)
    #: :attr:`class_names` 里算作物料的子集；没配时等于全部类别。
    materials: Tuple[str, ...] = tuple(DEFAULT_CLASS_NAMES)
    #: 模型输入边长（正方形）；``None`` 表示让代码去问模型自己。
    input_size: Optional[int] = None
    #: 解码阈值。
    conf_thresh: float = DEFAULT_CONF_THRESH
    iou_thresh: float = DEFAULT_IOU_THRESH
    #: 读取它的那份 config.json，只用于打印提示。
    source: Optional[Path] = None

    def resolve_backend(self, platform: Optional[str] = None) -> str:
        """本平台实际用哪个后端：显式配了就听配置的，``auto`` 才按平台猜。"""
        if self.backend != BACKEND_AUTO:
            return self.backend
        return BACKEND_ONNX if (platform or current_platform()) == "windows" else BACKEND_TENSORRT

    def path_for(self, backend: str) -> Path:
        """某个后端要加载的文件路径。"""
        return self.engine_path if backend == BACKEND_TENSORRT else self.onnx_path

    def model_path(self, platform: Optional[str] = None) -> Path:
        """本平台默认加载的模型文件路径。"""
        return self.path_for(self.resolve_backend(platform))


def load_model_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> ModelSettings:
    """读 ``config.json`` 的 ``model`` 块；整块缺失时返回内置默认值。"""
    config_path = Path(config_path)
    document = _read_json(config_path, "vision configuration") if config_path.exists() else {}
    root = _mapping(document, "root")
    return _model_from_root(root, config_path)


def _model_from_root(root: Mapping[str, object], config_path: Path) -> ModelSettings:
    """从**已经读到的**根对象里取 ``model`` 块。

    ``load_vision_config`` 走这条：它已经有 root 了，不该为了 model 再读一遍同一个文件
    （那会在"读配置"这件事上留下第二个入口，也让改成自定义路径的会话悄悄回读默认文件）。
    """
    value = root.get("model")
    if value is None:
        return ModelSettings(source=config_path)
    return _model_settings(_mapping(value, "model"), config_path)


def default_model_path(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """本平台默认加载的模型文件，取自 ``config.json`` 的 ``model`` 块。

    ``model.backend`` 是 ``auto``（默认）时：Windows 取 ``model.onnx``（OpenCV 推理），
    Jetson 取 ``model.engine``（TensorRT 推理）。想在某平台上强制另一条通道，
    把 ``backend`` 写成 ``"onnx"`` 或 ``"tensorrt"``。
    """
    return load_model_settings(config_path).model_path()


def require_model_path(settings: "ModelSettings | None" = None) -> Path:
    """同 :func:`default_model_path`，但文件不存在时直接说清楚该做什么。"""
    settings = settings if settings is not None else load_model_settings()
    path = settings.model_path()
    if path.is_file():
        return path
    if settings.resolve_backend() == BACKEND_TENSORRT:
        raise FileNotFoundError(
            f"未找到 TensorRT 引擎：{path}\n"
            "引擎与板子/GPU/TensorRT 版本绑定，只能在 Jetson 上生成：\n"
            "    python -m modules.vision.training.build_engine\n"
            "（换成 OpenCV 的 ONNX 通道：把 modules/vision/config.json 的 "
            'model.backend 改成 "onnx"）'
        )
    raise FileNotFoundError(
        f"未找到模型文件：{path}\n"
        "把导出的模型放到这个路径，或改 modules/vision/config.json 的 model.onnx"
    )


def resolve_target_settings(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Tuple[TargetSettings, Optional[str]]:
    """读 ``config.json`` 的 ``targets`` 块，**读不出来也不抛异常**。

    返回 ``(参数, 原因)``：成功时原因是 ``None``；文件坏了、字段写错时返回内置
    默认值 + 一句中文原因，由调用方决定怎么提示。

    为什么不做成"读失败就报错退出"：``targets`` 是**把识别调得更准**的那部分
    （物料颜色、工作距离先验），不是"没有就跑不动"的那部分。预览入口在用户面前
    开着，为了一个写错的字段直接退出，比按默认值跑一遍、把原因打在终端上更糟。
    真正需要这些参数的抓取流程会在自己的入口显式要求它们（``VisionConfig``
    的 ``require_*``）。
    """
    try:
        return load_vision_config(Path(config_path)).targets, None
    except ValueError as error:
        return TargetSettings(), str(error)


def _model_settings(document: Mapping[str, object], source: Path) -> ModelSettings:
    backend = document.get("backend", BACKEND_AUTO)
    if backend not in BACKENDS:
        raise ValueError(f"model.backend 必须是 {'/'.join(BACKENDS)} 之一，实际是 {backend!r}")
    class_names = _class_names(document.get("class_names"))
    materials = _materials(document.get("materials"), class_names)
    return ModelSettings(
        backend=str(backend),
        onnx_path=_model_file(document.get("onnx"), DEFAULT_ONNX_MODEL, "model.onnx"),
        engine_path=_model_file(document.get("engine"), DEFAULT_ENGINE_MODEL, "model.engine"),
        class_names=class_names,
        materials=materials,
        input_size=_optional_input_size(document.get("input_size")),
        conf_thresh=_probability(
            document.get("conf_thresh"), DEFAULT_CONF_THRESH, "model.conf_thresh"
        ),
        iou_thresh=_probability(document.get("iou_thresh"), DEFAULT_IOU_THRESH, "model.iou_thresh"),
        source=source,
    )


def _model_file(value: object, fallback: str, name: str) -> Path:
    if value is None:
        return _project_file(fallback)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串路径，或 null 表示用默认值 {fallback}")
    return _project_file(value.strip())


def _class_names(value: object) -> Tuple[str, ...]:
    """读类别表；``null`` 表示沿用内置表（见 ``yolov8.DEFAULT_CLASS_NAMES``）。"""
    if value is None:
        return tuple(DEFAULT_CLASS_NAMES)
    return _name_list(value, "model.class_names")


def _materials(value: object, class_names: Tuple[str, ...]) -> Tuple[str, ...]:
    """读"哪些类别算物料"；``null`` 表示全部类别都算。"""
    if value is None:
        return class_names
    names = _name_list(value, "model.materials")
    unknown = [name for name in names if name not in set(class_names)]
    if unknown:
        # 写错一个名字就会让那一类永远被判成"非物料"，静默漏检；这里直接拒绝启动。
        raise ValueError(
            f"model.materials 里的 {', '.join(unknown)} 不在 model.class_names "
            f"（{', '.join(class_names)}）里"
        )
    return names


def _name_list(value: object, name: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} 必须是非空的类别名列表，或 null")
    names = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}[{index}] 必须是非空字符串")
        names.append(item.strip())
    if len(set(names)) != len(names):
        raise ValueError(f"{name} 里有重复的类别名")
    return tuple(names)


def _optional_input_size(value: object) -> Optional[int]:
    """读输入边长；``null`` 表示让代码去问模型/引擎自己（问不出来才用兜底值）。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("model.input_size 必须是正整数（方形输入的边长），或 null")
    return int(value)


def _probability(value: object, fallback: float, name: str) -> float:
    if value is None:
        return float(fallback)
    number = _number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} 要在 0–1 之间")
    return number


@dataclass(frozen=True)
class TargetSettings:
    """抓取流程要区分的两类圆面目标。"""

    #: 物料顶部内圆的真实直径（毫米）。
    material_diameter_mm: float = DEFAULT_MATERIAL_DIAMETER_MM
    #: 夹爪上靶标的真实直径（毫米）；``None`` 表示还没装/没量。
    marker_diameter_mm: Optional[float] = None
    #: 物料内圆的参考颜色（BGR）；``None`` 表示不按颜色区分，全部圆面都当物料。
    material_bgr: Optional[Tuple[float, float, float]] = None
    #: 靶标的参考颜色（BGR）；``None`` 表示不按颜色挑靶标。
    marker_bgr: Optional[Tuple[float, float, float]] = None
    #: 颜色匹配容差。
    color_tolerance: float = DEFAULT_COLOR_TOLERANCE
    #: 相机到物料顶面的深度范围 ``(近, 远)`` 毫米；``None`` 表示不设先验。
    #: 给了它，物料定位的霍夫半径窗口就按针孔模型算，夹爪弧线一类的假圆会被排除。
    working_distance_mm: Optional[Tuple[float, float]] = None
    #: 物料调色板 ``{类别名: (B, G, R)}``。换一块料只在这里加一行，不用改代码；
    #: 给了它，物料定位就走颜色块（夹爪、台面、线材进不了掩膜）。空表示不按颜色区分。
    material_palette: Mapping[str, Tuple[float, float, float]] = field(default_factory=dict)

    # ---------------------------------------------------- 颜色块路径的放宽参数
    # 这几项原来写死在 ``material_blob`` / ``material_face`` 里。现场光照一变就得改代码，
    # 所以搬到这里；默认值与那些常量**逐字相同**，不配就与搬之前完全一致。
    #: 色相容差（OpenCV 0–179 标度）。参考色换灯以后 BGR 会整片平移，色相却基本不变，
    #: 所以掩膜是"BGR 距离够近 **或** 色相够近且不太暗"。调大能救回偏色的料，
    #: 但台面的色相离绿料只差 14——**别超过 12**，否则台面会整片进掩膜。
    hue_tolerance: float = 10.0
    #: 亮度下限：候选像素的 V 至少是参考色 V 的这个倍数。台面 V≈85、物料 V≈204–252，
    #: 所以这个比值必须**大于 0.42**，否则台面也会被当成料。
    value_ratio: float = 0.65
    #: 小于这么多像素的色块不当物料（噪点、反光、背景上一小块同色）。
    min_area_px: float = 500.0
    #: 边界起伏上限（轮廓点到形心距离的变异系数）；``None`` 表示不查这一项。
    max_boundary_variation: Optional[float] = 0.35
    #: 填充率下限（等效半径 / 外接圆半径）；``None`` 表示不查这一项。
    min_fill_ratio: Optional[float] = 0.65
    #: 形状判据"像不像物料顶面"：内部灰度标准差上限、圆周边界支撑比例下限。
    #: 放宽这两项会让夹爪圆弧、背景平地也更容易混进来。
    interior_std_max: float = 12.0
    edge_support_min: float = 0.55
    #: 严格颜色容差**一个色块都没找到**时的第二遍倍数（只放大 BGR 容差，
    #: 不动色相与亮度下限——那两个一放就会把台面收进来）。``1.0`` 表示不放松。
    colour_relax_factor: float = 1.5

    @property
    def can_split_by_color(self) -> bool:
        """两类目标都有参考颜色时才可能在同一帧里把它们分开。"""
        return self.material_bgr is not None and self.marker_bgr is not None


@dataclass(frozen=True)
class VisionConfig:
    """Which camera to open, what the two targets look like, and how it was calibrated."""

    camera_ids: Mapping[str, int] = field(default_factory=dict)
    targets: TargetSettings = field(default_factory=TargetSettings)
    #: 像素焦距；``None`` 表示沿用粗标定或调用方自备值。
    focal_px: Optional[float] = None
    #: 镜头到台面的深度（毫米）；换算离台面高度必需。
    table_depth_mm: Optional[float] = None
    #: 平面标定结果；``None`` 表示还没做实验 A。
    plane: Optional[PlaneCalibration] = None
    #: ``config.json`` 的 ``station`` 块：三圆环工位的真实尺寸与像素→毫米换算。
    #: 和 ``targets`` 一样是**手改**的：换工位尺寸改这里，不用改代码。
    station_geometry: StationGeometrySettings = field(default_factory=StationGeometrySettings)
    #: ``config.json`` 的 ``station`` 块：三圆环的检测阈值。
    station_detection: StationDetectionSettings = field(default_factory=StationDetectionSettings)
    #: ``config.json`` 的 ``station`` 块：参考工位与验收门槛。
    station_reference: StationReferenceSettings = field(default_factory=StationReferenceSettings)
    #: ``config.json`` 的 ``stacked_ring`` 块：叠放位（中环 + 上下基准线）的检测参数。
    stacked_ring: StackedRingSettings = field(default_factory=StackedRingSettings)
    #: ``config.json`` 的 ``ring_target`` 块：模型里哪几个类别是环、各自算几号。
    ring_target: RingTargetSettings = field(default_factory=RingTargetSettings)
    #: ``config.json`` 的 ``model`` 块：用哪个模型、哪些类别算物料、输入边长和阈值。
    #: 放在**字段尾部**，已有的位置参数顺序不变。
    model: ModelSettings = field(default_factory=ModelSettings)

    def camera_id(self, platform: str) -> int:
        """Return the camera index for one platform or explain what is missing."""
        camera_id = self.camera_ids.get(platform)
        if camera_id is None:
            raise ValueError(f"config.json 缺少 {platform}.camera_id")
        return camera_id

    def require_plane(self) -> PlaneCalibration:
        """Return the plane calibration or point at the experiment that produces it."""
        if self.plane is None:
            raise ValueError(
                "还没有平面标定结果；先跑实验 A 写入 calibration.json"
                "（见 modules/grasp/docs/抓取坐标链路.md）"
            )
        return self.plane

    def require_table_depth(self) -> float:
        """Return the lens-to-table depth or explain how to measure it."""
        if self.table_depth_mm is None:
            raise ValueError(
                "calibration.json 里还没有 table_depth_mm；"
                "把已知直径的圆片平放在台面上量一次即可（见 modules/grasp/docs/抓取坐标链路.md）"
            )
        return self.table_depth_mm

    def require_marker_diameter(self) -> float:
        """Return the gripper-marker diameter or explain what is missing."""
        if self.targets.marker_diameter_mm is None:
            raise ValueError(
                "config.json 的 targets.marker_diameter_mm 还没填；"
                "相机就是靠这个已知直径反推爪上靶标的三维位置的"
            )
        return self.targets.marker_diameter_mm


def load_vision_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
) -> VisionConfig:
    """Load both vision files; a missing calibration file means "uncalibrated"."""
    document = _read_json(config_path, "vision configuration")
    root = _mapping(document, "root")

    calibration_document = (
        _read_json(calibration_path, "vision calibration") if calibration_path.exists() else {}
    )
    calibration_root = _mapping(calibration_document, "calibration root")

    return VisionConfig(
        camera_ids=_camera_ids(root),
        targets=_targets(root.get("targets")),
        focal_px=_optional_positive_number(
            calibration_root.get("focal_px"), "calibration.focal_px"
        ),
        table_depth_mm=_optional_positive_number(
            calibration_root.get("table_depth_mm"), "calibration.table_depth_mm"
        ),
        plane=_plane_calibration(calibration_root.get("plane")),
        # 工位识别的三个块都住在 config.json 的 ``station`` 里：它们一起描述同一个工位，
        # 拆成三个顶层键只会让"这个数到底属于谁"更难查。
        station_geometry=_station_geometry(root.get("station")),
        station_detection=_station_detection(root.get("station")),
        station_reference=_station_reference(root.get("station")),
        stacked_ring=_stacked_ring(root.get("stacked_ring")),
        ring_target=_ring_target(root.get("ring_target")),
        # 同一份 root 里取 model：一个会话只读一次这个文件，绝不回读默认路径。
        model=_model_from_root(root, Path(config_path)),
    )


def save_vision_calibration(
    *,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    plane: Optional[PlaneCalibration] = None,
    focal_px: Optional[float] = None,
    table_depth_mm: Optional[float] = None,
) -> None:
    """Write ``calibration.json``, keeping any field the caller does not pass.

    只改传入的字段：标完焦距不会把已有的平面标定抹掉，反之亦然。
    """
    existing = (
        _mapping(_read_json(calibration_path, "vision calibration"), "calibration root")
        if calibration_path.exists()
        else {}
    )
    document = dict(existing)
    if focal_px is not None:
        document["focal_px"] = float(focal_px)
    if table_depth_mm is not None:
        document["table_depth_mm"] = float(table_depth_mm)
    if plane is not None:
        document["plane"] = {
            "center_x_mm": plane.center_x_mm,
            "center_y_mm": plane.center_y_mm,
            "rotation_degrees": plane.rotation_degrees,
            "rms_mm": plane.rms_mm,
            "samples": plane.samples,
        }
    calibration_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {name}: {path}") from error


def _camera_ids(root: Mapping[str, object]) -> Mapping[str, int]:
    found = {}
    for platform in PLATFORMS:
        profile = root.get(platform)
        if profile is None:
            continue
        camera_id = _mapping(profile, platform).get("camera_id")
        if camera_id is None:
            continue
        if type(camera_id) is not int or camera_id < 0:
            raise ValueError(f"{platform}.camera_id must be a non-negative integer")
        found[platform] = camera_id
    return found


def _targets(value: object) -> TargetSettings:
    if value is None:
        return TargetSettings()
    document = _mapping(value, "targets")
    material_diameter = document.get("material_diameter_mm")
    return TargetSettings(
        material_diameter_mm=(
            DEFAULT_MATERIAL_DIAMETER_MM
            if material_diameter is None
            else _positive_number(material_diameter, "targets.material_diameter_mm")
        ),
        marker_diameter_mm=_optional_positive_number(
            document.get("marker_diameter_mm"), "targets.marker_diameter_mm"
        ),
        material_bgr=_optional_bgr(document.get("material_bgr"), "targets.material_bgr"),
        marker_bgr=_optional_bgr(document.get("marker_bgr"), "targets.marker_bgr"),
        color_tolerance=_positive_number(
            document.get("color_tolerance", DEFAULT_COLOR_TOLERANCE),
            "targets.color_tolerance",
        ),
        working_distance_mm=_optional_depth_range(
            document.get("working_distance_mm"), "targets.working_distance_mm"
        ),
        material_palette=_material_palette(
            document.get("material_palette"), "targets.material_palette"
        ),
        hue_tolerance=_positive_number(
            document.get("hue_tolerance", TargetSettings().hue_tolerance),
            "targets.hue_tolerance",
        ),
        value_ratio=_positive_number(
            document.get("value_ratio", TargetSettings().value_ratio),
            "targets.value_ratio",
        ),
        min_area_px=_positive_number(
            document.get("min_area_px", TargetSettings().min_area_px),
            "targets.min_area_px",
        ),
        max_boundary_variation=_optional_positive_number(
            document.get("max_boundary_variation", TargetSettings().max_boundary_variation),
            "targets.max_boundary_variation",
        ),
        min_fill_ratio=_optional_positive_number(
            document.get("min_fill_ratio", TargetSettings().min_fill_ratio),
            "targets.min_fill_ratio",
        ),
        interior_std_max=_positive_number(
            document.get("interior_std_max", TargetSettings().interior_std_max),
            "targets.interior_std_max",
        ),
        edge_support_min=_positive_number(
            document.get("edge_support_min", TargetSettings().edge_support_min),
            "targets.edge_support_min",
        ),
        colour_relax_factor=_positive_number(
            document.get("colour_relax_factor", TargetSettings().colour_relax_factor),
            "targets.colour_relax_factor",
        ),
    )


def _station_geometry(value: object) -> StationGeometrySettings:
    """读 ``station`` 块里的**真实尺寸与换算**；整块缺失时全部走默认值。"""
    document = {} if value is None else _mapping(value, "station")
    defaults = StationGeometrySettings()
    return StationGeometrySettings(
        center_spacing_mm=_positive_number(
            document.get("center_spacing_mm", defaults.center_spacing_mm),
            "station.center_spacing_mm",
        ),
        outer_diameter_mm=_positive_number(
            document.get("outer_diameter_mm", defaults.outer_diameter_mm),
            "station.outer_diameter_mm",
        ),
        ring1_side=_ring1_side(document.get("ring1_side", defaults.ring1_side)),
        mm_per_px=_optional_positive_number(
            document.get("mm_per_px", defaults.mm_per_px), "station.mm_per_px"
        ),
        image_to_station_affine=_affine_2x2(
            document.get("image_to_station_affine", defaults.image_to_station_affine),
            "station.image_to_station_affine",
        ),
    )


def _station_detection(value: object) -> StationDetectionSettings:
    """读 ``station`` 块里的**检测阈值**。"""
    document = {} if value is None else _mapping(value, "station")
    defaults = StationDetectionSettings()
    return StationDetectionSettings(
        canny_low=_positive_number(
            document.get("canny_low", defaults.canny_low), "station.canny_low"
        ),
        canny_high=_positive_number(
            document.get("canny_high", defaults.canny_high), "station.canny_high"
        ),
        min_contour_points=_positive_int(
            document.get("min_contour_points", defaults.min_contour_points),
            "station.min_contour_points",
        ),
        min_minor_axis_px=_positive_number(
            document.get("min_minor_axis_px", defaults.min_minor_axis_px),
            "station.min_minor_axis_px",
        ),
        max_major_axis_px=_positive_number(
            document.get("max_major_axis_px", defaults.max_major_axis_px),
            "station.max_major_axis_px",
        ),
        min_ellipse_aspect=_probability(
            document.get("min_ellipse_aspect", defaults.min_ellipse_aspect),
            defaults.min_ellipse_aspect,
            "station.min_ellipse_aspect",
        ),
        center_cluster_radius_px=_positive_number(
            document.get("center_cluster_radius_px", defaults.center_cluster_radius_px),
            "station.center_cluster_radius_px",
        ),
        diameter_band_gap_px=_positive_number(
            document.get("diameter_band_gap_px", defaults.diameter_band_gap_px),
            "station.diameter_band_gap_px",
        ),
        min_cluster_support=_positive_int(
            document.get("min_cluster_support", defaults.min_cluster_support),
            "station.min_cluster_support",
        ),
        min_diameter_bands=_positive_int(
            document.get("min_diameter_bands", defaults.min_diameter_bands),
            "station.min_diameter_bands",
        ),
        min_station_span_ratio=_probability(
            document.get("min_station_span_ratio", defaults.min_station_span_ratio),
            defaults.min_station_span_ratio,
            "station.min_station_span_ratio",
        ),
        max_collinearity_error_ratio=_positive_number(
            document.get("max_collinearity_error_ratio", defaults.max_collinearity_error_ratio),
            "station.max_collinearity_error_ratio",
        ),
    )


def _station_reference(value: object) -> StationReferenceSettings:
    """读 ``station`` 块里的**参考工位与验收门槛**。"""
    document = {} if value is None else _mapping(value, "station")
    defaults = StationReferenceSettings()
    return StationReferenceSettings(
        station_yaw_deg=_number(
            document.get("station_yaw_deg", defaults.station_yaw_deg),
            "station.station_yaw_deg",
        ),
        ring_centers_px=_point_list(
            document.get("ring_centers_px", defaults.ring_centers_px),
            "station.ring_centers_px",
            expected=3,
        ),
        minimum_confidence=_probability(
            document.get("minimum_confidence", defaults.minimum_confidence),
            defaults.minimum_confidence,
            "station.minimum_confidence",
        ),
        stable_frame_count=_positive_int(
            document.get("stable_frame_count", defaults.stable_frame_count),
            "station.stable_frame_count",
        ),
        maximum_center_jitter_px=_positive_number(
            document.get("maximum_center_jitter_px", defaults.maximum_center_jitter_px),
            "station.maximum_center_jitter_px",
        ),
        maximum_affine_residual=_positive_number(
            document.get("maximum_affine_residual", defaults.maximum_affine_residual),
            "station.maximum_affine_residual",
        ),
        perspective_ratio_range=_ratio_range(
            document.get("perspective_ratio_range", defaults.perspective_ratio_range),
            "station.perspective_ratio_range",
        ),
        maximum_position_error_mm=_positive_number(
            document.get("maximum_position_error_mm", defaults.maximum_position_error_mm),
            "station.maximum_position_error_mm",
        ),
        maximum_angle_error_deg=_positive_number(
            document.get("maximum_angle_error_deg", defaults.maximum_angle_error_deg),
            "station.maximum_angle_error_deg",
        ),
    )


def _stacked_ring(value: object) -> StackedRingSettings:
    """读 ``stacked_ring`` 块；整块缺失时全部走默认值。"""
    if value is None:
        return StackedRingSettings()
    document = _mapping(value, "stacked_ring")
    defaults = StackedRingSettings()
    return StackedRingSettings(
        middle_min_radius_px=_positive_number(
            document.get("middle_min_radius_px", defaults.middle_min_radius_px),
            "stacked_ring.middle_min_radius_px",
        ),
        middle_max_radius_px=_positive_number(
            document.get("middle_max_radius_px", defaults.middle_max_radius_px),
            "stacked_ring.middle_max_radius_px",
        ),
        middle_edge_support=_probability(
            document.get("middle_edge_support", defaults.middle_edge_support),
            defaults.middle_edge_support,
            "stacked_ring.middle_edge_support",
        ),
        middle_search_radius_px=_optional_positive_number(
            document.get("middle_search_radius_px", defaults.middle_search_radius_px),
            "stacked_ring.middle_search_radius_px",
        ),
        middle_reference_px=_optional_point(
            document.get("middle_reference_px", defaults.middle_reference_px),
            "stacked_ring.middle_reference_px",
        ),
        middle_max_reference_distance_px=_positive_number(
            document.get(
                "middle_max_reference_distance_px", defaults.middle_max_reference_distance_px
            ),
            "stacked_ring.middle_max_reference_distance_px",
        ),
        line_offset_mm=_positive_number(
            document.get("line_offset_mm", defaults.line_offset_mm),
            "stacked_ring.line_offset_mm",
        ),
        max_line_angle_deg=_positive_number(
            document.get("max_line_angle_deg", defaults.max_line_angle_deg),
            "stacked_ring.max_line_angle_deg",
        ),
        max_line_disagreement_deg=_positive_number(
            document.get("max_line_disagreement_deg", defaults.max_line_disagreement_deg),
            "stacked_ring.max_line_disagreement_deg",
        ),
        min_line_points=_positive_int(
            document.get("min_line_points", defaults.min_line_points),
            "stacked_ring.min_line_points",
        ),
        min_line_x_span_px=_positive_number(
            document.get("min_line_x_span_px", defaults.min_line_x_span_px),
            "stacked_ring.min_line_x_span_px",
        ),
        minimum_confidence=_probability(
            document.get("minimum_confidence", defaults.minimum_confidence),
            defaults.minimum_confidence,
            "stacked_ring.minimum_confidence",
        ),
        stable_frame_count=_positive_int(
            document.get("stable_frame_count", defaults.stable_frame_count),
            "stacked_ring.stable_frame_count",
        ),
        maximum_center_jitter_px=_positive_number(
            document.get("maximum_center_jitter_px", defaults.maximum_center_jitter_px),
            "stacked_ring.maximum_center_jitter_px",
        ),
    )


def _ring_target(value: object) -> RingTargetSettings:
    """读 ``ring_target`` 块；整块缺失时用随包的三个靶标类别与实测外径。"""
    if value is None:
        return RingTargetSettings()
    document = _mapping(value, "ring_target")
    defaults = RingTargetSettings()
    return RingTargetSettings(
        classes=_ring_classes(
            document.get("classes", dict(defaults.classes)), "ring_target.classes"
        ),
        outer_diameter_mm=_optional_positive_number(
            document.get("outer_diameter_mm", defaults.outer_diameter_mm),
            "ring_target.outer_diameter_mm",
        ),
        refine_center=_boolean(
            document.get("refine_center", defaults.refine_center),
            "ring_target.refine_center",
        ),
        refine_min_observations=_positive_int(
            document.get("refine_min_observations", defaults.refine_min_observations),
            "ring_target.refine_min_observations",
        ),
        refine_cluster_radius_px=_positive_number(
            document.get("refine_cluster_radius_px", defaults.refine_cluster_radius_px),
            "ring_target.refine_cluster_radius_px",
        ),
        refine_max_spread_px=_positive_number(
            document.get("refine_max_spread_px", defaults.refine_max_spread_px),
            "ring_target.refine_max_spread_px",
        ),
        refine_margin_ratio=_fraction(
            document.get("refine_margin_ratio", defaults.refine_margin_ratio),
            "ring_target.refine_margin_ratio",
        ),
        refine_min_diameter_ratio=_fraction(
            document.get("refine_min_diameter_ratio", defaults.refine_min_diameter_ratio),
            "ring_target.refine_min_diameter_ratio",
        ),
        refine_max_diameter_ratio=_positive_number(
            document.get("refine_max_diameter_ratio", defaults.refine_max_diameter_ratio),
            "ring_target.refine_max_diameter_ratio",
        ),
        refine_max_offset_ratio=_fraction(
            document.get("refine_max_offset_ratio", defaults.refine_max_offset_ratio),
            "ring_target.refine_max_offset_ratio",
        ),
    )


def _boolean(value: object, name: str) -> bool:
    """读一个开关；只接受真正的 ``true`` / ``false``，不把 ``"false"`` 当假。"""
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是 true 或 false，实际是 {value!r}")
    return value


def _fraction(value: object, name: str) -> float:
    """读一个 ``(0, 1]`` 之间的比例。"""
    number = _positive_number(value, name)
    if number > 1.0:
        raise ValueError(f"{name} 必须不超过 1（它是相对框长边的比例），实际是 {number}")
    return number


def _ring_classes(value: object, name: str) -> Mapping[str, int]:
    """读"类别名 → 环号"：名字非空、环号从 1 开始且互不重复。

    两个环都写成 2 号是写得出来的错，后果却很难看：``digit_for`` 只能返回一个，
    另一个环永远取不到（现场表现是"少认一个环"，没有任何报错）。所以这里直接拒绝。
    """
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} 必须是非空对象，例如 {{'tg1': 1, 'tg2': 2, 'tg3': 3}}")
    classes: "dict[str, int]" = {}
    for raw_name, raw_digit in value.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{name} 的键必须是非空类别名")
        number = _positive_int(raw_digit, f"{name}.{raw_name}")
        if number in classes.values():
            raise ValueError(f"{name} 里的环号不能重复：{raw_name} 又写了 {number}")
        classes[raw_name.strip()] = number
    return dict(classes)


def _ring1_side(value: object) -> str:
    """读"哪一端是环 1"。写错就报错，不静默当成 ``"left"``。"""
    if value not in RING1_SIDES:
        raise ValueError(f"station.ring1_side 必须是 {'/'.join(RING1_SIDES)} 之一，实际是 {value!r}")
    return str(value)


def _affine_2x2(
    value: object, name: str
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """读 2×2 矩阵 ``[[a, b], [c, d]]``；``null`` 表示没有仿射标定。"""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须是 2×2 的数字数组（[[a, b], [c, d]]），或 null")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(
                f"{name}[{row_index}] 必须是两个数字（[a, b]），实际是 {row!r}"
            )
        rows.append(
            tuple(_number(item, f"{name}[{row_index}]") for item in row)  # type: ignore[arg-type]
        )
    return (rows[0], rows[1])  # type: ignore[return-value]


def _ratio_range(value: object, name: str) -> Tuple[float, float]:
    """读 ``[低, 高]`` 比值区间；低必须小于高，否则这个区间永远选不中东西。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须是两元素 [低, 高] 数组")
    low = _positive_number(value[0], f"{name}[0]")
    high = _positive_number(value[1], f"{name}[1]")
    if low >= high:
        raise ValueError(f"{name} 的低值必须小于高值，实际是 [{low}, {high}]")
    return (low, high)


def _positive_int(value: object, name: str) -> int:
    """读正整数；显式排除 ``bool``（``True`` 是 ``int`` 的实例）。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return int(value)


def _optional_point(value: object, name: str) -> Optional[Tuple[float, float]]:
    """读一个 ``(x, y)`` 像素点；``null`` 表示不设这一项。"""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须是两元素 [x, y] 坐标，或 null")
    return (_number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]"))


def _point_list(value: object, name: str, *, expected: int) -> Tuple[Tuple[float, float], ...]:
    """读 ``expected`` 个 ``(x, y)`` 点，按顺序。

    个数不对就报错：三个参考圆心少给一个，检测器会拿一个默认坐标去算偏差，
    那种错误在现场表现为"怎么调都不对"，最难查。
    """
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError(f"{name} 必须正好 {expected} 个 [x, y] 坐标")
    points = []
    for index, item in enumerate(value):
        points.append(_optional_point(item, f"{name}[{index}]"))
    if any(point is None for point in points):
        raise ValueError(f"{name} 里不能有 null")
    return tuple(points)  # type: ignore[return-value]


def _material_palette(value: object, name: str) -> Mapping[str, Tuple[float, float, float]]:
    """读 ``{"green": [165, 252, 114], ...}``；空对象表示不按颜色区分。"""
    if value is None:
        return {}
    document = _mapping(value, name)
    palette = {}
    for key, colour in document.items():
        palette[str(key)] = _optional_bgr(colour, f"{name}.{key}")  # type: ignore[assignment]
    return palette


def _plane_calibration(value: object) -> Optional[PlaneCalibration]:
    if value is None:
        return None
    document = _mapping(value, "calibration.plane")
    fields = ("center_x_mm", "center_y_mm", "rotation_degrees")
    if all(document.get(field) is None for field in fields):
        return None
    return PlaneCalibration(
        center_x_mm=_number(document.get("center_x_mm"), "calibration.plane.center_x_mm"),
        center_y_mm=_number(document.get("center_y_mm"), "calibration.plane.center_y_mm"),
        rotation_degrees=_number(
            document.get("rotation_degrees"), "calibration.plane.rotation_degrees"
        ),
        rms_mm=_number(document.get("rms_mm", 0.0), "calibration.plane.rms_mm"),
        samples=int(document.get("samples", 0)),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _positive_number(value: object, name: str) -> float:
    number = _number(value, name)
    if not number > 0:
        raise ValueError(f"{name} must be positive")
    return number


def _optional_positive_number(value: object, name: str) -> Optional[float]:
    if value is None:
        return None
    return _positive_number(value, name)


def _optional_bgr(value: object, name: str) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element [B, G, R] list or null")
    return tuple(_number(channel, f"{name}[{index}]") for index, channel in enumerate(value))  # type: ignore[return-value]


def _optional_depth_range(value: object, name: str) -> Optional[Tuple[float, float]]:
    """读 ``[近, 远]`` 毫米；校验规则与针孔模型那边共用一份。"""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element [near_mm, far_mm] list or null")
    try:
        return normalize_depth_range((_number(value[0], name), _number(value[1], name)))
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error

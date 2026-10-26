"""``Vision``：一个真正拥有相机与定位器（含模型）的视觉会话。

## 为什么需要它

在此之前，"打开相机 + 加载模型 + 取帧 + 把 ``MaterialReport`` 翻成观测"这套装配散在
``modules/grasp/drive/camera_source.py`` 和几个 CLI 里，抓取模块因此反过来依赖视觉装配。
现在这套职责回到视觉模块，依赖方向变成单向的 ``grasp → vision``。

## 用法

```python
from modules.vision import Vision

with Vision.open(config_path="modules/vision/config.json") as vision:
    report = vision.locate()
    if report.best is not None:
        print(report.best.x_mm, report.best.y_mm, report.best.z_mm)
```

## 两种读取语义（刻意不同，别混用）

- :meth:`Vision.locate` 是**面向业务**的：读不到画面就是失败，抛 ``RuntimeError``；
  画面里没有合格物料是**正常的空报告**（``report.best is None``），不是失败。
- :meth:`Vision.frame` 是**面向闭环**的窄接口：读不到帧返回 ``None``，沿用伺服侧
  "这一帧没看见"的既有契约（:class:`modules.grasp.drive.servo_observation.FrameSource`）。

两者共用一次 :meth:`Vision._read_report`，所以"一次调用只读一帧、只推理一次"只有一处实现。

## 画面停稳（抓取前）

:meth:`Vision.scene_stability` / :meth:`Vision.scene_stable` 连拍几帧、比**像素差**，
回答"整幅画面这一刻静止没有"。它不依赖检测：画面里没有物料也能回答，所以适合在
"爪子要往下走"之前做最后一道确认——物块在转盘上时，对准成功不等于它已经停稳。
只想等一小会儿直到静止用 :meth:`Vision.wait_scene_stable`；判据与阈值见
:mod:`modules.vision.drive.scene_stability`。

## 一次会话只装配一次

``open()`` 校验配置与相机编号、加载模型、打开相机；``locate()`` 只是读帧和推理。
模型在 :func:`_build_locator` 里真正加载（**不是** ``MaterialLocator.__enter__``——
那个只返回自身，不能证明模型就绪），失败时自行释放已经构造好的检测器。

## 焦距

``_open_config`` 不传 ``settings.focal_px``：主线沿用既有的**粗焦距策略**
（``modules.vision.drive.pinhole`` 的实拍样本值），只有 CLI 显式 ``--focal`` 才覆盖。
这是刻意保持的既有行为，不是遗漏——改成标定焦距会改变毫米坐标和控制结果。
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

from modules.vision.config import VisionConfig
# 这两个默认值只定义一次，就住在 sequence 里（那里是判据的出处）；本模块转发它们，
# 免得"停止判据"在两个地方各写一个数。
from modules.vision.drive.sequence import DEFAULT_MIN_SAMPLES, DEFAULT_STILL_MM
from modules.vision.drive.scene_stability import (
    DEFAULT_MAX_CHANGED_RATIO,
    DEFAULT_MAX_MEAN_DIFF,
    DEFAULT_PIXEL_DIFF_THRESHOLD,
    DEFAULT_STABLE_FRAMES,
)

#: 一次多帧融合连拍几帧。实测本机一帧定位约 178 ms，7 帧约 1.25 s——
#: 短到转盘转过的那点角度可以用直线近似，长到够剔掉一两个野值。
DEFAULT_MEASURE_FRAMES = 7

#: 认为"已经停稳"的最大批内偏差（毫米）。这是**批内一致性**，比伺服容差严：
#: 容差管的是"要不要再动一下"，这里管的是"这个读数本身稳不稳"。
DEFAULT_SETTLE_MM = 1.0

#: :meth:`Vision.measure_settled` 的重试上限（秒）。与抓取侧
#: ``ServoConfig.settle_timeout_seconds`` 的默认值一致：同一台机器上"等停稳"
#: 的耐心不该有两个数。
DEFAULT_SETTLE_TIMEOUT_SECONDS = 4.0

#: :meth:`Vision.wait_scene_stable` 默认最多等几秒。转盘停稳通常在一两秒内，
#: 给到 5 s 是为了挡住"画面一直在动"时不至于无限等下去；"从头到尾没见它动过"
#: 也由这个上限收尾。
DEFAULT_SCENE_STABLE_WAIT_SECONDS = 5.0

#: :meth:`Vision.wait_scene_stable` 两次判断之间的间隔（秒）。
#:
#: 判一次要连拍 4 帧（约 130 ms），所以 0.3 s 的间隔意味着大约每 0.3 s 看一次、
#: 每次看 130 ms——采样密度够抓到"刚停下"，又不会把 CPU 全占在取帧上。
DEFAULT_SCENE_STABLE_POLL_SECONDS = 0.3

#: :meth:`Vision.wait_scene_stable` 至少看这么久（秒）才允许放行。
#:
#: 一次判断只连拍 4 帧（约 130 ms），"这一眼没动"太便宜：转盘刚接到转动指令、物块正要
#: 被带走之前，画面也能看着是静的，照这一眼放行就会跑过去扑空。所以头几眼的静止一律
#: 不作数，看满这段窗口——足够把"物块要被带走"那次运动等出来，再等它停下来。
DEFAULT_SCENE_STABLE_MIN_SECONDS = 3.5

if TYPE_CHECKING:
    from modules.vision.drive.camera import USBCamera as Camera
    from modules.vision.drive.material_locate import MaterialLocator
    from modules.vision.drive.motion_gate import MotionGate
    from modules.vision.drive.numbered_ring import RingTarget, RingTargetLocator
    from modules.vision.drive.pixel_measure import MillimetreMeasurement, PixelMeasurement
    from modules.vision.drive.ring_station import RingStationDetector, RingStationDetection
    from modules.vision.drive.scene_stability import SceneStability
    from modules.vision.drive.sequence import Fusion, TimedPoint
    from modules.vision.drive.sighting import MaterialSighting
    from modules.vision.drive.stacked_ring import StackedRingDetector, StackedRingDetection
    from modules.vision.reports import MaterialReport, RegionReport


def _load_settings(
    config_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
) -> VisionConfig:
    """读一个会话要用的视觉配置快照。

    方法内导入配置模块（保持 ``import modules.vision`` 不预加载任何后端 / CLI）。
    ``None`` 解析为配置模块自己的 ``DEFAULT_*`` 常量——默认值锚定模块位置，
    不受当前工作目录影响。

    整条链上**只调用一次** ``load_vision_config``：模型设置从同一份 root 里取，
    不会为了 ``model`` 再读一遍默认文件（那会让自定义路径的会话悄悄回读默认配置）。
    """
    from modules.vision.config import (
        DEFAULT_CALIBRATION_PATH,
        DEFAULT_CONFIG_PATH,
        load_vision_config,
    )

    return load_vision_config(
        DEFAULT_CONFIG_PATH if config_path is None else Path(config_path),
        DEFAULT_CALIBRATION_PATH if calibration_path is None else Path(calibration_path),
    )


def _build_locator(settings: VisionConfig, *, focal_px: Optional[float] = None) -> "MaterialLocator":
    """按快照真实加载 detector 并构造定位器；失败时自行关闭已构造的 detector。

    这里的局部 import 是**明确的显式后端加载边界**：包初始化不能把 ONNX/TensorRT
    和定位算法一起拖进来。它不是为了绕开依赖方向——上层仍然只依赖 vision。
    """
    from modules.vision.detector_factory import create_configured_detector
    from modules.vision.drive.material_locate import MaterialLocator
    from modules.vision.drive.material_yolo import MaterialYOLO

    detector = create_configured_detector(settings=settings.model)
    try:
        targets = settings.targets
        yolo = MaterialYOLO(
            detector=detector,
            settings=settings.model,
            materials=settings.model.materials,
        )
        return MaterialLocator(
            yolo=yolo,
            focal_px=focal_px,
            depth_range_mm=targets.working_distance_mm,
            material_bgr=targets.material_bgr,
            material_palette=targets.material_palette or None,
            colour_tolerance=targets.color_tolerance,
            top_diameter_mm=targets.material_diameter_mm,
            # 颜色块路径与形状判据的放宽参数：现场光照一变就要调，从 targets 块读。
            hue_tolerance=targets.hue_tolerance,
            value_ratio=targets.value_ratio,
            min_area_px=targets.min_area_px,
            max_boundary_variation=targets.max_boundary_variation,
            min_fill_ratio=targets.min_fill_ratio,
            interior_std_max=targets.interior_std_max,
            edge_support_min=targets.edge_support_min,
            colour_relax_factor=targets.colour_relax_factor,
        )
    except BaseException as primary:
        # 构造未返回的对象只能由构造器自己收尾：不关掉 detector 就是泄漏一个已加载的模型。
        try:
            detector.close()
        except Exception as cleanup:
            raise primary.with_traceback(primary.__traceback__) from cleanup
        raise


def _new_camera(camera_id: int):
    """只构造 1280×720 相机，**不**调用 ``initialize``（打开设备是调用方的事）。"""
    from modules.vision.drive.camera import USBCamera

    # 60 fps：像素差分那条路（``scene_stability``）要的是同一个观察窗口里对比组更多，
    # 30 fps 下 4 帧就是 100 ms，判断本身要 130–150 ms，帧率是白等的那一段。
    # 再高没用：YOLO 主算路径一帧 27–29 ms，超过 60 fps 之后限速的是推理不是相机。
    return USBCamera(camera_id, width=1280, height=720, fps=60)


def _override_settings(
    settings: VisionConfig,
    *,
    model_path: Optional[Path] = None,
    camera_id: Optional[int] = None,
    conf_thresh: Optional[float] = None,
    depth_range_mm: Optional[Tuple[float, float]] = None,
    material_bgr: Optional[Tuple[float, float, float]] = None,
    material_palette=None,
    colour_tolerance: Optional[float] = None,
    top_diameter_mm: Optional[float] = None,
    hue_tolerance: Optional[float] = None,
    value_ratio: Optional[float] = None,
    min_area_px: Optional[float] = None,
    interior_std_max: Optional[float] = None,
    edge_support_min: Optional[float] = None,
    colour_relax_factor: Optional[float] = None,
) -> VisionConfig:
    """为**本次调用**生成一份覆盖后的快照；``None`` 表示不覆盖，不读写文件。

    只按需 ``replace`` 对应的字段，不做通用配置合并框架。数值 ``0`` 不能被 ``or``
    判成"没给"：每一处都用 ``is not None`` 判断（``0`` 会走到校验并被拒绝，
    那才是对的——"安静地当成没覆盖"会让人以为覆盖生效了）。

    校验复用 :mod:`modules.vision.config` 里已有的规则函数，不另写一套接受范围。
    """
    if camera_id is not None:
        from modules.vision.config import current_platform

        # 复用 _camera_ids 的整数/非负检查规则：写错序号要当场报错，而不是到开相机才炸。
        if type(camera_id) is not int or camera_id < 0:
            raise ValueError("camera_id must be a non-negative integer")
        camera_ids = dict(settings.camera_ids)
        camera_ids[current_platform()] = camera_id
        settings = replace(settings, camera_ids=camera_ids)

    if model_path is not None:
        from modules.vision.config import BACKEND_ONNX, BACKEND_TENSORRT

        path = Path(model_path)
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            model = replace(settings.model, onnx_path=path, backend=BACKEND_ONNX)
        elif suffix == ".engine":
            model = replace(settings.model, engine_path=path, backend=BACKEND_TENSORRT)
        else:
            raise ValueError(f"unsupported vision model format: {path.suffix}")
        # 显式 CLI 相对路径保留原 cwd 解释（Path 原样交给工厂），
        # 配置内相对路径仍按 PROJECT_ROOT 解析——那是 config._project_file 的事。
        settings = replace(settings, model=model)

    if conf_thresh is not None:
        from modules.vision.config import _probability
        from modules.vision.drive.yolov8 import DEFAULT_CONF_THRESH

        value = _probability(conf_thresh, DEFAULT_CONF_THRESH, "model.conf_thresh")
        settings = replace(settings, model=replace(settings.model, conf_thresh=value))

    targets = settings.targets
    if depth_range_mm is not None:
        from modules.vision.config import _optional_depth_range

        targets = replace(
            targets,
            working_distance_mm=_optional_depth_range(
                depth_range_mm, "targets.working_distance_mm"
            ),
        )
    if material_bgr is not None:
        from modules.vision.config import _optional_bgr

        targets = replace(
            targets, material_bgr=_optional_bgr(material_bgr, "targets.material_bgr")
        )
    if material_palette is not None:
        from modules.vision.config import _material_palette

        targets = replace(
            targets,
            material_palette=_material_palette(material_palette, "targets.material_palette"),
        )
    if colour_tolerance is not None:
        from modules.vision.config import _positive_number

        targets = replace(
            targets,
            color_tolerance=_positive_number(colour_tolerance, "targets.color_tolerance"),
        )
    if top_diameter_mm is not None:
        from modules.vision.config import _positive_number

        targets = replace(
            targets,
            material_diameter_mm=_positive_number(
                top_diameter_mm, "targets.material_diameter_mm"
            ),
        )
    # 颜色块路径与形状判据的放宽参数（现场调光照时最常用的几项）。
    for name, value in (
        ("hue_tolerance", hue_tolerance),
        ("value_ratio", value_ratio),
        ("min_area_px", min_area_px),
        ("interior_std_max", interior_std_max),
        ("edge_support_min", edge_support_min),
        ("colour_relax_factor", colour_relax_factor),
    ):
        if value is None:
            continue
        from modules.vision.config import _positive_number

        targets = replace(targets, **{name: _positive_number(value, f"targets.{name}")})
    if targets is not settings.targets:
        settings = replace(settings, targets=targets)
    return settings


def _sighting_from_region(
    region: "RegionReport",
    material_id: int,
    *,
    received_at: Optional[float] = None,
) -> "MaterialSighting":
    """一个 :class:`~modules.vision.reports.RegionReport` → 一条 ``MaterialSighting``。

    单位与字段名两边本来就一致（都是毫米、都是相机系右/下/前为正），所以这里只是搬运，
    **不做任何重新换算**——重复换算会引入第二套尺度。有两种换算必须保留方向：
    像素半径 → 直径（×2）、边缘支撑比例 → 百分比（×100）。
    """
    from modules.vision.drive.sighting import MaterialSighting

    face = region.face
    return MaterialSighting(
        material_id=material_id,
        x_mm=float(region.x_mm),
        y_mm=float(region.y_mm),
        z_mm=float(region.distance_mm),
        pixel_u=float(region.center_x_px),
        pixel_v=float(region.center_y_px),
        color=region.name,
        confidence=region.confidence,
        diameter_px=float(2 * region.region.radius),
        inner_sigma=float(face.interior_std),
        edge_support=float(face.edge_support) * 100.0,
        edge_directions=int(face.sampled_rays),
        received_at=received_at,
        raw=region.summary(),
    )


def _sightings_from_report(
    report: "MaterialReport",
    *,
    received_at: Optional[float] = None,
) -> "Tuple[MaterialSighting, ...]":
    """整帧报告 → 观测元组；编号与 ``MaterialReport.summary()`` 的「物料 N:」一致。

    枚举的是 ``report.faces``（只含通过校验的顶面）并 ``start=1``：被丢掉的候选
    （夹爪圆弧、背景纹理）**不进编号**。先枚举 ``regions`` 再过滤会留下编号空洞，
    于是 ``ObjectSelector(material_id=3)`` 会指到一个不存在的物料上。
    """
    return tuple(
        _sighting_from_region(region, index, received_at=received_at)
        for index, region in enumerate(report.faces, start=1)
    )


def _timed_point_from_report(
    report: "MaterialReport", stamp: float
) -> "Optional[TimedPoint]":
    """整帧报告 → 一个带时刻的观测点；这一帧没有合格物料时返回 ``None``。

    取 ``report.best`` 的**判据**而不是另一个近似规则：下游（抓取、逆解）要的就是
    这一个，多一套"取哪个圆"的规则只会让两处结果对不上。
    """
    from modules.vision.drive.sequence import TimedPoint

    best = report.best
    if best is None:
        return None
    return TimedPoint(
        stamp=stamp,
        x_mm=float(best.x_mm),
        y_mm=float(best.y_mm),
        z_mm=float(best.distance_mm),
        diameter_px=float(2 * best.region.radius),
        color=best.name,
    )


def _motion_position(fusion: "Fusion") -> "Tuple[float, float]":
    """一次融合结果 → 门控用的位置，**单位毫米**（取 x/y，忽略深度）。

    这里刻意只用 x/y：深度对"动没动"没有发言权，而且它是最不可靠的那个读数
    （README 明确写了"由框反推的深度偏近"）。

    **单位说明**：:class:`~modules.vision.drive.motion_gate.MotionGate` 的阈值默认按
    像素/秒给（它本来就是为画面位移设计的）。走会话这条路径时喂进去的是**毫米**，
    所以要么传一个按毫米标定的门控，要么用
    :data:`DEFAULT_SETTLED_MOTION_MM_S` 这两个默认值——它们就是按毫米/秒定的。
    刻意**不做**毫米↔像素的自动换算：那需要焦距，而焦距标定目前还是粗标定值，
    拿它去换单位等于把一个不确定的数引进"动没动"这个判断里。
    """
    point = fusion.point
    return (point.x_mm, point.y_mm)


#: 会话路径上"确实在动 / 已经停下"的速度阈值（毫米/秒），见 :func:`_motion_position`。
#: 数值取 30 / 25 mm/s：与门控模块的像素阈值同量级，在 0.5 mm/px 的画幅上大致相当。
DEFAULT_SETTLED_MOTION_MM_S = (30.0, 25.0)


def _settled_motion_gate() -> "MotionGate":
    """建一个按**毫米/秒**标定的门控，供 :meth:`Vision.measure_settled` 使用。"""
    from modules.vision.drive.motion_gate import MotionGate

    moving, near_stop = DEFAULT_SETTLED_MOTION_MM_S
    return MotionGate(moving_speed_px_s=moving, near_stop_speed_px_s=near_stop)


class Vision:
    """一次视觉会话：拥有相机与定位器（含模型），可重复定位。

    构造不打开设备（见 :meth:`open`）。关闭后任何操作立即失败，**不会自动重开**。
    """

    def __init__(self, settings: VisionConfig) -> None:
        self._settings = settings
        #: 相机与定位器在 ``open`` 之前确实还不存在，所以是可空字段；就绪后由
        #: :meth:`_require_resources` 缩窄成非空，动作里只用那个局部变量。
        self._camera: Optional[Camera] = None
        self._locator: Optional[MaterialLocator] = None
        #: 工位识别的两个检测器。它们在 ``open`` 之后**懒建**（第一次调用才构造），
        #: 因为纯经典 CV 的检测器不需要模型，也不该在"只做物料定位"的会话里被建出来。
        #: 建好就缓存：配置在 ``open`` 时已定死，检测器内部的跨帧稳定窗口要连续。
        self._station: Optional["RingStationDetector"] = None
        self._stacked: Optional["StackedRingDetector"] = None
        #: 视野里那**一个**环（窄视野，相机装臂上时的常用入口）。
        #: 它**不拥有**模型：检测器由物料定位器持有，这里只借同一个实例（见
        #: :meth:`_ring_target_locator`）。
        self._ring_target: Optional["RingTargetLocator"] = None
        self._closed = False
        self._clock: Callable[[], float] = time.monotonic
        #: 读帧前丢掉几帧缓冲。摄像头没有队列，这是既有的启发式新鲜度处理。
        self._drain_frames = 2
        #: 已产出的帧数。
        self.frames = 0
        #: 累计读失败次数；连续失败说明摄像头掉线。
        self.misses = 0
        #: 最近一次 :meth:`scene_stability` 的结论（诊断用）。
        #:
        #: 等停稳超时之后，"是阈值太紧还是等得太短"只能靠**判据当时看到的实测值**判断；
        #: 事后复测往往已经停了，两个数对不上就说明是等得不够长。只作观测记录，
        #: 不影响任何判定。
        self.last_scene_stability: Optional["SceneStability"] = None

    # ------------------------------------------------------------- 创建
    @classmethod
    def open(
        cls,
        *,
        config_path: Optional[Path] = None,
        calibration_path: Optional[Path] = None,
    ) -> "Vision":
        """读取指定配置，加载模型并打开相机；返回拥有这些资源的会话。"""
        settings = _load_settings(config_path, calibration_path)
        return cls._open_config(settings)

    @classmethod
    def _open_config(cls, settings: VisionConfig, *, focal_px: Optional[float] = None) -> "Vision":
        """按**已经读好的快照**开一个会话（CLI 覆盖走这条路，不再读第二个文件）。

        顺序是有意的：``camera_id`` 的检查在分配模型之前，模型的加载在打开相机之前。
        于是"配置写错"不会先占住一个模型，"模型缺失"也不会先占住摄像头。
        """
        from modules.vision.config import current_platform

        session = cls(settings)
        camera_id = settings.camera_id(current_platform())
        try:
            session._locator = _build_locator(settings, focal_px=focal_px)
            session._camera = _new_camera(camera_id)
            session._camera.initialize()
            return session
        except BaseException as primary:
            # 中断（KeyboardInterrupt 等）也要收尾，然后原样再抛。
            try:
                session.close()
            except Exception as cleanup:
                raise primary.with_traceback(primary.__traceback__) from cleanup
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Vision 已关闭")

    def _require_resources(self) -> "Tuple[Camera, MaterialLocator]":
        """校验会话开启并返回**确定存在**的相机与定位器。

        构造之后、``open`` 完成之前这两个字段确实是 ``None``，所以它们如实标成可空；
        就绪性只在这一处判定，动作方法用返回的局部变量，不再各自判空。
        """
        self._require_open()
        camera, locator = self._camera, self._locator
        if camera is None or locator is None:
            raise RuntimeError(
                "Vision 会话尚未就绪（请用 Vision.open(...) 创建，不要直接构造）"
            )
        return camera, locator

    @property
    def closed(self) -> bool:
        """会话是否已经关闭；关闭后 ``locate`` / ``frame`` 统一抛 ``RuntimeError``。"""
        return self._closed

    # ------------------------------------------------------------- 取帧与定位
    def _read_report(
        self, *, timeout: Optional[float] = None, want=None
    ) -> "Optional[Tuple[MaterialReport, float]]":
        """丢掉缓冲里的旧画面，读一帧并定位；读不到返回 ``None``。

        返回 ``(report, stamp)``：``stamp`` 是**读到帧之后、推理之前**的时刻。
        保留它是因为闭环要靠它判断"手上这帧是不是过期了"；这是启发式新鲜度处理，
        不宣称等于相机曝光时刻，也不能用推理结束时间冒充。

        ``want`` 原样交给定位器，决定 :attr:`MaterialReport.best` 挑哪一个物料
        （``frame`` 那条闭环路径不传它：那边由 ``ObjectSelector`` 选观测）。
        """
        camera, locator = self._require_resources()
        # 机械臂刚动完时，摄像头 / OpenCV 的缓冲里可能还压着动作**之前**的画面，
        # 直接读会拿到旧坐标，闭环就会以为没动过。先丢几帧再取。
        for _ in range(max(1, self._drain_frames)):
            camera.read()

        deadline = None if timeout is None else self._clock() + max(0.0, timeout)
        while True:
            frame = camera.read()
            if frame is not None:
                break
            self.misses += 1
            if deadline is None or self._clock() >= deadline:
                # 失败通过返回值传达，不在业务路径上打印或弹菜单。
                return None
            time.sleep(0.05)

        stamp = self._clock()
        report = locator.locate(frame, want=want)
        self.frames += 1
        return report, stamp

    def locate(self, want=None) -> "MaterialReport":
        """取帧并定位一次；``want`` 指定**要哪一个物料**。

        - ``None``（默认，保持原行为）：认出来的里置信度最高的那个，否则半径最大的
          已接受顶面；
        - ``"red"``：类别名，大小写与前后空格无关（``locate("Red")`` 等价）；
        - ``2``：编号，与报告 ``summary()`` 打印的「物料 N:」**同一个编号**（从 1 开始，
          只数通过校验的顶面）。

        指定的那一种不在画面里时 ``report.best is None``——**不回退**到别的物料：
        "我要红的、画面里只有蓝的"必须报"没有"，拿蓝的顶上等于让机械臂朝错的目标动，
        而动作本身看起来完全正常。画面里的其它候选仍然照常列在报告里，便于看清到底
        看到了什么。

        :raises RuntimeError: 读不到摄像头画面（设备掉线）。画面里没有合格物料
            是**正常的空报告**（``report.best is None``），不是失败。
        :raises ValueError: ``want`` 写法不合法（**在取帧之前**就报出来）。
        """
        result = self._read_report(want=want)
        if result is None:
            raise RuntimeError("读不到摄像头画面")
        return result[0]

    def frame(
        self,
        *,
        timeout: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
        min_received_at: Optional[float] = None,
    ) -> "Optional[Tuple[MaterialSighting, ...]]":
        """取一帧并定位，返回观测元组；读不到画面时返回 ``None``。

        这是**面向闭环的窄观测接口**（形状与 ``CameraLink`` 一致），供抓取和测试使用。
        ``max_age_seconds`` / ``min_received_at`` 保留原有窄接口形状：具体的新鲜度与
        停稳筛选在 :mod:`modules.grasp.drive.servo_observation`，这里只提供时间戳。
        """
        result = self._read_report(timeout=timeout)
        if result is None:
            return None
        report, stamp = result
        return _sightings_from_report(report, received_at=stamp)

    # ------------------------------------------------------------- 多帧融合
    def measure(
        self,
        *,
        frames: int = DEFAULT_MEASURE_FRAMES,
        timeout: Optional[float] = None,
        still_mm: float = DEFAULT_STILL_MM,
        settle_mm: Optional[float] = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> "Optional[Fusion]":
        """连拍若干帧，稳健融合成**一个**值；有效读数不够时返回 ``None``。

        这是给"物料在转盘上会动、会停"准备的：一帧定位既躲不开运动模糊，也说不清
        "现在到底稳没稳"。这里连拍 ``frames`` 帧，交给
        :func:`modules.vision.drive.sequence.fuse` 剔除与其余帧不一致的那些，
        再给出值、偏差、速度，以及**"还在动"这个结论本身**。

        三条要点：

        - **返回 ``None`` 表示"这批读数不够（少于 ``min_samples`` 个有效值）"**，
          不要把 ``None`` 当成"位置是 0"或者退回单帧值——那正是要挡掉的东西。
        - ``moving`` 为 ``True`` 时那个值仍然可用（它是按拟合速度外推到参考时刻的），
          但调用方应当知道"目标还在动"：要么等它停，要么接受这个外推。
        - ``settle_mm`` 给定时**会提前收工**：攒够 ``min_samples`` 帧、没在动、
          且最大偏差不超过它，就不用再拍满 ``frames`` 帧。省下的是几帧 ×
          约 178 ms 的定位时间。默认 ``None`` = 一定拍满，行为可预测。

        :raises RuntimeError: 会话未就绪或已关闭（与 :meth:`locate` 一致）。
        """
        from modules.vision.drive.sequence import fuse

        if frames < 1:
            raise ValueError("frames must be positive")
        samples = []
        for _ in range(frames):
            result = self._read_report(timeout=timeout)
            if result is None:
                # 读不到画面与"读到了但没有物料"在这里是同一件事：这一帧没有观测。
                samples.append(None)
                continue
            report, stamp = result
            samples.append(_timed_point_from_report(report, stamp))
            if settle_mm is not None:
                partial = fuse(
                    samples, still_mm=still_mm, min_samples=min_samples
                )
                if (
                    partial is not None
                    and not partial.moving
                    and partial.spread_mm <= settle_mm
                ):
                    return partial
        return fuse(samples, still_mm=still_mm, min_samples=min_samples)

    def measure_settled(
        self,
        *,
        timeout: float = DEFAULT_SETTLE_TIMEOUT_SECONDS,
        frames: int = DEFAULT_MEASURE_FRAMES,
        still_mm: float = DEFAULT_STILL_MM,
        settle_mm: float = DEFAULT_SETTLE_MM,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        require_motion: bool = False,
        motion: "Optional[MotionGate]" = None,
    ) -> "Optional[Fusion]":
        """反复连拍，直到拿到一次"已停稳"的融合（或者超时，返回最后一次）。

        转盘停止、机械臂停下的那一刻，前一批读数多半还在动（``moving``），
        所以"拍一批就下结论"是不够的。这里按批重试，直到 :attr:`Fusion.confident`
        成立；超时则返回**最后一批**——它带着 ``moving`` 与 ``spread_mm``，
        调用方自己决定要不要用，而不是被悄悄塞一个"看起来没事"的值。

        读不到画面（相机掉线）不算"还在动"：连续读失败会一直重试到超时，
        与 :func:`modules.grasp.drive.servo_observation.measure_settled` 的取舍一致。

        ``require_motion`` / ``motion`` 补的是**另一件事**：:attr:`Fusion.confident`
        只说明"这一批读数内部一致"，它答不了"转盘到底转过没有"。画面本来就静止时，
        读数连续几批都一样，看起来和"转完停稳了"完全一致——于是上一轮的残留状态会被
        当成这一轮的结果。给 ``require_motion=True`` 就要求**先看见它真的动过**
        （判据见 :mod:`modules.vision.drive.motion_gate`），没动过就一直等到超时。

        ``motion`` 可以传一个自己的门控进来跨调用累积状态（转盘一个完整周期跨多次
        ``measure_settled`` 时用得上）；不给就现建一个。**只有 ``require_motion``
        为真时才会用到它。**
        """
        gate = motion
        if require_motion and gate is None:
            gate = _settled_motion_gate()

        deadline = self._clock() + max(0.0, timeout)
        last = None
        while True:
            current = self.measure(
                frames=frames, still_mm=still_mm, settle_mm=settle_mm,
                min_samples=min_samples,
            )
            if current is not None:
                last = current
                if current.confident:
                    if not require_motion:
                        return current
                    # 门控喂的是**毫米**位置（见 _motion_position 的单位说明）。
                    verdict = gate.observe(
                        _motion_position(current),
                        current.point.stamp,
                        key=current.point.color,
                    )
                    if verdict.report_now:
                        return current
            if self._clock() >= deadline:
                return last

    # ------------------------------------------------------------- 画面级停稳
    def scene_stability(
        self,
        *,
        frames: int = DEFAULT_STABLE_FRAMES,
        pixel_threshold: int = DEFAULT_PIXEL_DIFF_THRESHOLD,
        max_mean_diff: float = DEFAULT_MAX_MEAN_DIFF,
        max_changed_ratio: float = DEFAULT_MAX_CHANGED_RATIO,
        timeout: Optional[float] = None,
    ) -> "SceneStability":
        """连拍几帧，量**整幅画面**的像素差，回答"现在画面静止没有"。

        与 :meth:`measure_settled` 的分工：那个看**检测到的物料位置**稳不稳（没有目标
        就答不了），这个只看像素（画面里有没有物料都能回答）。抓取前"下降的那几百毫秒
        里物块会不会跟着转盘走掉"就是后者要回答的问题::

            if robot.vision.scene_stable():
                robot.arm.move_axis_to_motor_degrees("lift", 530)
                robot.gripper.close_jaws()

        只比**相邻帧**：``frames`` 帧产生 ``frames - 1`` 组比较，每一组都要小于阈值。
        只比首尾两帧会被"转过去又转回来"骗过去。

        :param frames: 连拍几帧（至少 2）。
        :param pixel_threshold: 逐像素灰度差超过它才算"变了"（0–255）。
        :param max_mean_diff: 整幅图平均灰度差的允许上限。
        :param max_changed_ratio: 变化像素占比的允许上限（0–1）。
        :param timeout: 整段连拍的读取超时（秒）；``None`` = 不设总时限。
        :raises RuntimeError: 会话未就绪 / 已关闭，或读不到摄像头画面。
        :raises ValueError: 帧数或阈值不合法。
        """
        from modules.vision.drive.scene_stability import evaluate_stability

        burst = self._read_burst(frames, timeout=timeout)
        result = evaluate_stability(
            burst,
            pixel_threshold=pixel_threshold,
            max_mean_diff=max_mean_diff,
            max_changed_ratio=max_changed_ratio,
        )
        # 留一份给诊断：等停稳超时后要能报出"判据当时差了多少"，而不是只报一句还在动。
        self.last_scene_stability = result
        return result

    def scene_stable(
        self,
        *,
        frames: int = DEFAULT_STABLE_FRAMES,
        pixel_threshold: int = DEFAULT_PIXEL_DIFF_THRESHOLD,
        max_mean_diff: float = DEFAULT_MAX_MEAN_DIFF,
        max_changed_ratio: float = DEFAULT_MAX_CHANGED_RATIO,
        timeout: Optional[float] = None,
    ) -> bool:
        """布尔简写：**整幅画面静止**（可以下降）时返回 ``True``。

        参数与 :meth:`scene_stability` 完全相同；要实测值（差多少灰阶、多少像素变了）
        用那个，不要在这里改判据。
        """
        return self.scene_stability(
            frames=frames,
            pixel_threshold=pixel_threshold,
            max_mean_diff=max_mean_diff,
            max_changed_ratio=max_changed_ratio,
            timeout=timeout,
        ).stable

    def wait_scene_stable(
        self,
        *,
        timeout: float = DEFAULT_SCENE_STABLE_WAIT_SECONDS,
        interval: float = DEFAULT_SCENE_STABLE_POLL_SECONDS,
        min_seconds: float = DEFAULT_SCENE_STABLE_MIN_SECONDS,
        frames: int = DEFAULT_STABLE_FRAMES,
        pixel_threshold: int = DEFAULT_PIXEL_DIFF_THRESHOLD,
        max_mean_diff: float = DEFAULT_MAX_MEAN_DIFF,
        max_changed_ratio: float = DEFAULT_MAX_CHANGED_RATIO,
    ) -> bool:
        """等到画面**看满最小等待之后仍然静止**；返回是否等到了。

        进来那一眼"静止"一律不作数：一次判断只连拍 4 帧（约 130 ms），"这一眼没动"
        太便宜——转盘刚接到转动指令、物块正要被带走之前，画面也能看着是静的，
        照这一眼放行就会"跑过去扑空"（现场遇到的就是这个）。

        所以每隔 ``interval`` 秒看一眼，**两条同时满足**才放行：

        - 从进来算起已经看了至少 ``min_seconds`` 秒（默认 3.5 s，头几眼的静止不作数）；
        - 这一眼没有大变化（画面静止）。

        最小等待之内、或哪一眼还在动，就接着看；到 ``timeout`` 还没等到返回 ``False``——
        宁可让调用方不下降，也不要落下去扑空::

            if not robot.vision.wait_scene_stable(timeout=8.0):
                raise RuntimeError("还没停稳，先别下降")

        ``min_seconds`` 只管"最早什么时候允许放行"，不改变判据；它比 ``timeout`` 大时
        按 ``timeout`` 算（等待上限由调用方定，这个函数不擅自延长）。判一次要连拍几帧
        （默认 4 帧，约 130 ms），所以 ``interval`` 是**两次判断开始之间**的间隔
        （判一次比它久时，下一次立刻开始，不再叠加等待）。

        返回 ``False`` 时，:attr:`last_scene_stability` 留着最后一次判断的实测值：
        "明明已经停了却报还在动"要先分清是阈值太紧（只超一点）还是等得太短
        （事后复测已经很小）。

        :raises ValueError: ``timeout`` / ``interval`` / ``min_seconds`` 为负，
            或其他参数不合法。
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError(f"timeout 必须是非负的有限数，收到 {timeout!r}")
        if not math.isfinite(interval) or interval < 0:
            raise ValueError(f"interval 必须是非负的有限数，收到 {interval!r}")
        if not math.isfinite(min_seconds) or min_seconds < 0:
            raise ValueError(f"min_seconds 必须是非负的有限数，收到 {min_seconds!r}")

        started_at = self._clock()
        deadline = started_at + timeout
        # 头 min_seconds 秒内的"静止"不作数：那可能只是物块被带走前的一瞬间。
        # 上限是 timeout——等多久由调用方定，这里不擅自延长。
        hold_until = started_at + min(min_seconds, timeout)
        while True:
            started = self._clock()
            if (
                self.scene_stable(
                    frames=frames,
                    pixel_threshold=pixel_threshold,
                    max_mean_diff=max_mean_diff,
                    max_changed_ratio=max_changed_ratio,
                    timeout=max(0.0, deadline - started),
                )
                and self._clock() >= hold_until
            ):
                return True
            if self._clock() >= deadline:
                return False
            # 间隔按"上次开始判 → 下次开始判"算：判一次本身要连拍几帧，不再叠加。
            pause = max(0.0, interval - (self._clock() - started))
            # 别睡过 deadline：到点之后再判一眼就下结论，不白等。
            time.sleep(min(pause, max(0.0, deadline - self._clock())))

    # ------------------------------------------------------------- 两点像素测量
    def pixel_distance(self, start, end) -> "PixelMeasurement":
        """量画面上两点之间的 ΔX / ΔY（有符号）与直线距离，单位像素。

        **不读帧、不开设备**：只是算术，所以会话没开相机也能用（``open`` 之前就可用）。
        画面尺寸取自配置里本平台的相机画幅，用来判断端点是否落在画外。

        想连毫米一起要，见 :meth:`pixel_distance_mm`；点从鼠标来还是从别的检测器来，
        由调用方决定——本方法不猜。

        :raises ValueError: 端点写法不合法（不是两元素数字坐标）。
        """
        from modules.vision.config import current_platform
        from modules.vision.drive.pixel_measure import pixel_distance

        return pixel_distance(start, end, image_size=self._frame_size())

    def pixel_distance_mm(
        self,
        start,
        end,
        *,
        working_distance_mm: Optional[float] = None,
        focal_px: Optional[float] = None,
    ) -> "MillimetreMeasurement":
        """同 :meth:`pixel_distance`，再按针孔模型换算成毫米。

        ``working_distance_mm`` 是**被测平面**到镜头的距离；不给就用
        ``calibration.json`` 的 ``table_depth_mm``。两者都没有时**报错**而不是猜一个数——
        比例尺错一个倍数，量出来的毫米就是错的，而它看起来完全正常。

        ``focal_px`` 不给就用本会话的焦距（``calibration.json`` 的标定值，或粗标定兜底）。

        :raises ValueError: 焦距或工作距离没标定、或取值不合法。
        """
        from modules.vision.config import current_platform
        from modules.vision.drive.pixel_measure import to_millimetres

        measured = self.pixel_distance(start, end)
        distance = (
            self._settings.table_depth_mm
            if working_distance_mm is None
            else float(working_distance_mm)
        )
        if distance is None:
            raise ValueError(
                "没有工作距离，量不出毫米：传 working_distance_mm，"
                "或先用 python -m modules.vision.test.manual.calibrate_vision 写入 "
                "calibration.json 的 table_depth_mm"
            )
        return to_millimetres(
            measured,
            focal_px=self._effective_focal_px(focal_px),
            working_distance_mm=distance,
        )

    def _frame_size(self) -> "Tuple[int, int]":
        """本平台相机的画幅 ``(宽, 高)``。相机没开时也能答（配置里就写着）。"""
        camera = self._camera
        if camera is not None:
            width = getattr(camera, "width", None)
            height = getattr(camera, "height", None)
            if width and height:
                return (int(width), int(height))
        return (1280, 720)

    def _effective_focal_px(self, override: Optional[float] = None) -> float:
        """本会话实际用的像素焦距：显式覆盖 > 标定值 > 粗标定兜底。

        ``resolve_focal_px`` 要画幅才能把"1280 宽时的粗焦距"等比换算到本画幅，
        所以这里必须把 ``_frame_size()`` 传进去——漏传会直接 ``TypeError``
        （这个入口在没有显式焦距时就会被走到）。
        """
        from modules.vision.drive.pinhole import resolve_focal_px

        if override is not None:
            if override <= 0:
                raise ValueError("focal_px must be positive")
            return float(override)
        width, height = self._frame_size()
        return float(resolve_focal_px(width, height, self._settings.focal_px))

    # ------------------------------------------------------------- 工位识别
    def detect_ring_station(self, *, timeout: Optional[float] = None) -> "RingStationDetection":
        """读一帧，找三圆环工位，返回圆心、偏角与偏差。

        **与 :meth:`locate` 的语义差异是有意的**：``locate`` 读不到画面就抛
        ``RuntimeError``（那是业务失败）；这里**画面里没有工位是正常结果**
        （``detection.detected is False``，``detail`` 说明原因），而读不到画面仍然抛
        ``RuntimeError``——两种情况不该用同一个返回值表达。

        检测器按会话缓存（配置在 ``open`` 时就定死了），所以跨帧的稳定窗口是连续的。

        :raises RuntimeError: 会话未就绪或已关闭，或读不到摄像头画面。
        """
        frame = self._read_frame(timeout=timeout)
        return self._ring_station_detector().detect(frame)

    def detect_stacked_ring(self, *, timeout: Optional[float] = None) -> "StackedRingDetection":
        """读一帧，找叠放位（中环 + 上下基准线），返回圆心、偏角与偏差。

        语义与 :meth:`detect_ring_station` 一致：没识别到是正常结果，
        读不到画面才抛 ``RuntimeError``。

        :raises RuntimeError: 会话未就绪或已关闭，或读不到摄像头画面。
        """
        frame = self._read_frame(timeout=timeout)
        return self._stacked_ring_detector().detect(frame)

    def _read_frame(self, *, timeout: Optional[float] = None):
        """丢缓冲、读一帧 BGR；读不到抛 ``RuntimeError``（与 :meth:`locate` 一致）。

        与 :meth:`_read_report` 的区别是**只读帧、不推理**：工位识别用的是经典 CV，
        不该为了它把 YOLO 也叫起来。
        """
        camera, _locator = self._require_resources()
        for _ in range(max(1, self._drain_frames)):
            camera.read()
        deadline = None if timeout is None else self._clock() + max(0.0, timeout)
        while True:
            frame = camera.read()
            if frame is not None:
                self.frames += 1
                return frame
            self.misses += 1
            if deadline is None or self._clock() >= deadline:
                raise RuntimeError("读不到摄像头画面")

    def _read_burst(
        self, count: int, *, timeout: Optional[float] = None
    ) -> "Tuple[object, ...]":
        """先丢一次缓冲，再**连续**读 ``count`` 帧（中间不再丢帧）。

        :meth:`_read_frame` 每次只取一帧、每次都要丢缓冲，适合单帧定位；这里要的是
        时间上相邻的几帧，所以只在开头丢一次。默认 4 帧约 130 ms（30 fps），正好是
        "看一小段时间里画面动没动"的窗口。
        """
        camera, _locator = self._require_resources()
        if type(count) is not int or count < 2:
            raise ValueError(f"至少需要两帧才能比较（收到 {count!r}）")
        for _ in range(max(1, self._drain_frames)):
            camera.read()
        deadline = None if timeout is None else self._clock() + max(0.0, timeout)
        frames: list = []
        while len(frames) < count:
            frame = camera.read()
            if frame is not None:
                frames.append(frame)
                self.frames += 1
                continue
            self.misses += 1
            if deadline is not None and self._clock() >= deadline:
                raise RuntimeError("读不到摄像头画面")
            time.sleep(0.01)
        return tuple(frames)

    def _ring_station_detector(self) -> "RingStationDetector":
        """懒建并缓存三圆环检测器：配置在 ``open`` 时就定死了，中途不会回读文件。"""
        from modules.vision.drive.ring_station import RingStationDetector

        if self._station is None:
            self._station = RingStationDetector(
                geometry=self._settings.station_geometry,
                detection=self._settings.station_detection,
                reference=self._settings.station_reference,
            )
        return self._station

    def _stacked_ring_detector(self) -> "StackedRingDetector":
        """懒建并缓存叠放位检测器。

        中环参考圆心取 ``station.ring_centers_px`` 的**中间那个**：叠放位的中环和
        三圆环的中间环是同一个工位位置，配两份只会让它们漂移。
        """
        from modules.vision.drive.stacked_ring import StackedRingDetector

        if self._stacked is None:
            self._stacked = StackedRingDetector(
                stacked=self._settings.stacked_ring,
                geometry=self._settings.station_geometry,
                middle_reference_px=self._settings.station_reference.ring_centers_px[1],
            )
        return self._stacked

    # ------------------------------------------------------------- 视野里的那一个环
    def locate_ring(
        self, *, timeout: Optional[float] = None, digit: Optional[int] = None
    ) -> "Optional[RingTarget]":
        """读一帧，找出**视野里那一个环**的环心（像素 + 相机系毫米 + 环号）。

        为"相机视野装不下整排环"的场合准备：相机装在臂上、镜头离纸很近时，画面里
        通常只有一个环，而"只看到一个环"是**正常取景**，不是错误。

        环心与环号都来自模型（``models/best.onnx`` 的 ``tg1`` / ``tg2`` / ``tg3``
        三类），所以这条路径**看不到模型就什么也给不出**——它不再是一套独立的经典 CV。

        ``digit`` 给了就只找**这个环号**：画面里同时看得见两个环时，模型给分高的那个
        未必是你要的（校准 2 号环时按分数挑到 3 号，`require_digit` 只会报"是 3 号"，
        而 2 号就在画面里）。没给则照旧"整框可见优先、再比置信度"。

        与 :meth:`locate` 相同的取舍：

        - 读不到画面抛 ``RuntimeError``（设备问题）；
        - 画面里**没有环**是正常结果，返回 ``None``（不是异常）；
        - 框被画面裁掉、或没有比例尺时照样返回，但
          :meth:`RingTarget.reading` 返回 ``None``——不拿偏过的框心冒充环心。

        选哪个框：优先"整框都在画面里"的那个（只有它才可能给出真环心），再比置信度。

        :raises RuntimeError: 会话未就绪或已关闭，或读不到摄像头画面。
        """
        frame = self._read_frame(timeout=timeout)
        return self._ring_target_locator().detect(frame, digit=digit)

    def locate_rings(
        self, *, timeout: Optional[float] = None
    ) -> "Dict[int, RingTarget]":
        """读**一帧**，把画面里每个环号各取一个：``{环号: RingTarget}``。

        与 :meth:`locate_ring` 的区别是"全都要"：三环同框时用这三条读数定出那条直线
        （中间环实测、两端按已知间距外推），比逐环各拍一帧省两次取帧，也不会因为
        机械臂动了半步就换了一套坐标系。

        语义与 :meth:`locate_ring` 一致：画面里一个环都没有是正常结果（空字典），
        读不到画面才抛 ``RuntimeError``；贴边 / 没比例尺的环照样返回，只是
        ``RingTarget.reading()`` 为 ``None``。

        :raises RuntimeError: 会话未就绪或已关闭，或读不到摄像头画面。
        """
        frame = self._read_frame(timeout=timeout)
        return self._ring_target_locator().detect_all(frame)

    @property
    def ring_center_spacing_mm(self) -> float:
        """相邻两个地面圆环的圆心间距（毫米，``station.center_spacing_mm``）。

        三环成排时两端环心就是"中间环沿直线退/进这么远"，是**装好的物理真值**，
        不该在业务代码里再写一份常量。
        """
        return float(self._settings.station_geometry.center_spacing_mm)

    def ring_center(self, *, timeout: Optional[float] = None) -> "Optional[Tuple[float, float]]":
        """视野里那个环的**相机系中心** ``(x_mm, y_mm)``，可直接喂 ``align_xy(reading=...)``。

        画面里没有环、没有比例尺、或检测框被画面裁掉时返回 ``None``——与
        :meth:`locate_ring` 一致，不拿像素坐标冒充毫米，也不拿偏过的框心冒充环心。

        :raises RuntimeError: 会话未就绪或已关闭，或读不到摄像头画面。
        """
        target = self.locate_ring(timeout=timeout)
        return None if target is None else target.reading()

    def _ring_target_locator(self) -> "RingTargetLocator":
        """懒建并缓存单环定位器：配置在 ``open`` 时就定死了。

        检测器**借**物料定位器那一个（``locator.yolo.detector``）：同一个会话只该加载
        一次模型，靶标与物料跑的是同一份权重。单环定位器自己不持有、也不关闭它——
        关闭由会话统一收尾。
        """
        from modules.vision.drive.numbered_ring import RingTargetLocator

        if self._ring_target is None:
            _, locator = self._require_resources()
            self._ring_target = RingTargetLocator(
                locator.yolo.detector,
                settings=self._settings.ring_target,
                geometry=self._settings.station_geometry,
                # 精修沿用 station 那一套 Canny/椭圆阈值：同一台相机、同一种图案，
                # 另配一份只会在两处漂移。
                detection=self._settings.station_detection,
                # 针孔标定优先：环心读数必须与 RegionReport 的毫米同一套单位，
                # align_xy 的目标点（tx, ty）才对得上。
                focal_px=self._effective_focal_px(),
                table_depth_mm=self._settings.table_depth_mm,
            )
        return self._ring_target

    # ------------------------------------------------------------- 收尾
    def close(self) -> None:
        """释放所有已取得的资源；重复关闭不再执行资源动作。

        相机与定位器分别尝试：一个失败不阻断另一个。全部尝试之后才报告，
        错误清单包含每个失败阶段，以第一项为 ``__cause__``。
        """
        if self._closed:
            return
        self._closed = True
        camera, locator = self._camera, self._locator
        self._camera = self._locator = None
        errors = []
        if camera is not None:
            try:
                camera.release()
            except Exception as error:
                errors.append(("camera.release", error))
        if locator is not None:
            try:
                locator.close()
            except Exception as error:
                errors.append(("locator.close", error))
        if errors:
            details = "; ".join(
                "{}: {}".format(stage, error) for stage, error in errors
            )
            raise RuntimeError("Vision 清理失败：" + details) from errors[0][1]

    def __enter__(self) -> "Vision":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """收尾失败不吞掉：没有业务异常就直接抛清理错误，有则把清理错误挂成 cause。"""
        try:
            self.close()
        except Exception as cleanup:
            if exc is not None:
                raise exc.with_traceback(traceback) from cleanup
            raise
        return False

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

import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from modules.vision.config import VisionConfig

if TYPE_CHECKING:
    from modules.vision.drive.camera import USBCamera as Camera
    from modules.vision.drive.material_locate import MaterialLocator
    from modules.vision.drive.sighting import MaterialSighting
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

    return USBCamera(camera_id, width=1280, height=720)


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
        self._closed = False
        self._clock: Callable[[], float] = time.monotonic
        #: 读帧前丢掉几帧缓冲。摄像头没有队列，这是既有的启发式新鲜度处理。
        self._drain_frames = 2
        #: 已产出的帧数。
        self.frames = 0
        #: 累计读失败次数；连续失败说明摄像头掉线。
        self.misses = 0

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
        self, *, timeout: Optional[float] = None
    ) -> "Optional[Tuple[MaterialReport, float]]":
        """丢掉缓冲里的旧画面，读一帧并定位；读不到返回 ``None``。

        返回 ``(report, stamp)``：``stamp`` 是**读到帧之后、推理之前**的时刻。
        保留它是因为闭环要靠它判断"手上这帧是不是过期了"；这是启发式新鲜度处理，
        不宣称等于相机曝光时刻，也不能用推理结束时间冒充。
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
        report = locator.locate(frame)
        self.frames += 1
        return report, stamp

    def locate(self) -> "MaterialReport":
        """取帧并定位一次。

        :raises RuntimeError: 读不到摄像头画面（设备掉线）。画面里没有合格物料
            是**正常的空报告**（``report.best is None``），不是失败。
        """
        result = self._read_report()
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

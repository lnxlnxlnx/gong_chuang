"""USB 摄像头 + ``MaterialLocator`` → 与 ``CameraLink`` **同形**的观测源。

## 为什么需要这一层

``modules/grasp/drive/eye_in_hand.py`` 只要求观测源提供一件事：

```python
frame(timeout=..., min_received_at=...) -> tuple[MaterialSighting, ...] | None
```

远端镜头（TCP）能满足它，本机 USB 摄像头 + 现成的 :class:`~modules.vision.drive.material_locate.MaterialLocator`
也能——后者只是要有人把 ``MaterialReport`` 翻译成 :class:`MaterialSighting`。
这个模块就干这件事，于是伺服逻辑对"观测从哪来"完全无感。

## 编号必须和 ``MaterialReport.summary()`` 一致

``summary()`` 打印的是 ``物料 1:``、``物料 2:``……，编号只在**通过校验的顶面**
（``report.faces``）上从 1 开始数。这里的 ``material_id`` 照着它编，
所以你在终端看到的「物料 3」和 ``ObjectSelector(material_id=3)`` 指的是同一个东西。
被丢掉的候选（夹爪圆弧、背景纹理）**不进编号**，它们不是物料。

## 新鲜度

``min_received_at`` 的语义是"只要不早于这个时刻的观测"。摄像头这边没有队列，
所以它对应的是另一件事：**取帧前先丢掉几帧**。机械臂刚动完时，摄像头/OpenCV 的缓冲里
可能还压着动作**之前**的画面，直接读会拿到旧坐标，闭环就会以为没动过。
丢掉 ``drain_frames`` 帧再取，拿到的基本就是当前的画面。

这个办法是启发式的（不是时间戳那种硬保证），所以 ``drain_frames`` 留成参数，
真机上如果发现闭环"看不到自己刚走的位移"，把它调大。
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence, Tuple

from modules.vision.drive.material_locate import MaterialReport, RegionReport
from modules.vision.drive.sighting import MaterialSighting

#: 取帧前默认丢掉几帧。USB 摄像头的缓冲通常是 1–3 帧。
DEFAULT_DRAIN_FRAMES = 2


def sighting_from_region(
    region: RegionReport,
    material_id: int,
    *,
    received_at: Optional[float] = None,
) -> MaterialSighting:
    """一个 ``RegionReport`` → 一条 :class:`MaterialSighting`。

    单位与字段名两边本来就一致（都是毫米、都是相机系右/下/前为正），
    所以这里只是搬运，**不做任何重新换算**——重复换算会引入第二套尺度。
    """
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


def sightings_from_report(
    report: MaterialReport,
    *,
    received_at: Optional[float] = None,
) -> Tuple[MaterialSighting, ...]:
    """整帧报告 → 观测元组；编号与 ``MaterialReport.summary()`` 的「物料 N:」一致。"""
    return tuple(
        sighting_from_region(region, index, received_at=received_at)
        for index, region in enumerate(report.faces, start=1)
    )


class CameraFrameSource:
    """把一个已打开的摄像头和定位器包成观测源。

    ``camera`` 需要有 ``read() -> frame | None``；``locator`` 需要有
    ``locate(frame) -> MaterialReport``。两个都注入，所以离线测试可以塞假件。
    """

    def __init__(
        self,
        camera,
        locator,
        *,
        drain_frames: int = DEFAULT_DRAIN_FRAMES,
        clock: Callable[[], float] = time.monotonic,
        on_message: Optional[Callable[[str], None]] = None,
        owns_camera: bool = False,
        owns_locator: bool = False,
    ) -> None:
        if drain_frames < 0:
            raise ValueError("drain_frames 不能是负数")
        self.camera = camera
        self.locator = locator
        self.drain_frames = drain_frames
        self._clock = clock
        self._say = on_message if on_message is not None else (lambda _message: None)
        #: 由本对象负责关闭的资源（``from_vision_config`` 建的那些）。
        self._owns_camera = owns_camera
        self._owns_locator = owns_locator
        #: 累计读失败次数；连续失败说明摄像头掉线。
        self.misses = 0
        #: 已产出的帧数。
        self.frames = 0

    @classmethod
    def from_vision_config(
        cls,
        *,
        camera_id: Optional[int] = None,
        platform: Optional[str] = None,
        drain_frames: int = DEFAULT_DRAIN_FRAMES,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> "CameraFrameSource":
        """按 ``modules/vision/config.json`` 打开摄像头与定位器。

        参数来源与菜单 18（``modules/vision/test/manual/material_locate.py``）**完全一致**：
        摄像头序号取自平台的 ``camera_id``，物料直径/颜色/工作距离取自 ``targets``，
        模型路径取自 ``model`` 块。两处读同一份配置，所以在这里调好的参数在那边也成立。

        :raises FileNotFoundError: 模型文件不存在。宁可在开摄像头之前说清楚。
        """
        from pathlib import Path

        from modules.vision import (
            MaterialLocator,
            USBCamera,
            default_model_path,
            load_camera_id,
        )
        from modules.vision.config import current_platform, load_vision_config

        settings = load_vision_config()
        targets = settings.targets
        model_path = default_model_path()
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"找不到模型文件：{model_path}；路径写在 modules/vision/config.json 的 model 块。"
                "Jetson 上 engine 要用 build_model/build_engine 在板上生成"
            )

        if camera_id is None:
            # ``load_camera_id`` 要的是平台名（``windows`` / ``jetson``），
            # 与 ``modules/vision/config.json`` 里的两套 profile 对应。
            camera_id = load_camera_id(platform or current_platform())

        locator = MaterialLocator(
            model_path=model_path,
            depth_range_mm=targets.working_distance_mm,
            material_bgr=targets.material_bgr,
            material_palette=targets.material_palette or None,
            colour_tolerance=targets.color_tolerance,
            top_diameter_mm=targets.material_diameter_mm,
        )
        # 注意 ``USBCamera.__enter__`` **不**打开设备，必须先 ``initialize()``
        # （``__enter__`` 只是返回自身）。定位器那边则是标准的 ``with`` 协议。
        camera = USBCamera(camera_id, width=1280, height=720)
        try:
            locator.__enter__()
            camera.initialize()
        except Exception:
            camera.release()
            _exit_quietly(locator)
            raise
        return cls(
            camera,
            locator,
            drain_frames=drain_frames,
            on_message=on_message,
            owns_camera=True,
            owns_locator=True,
        )

    # ------------------------------------------------------------- 取帧
    def frame(
        self,
        *,
        timeout: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
        min_received_at: Optional[float] = None,
    ) -> Optional[Tuple[MaterialSighting, ...]]:
        """取一帧并定位，返回观测元组；读不到画面时返回 ``None``。

        ``timeout`` 只用于"一直读不到就放弃"的重试上限——摄像头没有队列可等，
        所以这里按零点几秒的间隔重试，而不是阻塞在一个空的队列上。

        **每次读都先丢几帧**，不管有没有传 ``min_received_at``：第一次读同样可能拿到
        驱动缓冲里的旧画面（实测第一次读数会偏一两毫米，随后才稳定），
        而伺服容差只有几毫米——那种偏差足以让闭环算错方向。
        """
        # 丢掉缓冲里的旧画面。``CAP_PROP_BUFFERSIZE=1`` 时最多一帧，
        # 但多丢一帧的代价（约 30 ms）远小于读到一个过时坐标的代价。
        for _ in range(max(1, self.drain_frames)):
            self.camera.read()

        deadline = None if timeout is None else self._clock() + max(0.0, timeout)
        while True:
            frame = self.camera.read()
            if frame is not None:
                break
            self.misses += 1
            if deadline is None or self._clock() >= deadline:
                self._say("读不到摄像头画面")
                return None
            time.sleep(0.05)

        stamp = self._clock()
        report = self.locator.locate(frame)
        self.frames += 1
        return sightings_from_report(report, received_at=stamp)

    def close(self) -> None:
        """关掉自己开的资源；注入进来的（测试用）不动。"""
        if self._owns_camera:
            _exit_quietly(self.camera)
            self._owns_camera = False
        if self._owns_locator:
            _exit_quietly(self.locator)
            self._owns_locator = False

    def __enter__(self) -> "CameraFrameSource":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def _exit_quietly(resource) -> None:
    """按 ``with`` 协议关一个资源，不让收尾失败盖住调用方的异常。"""
    exit_method = getattr(resource, "__exit__", None)
    if exit_method is None:
        return
    try:
        exit_method(None, None, None)
    except Exception:  # noqa: BLE001 - 收尾失败只该被忽略，不该冒泡
        pass

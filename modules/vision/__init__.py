"""视觉模块的公开入口。

只导出三个符号，全部有明确用途：

- :class:`~modules.vision.runtime.Vision` —— 拥有相机与模型的视觉会话；
- :class:`~modules.vision.reports.MaterialReport` —— 一帧的定位报告；
- :class:`~modules.vision.reports.RegionReport` —— 报告里的一个圆面。

用法::

    from modules.vision import Vision

    with Vision.open(config_path="modules/vision/config.json") as vision:
        report = vision.locate()          # 一帧
        if report.best is not None:
            print(report.best.x_mm, report.best.y_mm, report.best.z_mm)

        fused = vision.measure()          # 多帧稳健融合（物料会动时用这个）
        if fused is not None and fused.confident:
            print(fused.point.x_mm, fused.point.y_mm, fused.point.z_mm)

## 一帧还是多帧

``locate()`` / ``frame()`` 是**单帧**接口：快，但读数的好坏全押在这一帧上。物料放在
转盘上（会转、会停）时单帧不够用：运动模糊会让某一帧整个检不到，而"停稳了没有"
是单帧回答不了的问题。

``measure()`` 连拍若干帧、剔掉与其余帧不一致的那些，给出一个值**以及**
"目标还在不在动"。判据与取舍在 :mod:`modules.vision.drive.sequence`；只想等到停稳
就用 :meth:`~modules.vision.runtime.Vision.measure_settled`。

``scene_stable()`` 回答的是**另一个**问题：整幅画面此刻静止没有。它只看像素差，
不需要画面里有目标，用于抓取下降前的最后确认（物块在转盘上时尤其重要）。
等待它变静止用 ``wait_scene_stable()``；判据在 :mod:`modules.vision.drive.scene_stability`。

## 其余能力去哪里找

这里刻意**不再**维护"符号 → 模块"的惰性映射表。工具、算法和配置各有真实归属，
需要什么就从定义处导入：

- 相机 / 画面检查 → :mod:`modules.vision.drive.camera`
- 预览窗口 → :mod:`modules.vision.drive.preview`；两线程预览 → :mod:`modules.vision.drive.stream`
- 模型路径与设置 → :mod:`modules.vision.config`；检测器工厂 → :mod:`modules.vision.detector_factory`
- 帧级定位算法 → :mod:`modules.vision.drive.material_locate` 等 ``drive`` 子模块
- 工位识别（三圆环 / 叠放位） → :mod:`modules.vision.drive.ring_station`、:mod:`modules.vision.drive.stacked_ring`
- 环心与环号（机械臂对准用） → :mod:`modules.vision.drive.numbered_ring` 的 ``RingTargetLocator``：
  环心与"是几号环"都由模型（``tg1`` / ``tg2`` / ``tg3`` 三类）给出
- 圆心 / 半径拟合 → :mod:`modules.vision.drive.circle_fit`；多帧融合 → :mod:`modules.vision.drive.sequence`
- 单目测距 → :mod:`modules.vision.drive.distance`；默认类别表 → :mod:`modules.vision.drive.yolov8`

导入本包**不读配置、不开设备、不加载推理后端**：``Vision.open()`` 才做那些事。
``hough_lock`` / ``material_locate`` 这类 ``python -m`` 自测入口也不会因为在包初始化时
被导入过而触发 runpy 的 "found in sys.modules" 警告。
"""

from __future__ import annotations

from .reports import MaterialReport, RegionReport
from .runtime import Vision

__all__ = [
    "MaterialReport",
    "RegionReport",
    "Vision",
]

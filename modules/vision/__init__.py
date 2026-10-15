"""视觉模块的公开入口。

只导出三个符号，全部有明确用途：

- :class:`~modules.vision.runtime.Vision` —— 拥有相机与模型的视觉会话；
- :class:`~modules.vision.reports.MaterialReport` —— 一帧的定位报告；
- :class:`~modules.vision.reports.RegionReport` —— 报告里的一个圆面。

用法::

    from modules.vision import Vision

    with Vision.open(config_path="modules/vision/config.json") as vision:
        report = vision.locate()
        if report.best is not None:
            print(report.best.x_mm, report.best.y_mm, report.best.z_mm)

## 其余能力去哪里找

这里刻意**不再**维护"符号 → 模块"的惰性映射表。工具、算法和配置各有真实归属，
需要什么就从定义处导入：

- 相机 / 画面检查 → :mod:`modules.vision.drive.camera`
- 预览窗口 → :mod:`modules.vision.drive.preview`；两线程预览 → :mod:`modules.vision.drive.stream`
- 模型路径与设置 → :mod:`modules.vision.config`；检测器工厂 → :mod:`modules.vision.detector_factory`
- 帧级定位算法 → :mod:`modules.vision.drive.material_locate` 等 ``drive`` 子模块
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

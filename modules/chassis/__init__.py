"""麦轮运动学、底盘控制、地图路线与底盘演示。

公开入口：

```python
from modules.chassis import MecanumChassis

with MecanumChassis.open() as chassis:
    chassis.forward(rpm=30, duration=1)
```

:class:`~modules.chassis.runtime.MecanumChassis` 的真实定义在
:mod:`modules.chassis.runtime`（原来在 ``drive/chassis.py``，已退出，不留转发壳）。
``modules.chassis.config`` 只负责读并校验配置；**装配**（建驱动、开或借总线）在
:meth:`MecanumChassis.open`，原来的 ``create_drive`` / ``create_chassis`` 工厂已退出——
它们会重新读文件、开第二条连接。

导入本包**不读配置、不开串口**。
"""

from __future__ import annotations

from .runtime import MecanumChassis

__all__ = [
    "MecanumChassis",
]

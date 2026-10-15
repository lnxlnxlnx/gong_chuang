"""幻尔总线舵机驱动、夹爪与转盘控制。

公开入口：

```python
from modules.servo import GripperServo

with GripperServo.open() as gripper:
    report = gripper.close_jaws()
```

- :class:`~modules.servo.runtime.GripperServo` —— 夹爪上层（生命周期 + 开合语义）；
- :class:`~modules.servo.runtime.GripperReport` —— 一次开合的结果。

``GripperServo`` 的真实定义在 :mod:`modules.servo.runtime`（原来在 ``drive/gripper.py``，
已退出，不留转发壳）。协议层（:mod:`modules.servo.drive.bus_servo`）与转盘
（:mod:`modules.servo.drive.turntable`）保持原位。
导入本包**不读配置、不开串口**。
"""

from __future__ import annotations

from .runtime import GripperReport, GripperServo

__all__ = [
    "GripperReport",
    "GripperServo",
]

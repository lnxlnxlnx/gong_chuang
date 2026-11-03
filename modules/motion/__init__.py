"""动作票据与动作组：让多个模块的动作"各自发出、一起等"。

公开入口：

```python
from modules.motion import MotionTicket

batch = robot.motion.begin()
batch.add(robot.arm.move_axes({"lift": 800, "extension": -200}, wait=False))
batch.wait()
```

- :class:`~modules.motion.runtime.MotionTicket` —— 一次已发出、还没走完的动作；
- :class:`~modules.motion.runtime.MotionBatch` —— 一组票据，一个 ``wait()``；
- :class:`~modules.motion.runtime.MotionController` —— 会话级入口（``robot.motion``）。

真实定义在 :mod:`modules.motion.runtime`。导入本包**不读配置、不开串口、不建线程**，
也不导入任何设备模块——"谁动、动多少"仍旧由各模块自己决定。
"""

from __future__ import annotations

from .runtime import (
    CANCELLED,
    DONE,
    FAILED,
    PENDING,
    STATES,
    MotionBatch,
    MotionBatchError,
    MotionBatchItem,
    MotionBatchReport,
    MotionController,
    MotionTicket,
    MotionTimeoutError,
)

__all__ = [
    "CANCELLED",
    "DONE",
    "FAILED",
    "PENDING",
    "STATES",
    "MotionBatch",
    "MotionBatchError",
    "MotionBatchItem",
    "MotionBatchReport",
    "MotionController",
    "MotionTicket",
    "MotionTimeoutError",
]

"""机械臂模块：三轴相对脉冲原语、绝对电机角度目标与一次多轴同时启动。

公开入口：

```python
from modules.arm import AXES_SAME_START, AXES_SERIAL, RobotArm

with RobotArm.open() as arm:
    arm.move_up(pulses=800)
    arm.move_axes({"lift": 4000, "extension": -8000})          # 三轴同时起步
    arm.move_axes({"base": 5000, "extension": 200}, mode=AXES_SERIAL)
```

- :class:`~modules.arm.runtime.RobotArm` —— 生命周期、绝对电机角度目标与多轴动作；
- :class:`~modules.arm.runtime.MotionReport` —— 一次相对动作的命令量；
- :data:`~modules.arm.runtime.AXES_SAME_START` / :data:`~modules.arm.runtime.AXES_SERIAL`
  —— :meth:`~modules.arm.runtime.RobotArm.move_axes` 的两种模式。

``wait=False`` 的动作返回 :class:`modules.motion.MotionTicket`（票据），``wait=True`` 返回
``MotionReport``：报告是"结论"，票据是"结论还没到"，见 :mod:`modules.motion`。

真实定义在 :mod:`modules.arm.runtime`。
导入本包**不读配置、不开串口**：``RobotArm.open()`` 才做那些事。
"""

from __future__ import annotations

from .runtime import (
    AXES_MODES,
    AXES_SAME_START,
    AXES_SERIAL,
    WAIT_ESTIMATE,
    WAIT_IN_POSITION,
    MotionReport,
    RobotArm,
)

__all__ = [
    "AXES_MODES",
    "AXES_SAME_START",
    "AXES_SERIAL",
    "MotionReport",
    "RobotArm",
    "WAIT_ESTIMATE",
    "WAIT_IN_POSITION",
]

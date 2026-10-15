"""机械臂模块：三轴相对脉冲原语 + 绝对圆柱 / XYZ 双坐标定位。

公开入口只有两个符号：

```python
from modules.arm import RobotArm

with RobotArm.open() as arm:
    arm.move_up(pulses=800)
```

- :class:`~modules.arm.runtime.RobotArm` —— 生命周期、位姿与两个绝对坐标入口；
- :class:`~modules.arm.runtime.MotionReport` —— 一次相对动作的命令量。

真实定义在 :mod:`modules.arm.runtime`（原来在 ``drive/robot_arm.py``，已退出，不留转发壳）。
导入本包**不读配置、不开串口**：``RobotArm.open()`` 才做那些事。
"""

from __future__ import annotations

from .runtime import MotionReport, RobotArm

__all__ = [
    "MotionReport",
    "RobotArm",
]

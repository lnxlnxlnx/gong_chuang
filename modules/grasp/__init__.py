"""摄像头识别 + 机械臂抓取：把视觉、机械臂和夹爪接成一条抓取流程。

公开入口只有 ``Grasp`` 与它对外承诺的结果 / 异常类型：

```python
from modules.grasp import Grasp

with Grasp.open(
    config_path="modules/grasp/eye_in_hand.json",
    vision_config_path="modules/vision/config.json",
) as grasp:
    result = grasp.align_xy()
```

- :class:`~modules.grasp.runtime.Grasp` —— 拥有机械臂与视觉源的会话；
- :class:`~modules.grasp.drive.servo_model.ServoResult` / ``GraspReadyResult`` —— 动作结果；
- :class:`~modules.grasp.drive.servo_model.ServoError` —— 伺服无法继续。

## 为什么要显式从定义处导出

``Grasp`` 的真实定义在 :mod:`modules.grasp.runtime`。这里显式再导出它，**不是**兼容壳：
旧的门面函数没有留，消费者同一批迁到了类接口。工具装配（``load_*``）和全部内部助手
（``_read_object`` / ``_move_axes`` / ``probe_jacobian`` …）**不从这里导出**——
需要它们的现场工具直接从定义处导入。

导入本包**不读配置、不开设备**，也不会预加载 ``python -m modules.grasp.drive.eye_in_hand``
的目标模块（控制算法在动作方法内才导入）。
"""

from __future__ import annotations

from .drive.servo_model import GraspReadyResult, ServoError, ServoResult
from .runtime import Grasp

__all__ = [
    "Grasp",
    "GraspReadyResult",
    "ServoError",
    "ServoResult",
]

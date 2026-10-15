"""``Grasp``：一次抓取会话，真正拥有机械臂与一个视觉源。

## 为什么需要它

在此之前，"读五份配置 → 开视觉源 → 开机械臂 → 使能 → 跑动作 → 收尾"这套装配藏在
``modules/grasp/test/manual/align_xy.py`` 的 CLI 里，抓取主流程本身是个要调用方凑齐
``arm`` / ``link`` / ``selector`` / ``target`` / ``jacobian`` / ``config`` 六个参数的自由函数。
上层因此必须知道相机和机械臂怎么装、哪几项标定是必需的、异常时先停哪个。

现在这些都在对象里：

```python
from modules.grasp import Grasp

with Grasp.open(
    config_path="modules/grasp/eye_in_hand.json",
    vision_config_path="modules/vision/config.json",
) as grasp:
    result = grasp.align_and_descend()
```

## 资源归属

```
Grasp 实例
  ├─ 一个机械臂连接（串口）
  ├─ 一个观测源（本机路线是 Vision，TCP 路线是 CameraLink）
  └─ 本次的配置、目标、雅可比、选择条件与可选升降尺度
```

``open`` 返回即完成资源移交；返回前的失败由取得资源的那一方自己收尾。关闭时
``arm.stop_all`` / ``arm.disable_all`` / ``arm.close`` / ``source.close`` **每项独立尝试**，
一项失败不跳过其余；重复关闭不再发任何动作。关闭后的实例立即失败，不自动重连。

## 校验必须早于设备

普通动作需要 ``target`` 与 ``jacobian``。``RobotArm.__init__`` **已经打开了串口**，
所以"缺标定"必须在构造 ``RobotArm`` 之前就报出来——见 :meth:`Grasp._open_settings`
里的顺序。这也正是"缺标定所以不能跑标定"那个循环的解法：状态 / 示教 / 标定这些
工具各有自己的前置条件，它们**不走** :meth:`Grasp.open`（详见菜单 21 的实现）。

## 包导入不预加载控制算法

本模块顶层只依赖标准库与 ``servo_model``；实际的伺服算法在**动作方法内**才从
:mod:`modules.grasp.drive.eye_in_hand` 导入。这样 ``import modules.grasp`` 不会把
``python -m modules.grasp.drive.eye_in_hand`` 的目标模块提前塞进 ``sys.modules``
（那会让 runpy 报 "found in sys.modules"）。
"""

from __future__ import annotations

import time

from modules.grasp.drive.servo_model import GraspReadyResult, ServoError


def _new_arm(port, arm_config):
    """真实构造机械臂的唯一位置；测试在这里注入假臂。

    ``RobotArm.__init__`` 就会打开串口，所以调用它之前必须已经做完配置校验
    （见 :meth:`Grasp._open_settings`）。
    """
    from modules.arm.drive.robot_arm import RobotArm

    return RobotArm(port, config=arm_config)


class Grasp:
    """一次抓取会话：拥有机械臂与视觉源，负责就位流程的资源生命周期。"""

    def __init__(
        self,
        *,
        config,
        selector,
        target,
        jacobian,
        lift_scale,
        on_message=None,
    ) -> None:
        """只保存**已经读好的**配置与值对象：不读文件、不碰设备。"""
        self._config = config
        self._selector = selector
        self._target = target
        self._jacobian = jacobian
        self._lift_scale = lift_scale
        #: 进度回调；没给就安静。正常公开动作默认不打印任何东西。
        self._say = on_message if on_message is not None else (lambda message: None)
        self._arm = None
        self._source = None
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Grasp 已关闭")

    # ------------------------------------------------------------- 创建
    @classmethod
    def open(
        cls,
        *,
        config_path=None,
        vision_config_path=None,
        vision_calibration_path=None,
        arm_config_path=None,
        motor_config_path=None,
    ) -> "Grasp":
        """从现有配置创建本机视觉、机械臂和普通动作会话。

        五个路径**各自**解析默认值，不能混用：抓取参数来自 ``eye_in_hand.json``，
        视觉来自 vision 模块的 ``config.json`` / ``calibration.json``，
        机械臂来自 ``modules/arm/config.json``，串口来自 ``modules/motor/config.json``。
        """
        from pathlib import Path

        from modules.arm.config import DEFAULT_CONFIG_PATH as arm_default
        from modules.arm.config import load_arm_config
        from modules.motor.config import DEFAULT_CONFIG_PATH as motor_default
        from modules.motor.config import load_bus_config
        from modules.vision.runtime import Vision, _load_settings

        from .drive.servo_model import (
            DEFAULT_CONFIG_PATH,
            load_jacobian,
            load_lift_scale,
            load_object_selector,
            load_servo_config,
            load_target,
        )

        grasp_path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        arm_path = arm_default if arm_config_path is None else Path(arm_config_path)
        motor_path = motor_default if motor_config_path is None else Path(motor_config_path)

        # 五份配置一次读好，构造后只使用这些值——动作方法里不再读文件。
        config = load_servo_config(grasp_path)
        selector = load_object_selector(grasp_path)
        target = load_target(grasp_path)
        jacobian = load_jacobian(grasp_path)
        lift_scale = load_lift_scale(grasp_path)
        arm_config = load_arm_config(config_path=arm_path)
        # 注意 load_bus_config 的第一个位置参数是 platform，路径只能走关键字。
        port = load_bus_config(config_path=motor_path).port
        # 视觉快照在**打开任何设备之前**读好；真正开设备由 source_factory 负责。
        vision_settings = _load_settings(vision_config_path, vision_calibration_path)
        return cls._open_settings(
            config=config,
            selector=selector,
            target=target,
            jacobian=jacobian,
            lift_scale=lift_scale,
            arm_config=arm_config,
            port=port,
            source_factory=lambda: Vision._open_config(vision_settings),
        )

    @classmethod
    def _open_settings(
        cls,
        *,
        config,
        selector,
        target,
        jacobian,
        lift_scale,
        arm_config,
        port,
        source_factory,
        on_message=None,
    ) -> "Grasp":
        """**普通动作唯一的设备创建路径**（public ``open`` 与菜单 21 都走这里）。

        顺序是硬要求：普通动作的配置校验**不能**挪到 ``RobotArm`` 构造之后——那个构造
        已经打开了串口。校验通过之后才建会话、取观测源、开机械臂、使能。
        任何一步失败都触发 :meth:`close`，已取得的资源全部尝试释放。
        """
        if target is None:
            raise ServoError("缺少 target；先运行现有设目标或示教工具")
        if jacobian is None:
            raise ServoError("缺少 jacobian；先运行现有标定工具")
        # 复用原有的奇异判据（pulses_for 自查行列式），不新增阈值。
        jacobian.pulses_for(0.0, 0.0)
        session = cls(
            config=config,
            selector=selector,
            target=target,
            jacobian=jacobian,
            lift_scale=lift_scale,
            on_message=on_message,
        )
        try:
            # source_factory 返回即**转移所有权**；返回前失败由 factory 自己收尾。
            session._source = source_factory()
            session._arm = _new_arm(port, arm_config)
            session._arm.initialize()
            session._arm.enable_all()
            return session
        except BaseException as primary:
            try:
                session.close()
            except Exception as cleanup:
                raise primary.with_traceback(primary.__traceback__) from cleanup
            raise

    # ------------------------------------------------------------- 动作
    def align_xy(self):
        """包含原 ``run_align`` 的可选悬停，再按原规则检查安全高度并对准。

        只动平面（底座 + 伸缩），**不下降、不后缩**。
        """
        self._require_open()
        from modules.grasp.drive.eye_in_hand import align_x_y, hover_above_object

        if self._config.hover_z_mm is not None:
            hover_above_object(
                self._arm,
                self._source,
                self._selector,
                self._config,
                scale=self._lift_scale,
                on_message=self._say,
            )
        return align_x_y(
            self._arm,
            self._source,
            self._selector,
            self._target,
            self._jacobian,
            self._config,
            lift_scale=self._lift_scale,
            on_message=self._say,
        )

    def align_and_descend(self, *, z_pulses=None, retract_pulses=None) -> GraspReadyResult:
        """执行抓取前就位；**不闭合夹爪，也不承诺完成搬运**。

        顺序不是随便定的：

        1. **先到物块上方的悬停高度**（配了 ``hover_z_mm`` 才动），再确认转底盘前的
           安全高度：x/y 的每一步都要转底盘，低位转过去会撞到台面上的物块。整个流程
           只在这里查一次——后面三步都不改高度。悬停高度决定下降的**起点**，而下降是
           开环的固定脉冲数，所以它同时决定了落点。
        2. **x/y 对准**（闭环，只看画面）。
        3. **z 下降**：平面已对准才允许下降，按配置走固定脉冲。
        4. **伸缩后缩**：按配置后缩到夹持位。

        第 3、4 步是开环的（爪子压下去之后就看不到物块了），**第 2 步的结果就是它们
        唯一的安全依据**：平面误差超限时 ``descend_z`` 会拒绝下降、一个动作帧都不发。

        这一步**不**调用 :meth:`align_xy`：那会重复悬停与安全检查，改变原来只查一次的主链。
        悬停之后的第 2 步用 ``ensure_safe_height=False``，跳过的仅仅是"刚刚已经做过、
        而且这几步都不改高度"的那一次重复检查。

        :param z_pulses: 覆盖下降脉冲数（只这一次）；``None`` 用配置值。
        :param retract_pulses: 覆盖后缩脉冲数（只这一次）；``None`` 用配置值。
        :raises ServoError: 任何一步失败；异常信息里会写清卡在哪一步。
        """
        self._require_open()
        from modules.grasp.drive.eye_in_hand import (
            align_x_y,
            descend_z,
            hover_above_object,
            raise_before_rotating,
            retract_extension,
        )

        say = self._say
        started = time.monotonic()
        say("① 先到物块上方的悬停高度（配了 hover_z_mm 才动），再确认转底盘前的安全高度")
        hovered = hover_above_object(
            self._arm, self._source, self._selector, self._config,
            scale=self._lift_scale, on_message=say,
        )
        raised = raise_before_rotating(
            self._arm, self._source, self._selector, self._config,
            scale=self._lift_scale, on_message=say,
        )
        lift_seconds = time.monotonic() - started

        started = time.monotonic()
        say("② x/y 对准（闭环，只动底座与伸缩）")
        aligned = align_x_y(
            self._arm,
            self._source,
            self._selector,
            self._target,
            self._jacobian,
            self._config,
            ensure_safe_height=False,
            lift_scale=self._lift_scale,
            on_message=say,
        )
        align_seconds = time.monotonic() - started

        started = time.monotonic()
        say("③ z 下降（平面已对准，按固定脉冲）")
        descent = descend_z(
            self._arm, self._source, self._selector, self._config,
            pulses=z_pulses, scale=self._lift_scale, on_message=say,
        )
        descend_seconds = time.monotonic() - started

        started = time.monotonic()
        say("④ 伸缩后缩")
        retracted = retract_extension(
            self._arm, self._config, pulses=retract_pulses, on_message=say
        )
        retract_seconds = time.monotonic() - started

        return GraspReadyResult(
            raised_pulses=raised,
            aligned=aligned,
            descent=descent,
            retracted_pulses=retracted,
            lift_seconds=lift_seconds,
            align_seconds=align_seconds,
            descend_seconds=descend_seconds,
            retract_seconds=retract_seconds,
            hover_pulses=hovered.pulses,
        )

    # ------------------------------------------------------------- 收尾
    def close(self) -> None:
        """依次尝试停止、失能、关串口、关观测源；全部尝试后报告清理错误。

        **限制如实说明**：``RobotArm.stop_all`` / ``disable_all`` 内部按轴顺序执行，
        本方法保证的是"这两项以及后面的串口、视觉收尾分别被尝试"，
        不是"每个轴都能各自容错"——那要改的是机械臂驱动，不在本次接口重构范围里。
        """
        if self._closed:
            return
        self._closed = True
        arm, source = self._arm, self._source
        self._arm = self._source = None
        actions = []
        if arm is not None:
            actions.extend(
                (
                    ("arm.stop_all", arm.stop_all),
                    ("arm.disable_all", arm.disable_all),
                    ("arm.close", arm.close),
                )
            )
        if source is not None:
            actions.append(("source.close", source.close))
        errors = []
        for stage, action in actions:
            try:
                action()
            except Exception as error:
                errors.append((stage, error))
        if errors:
            details = "; ".join("{}: {}".format(stage, error) for stage, error in errors)
            raise RuntimeError("Grasp 清理失败：" + details) from errors[0][1]

    def __enter__(self) -> "Grasp":
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

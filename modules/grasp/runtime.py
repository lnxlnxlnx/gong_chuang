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
``arm.close`` 与 ``source.close`` **每项独立尝试**，一项失败不跳过其余；``arm.close``
内部自己按 停 → 失能 → 关串口 逐项试行（统一生命周期里上层不再重复 stop / disable），
重复关闭不再发任何动作。关闭后的实例立即失败，不自动重连。

## 校验必须早于设备

普通动作需要 ``target`` 与 ``jacobian``。``RobotArm._from_settings`` **已经打开了串口**，
所以"缺标定"必须在装配机械臂之前就报出来——见 :meth:`Grasp._open_settings`
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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from modules.grasp.drive.servo_model import (
    GraspReadyResult,
    HoverResult,
    LiftScale,
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoResult,
    ServoTarget,
)
from modules.grasp.drive.servo_observation import ClosableFrameSource

if TYPE_CHECKING:
    from modules.arm.config import ArmConfig
    from modules.arm.runtime import RobotArm
    from modules.vision.runtime import Vision


class GraspSettings(TypedDict):
    """已通过普通抓取动作前置校验的配置快照。

    ``target`` 与 ``jacobian`` 在这里是**非空**的：进这份快照之前必须已经检查过
    （见 :meth:`Grasp._require_action_ready`），否则"缺标定就在开设备前失败"这条不变式
    会从类型上消失。``lift_scale`` 保持可空——它本来就是可选的。
    """

    config: ServoConfig
    selector: ObjectSelector
    target: ServoTarget
    jacobian: ServoJacobian
    lift_scale: LiftScale | None


def _new_arm(port: str, arm_config) -> RobotArm:
    """真实构造机械臂的唯一位置；测试在这里注入假臂。

    ``RobotArm`` 的构造会打开串口，所以调用它之前必须已经做完配置校验
    （见 :meth:`Grasp._open_settings`）。
    """
    from modules.arm.runtime import RobotArm

    return RobotArm._from_settings(config=arm_config, port=port)


class Grasp:
    """一次抓取会话：拥有机械臂与视觉源，负责就位流程的资源生命周期。"""

    def __init__(
        self,
        *,
        config: ServoConfig,
        selector: ObjectSelector,
        target: ServoTarget,
        jacobian: ServoJacobian,
        lift_scale: LiftScale | None,
        on_message: Callable[[str], None] | None = None,
    ) -> None:
        """只保存**已经读好的**配置与值对象：不读文件、不碰设备。"""
        self._config = config
        self._selector = selector
        self._target = target
        self._jacobian = jacobian
        self._lift_scale = lift_scale
        #: 进度回调；没给就安静。正常公开动作默认不打印任何东西。
        self._say: Callable[[str], None] = (
            on_message if on_message is not None else (lambda message: None)
        )
        #: 设备在 ``open`` / ``_from_modules_settings`` 之前确实还不存在，所以如实可空；
        #: 动作边界用 :meth:`_require_resources` 缩窄。
        self._arm: RobotArm | None = None
        self._source: ClosableFrameSource | None = None
        self._closed = False
        #: 设备是**借来的**吗（``from_modules``）：借来的不收尾，只结束本会话。
        self._borrowed = False

    @property
    def closed(self) -> bool:
        """会话是否已经关闭；关闭后动作方法统一抛 ``RuntimeError``，不自动重开。"""
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Grasp 已关闭")

    def _require_resources(self) -> tuple[RobotArm, ClosableFrameSource]:
        """校验会话开启并返回**确定存在**的机械臂与观测源。"""
        self._require_open()
        arm, source = self._arm, self._source
        if arm is None or source is None:
            raise RuntimeError(
                "Grasp 会话尚未就绪（请用 Grasp.open(...) 或 Grasp.from_modules(...) 创建）"
            )
        return arm, source

    # ------------------------------------------------------------- 创建
    @classmethod
    def open(
        cls,
        *,
        config_path: Path | None = None,
        vision_config_path: Path | None = None,
        vision_calibration_path: Path | None = None,
        arm_config_path: Path | None = None,
        motor_config_path: Path | None = None,
    ) -> Grasp:
        """从现有配置创建本机视觉、机械臂和普通动作会话。

        五个路径**各自**解析默认值，不能混用：抓取参数来自 ``eye_in_hand.json``，
        视觉来自 vision 模块的 ``config.json`` / ``calibration.json``，
        机械臂来自 ``modules/arm/config.json``，串口来自 ``modules/motor/config.json``。
        """
        from modules.arm.config import DEFAULT_CONFIG_PATH as arm_default
        from modules.arm.config import load_arm_config
        from modules.grasp.drive.servo_model import (
            DEFAULT_CONFIG_PATH,
            load_jacobian,
            load_lift_scale,
            load_object_selector,
            load_servo_config,
            load_target,
        )
        from modules.motor.config import DEFAULT_CONFIG_PATH as motor_default
        from modules.motor.config import load_bus_config
        from modules.vision.runtime import Vision, _load_settings

        grasp_path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        arm_path = arm_default if arm_config_path is None else Path(arm_config_path)
        motor_path = motor_default if motor_config_path is None else Path(motor_config_path)

        # 五份配置一次读好，构造后只使用这些值——动作方法里不再读文件。
        config: ServoConfig = load_servo_config(grasp_path)
        selector: ObjectSelector = load_object_selector(grasp_path)
        target: ServoTarget | None = load_target(grasp_path)
        jacobian: ServoJacobian | None = load_jacobian(grasp_path)
        lift_scale: LiftScale | None = load_lift_scale(grasp_path)
        # 在**打开任何设备之前**取得非空 target / jacobian（含奇异判据）。
        target, jacobian = cls._require_action_ready(target=target, jacobian=jacobian)
        arm_config = load_arm_config(config_path=arm_path)
        # 注意 load_bus_config 的第一个位置参数是 platform，路径只能走关键字。
        port: str = load_bus_config(config_path=motor_path).port
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
        config: ServoConfig,
        selector: ObjectSelector,
        target: ServoTarget,
        jacobian: ServoJacobian,
        lift_scale: LiftScale | None,
        arm_config: ArmConfig,
        port: str,
        source_factory: Callable[[], ClosableFrameSource],
        on_message: Callable[[str], None] | None = None,
    ) -> Grasp:
        """**普通动作唯一的设备创建路径**（public ``open`` 与菜单 21 都走这里）。

        顺序是硬要求：普通动作的配置校验**不能**挪到装配机械臂之后——``_new_arm``
        已经打开了串口。校验通过之后才建会话、取观测源、装配机械臂；
        ``RobotArm._from_settings`` 返回的就是**已使能**的就绪对象，所以这里不再
        补一次 ``initialize`` / ``enable_all``（统一生命周期，见
        ``docs/模块调用与统一初始化方案.md`` §2.2）。
        任何一步失败都触发 :meth:`close`，已取得的资源全部尝试释放。
        """
        cls._require_action_ready(target=target, jacobian=jacobian)
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
            # ``_from_settings`` 返回时已经使能过：这里不再补 initialize / enable_all。
            session._arm = _new_arm(port, arm_config)
            return session
        except BaseException as primary:
            try:
                session.close()
            except Exception as cleanup:
                raise primary.with_traceback(primary.__traceback__) from cleanup
            raise

    @classmethod
    def from_modules(
        cls,
        *,
        arm: RobotArm,
        vision: Vision,
        config_path: Path | None = None,
        on_message: Callable[[str], None] | None = None,
    ) -> Grasp:
        """**借用**已经装配好的 arm / vision，读完 eye_in_hand 配置后建立会话。

        ``Robot`` 集成路径用这个：同一个会话里 arm 与 vision 只开一份，抓取复用它。

        - **不打开任何设备**、不重新使能、不读另一份视觉/机械臂配置；
        - 只读 ``eye_in_hand.json`` 里的 selector / target / jacobian / lift_scale，
          并做与传统 ``open`` 完全相同的那套校验（含雅可比奇异判据）；
        - 借来的设备**不归本会话收尾**：:meth:`close` 只结束本会话，
          不 ``stop`` / ``disable`` / ``close`` 它们；设备最终由装配者（``Robot``）关闭。

        这里 ``vision`` 明确是 :class:`~modules.vision.runtime.Vision`：借用分支**不**关闭它，
        所以它不需要 ``ClosableFrameSource`` 那层能力，类型也就更窄。

        :raises ServoError: 缺 target / jacobian，或雅可比奇异。
        """
        from modules.grasp.drive.servo_model import (
            DEFAULT_CONFIG_PATH,
            load_jacobian,
            load_lift_scale,
            load_object_selector,
            load_servo_config,
            load_target,
        )

        grasp_path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        config: ServoConfig = load_servo_config(grasp_path)
        selector: ObjectSelector = load_object_selector(grasp_path)
        target: ServoTarget | None = load_target(grasp_path)
        jacobian: ServoJacobian | None = load_jacobian(grasp_path)
        lift_scale: LiftScale | None = load_lift_scale(grasp_path)
        target, jacobian = cls._require_action_ready(target=target, jacobian=jacobian)
        return cls._from_modules_settings(
            config=config,
            selector=selector,
            target=target,
            jacobian=jacobian,
            lift_scale=lift_scale,
            arm=arm,
            vision=vision,
            on_message=on_message,
        )

    @classmethod
    def _from_modules_settings(
        cls,
        *,
        config: ServoConfig,
        selector: ObjectSelector,
        target: ServoTarget,
        jacobian: ServoJacobian,
        lift_scale: LiftScale | None,
        arm: RobotArm,
        vision: Vision,
        on_message: Callable[[str], None] | None = None,
    ) -> Grasp:
        """借用装配的**唯一接缝**：``Robot`` 在开设备前预校验配置后也走这里。

        与 :meth:`_open_settings` 共用同一份参数校验与同一套动作实现，只是不创建设备。
        签名里 ``target`` / ``jacobian`` 是**非空**的（快照类型已经保证），这里再查一次是
        **纵深防御**：万一有人绕过公开入口直接调这个接缝传入空值，也必须在装配前拦住，
        而不是等到动作时才炸。
        """
        cls._require_action_ready(target=target, jacobian=jacobian)
        session = cls(
            config=config,
            selector=selector,
            target=target,
            jacobian=jacobian,
            lift_scale=lift_scale,
            on_message=on_message,
        )
        session._arm = arm
        session._source = vision
        #: 借来的设备不归本会话收尾；``Robot`` 负责它们的最终关闭。
        session._borrowed = True
        return session

    @staticmethod
    def _require_action_ready(
        *, target: ServoTarget | None, jacobian: ServoJacobian | None
    ) -> tuple[ServoTarget, ServoJacobian]:
        """普通动作的前置校验：必须在**打开任何设备之前**完成。

        返回缩窄后的 ``(target, jacobian)``：调用方拿到的就是非空值，不必在装配完成后
        再判一次空——那正是"缺标定却先开了设备"的入口。
        """
        if target is None:
            raise ServoError("缺少 target；先运行现有设目标或示教工具")
        if jacobian is None:
            raise ServoError("缺少 jacobian；先运行现有标定工具")
        # 复用原有的奇异判据（pulses_for 自查行列式），不新增阈值。
        jacobian.pulses_for(0.0, 0.0)
        return target, jacobian

    # ------------------------------------------------------------- 动作
    def align_xy(
        self, *, hover: bool = False, ensure_safe_height: bool = False
    ) -> ServoResult:
        """只做 x/y 对准：**默认一步都不动高度**，高度由调用方负责。

        只动平面（底座 + 伸缩），**不下降、不后缩**。

        :param hover: 要不要先把高度伺服到 ``hover_z_mm``。那一步会**自己升降**——物块放得
            高时它会一路往上抬到行程端点（抬升方向没有上限），所以默认不做；要做就传
            ``hover=True``。配置里没写 ``hover_z_mm`` 时这一步本来就不发生。
        :param ensure_safe_height: 转底盘之前要不要先确认已经在安全高度。**默认不做**，
            因为默认值要跟"不动高度"一致。这意味着**低位转底盘会撞到台面上的物块**
            （现场发生过），所以调用方必须先自己点动到够高，或者显式传 ``True`` 把这次
            检查要回来。传 ``True`` 时底层拿到的是它自己的默认值（关键字一个都不传）。
        """
        arm, source = self._require_resources()
        from modules.grasp.drive.eye_in_hand import align_x_y, hover_above_object

        if hover and self._config.hover_z_mm is not None:
            hover_above_object(
                arm,
                source,
                self._selector,
                self._config,
                scale=self._lift_scale,
                on_message=self._say,
            )
        if not ensure_safe_height:
            return align_x_y(
                arm,
                source,
                self._selector,
                self._target,
                self._jacobian,
                self._config,
                ensure_safe_height=False,
                lift_scale=self._lift_scale,
                on_message=self._say,
            )
        return align_x_y(
            arm,
            source,
            self._selector,
            self._target,
            self._jacobian,
            self._config,
            lift_scale=self._lift_scale,
            on_message=self._say,
        )

    def align_and_descend(
        self,
        *,
        z_pulses=None,
        retract_pulses=None,
        hover: bool = True,
        ensure_safe_height: bool = True,
    ) -> GraspReadyResult:
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
        :param hover: 第 1 步里的悬停要不要做（默认做）。悬停会**自己升降**：它把"镜头到
            物块"的距离伺服到 ``hover_z_mm``，物块放得高时它会一路往上抬——没有上限，
            只有"走了脉冲 z 却没变"的事后判据，所以会顶到行程端点。想让高度完全由自己
            掌握就传 ``hover=False``。
        :param ensure_safe_height: 要不要在转底盘之前把高度抬到安全高度（默认要）。传
            ``False`` = 高度由调用方自己负责，本方法一次都不动升降（除了第 3 步那次下降）。
            **低位转底盘会撞到台面上的物块**，关掉它之前请先自己点动到够高。
        :raises ServoError: 任何一步失败；异常信息里会写清卡在哪一步。

        高度自己管的一条典型用法（第 1 步整体让开，下降脉冲按现场改）::

            robot.arm.move_up(pulses=600)          # 自己点动到够高
            result = robot.grasp.align_and_descend(
                hover=False, ensure_safe_height=False, z_pulses=600
            )
        """
        arm, source = self._require_resources()
        from modules.grasp.drive.eye_in_hand import (
            align_x_y,
            descend_z,
            hover_above_object,
            raise_before_rotating,
            retract_extension,
        )

        say = self._say
        started = time.monotonic()
        if hover or ensure_safe_height:
            say("① 先到物块上方的悬停高度（配了 hover_z_mm 才动），再确认转底盘前的安全高度")
        else:
            say("① 不动高度：转底盘前的高度由调用方自己保证（低位转动会撞到台面上的物块）")
        hovered = (
            hover_above_object(
                arm, source, self._selector, self._config,
                scale=self._lift_scale, on_message=say,
            )
            if hover
            else HoverResult(configured=False)
        )
        raised = (
            raise_before_rotating(
                arm, source, self._selector, self._config,
                scale=self._lift_scale, on_message=say,
            )
            if ensure_safe_height
            else 0
        )
        lift_seconds = time.monotonic() - started

        started = time.monotonic()
        say("② x/y 对准（闭环，只动底座与伸缩）")
        aligned = align_x_y(
            arm,
            source,
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
            arm, source, self._selector, self._config,
            pulses=z_pulses, scale=self._lift_scale, on_message=say,
        )
        descend_seconds = time.monotonic() - started

        started = time.monotonic()
        say("④ 伸缩后缩")
        retracted = retract_extension(
            arm, self._config, pulses=retract_pulses, on_message=say
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
        """依次尝试关闭机械臂与观测源；全部尝试后报告清理错误。

        ``arm.close`` 内部已经按 停 → 失能 → 关串口 逐项试行（统一生命周期里上层
        不再重复 stop / disable），这里保证的是"机械臂与观测源两项各自被尝试"，
        不是"每个轴都能各自容错"——那要改的是机械臂驱动，不在本次接口重构范围里。

        **借用模式**（:meth:`from_modules`）下设备不归本会话：只结束本会话、把引用放开，
        不 stop / disable / close 它们——同一个 Robot 会话里 arm 与 vision 还在被用。
        """
        if self._closed:
            return
        self._closed = True
        arm, source = self._arm, self._source
        self._arm = self._source = None
        if self._borrowed:
            return
        actions = []
        if arm is not None:
            actions.append(("arm.close", arm.close))
        if source is not None:
            actions.append(("source.close", source.close))
        errors = []
        for stage, action in actions:
            try:
                action()
            except Exception as error:
                errors.append((stage, error))
        if errors:
            details = "; ".join(f"{stage}: {error}" for stage, error in errors)
            raise RuntimeError("Grasp 清理失败：" + details) from errors[0][1]

    def __enter__(self) -> Grasp:
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

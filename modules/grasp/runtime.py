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

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
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
    from modules.grasp.drive.ring_plan import RingBasePlan
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


#: 环心读数**可以当物理毫米用**的比例尺来源。雅可比是按物理毫米标定的，其余来源
#: （``station.mm_per_px`` 是给固定相机标的值）会整体差一个倍数，而计划看起来完全正常。
PHYSICAL_RING_SCALES = ("depth/focal", "ring box")


def _new_arm(port: str, arm_config) -> RobotArm:
    """真实构造机械臂的唯一位置；测试在这里注入假臂。

    ``RobotArm`` 的构造会打开串口，所以调用它之前必须已经做完配置校验
    （见 :meth:`Grasp._open_settings`）。
    """
    from modules.arm.runtime import RobotArm

    return RobotArm._from_settings(config=arm_config, port=port)


def _round_pulses(value: float) -> int:
    """四舍五入到整数脉冲（``.5`` 远离零，与 ``ring_plan`` 的取整一致）。"""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


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
        from modules.motor.config import DEFAULT_HARDWARE_PATH as motor_default
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
        port: str = load_bus_config(hardware_path=motor_path).port
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
    @staticmethod
    def _selector_for(want, default: ObjectSelector) -> ObjectSelector:
        """本次动作按哪个物块判别：``want`` 给了就用它，没给就用配置里的那一份。

        为什么要按次传：``eye_in_hand.json`` 里的 ``object`` 是**默认值**，而"一次
        main.py 里依次抓红、绿、蓝"要求判别条件每抓一个就换一次——写死在文件里就得
        抓一个改一次文件。``want`` 与 ``vision.locate("red")`` 是**同一套说法**
        （类别名，或从 1 开始的编号），见 :meth:`ObjectSelector.from_want`。
        """
        if want is None:
            return default
        return ObjectSelector.from_want(want)

    def align_xy(
        self,
        *,
        hover: bool = False,
        ensure_safe_height: bool = False,
        single_step: bool = False,
        reading=None,
        want=None,
        tolerance_mm: float | None = None,
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
        :param single_step: **单帧一步**：最多取一帧、走一步就返回，动作之后不再测量。
        :param reading: 已经量好的平面读数 ``(x_mm, y_mm)``；给了它连那一帧都不取。
            "``locate`` 一次 + 走一步"的跟随循环就用它::

                while True:
                    report = robot.vision.locate("green")
                    if report.best is None:
                        continue
                    robot.grasp.align_xy(reading=(report.best.x_mm, report.best.y_mm))

            每轮只取**一帧**、只走**一步**；``reading`` 给了就等于 ``single_step``。
            结果里的 ``unverified`` 为真——读数说的是动作**之前**那一次。
        :param tolerance_mm: 这一次的到位门槛（毫米）；``None`` 用
            ``eye_in_hand.json`` 的 ``tolerance_mm``（你那份是 6.0）。**比配置更严**时
            多用几步，更松时可能一步就收工；``max_iterations`` 和单步脉冲上限都不变，
            所以收得太严会以"没进容差"报错，而不是一直磨。负数 / 0 直接报错。
        :param want: 这一次要抓**哪一个**物料：类别名（``"red"``）或从 1 开始的编号，
            与 ``vision.locate(want)`` 同一套说法。**不传就用 ``eye_in_hand.json`` 里
            ``object`` 那一份**（那份只是默认值）。一次程序里依次抓多种物料就靠它::

                for colour in ("red", "green", "blue"):
                    report = robot.vision.locate(colour)
                    if report.best is None:
                        continue
                    robot.grasp.align_xy(
                        reading=(report.best.x_mm, report.best.y_mm), want=colour
                    )

        物块放在转盘上、调用时还没转进视野时，**第一次测量**会按
        ``eye_in_hand.json`` 的 ``target_wait_seconds`` 等它出现（``null`` = 一直等，
        ``0`` = 保持旧语义），而不是 4 s 就报"没能稳定看到物块"；物块出现之后才按
        ``settle_timeout_seconds`` 等它停稳。之后的测量仍按 4 s 收工，爪子自己挡住物块时
        照旧能按最后读数收工。
        """
        arm, source = self._require_resources()
        from modules.grasp.drive.eye_in_hand import align_x_y, hover_above_object

        selector = self._selector_for(want, self._selector)
        if hover and self._config.hover_z_mm is not None:
            hover_above_object(
                arm,
                source,
                selector,
                self._config,
                scale=self._lift_scale,
                on_message=self._say,
            )
        # 两条分支只差"要不要自己查安全高度"，其余关键字一模一样：这样加参数时不会
        # 只改到其中一条（以前那种写法正是这么漂移的）。
        options = dict(
            single_step=single_step,
            reading=reading,
            lift_scale=self._lift_scale,
            on_message=self._say,
        )
        # 门槛只改这一次：``ServoConfig`` 是 frozen 的，``replace`` 顺带跑它自己的校验
        # （<=0 直接 ValueError），也顺带把由它派生的"停稳细档"（0.25×容差）一起改掉。
        config = (
            self._config
            if tolerance_mm is None
            else replace(self._config, tolerance_mm=float(tolerance_mm))
        )
        if not ensure_safe_height:
            return align_x_y(
                arm,
                source,
                selector,
                self._target,
                self._jacobian,
                config,
                ensure_safe_height=False,
                **options,
            )
        return align_x_y(
            arm,
            source,
            selector,
            self._target,
            self._jacobian,
            config,
            **options,
        )

    def plan_ring(
        self,
        *,
        require_digit: int | None = None,
        timeout: float | None = None,
        scale_mm_per_px: float | None = None,
    ) -> "RingBasePlan":
        """读**一帧**、定位环心，解算各电机的**绝对位置**；**不动任何设备**。

        这就是"先定位、再解算"的一条命令。返回值里各轴的绝对位置按轴名存放，
        可以直接存起来、**之后再执行**::

            plan = robot.grasp.plan_ring()          # 只算，一帧都不动
            ...                                     # 中间去抓物料、摆料
            robot.arm.move_axis_to_motor_degrees("base", plan.base)
            robot.arm.move_axis_to_motor_degrees("extension", plan.extension)
            # 或者一次走完：plan.apply(robot.arm)；只转底座：plan.apply(robot.arm, axes=("base",))

        ``plan.base`` / ``plan.extension`` 是**绝对角度**（电机轴口径，3200 脉冲 = 360°，
        不取模），``plan.degrees`` / ``plan.pulses`` 是同一位置的字典形式（键就是轴名），
        与 :meth:`modules.arm.runtime.RobotArm.move_axis_to_motor_degrees`、
        ``arm.turns_state`` 同一坐标。

        环心与环号来自模型（:meth:`modules.vision.runtime.Vision.locate_ring` 的
        ``tg1`` / ``tg2`` / ``tg3``），角度来自**实测雅可比 + 命令里程表**：

        - 底座与伸缩各是一个**绝对值**（与 ``move_axis_to_motor_degrees``、
          ``arm.turns_state`` 同一口径：3200 脉冲 = 360°，不取模），不是增量；
        - 只转底座时，爪子到了环的**角度**却不一定在同一条**半径**上，那点差就是计划里
          的伸缩补偿（``extension_pulses``）；
        - ``residual_mm`` 是"两轴都走之后"的预测残差。大角度下雅可比会变，**走完再看
          一帧重算一次**就能收敛（与 :meth:`align_xy` 的闭环是同一个道理）。

        ## 比例尺必须与雅可比一致

        读数用哪把尺，角度就按哪把尺结算：``jacobian`` 是"物理毫米/脉冲"，所以环心读数
        也必须是物理毫米。``depth/focal``（针孔标定）和 ``ring box``（已知外径 / 模型框）
        都满足；``station.mm_per_px`` 不满足（那是给固定相机标的值，装在臂上差好几倍）。
        这一档会**核对并报错**；确实知道换算关系时用 ``scale_mm_per_px`` 直接给。

        :param require_digit: 只接受指定环号（1/2/3）：**定位器按号筛框**（一帧里有好几个
            环也不会挑错），画面里确实没有这个号才报错，**不回退**到另一个环。
        :param timeout: 等一帧的超时（秒）。
        :param scale_mm_per_px: 与 ``jacobian`` 一致的毫米/像素；给了就按它换算读数。

        :raises ServoError: 画面里没有环、环心读数不可用（贴边 / 缺比例尺）、环号不符、
            比例尺对不上、雅可比奇异，或观测源不支持环心定位。
        """
        arm, source = self._require_resources()
        target = self._locate_ring_for_plan(
            source, timeout=timeout, require_digit=require_digit
        )
        return self._plan_for_ring(arm, target, scale_mm_per_px=scale_mm_per_px)

    def plan_ring_readings(
        self,
        readings: "dict[int, tuple[float, float]]",
        *,
        scale_source: str = "",
        offset_digit: int | None = None,
    ) -> "dict[int, RingBasePlan]":
        """一批**已经算好的**环心读数 → ``{环号: 各轴绝对位置}``；一帧都不拍。

        ``readings`` 是 ``{环号: (x_mm, y_mm)}``，与 :meth:`plan_ring` 的读数是同一套
        单位（相机系毫米）。三环成排时"2 号环实测 + 一条直线"就能把 1/3 号的环心算
        出来，再一次性解算三个落点——:func:`app.gripper.place_box_from_repo_to_floor_v3`
        就是这么用的；抓料/摆料之后拿 ``plan.apply(arm)`` 走到那个绝对位置。

        ``scale_source`` 是这批读数**来自那一帧**的比例尺来源
        （``RingTarget.scale_source``）。它与雅可比必须是同一把尺，这条不变式与
        :meth:`plan_ring` 完全一样：非物理毫米的来源在这里同样拒绝，而不是给一个
        像样的错角度。

        ``offset_digit`` 让这几份位置**共用同一个环号的落料补偿**（``ring_offsets``）。
        默认各用各的——那适合"每个环都是自己量出来的"。外推出来的点没有自己的补偿表，
        而且三份补偿不同会把"这三个落点该在一条直线上"直接搅乱（现场配的
        ``ring_offsets`` 各环差到 16 mm），所以一条直线上的三个点用同一个补偿号。

        :raises ServoError: 比例尺不是物理毫米，或雅可比奇异。
        """
        self._require_physical_ring_scale(scale_source, None)
        arm, _source = self._require_resources()
        offset = None if offset_digit is None else int(offset_digit)
        return {
            int(digit): self._plan_from_reading(
                arm, reading, digit=int(digit), offset_digit=offset
            )
            for digit, reading in readings.items()
        }

    def align_ring(
        self,
        *,
        require_digit: int | None = None,
        timeout: float | None = None,
        scale_mm_per_px: float | None = None,
        axes: "Sequence[str]" = ("base", "extension"),
        max_iterations: int = 4,
        gain: float = 0.9,
        tolerance_mm: float | None = None,
    ) -> "RingAlignResult":
        """闭环把环心推到目标读数：**测一帧 → 算 → 走 → 再测**，直到进容差。

        一次 :meth:`plan_ring` 只做一次线性化解算，它同时依赖两件事：

        - 雅可比是不是**现在这台机器、这个高度**的（``eye_in_hand.json`` 里那份是按
          物料标定的，换高度/换装法就会过期）；
        - 环心读数的比例尺与雅可比是不是同一把尺（``ring box`` 是估的，±10%）。

        任一对不上，一步就走不准——这正是"转一次再算一次、偏差还是很大"的原因。
        这里每走一步都用**实测位移 / 预测位移**把响应比在线学出来，下一步按它折算，
        所以两三步就能收敛；而且把响应比打印出来：**它不是 1，就说明该重标了**。

        与 :meth:`plan_ring` 相同的拒绝（没有环 / 读数不可用 / 环号不符 / 尺不对 /
        雅可比奇异），另外：

        :param axes: 只动哪些轴；``("base",)`` = 只转底座。
        :param max_iterations: 最多走几步（默认 4）。
        :param gain: 每步取解算值的多少（默认 0.9，留一点余量防过冲）。
        :param tolerance_mm: 到位门槛；``None`` 用 ``eye_in_hand.json`` 的 ``tolerance_mm``。
        :raises ServoError: 任何一步看不见环、解算失败；或响应比 ≤ 0（方向反了：
            那是标定/接线问题，硬走只会越走越远）。
        """
        arm, source = self._require_resources()
        from modules.grasp.drive.ring_plan import RingAlignResult, RingAlignStep
        from modules.grasp.drive.servo_motion import _move_axis

        wanted = tuple(axes)
        unknown = [name for name in wanted if name not in ("base", "extension")]
        if unknown:
            raise ValueError(
                f"align_ring 只认 base / extension，收到 {unknown}"
            )
        tolerance = float(
            self._config.tolerance_mm if tolerance_mm is None else tolerance_mm
        )
        steps: list[RingAlignStep] = []
        scale = 1.0  # 步长的在线折算：1/响应比
        response: float | None = None
        previous_reading: tuple[float, float] | None = None
        previous_step: tuple[int, int] | None = None
        note = ""
        aligned = False
        final_reading: tuple[float, float] | None = None
        digit: int | None = None
        final_plan = None
        final_source = ""
        for attempt in range(1, max(1, int(max_iterations)) + 1):
            target = self._locate_ring_for_plan(
                source, timeout=timeout, require_digit=require_digit, settle=True
            )
            plan = self._plan_for_ring(arm, target, scale_mm_per_px=scale_mm_per_px)
            digit = plan.digit
            final_reading = plan.reading
            final_plan = plan
            final_source = str(target.scale_source)
            error = math.hypot(plan.error_mm[0], plan.error_mm[1])
            step = RingAlignStep(
                attempt=attempt,
                reading=plan.reading,
                error_mm=error,
                base_pulses=0,
                extension_pulses=0,
            )
            # 先结算**上一步**的账：实测挪了多少，按雅可比该挪多少。
            if attempt > 1 and previous_step is not None and previous_reading is not None:
                predicted = self._jacobian.displacement_for(*previous_step)
                measured = (
                    plan.reading[0] - previous_reading[0],
                    plan.reading[1] - previous_reading[1],
                )
                denominator = predicted[0] ** 2 + predicted[1] ** 2
                if denominator > 1e-12:
                    response = (
                        measured[0] * predicted[0] + measured[1] * predicted[1]
                    ) / denominator
                    step = replace(
                        step, measured_mm=measured, predicted_mm=predicted, response=response
                    )
                    if response <= 0.05:
                        steps.append(step)
                        note = (
                            "实测位移与雅可比预测**方向相反或几乎没有**，再走只会越走越远："
                            "先重标雅可比（菜单 21 的 --calibrate），或检查底座/伸缩的接线方向"
                        )
                        break
                    # 响应比 = 真实的灵敏度 / 配置的灵敏度；下一步按它折算。
                    scale = min(5.0, max(0.2, 1.0 / response))
            if error <= tolerance:
                aligned = True
                steps.append(step)
                break
            base_step = (
                _round_pulses(plan.base_pulses * gain * scale) if "base" in wanted else 0
            )
            extension_step = (
                _round_pulses(plan.extension_pulses * gain * scale)
                if "extension" in wanted
                else 0
            )
            if base_step == 0 and extension_step == 0:
                steps.append(step)
                note = "步长不足一个脉冲，停下：把增益调大或检查读数是不是抖"
                break
            _move_axis(arm, "base", base_step) if base_step else None
            _move_axis(arm, "extension", extension_step) if extension_step else None
            steps.append(
                replace(step, base_pulses=base_step, extension_pulses=extension_step)
            )
            previous_reading = plan.reading
            previous_step = (base_step, extension_step)
        else:
            note = note or f"走了 {max_iterations} 步还没进容差，先看响应比是不是一直不变"

        final_error = None
        if final_reading is not None and self._target is not None:
            # 残差要跟**环心那一次的目标**比（含落料补偿），否则报出来的数与闭环停下的
            # 判据不是同一个口径：补偿 7.5 mm 时它会一直显示差 7.5 mm。
            ring_target = self._ring_target(digit)
            final_error = math.hypot(
                final_reading[0] - ring_target.x_mm,
                final_reading[1] - ring_target.y_mm,
            )
        return RingAlignResult(
            steps=tuple(steps),
            aligned=aligned,
            final_reading=final_reading,
            final_error_mm=final_error,
            tolerance_mm=tolerance,
            digit=digit,
            scale_source=final_source,
            response=response,
            # 收敛时那一份计划：字段与 ``plan_ring()`` 完全同一套，可以直接存下来
            # （"先对准、之后回到这个绝对位置"就靠它）。
            plan=final_plan,
            note=note,
        )

    #: 等环心读数停稳时最多连读几帧。3 帧够滤掉底座摆完的那点余振，又不会在
    #: "一直等不到稳定"时卡住——等不到就用最后一帧，和现在完全一样。
    RING_SETTLE_READS = 3

    def _locate_ring_for_plan(
        self, source, *, timeout, require_digit, settle: bool = False
    ):
        """读一帧并校验"这一个环能不能拿去算角度"；不能就说清楚为什么。

        ``settle=True``（闭环用）时连读几帧，直到相邻两次读数足够接近
        （``settle_tolerance_mm``）。为什么要：每步走完只按**估算耗时 + 0.15 s** 睡一下
        就去测下一帧，而底座是大惯量摆动，那点时间不够；读数抖 1–2 px，下一步的步长是
        **按这个读数算的**，于是抖动量直接变成来回摆。单帧定位（``plan_ring``）保持一帧，
        那条路是人看着按的，多读几帧只是白等。
        """
        target = self._validated_ring(
            source, timeout=timeout, require_digit=require_digit
        )
        if not settle:
            return target
        previous = target.reading()
        for _ in range(self.RING_SETTLE_READS - 1):
            if previous is None:
                break
            following = self._validated_ring(
                source, timeout=timeout, require_digit=require_digit
            )
            reading = following.reading()
            target = following
            if reading is not None and math.hypot(
                reading[0] - previous[0], reading[1] - previous[1]
            ) <= self._config.settle_tolerance_mm:
                break
            previous = reading
        return target

    def _validated_ring(self, source, *, timeout, require_digit):
        """一帧环心 + 三项校验（有环 / 读数可用 / 环号对得上）。

        ``require_digit`` **下传给定位器**，让它只在这个号里挑框：一帧里同时有 1/2/3 号环
        是正常取景，按"取分最高的那个再看是不是它"会随机报"画面里是 1 号，不是 2 号"——
        分最高的那个通常就不是你要的那个。后面那道环号检查留着，兜住不认 ``digit`` 的源。
        """
        locate = getattr(source, "locate_ring", None)
        if locate is None:
            raise ServoError(
                f"这个观测源没有环心定位（{type(source).__name__}）："
                "环心是相机这一侧的能力，先确认会话用的是本机视觉源（Vision）"
            )
        target = locate(timeout=timeout, digit=require_digit)
        if target is None:
            raise ServoError(
                "画面里没有环：把靶标移到视野里、或确认模型类别表里有 tg1/tg2/tg3"
                "（modules/vision/config.json 的 model.class_names）"
            )
        if target.reading() is None:
            raise ServoError(self._unusable_ring_reason(target))
        if require_digit is not None and int(target.digit) != int(require_digit):
            raise ServoError(
                f"画面里的环是 {target.digit} 号，不是要求的 {require_digit} 号；"
                "拿另一个环的角度去转，机械臂会朝 95 mm 外的位置动，而动作看起来完全正常"
            )
        return target

    def _ring_target(self, digit: int | None = None) -> ServoTarget:
        """环心那一次伺服的**目标读数**：配置里的 ``target`` 加上落料补偿。

        补偿只加在环心路径上（``ServoConfig.ring_offset``）：物料对准仍然对着
        ``self._target``。两边都加等于整体平移，料与环的相对偏差一点没变。

        闭环看不见"料落在环心的哪一侧"（镜头光轴与爪口有固定偏差、料在爪里也会
        偏），所以这份偏差只能量出来写进配置；落点误差 = 补偿量 − 实际偏移。

        ``digit`` 给了就用**这个环号那一份**补偿（``ring_offsets``）：同一份补偿放到不同
        环上会差一个方向，因为回程要转过的角度不同。没配这个号就退回全局那一对。
        """
        offset_x, offset_y = self._config.ring_offset(digit)
        return ServoTarget(
            x_mm=self._target.x_mm + offset_x,
            y_mm=self._target.y_mm + offset_y,
        )

    def _plan_for_ring(self, arm, target, *, scale_mm_per_px):
        """一帧已定位的环 → 各轴绝对位置（闭环与单次计划共用这一处）。"""
        reading = self._ring_reading_in_jacobian_units(
            target, target.reading(), scale_mm_per_px
        )
        return self._plan_from_reading(
            arm, reading, digit=int(target.digit), score=float(target.score)
        )

    def _plan_from_reading(self, arm, reading, *, digit, score=None, offset_digit=None):
        """读数**已经与雅可比同一把尺**（比例尺由调用方核过）→ 各轴绝对位置。

        ``offset_digit`` 给了就用那个环号的落料补偿（一条直线上的几个点共用一份）；
        ``digit`` 仍然是这份计划自己的环号，写进结果里给人看。
        """
        from modules.grasp.drive.ring_plan import plan_ring_base

        ring_target = self._ring_target(digit if offset_digit is None else offset_digit)
        plan = plan_ring_base(
            reading=reading,
            target=(ring_target.x_mm, ring_target.y_mm),
            jacobian=self._jacobian,
            base_pulses=arm.turns_state.of("base").pulses,
            extension_pulses=arm.turns_state.of("extension").pulses,
            pulses_per_revolution=arm.pulses_per_revolution,
            reduction_ratio=arm.config.base_reduction,
            tolerance_mm=self._config.tolerance_mm,
        )
        return replace(plan, digit=int(digit), score=score)

    @staticmethod
    def _unusable_ring_reason(target) -> str:
        """环心读数为什么不能用——贴边和缺比例尺是两件完全不同的事。"""
        if target.clipped:
            return (
                "环心读数不可用：检测框贴到画面边缘，框心已经不是环心了"
                "（实测能偏到 80 px 以上）。把靶标往画面中间挪一点再试"
            )
        return (
            "环心读数不可用：没配比例尺。给 modules/vision/calibration.json 标上 "
            "focal_px 与 table_depth_mm（菜单 25），或配 station.mm_per_px"
        )

    def _ring_reading_in_jacobian_units(
        self, target, reading: tuple[float, float], scale_mm_per_px: float | None
    ) -> tuple[float, float]:
        """把环心读数换算到**标定雅可比时那把尺**上；对不上就拒绝。

        雅可比是"物理毫米/脉冲"，所以环心读数只要也是**物理毫米**就能直接用——
        ``depth/focal``（针孔标定）和 ``ring box``（模型框 + 已知外径）都是，
        ``station.mm_per_px`` 不是（那是给固定相机标的值，装在臂上会差好几倍）。
        """
        current = target.scale_mm_per_px
        if scale_mm_per_px is not None:
            if not scale_mm_per_px > 0:
                raise ServoError(
                    f"scale_mm_per_px 必须是正数（毫米/像素），收到 {scale_mm_per_px!r}"
                )
            if current is None or current <= 0:  # pragma: no cover - 读数为空时早已返回
                raise ServoError("环心读数没有比例尺，换不了单位")
            factor = float(scale_mm_per_px) / float(current)
            return (reading[0] * factor, reading[1] * factor)
        self._require_physical_ring_scale(target.scale_source, current)
        return reading

    @staticmethod
    def _require_physical_ring_scale(
        scale_source: str, scale_mm_per_px: float | None
    ) -> None:
        """环心读数的比例尺必须是**物理毫米**（与雅可比同一把尺）；不是就拒绝。

        ``station.mm_per_px`` 是给固定相机标的值，相机装在臂上时它会差好几倍：
        照着算出来的角度看起来完全正常，爪子却朝错的地方去。
        """
        if scale_source in PHYSICAL_RING_SCALES:
            return
        measured = "无" if scale_mm_per_px is None else f"{scale_mm_per_px:.4f} mm/px"
        raise ServoError(
            f"环心读数的比例尺是 {scale_source or '未知'}（{measured}），"
            "不是物理毫米，而 jacobian 是按物理毫米标定的：直接算角度会整体差一个倍数，"
            "而计划看起来完全正常。\n"
            "  两种做法：\n"
            "  1) 给 modules/vision/config.json 的 ring_target 填 outer_diameter_mm"
            "（靶标图案外径，现场 95 mm），环心读数就会用“外径 / 模型框”换算，"
            "与镜头距离无关；\n"
            "  2) 知道与雅可比一致的毫米/像素时显式传进来："
            "robot.grasp.plan_ring(scale_mm_per_px=0.183)"
        )

    def align_and_descend(
        self,
        *,
        z_pulses=None,
        retract_pulses=None,
        hover: bool = True,
        ensure_safe_height: bool = True,
        want=None,
    ) -> GraspReadyResult:
        # NOTE: 这个函数不再使用
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
        :param want: 这一次要抓**哪一个**物料：类别名（``"red"``）或从 1 开始的编号，
            与 ``vision.locate(want)`` 同一套说法；不传就用 ``eye_in_hand.json`` 里
            ``object`` 那一份。**一次程序里依次抓多个物体**就靠它::

                for colour in ("red", "green", "blue"):
                    robot.grasp.align_and_descend(want=colour)
                    robot.gripper.close_jaws()
                    robot.gripper.set_position(250)      # 张开，放下一个
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
        # 这一次抓哪一个：按次传的 want 优先，否则用配置里那份默认。
        selector = self._selector_for(want, self._selector)
        started = time.monotonic()
        if hover or ensure_safe_height:
            say("① 先到物块上方的悬停高度（配了 hover_z_mm 才动），再确认转底盘前的安全高度")
        else:
            say("① 不动高度：转底盘前的高度由调用方自己保证（低位转动会撞到台面上的物块）")
        hovered = (
            hover_above_object(
                arm, source, selector, self._config,
                scale=self._lift_scale, on_message=say,
            )
            if hover
            else HoverResult(configured=False)
        )
        raised = (
            raise_before_rotating(
                arm, source, selector, self._config,
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
            selector,
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
            arm, source, selector, self._config,
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

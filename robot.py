"""项目总入口：一个会话里装配 arm / gripper / vision / chassis，需要时再挂上 grasp。

## 为什么要它

每个模块都有自己的 ``open``，但**同一台设备只能被打开一次**：

- arm 与 chassis 用的是**同一根** Emm42 步进总线（``modules/motor/config.json`` 的端口）；
- grasp 需要 arm 与 vision，而它不该再开第二份。

所以"上层只写一条 with"这件事必须由一个**总所有者**来做。这就是 ``Robot``：

```python
from robot import Robot

with Robot.open_all() as robot:
    report = robot.vision.locate()

with Robot.open(arm=True, gripper=True) as robot:
    robot.arm.move_up(pulses=800)
    robot.gripper.open_jaws()

with Robot.open(grasp=True, gripper=True) as robot:
    result = robot.grasp.align_and_descend()
    robot.gripper.close_jaws()          # 是否闭爪由业务显式决定
```

## 资源关系

```
Robot 会话（总所有者）
├─ 一条 Emm42 步进连接         ← 由 Robot 打开，最后关闭
│  ├─ arm     借用（地址 5/6/7）
│  └─ chassis 借用（配置里的四轮地址）
├─ gripper：独立舵机连接
├─ vision：一个相机 + 一个模型
└─ grasp：借用上面同一个 arm 与 vision，不新开设备
```

## 它不做什么

- **不自动回零、不移动、不开闭爪**：这些是明确的动作，由上层调用；
- 不做"USB 失败就换 TCP"一类的自动降级，也没有设备注册表或线程池；
- 没有总配置文件：每个模块的路径单独具名，``None`` 表示用那个模块自己的默认文件。

## 一次会话只读一次 motor 配置

``motor_config_path`` 只读一次，arm 与 chassis 共用那一个端口快照；两者的电机地址
不能冲突——冲突会在**打开第一条连接之前**报出来。
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

if TYPE_CHECKING:
    # 只给注解用的导入：真实装配需要的导入留在各自的装配边界（``open`` 内部），
    # 这样 ``import robot`` 不会顺带加载 cv2 / 模型 / 串口后端。
    from modules.arm.config import ArmConfig
    from modules.arm.runtime import RobotArm
    from modules.chassis.config import HardwareProfile
    from modules.chassis.runtime import MecanumChassis
    from modules.grasp.runtime import Grasp, GraspSettings
    from modules.motor.drive.emm42 import SerialPort
    from modules.servo.runtime import GripperServo
    from modules.vision.config import VisionConfig
    from modules.vision.runtime import Vision
T = TypeVar("T")

#: 本轮的"全部"模块：四个设备模块 + 抓取流程。grasp 借用 arm/vision，不额外开设备。
ALL_MODULES: Tuple[str, ...] = ("arm", "gripper", "vision", "chassis", "grasp")


class Closable(Protocol):
    """只声明"能被关掉"这一件事的窄协议。

    清理集合用它，是为了让 :meth:`Robot._attempt` 接受任意可关闭对象，
    而**不是**把模块字段或属性放宽成这个协议——那两者仍是各自的具体类。
    """

    def close(self) -> None: ...


class Robot:
    """把四个设备模块（加抓取流程）装配成一个会话，并统一收尾。"""

    def __init__(
        self,
        *,
        arm: Optional["RobotArm"] = None,
        gripper: Optional["GripperServo"] = None,
        vision: Optional["Vision"] = None,
        chassis: Optional["MecanumChassis"] = None,
        grasp: Optional["Grasp"] = None,
        bus: Optional["SerialPort"] = None,
    ) -> None:
        """只保存**已经装配好的**模块；本构造器不做设备 I/O。

        ``bus`` 是那条共享的步进连接（如果有模块用到它），由 :meth:`close` 最后关闭。
        字段类型如实是 ``Optional[具体模块类]``：本次会话确实可能没启用某个模块。
        """
        self._arm: Optional[RobotArm] = arm
        self._gripper: Optional[GripperServo] = gripper
        self._vision: Optional[Vision] = vision
        self._chassis: Optional[MecanumChassis] = chassis
        self._grasp: Optional[Grasp] = grasp
        self._bus: Optional[SerialPort] = bus
        self._owns_bus = bus is not None
        self._closed = False

    # ------------------------------------------------------------- 属性
    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Robot 会话已关闭")

    def _module(self, name: str, value: Optional[T]) -> T:
        """取一个模块；未启用时**立刻**说清楚，而不是在动作时偷偷打开设备。

        返回类型是非空 ``T``：上层拿到对象后不必反复判空。
        """
        self._require_open()
        if value is None:
            raise RuntimeError(
                f"本次会话没有启用 {name}；在 Robot.open(...) 里显式打开它"
                "（没有启用的模块不会在动作时自动打开设备）"
            )
        return value

    @property
    def arm(self) -> "RobotArm":
        """机械臂模块实例；未启用时抛 ``RuntimeError``。"""
        return self._module("arm", self._arm)

    @property
    def gripper(self) -> "GripperServo":
        """夹爪模块实例；未启用时抛 ``RuntimeError``。"""
        return self._module("gripper", self._gripper)

    @property
    def vision(self) -> "Vision":
        """视觉模块实例；未启用时抛 ``RuntimeError``。"""
        return self._module("vision", self._vision)

    @property
    def chassis(self) -> "MecanumChassis":
        """底盘模块实例；未启用时抛 ``RuntimeError``。"""
        return self._module("chassis", self._chassis)

    @property
    def grasp(self) -> "Grasp":
        """抓取会话；未启用时抛 ``RuntimeError``。"""
        return self._module("grasp", self._grasp)

    @property
    def modules(self) -> Tuple[str, ...]:
        """本次真正装配起来的模块名（供日志与验收用）。"""
        found: List[str] = []
        for name, value in (
            ("arm", self._arm),
            ("gripper", self._gripper),
            ("vision", self._vision),
            ("chassis", self._chassis),
            ("grasp", self._grasp),
        ):
            if value is not None:
                found.append(name)
        return tuple(found)

    @property
    def closed(self) -> bool:
        """会话是否已经关闭；关闭后取用任何模块都失败。"""
        return self._closed

    # ------------------------------------------------------------- 创建
    @classmethod
    def open(
        cls,
        *,
        arm: bool = False,
        gripper: bool = False,
        vision: bool = False,
        chassis: bool = False,
        grasp: bool = False,
        arm_config_path: Optional[Path] = None,
        motor_config_path: Optional[Path] = None,
        servo_config_path: Optional[Path] = None,
        vision_config_path: Optional[Path] = None,
        vision_calibration_path: Optional[Path] = None,
        chassis_config_path: Optional[Path] = None,
        grasp_config_path: Optional[Path] = None,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> "Robot":
        """按需装配本次要用的模块；至少明确选择一个。

        ``grasp=True`` **明确包含 arm 与 vision 两项依赖**（它们由本次会话提供，
        不再另开设备），但**不**包含 gripper，也不自动回零或闭爪。

        所有被选模块的配置**先全部校验完**，再打开第一个设备：这样"配置写错"不会
        留下半开的连接。任何一项目开启失败，已经开好的都会被清理，不返回半就绪对象。

        路径语义见模块文档；``servo_config_path`` 同时给夹爪参数与舵机总线两个加载器，
        ``motor_config_path`` 是 arm/chassis 共用的那一份步进总线配置。
        """
        from modules.arm.config import DEFAULT_CONFIG_PATH as arm_default
        from modules.arm.config import load_arm_config
        from modules.motor.config import DEFAULT_CONFIG_PATH as motor_default
        from modules.motor.config import load_bus_config

        if not any((arm, gripper, vision, chassis, grasp)):
            raise ValueError(
                "Robot.open 至少要明确选择一个模块（arm / gripper / vision / chassis / grasp）"
            )
        # grasp 需要 arm 与 vision：显式打开它们，并让本次会话共用同一份。
        if grasp:
            arm = True
            vision = True

        # ---- 阶段一：把所有配置读完，一个设备都不开 ----
        # 这些局部变量在"没选那个模块"时确实是 None，所以如实可空；装配点再逐个
        # 用**实际值**判空缩窄，不靠另一个布尔变量替它们担保。
        arm_config: Optional[ArmConfig] = None
        port: Optional[str] = None
        chassis_profile: Optional[HardwareProfile] = None
        vision_settings: Optional[VisionConfig] = None
        grasp_settings: Optional[GraspSettings] = None

        if arm:
            arm_path = arm_default if arm_config_path is None else Path(arm_config_path)
            arm_config = load_arm_config(config_path=arm_path)
        if arm or chassis:
            motor_path = motor_default if motor_config_path is None else Path(motor_config_path)
            # 注意 load_bus_config 的第一个位置参数是 platform，路径只能走关键字。
            port = load_bus_config(config_path=motor_path).port

        if chassis:
            from modules.chassis.runtime import MecanumChassis

            chassis_profile = MecanumChassis.load_hardware_profile(chassis_config_path)
            if arm_config is not None and chassis_profile is not None:
                cls._require_distinct_motor_addresses(arm_config, chassis_profile)

        if gripper:
            # 只**校验**夹爪那份配置（夹爪参数 + 舵机总线）；真正的装配在开启阶段。
            from modules.servo.config import DEFAULT_CONFIG_PATH as servo_default
            from modules.servo.config import load_gripper_config
            from modules.servo.drive.bus_servo import load_bus_config as load_servo_bus

            servo_path = servo_default if servo_config_path is None else Path(servo_config_path)
            load_gripper_config(servo_path)
            load_servo_bus(config_path=servo_path)

        vision_settings = None
        if vision:
            from modules.vision.runtime import _load_settings

            vision_settings = _load_settings(vision_config_path, vision_calibration_path)

        if grasp:
            grasp_settings = cls._load_grasp_settings(grasp_config_path)

        # ---- 阶段二：开始开设备；失败就把已经开的全部收掉 ----
        session = cls()
        try:
            # arm 与 chassis 共用**一条**步进连接。端口是这条链的硬前置条件：
            # 选了 arm/chassis 却没有端口，就在这里失败，而不是把 None 传下去。
            if arm or chassis:
                if port is None:
                    raise RuntimeError(
                        "选了 arm/chassis 却没有可用的步进总线端口；"
                        "请检查 modules/motor/config.json 的端口配置"
                    )
                session._bus = session._open_shared_bus(port)
                # 这条连接是**本会话开的**，所以由本会话最后关闭。
                session._owns_bus = True

            if arm:
                from modules.arm.runtime import RobotArm

                if arm_config is None:
                    raise RuntimeError("选了 arm 却没有读到 arm 配置")
                # 就地判空：不靠上面那个 `arm or chassis` 分支替这次装配担保端口存在。
                if port is None:
                    raise RuntimeError("选了 arm 却没有可用的步进总线端口")
                session._arm = RobotArm._from_settings(
                    config=arm_config,
                    port=port,
                    bus=session._bus,
                    owns_bus=False,
                )

            if chassis:
                from modules.chassis.runtime import MecanumChassis

                if chassis_profile is None:
                    raise RuntimeError("选了 chassis 却没有读到 chassis 配置")
                if port is None:
                    raise RuntimeError("选了 chassis 却没有可用的步进总线端口")
                session._chassis = MecanumChassis._from_settings(
                    profile=chassis_profile,
                    port=port,
                    bus=session._bus,
                    owns_bus=False,
                )

            if gripper:
                from modules.servo.runtime import GripperServo

                # 走夹爪自己的唯一装配：读同一份 servo 配置、开总线、上扭矩。
                session._gripper = GripperServo.open(config_path=servo_config_path)

            if vision:
                from modules.vision.runtime import Vision

                if vision_settings is None:
                    raise RuntimeError("选了 vision 却没有读到视觉配置")
                session._vision = Vision._open_config(vision_settings)

            if grasp:
                from modules.grasp.runtime import Grasp

                if grasp_settings is None:
                    raise RuntimeError("选了 grasp 却没有读到抓取配置快照")
                # 借用本次会话的同一个 arm 与 vision，不新开设备。
                session._grasp = Grasp._from_modules_settings(
                    config=grasp_settings["config"],
                    selector=grasp_settings["selector"],
                    target=grasp_settings["target"],
                    jacobian=grasp_settings["jacobian"],
                    lift_scale=grasp_settings["lift_scale"],
                    arm=session.arm,
                    vision=session.vision,
                    on_message=on_message,
                )
            return session
        except BaseException as primary:
            try:
                session.close()
            except Exception as cleanup:
                raise primary.with_traceback(primary.__traceback__) from cleanup
            raise

    @classmethod
    def open_all(
        cls,
        *,
        arm_config_path: Optional[Path] = None,
        motor_config_path: Optional[Path] = None,
        servo_config_path: Optional[Path] = None,
        vision_config_path: Optional[Path] = None,
        vision_calibration_path: Optional[Path] = None,
        chassis_config_path: Optional[Path] = None,
        grasp_config_path: Optional[Path] = None,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> "Robot":
        """一键开启本轮的全部五项；配置参数与 :meth:`open` 同名同义。

        内部复用 :meth:`open` 的**唯一装配**，不复制初始化逻辑。
        "全部"要求每一项的启动配置都有效（含抓取 target / Jacobian）；任一项失败就
        清理已打开的资源并抛错，**不静默跳过模块**。它不自动回零、不移动、不开闭爪。
        """
        return cls.open(
            arm=True,
            gripper=True,
            vision=True,
            chassis=True,
            grasp=True,
            arm_config_path=arm_config_path,
            motor_config_path=motor_config_path,
            servo_config_path=servo_config_path,
            vision_config_path=vision_config_path,
            vision_calibration_path=vision_calibration_path,
            chassis_config_path=chassis_config_path,
            grasp_config_path=grasp_config_path,
            on_message=on_message,
        )

    # ------------------------------------------------------------- 装配助手
    @staticmethod
    def _require_distinct_motor_addresses(arm_config, chassis_profile) -> None:
        """arm 与 chassis 共用一条总线，地址不能冲突——在开连接之前就报出来。"""
        arm_ids = {arm_config.axes[name].motor_id for name in ("base", "lift", "extension")}
        chassis_ids = set(chassis_profile.motor_ids.values())
        clash = sorted(arm_ids & chassis_ids)
        if clash:
            raise ValueError(
                "arm 与 chassis 共用一条步进总线，但电机地址冲突："
                + "、".join(str(item) for item in clash)
                + "。请改 modules/arm/config.json 或 modules/chassis/config.json 的地址。"
            )

    @staticmethod
    def _open_shared_bus(port: str) -> "SerialPort":
        """打开那条共享的 Emm42 连接（arm 与 chassis 时序复用同一条）。

        返回值标成 :class:`~modules.motor.drive.emm42.SerialPort`：上层只依赖这个窄协议，
        真实实现仍是 pyserial 的 ``Serial``（它的 ``write`` 与协议声明一致，见那里的说明）。
        """
        from modules.motor.drive.emm42 import Emm42Driver, SerialPort as _SerialPort

        try:
            import serial
        except ImportError as error:  # pragma: no cover - 板子上的环境问题
            raise RuntimeError(
                "pyserial is required on Jetson. Install it in the target Python environment first."
            ) from error
        opened: _SerialPort = serial.Serial(
            port, baudrate=Emm42Driver.BAUDRATE, timeout=0.1
        )
        return opened

    @staticmethod
    def _load_grasp_settings(config_path: Optional[Path]) -> "GraspSettings":
        """读并校验 eye_in_hand 那份配置；返回可直接喂给 ``_from_modules_settings`` 的快照。

        与 ``Grasp.from_modules`` 共用同一份参数校验，所以校验时机同样是**开设备之前**；
        ``_require_action_ready`` 返回缩窄后的非空 target/jacobian，正是这份快照的类型要求。
        """
        from modules.grasp.drive.servo_model import (
            DEFAULT_CONFIG_PATH,
            load_jacobian,
            load_lift_scale,
            load_object_selector,
            load_servo_config,
            load_target,
        )
        from modules.grasp.runtime import Grasp, GraspSettings

        path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        target, jacobian = Grasp._require_action_ready(
            target=load_target(path), jacobian=load_jacobian(path)
        )
        return GraspSettings(
            config=load_servo_config(path),
            selector=load_object_selector(path),
            target=target,
            jacobian=jacobian,
            lift_scale=load_lift_scale(path),
        )

    # ------------------------------------------------------------- 收尾
    def close(self) -> None:
        """释放本会话管理的全部模块与自有连接。

        顺序（固定）：

        1. 先让**借用的**抓取会话失效——它借的是 arm 与 vision，不能比它们晚；
        2. 再按取得资源的**逆序**逐个调用模块 ``close``；
        3. 最后关闭那条共享的步进连接（如果本次开了）。

        第一项失败不阻断其他项；全部试完才报告，错误清单含每个失败阶段。
        重复调用不再发命令。**不自动开闭爪。**
        """
        if self._closed:
            return
        self._closed = True
        grasp, vision, gripper, chassis, arm = (
            self._grasp,
            self._vision,
            self._gripper,
            self._chassis,
            self._arm,
        )
        self._grasp = self._vision = self._gripper = self._chassis = self._arm = None
        bus, self._bus = self._bus, None

        errors: List[Tuple[str, Exception]] = []
        if grasp is not None:
            self._attempt(errors, "grasp.close", grasp.close)
        # 逆序：后取得的先放。
        for name, module in (
            ("vision.close", vision),
            ("gripper.close", gripper),
            ("chassis.close", chassis),
            ("arm.close", arm),
        ):
            if module is not None:
                self._attempt(errors, name, module.close)
        if bus is not None and self._owns_bus:
            self._attempt(errors, "bus.close", bus.close)
        self._owns_bus = False

        if errors:
            details = "; ".join(f"{stage}: {error}" for stage, error in errors)
            raise RuntimeError("Robot 清理失败：" + details) from errors[0][1]

    @staticmethod
    def _attempt(
        errors: List[Tuple[str, Exception]],
        stage: str,
        action: Callable[[], None],
    ) -> None:
        """执行一项收尾；失败记进清单，但绝不跳过后面的项。"""
        try:
            action()
        except Exception as error:
            errors.append((stage, error))

    def __enter__(self) -> "Robot":
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

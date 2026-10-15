"""真机入口：让爪子正下方对准物块（抓取第一步，只动平面）。

    python test.py            # 菜单选 21

## 观测从哪来

默认走**本机 USB 摄像头**：``modules/vision`` 的 ``MaterialLocator`` 直接读画面，
就是菜单 18 那条通路（你终端里看到的 `物料 1: @(543,382) …` 就是它打的）。
远端镜头（TCP 那种）用 ``--tcp``，两者对上层完全同形。

## 用法

```powershell
# **一键就位**：悬停高度 -> 安全高度 -> x/y 对准 -> z 下降 -> 伸缩后缩，每段都报耗时
python -m modules.grasp.test.manual.align_xy --align-and-descend

# 只看不动：实时打印物块读数、目标读数、误差，以及打算走多少脉冲
python -m modules.grasp.test.manual.align_xy --status

# 第一次用：两步标定
python -m modules.grasp.test.manual.align_xy --set-target   # 用手把爪子挪到正好套住物块，回车记录
python -m modules.grasp.test.manual.align_xy --calibrate    # 程序自己动两下，实测雅可比

# 对准一次（对准就停）
python -m modules.grasp.test.manual.align_xy

# **跟随**：物块被移动时持续跟着对准，Ctrl+C 退出
python -m modules.grasp.test.manual.align_xy --follow

# 平面稳定之后往下走：升降下降 + 伸缩后缩（两个固定脉冲数）
python -m modules.grasp.test.manual.align_xy --align-z
```

## 起始高度也在配置里：`hover_z_mm`

x/y 对准**不动高度**，所以"对准时爪子停在哪一层"原本只由 `safe_lift_z_mm` 兜下限
（低了才抬，高了不管）。而下降是**开环的固定脉冲数**，起点不固定、落点就不固定。

`eye_in_hand.json` 的 `hover_z_mm`（默认 100.0，和 `safe_lift_z_mm` 同值）让就位第一步
先把"镜头到物块的距离"伺服到这一层（高了降、低了抬，判据 `z_tolerance_mm`），
于是每次下降都从同一层开始：

- 必须 `>= safe_lift_z_mm`，否则直接拒绝加载（悬停得比安全高度还低 = 低位转底盘）；
- 往下降只降到 `safe_lift_z_mm` 为止；而且要**先知道一步走多远**（标定的 `lift_scale`，
  或先抬一段实测出灵敏度），在量出来之前绝不往下走；
- **改了它就要重测 `z_descend_pulses`**：起点换多少毫米，落点就换多少毫米；
- 临时试：`--hover-z 105`（只覆盖这一次）。设成 `null` = 完全不主动调整。

## 时间花在哪（本机实测）

| 环节 | 实测 | 说明 |
|---|---|---|
| `camera.read()` | 32 ms | 30 fps 一帧 |
| 每帧定位 | 178 ms | Hough + YOLO，**单帧地板价** |
| 一次取帧 | 221 ms | 丢 2 帧 + 读 + 定位 |
| 一次**停稳测量** | 537 ms | 闭环每测一次至少两次定位 |

所以"少测几次"比"少动几下"更值钱。几处提速（都在 ``modules/grasp/eye_in_hand.json`` 里可关）：

- 抬到安全高度：标过 ``lift_scale`` 就按"还差多少毫米"**一步算到位**（夹在
  ``safe_lift_max_pulses_per_step`` 内），而不是 500 脉冲一段地试——最多省十几段测量；
- 粗调阶段用 ``coarse_settle_tolerance_mm`` 判停稳（远的时候不必为了 0.4 mm 多等帧）；
- ``--follow`` **每一轮仍然查安全高度**，但用"单帧快判 + 余量（``safe_lift_quick_margin_mm``）"：
  够高就直接放行（一帧，约 0.22 s），读数偏低或落在余量内就退回完整的停稳测量（约 0.54 s）。
  判据一点没放宽，省的是"绝大多数轮次都够高"这件事上的重复测量；
- ``--align-and-descend`` 一次走完四步，并把每段耗时打出来。

## z 这一步是开环的：两个可改的脉冲数

x/y 是闭环（只在平面里动）；**平面一旦对准**，剩下的两步就是两个固定脉冲数，
都在 ``modules/grasp/eye_in_hand.json`` 里，现场直接改：

| 配置键 | 轴 | 动作 |
|---|---|---|
| ``z_descend_pulses`` | 升降（6 号） | 下降这么多脉冲到夹持高度 |
| ``extension_retract_pulses`` | 伸缩（7 号） | 再后缩这么多脉冲 |

``--align-z`` 会**先确认平面已经对准**（物块离光轴 ≤ ``z_center_tolerance_mm``）再走；
没对准就拒绝下降——脉冲数只保证往下走了那么多，不保证下面正好是物块。

试新数不用改文件：``--z-pulses`` / ``--retract-pulses`` 只覆盖这一次。
只有想"拿物块深度当反馈"时才用视觉伺服版 ``--align-z-visual``（要先标定升降灵敏度）。

## 这套机器上相机装在臂上

相机随爪子一起动，所以「物块在相机系的坐标」里混着机械臂自己的运动，
不能用世界坐标换算。这里走的是**视觉伺服**：把画面里的物块调到那个固定的目标读数
``(tx, ty)``——它就是爪子抓取点在相机坐标系里的位置。细节见
``modules/grasp/drive/eye_in_hand.py``。

**不需要** ``mm_per_motor_rev``、``limits``、平面标定，也不需要给爪子贴靶标：
每一步都用脉冲驱动、拿画面当反馈。代价是精度受"一帧能分辨多少毫米"限制，
而且没标定行程时只能靠脉冲预算兜底。
"""

from __future__ import annotations

MENU_ENTRY = (21, "x/y 对准物块（含一键就位）")

import argparse
import dataclasses
import math
import os
import time
from typing import Optional, Sequence

from modules.arm.config import load_arm_config
from modules.arm.runtime import RobotArm
from modules.grasp.drive.eye_in_hand import (
    align_x_y,
    align_z,
    descend_z,
    find_object,
    hover_above_object,
    probe_jacobian,
    probe_lift_scale,
    retract_extension,
)
# 配置与值类型只有一个定义处（servo_model），观测判据在 servo_observation，
# 脉冲原语在 servo_motion，实际控制算法在 eye_in_hand。
# 这里**不从包顶层导入内部工具**：那些名字是给上层用的会话接口，不是工具装配件。
from modules.grasp.drive.servo_motion import _move_lift
from modules.grasp.drive.servo_model import (
    DEFAULT_CONFIG_PATH,
    LiftScale,
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoTarget,
    load_jacobian,
    load_lift_scale,
    load_object_selector,
    load_servo_config,
    load_target,
    load_z_target,
    save_calibration,
)
from modules.grasp.drive.servo_observation import (
    ClosableFrameSource,
    FrameSource,
    measure_offset,
    select_object,
)
from modules.grasp.runtime import Grasp
from modules.motor.config import load_bus_config as load_motor_bus_config


def _say(message: str) -> None:
    print(f"  {message}")


def _open_source(arguments, *, vision_settings) -> ClosableFrameSource:
    """按参数开一个观测源；调用方负责 ``close()``。

    **本地分支收的是已经读好的快照**（``vision_settings``），不在这里再读一次配置：
    覆盖值（``--camera`` 等）在 main 里就已经落到快照上了，这里再读一遍会把它们丢掉。
    TCP 分支**不读、也不校验**本地视觉配置与摄像头——远端路线跟本机模型没关系，
    所以调用方对 TCP 传 ``vision_settings=None``。
    """
    if arguments.tcp:
        from modules.camera_link.drive.link import (
            CameraLink,
            CameraLinkError,
            load_camera_link_config,
        )

        config = load_camera_link_config()
        port = config.port if arguments.listen_port is None else arguments.listen_port
        try:
            link = CameraLink(config=config, port=port).start()
        except CameraLinkError as error:
            raise SystemExit(f"起不了接收链路：{error}") from error
        print(f"接收链路已监听 {link.host}:{link.port}（等远端镜头连上来）")
        return link

    from modules.vision.runtime import Vision

    # ``--camera`` 早已落进快照（见 main），所以这里直接问快照要序号：
    # 再读一次命令行等于给"覆盖没生效"留后门。
    camera_id = vision_settings.camera_id("windows" if os.name == "nt" else "jetson")
    source = Vision._open_config(vision_settings)
    print(f"已打开摄像头 {camera_id}，观测来自本机 MaterialLocator")
    return source


def run_watch(*, source: FrameSource, selector: ObjectSelector, config: ServoConfig) -> int:
    """只看不动：把每一帧的读数实时打出来。"""
    print("watch：只读不发帧。Ctrl+C 退出。\n")
    try:
        while True:
            sighting = _read_one(source, selector, config)
            extra = ""
            if sighting.color is not None:
                extra += f" {sighting.color}"
            if sighting.confidence is not None:
                extra += f" conf={sighting.confidence:.2f}"
            print(
                f"物块 {sighting.material_id}{extra}："
                f"xyz=({sighting.x_mm:+.2f}, {sighting.y_mm:+.2f}, {sighting.z_mm:.2f}) mm；"
                f"水平偏离镜头轴 {sighting.planar_distance_mm:.2f} mm"
            )
    except KeyboardInterrupt:
        print("\nwatch 结束。")
        return 0
    except ServoError as problem:
        print(f"读取失败：{problem}")
        return 1


def _read_one(source: FrameSource, selector: ObjectSelector, config: ServoConfig):
    from modules.grasp.drive.eye_in_hand import select_object

    frame = source.frame(timeout=config.frame_timeout_seconds)
    if frame is None:
        raise ServoError("等不到新的观测；确认摄像头有画面、且物块在视野里")
    return select_object(frame, selector)


def run_status(
    *,
    source: FrameSource,
    selector: ObjectSelector,
    target: Optional[ServoTarget],
    jacobian: Optional[ServoJacobian],
    config: ServoConfig,
) -> int:
    """只算不动：打印当前读数、目标读数、误差，以及打算怎么走。"""
    print("status：只读不发帧。Ctrl+C 退出。\n")
    if target is None:
        print("还没标定目标读数（tx, ty）；先跑 --set-target。")
    if jacobian is None:
        print("还没标定雅可比；先跑 --calibrate。（没有它算不出要走的脉冲）")
    try:
        while True:
            offset = measure_offset(source, selector, config)
            line = f"物块读数 ({offset[0]:+.2f}, {offset[1]:+.2f}) mm"
            if target is not None:
                error_x = offset[0] - target.x_mm
                error_y = offset[1] - target.y_mm
                line += (
                    f"｜目标 ({target.x_mm:+.2f}, {target.y_mm:+.2f})"
                    f"｜误差 {math.hypot(error_x, error_y):.2f} mm"
                )
                if jacobian is not None:
                    try:
                        base_pulses, radial_pulses = jacobian.pulses_for(error_x, error_y)
                    except ServoError as problem:
                        line += f"｜算不出脉冲：{problem}"
                    else:
                        line += f"｜应走 底座 {base_pulses:+.0f}、伸缩 {radial_pulses:+.0f} 脉冲"
            print(line)
    except KeyboardInterrupt:
        print("\nstatus 结束。")
        return 0
    except ServoError as problem:
        print(f"读取失败：{problem}")
        return 1


def run_set_target(
    *, source: FrameSource, selector: ObjectSelector, config: ServoConfig, centred: bool = False
) -> int:
    """把**当前**画面读数记成目标 ``(tx, ty)``。

    ``centred=True`` 时不读画面，直接把目标设成 ``(0, 0)``——那是"相机光轴正好穿过
    抓取点"这个假设。相机正装在爪子上方时它就是对的，而且**不用手工摆位**：
    先按这个假设跑，如果对准之后爪子明显偏了，再改用下面那种手工记录法。

    手工记录法：把物块放到爪子附近，用手把爪子挪到"正好套住物块"的位置，
    然后记录此刻的画面读数——那个读数**就是**抓取点在相机系里的位置。
    """
    if centred:
        target = ServoTarget(x_mm=0.0, y_mm=0.0)
        path = save_calibration(target=target)
        print(f"已按「相机光轴穿过抓取点」记下 {target.summary()}，写入 {path}")
        print("如果之后对准了但爪子明显偏，说明相机是偏装的：")
        print("  用手把爪子挪到正好套住物块的位置，再跑 --set-target 重新记录。")
        return 0

    print("把物块放到爪子附近，然后用手把爪子挪到「正好套住物块」的位置。")
    print("挪好之后按回车记录当前读数。")
    input("准备好了按回车：")
    try:
        offset = measure_offset(source, selector, config)
    except ServoError as problem:
        print(f"读取失败：{problem}")
        return 1
    target = ServoTarget(x_mm=offset[0], y_mm=offset[1])
    path = save_calibration(target=target)
    print(f"已记录 {target.summary()}，写入 {path}")
    print("接着跑 --calibrate 实测雅可比。")
    return 0


def run_calibrate(
    *, arm: RobotArm, source: FrameSource, selector: ObjectSelector, config: ServoConfig
) -> int:
    """实测雅可比：让程序自己动两下，看画面挪了多少。"""
    print(f"将用 {config.probe_pulses} 脉冲一段、每轴走两段同向位移，然后退回原位。")
    print("请确认机械臂四周没有障碍物。\n")
    try:
        jacobian = probe_jacobian(arm, source, selector, config, on_message=_say)
    except ServoError as problem:
        print(f"标定失败：{problem}")
        return 1
    path = save_calibration(jacobian=jacobian)
    print(f"\n{jacobian.summary()}")
    print(f"已写入 {path}")
    print("接着跑 --status 看数字对不对，再跑真动。")
    return 0


def run_align(*, grasp: Grasp) -> int:
    """真动：闭环把物块伺服到目标读数（对准就停）。

    动作由 :meth:`modules.grasp.runtime.Grasp.align_xy` 执行——它包含"配了
    ``hover_z_mm`` 就先悬停"这一步，再按原规则检查安全高度。这里只负责提示与打印；
    提示要用的几个值取自**本会话已有的**配置与值对象（工具内部使用，不是上层接口）。
    """
    config = grasp._config
    selector = grasp._selector
    target = grasp._target
    jacobian = grasp._jacobian
    print("开始 x/y 对准（只动 底座旋转 与 伸缩，高度不动）")
    print(f"  {selector.describe()}")
    print(f"  {target.summary()}；{jacobian.summary()}")
    print(f"  容差 {config.tolerance_mm:.1f} mm，最多测量 {config.max_iterations} 次")
    print(f"  每轴脉冲预算 {config.max_total_pulses}（没有标定行程，这是唯一的兜底）")
    if config.hover_z_mm is not None:
        print(
            f"  先把悬停高度伺服到 z={config.hover_z_mm:.1f} mm"
            f"（±{config.z_tolerance_mm:.1f}）——它决定下面那次固定脉冲下降的起点"
        )
    print()

    try:
        result = grasp.align_xy()
    except ServoError as problem:
        print(f"\nx/y 对准失败：{problem}")
        return 1
    print(f"\n{result.summary()}")
    print(
        f"下一步：--align-z（升降下降 {config.z_descend_pulses} 脉冲 + 伸缩后缩 "
        f"{config.extension_retract_pulses} 脉冲），然后合上夹爪。"
    )
    return 0


def run_follow(
    *,
    arm: RobotArm,
    source: FrameSource,
    selector,
    target,
    jacobian,
    config,
    lift_scale: Optional[LiftScale] = None,
) -> int:
    """跟随：物块被移动时持续把它伺服回目标读数。

    每一轮都会重新测量并按当前误差算脉冲；物块不动时误差本来就在容差内，
    于是那一轮只花一次测量、不发任何动作帧。所以"静止"和"被推动"是同一段代码。

    **安全高度每一轮都查，但用单帧快判**：跟随期间这套代码不动高度，可是别人可能动
    （另一个终端、手去拨）。所以判据一点都不放宽——只要"单帧读到的 z 比安全高度还高
    一个余量"就直接放行，否则退回一次完整的停稳测量。绝大多数轮次因此只花一帧
    （约 0.22 s）而不是一次停稳测量（约 0.54 s）。
    """
    print("开始跟随（只动 底座旋转 与 伸缩，高度不动）")
    print(f"  {target.summary()}；{jacobian.summary()}")
    print(f"  容差 {config.tolerance_mm:.1f} mm；物块移动就会跟着动，Ctrl+C 退出\n")

    # 悬停高度只在**开始**做一次：跟随期间高度不该动，每轮都调就成了自己跟自己打架。
    if config.hover_z_mm is not None:
        print(f"  先把悬停高度伺服到 z={config.hover_z_mm:.1f} mm（只做一次）")
        try:
            hover_above_object(
                arm, source, selector, config, scale=lift_scale, on_message=_say
            )
        except ServoError as problem:
            print(f"  悬停失败：{problem}（继续跟随）")

    rounds = 0
    moved_rounds = 0
    limited_notice = False
    try:
        while True:
            rounds += 1
            started = time.monotonic()
            try:
                result = align_x_y(
                    arm,
                    source,
                    selector,
                    target,
                    jacobian,
                    config,
                    quick_safe_lift=True,
                    lift_scale=lift_scale,
                )
            except ServoError as problem:
                # 跟随过程中"这一帧没看到物块"是常见的（手挡住、移出视野），
                # 不该让整条循环退出——报一句继续。
                print(f"  第 {rounds} 轮：{problem}")
                time.sleep(0.2)
                continue
            elapsed = time.monotonic() - started
            if result.base_pulses or result.radial_pulses:
                moved_rounds += 1
                print(
                    f"  第 {rounds} 轮：误差 {result.error_mm:.2f} mm -> "
                    f"底座 {result.base_pulses:+d}、伸缩 {result.radial_pulses:+d} 脉冲"
                    f"（{elapsed * 1000:.0f} ms，{result.iterations} 次测量）"
                )
            elif result.resolution_limited and not limited_notice:
                limited_notice = True
                print(
                    f"  误差 {result.error_mm:.2f} mm 已小于背隙地板"
                    f"（{config.min_step_mm:.2f} mm）：这是这台机器能做到的极限，不再微调。"
                    "想更准就得先把背隙调小（或把 min_step_mm 调小试试）。"
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print(f"\n跟随结束：共 {rounds} 轮，其中 {moved_rounds} 轮发了动作。")
        print(
            f"要对准之后往下走：--align-z（升降下降 {config.z_descend_pulses} 脉冲 + "
            f"伸缩后缩 {config.extension_retract_pulses} 脉冲）"
        )
        return 0


def run_teach(
    *,
    arm: RobotArm,
    source: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    step_pulses: int,
) -> int:
    """手动示教：小步挪动机械臂，读数实时显示，按 ``t`` 把当前读数记成目标。

    为什么需要它：目标读数 ``(tx, ty)`` 是"爪子抓取点在相机坐标系里的位置"，
    它是一个**物理安装常数**——只有把爪子真的挪到物块正上方，才知道那个读数是多少。
    程序推不出来，也不该猜：猜错的话，"对准"之后爪子会稳定地停在物块旁边。

    命令：

    - ``a`` / ``b``：底座逆时针 / 顺时针走一步；
    - ``c`` / ``d``：伸缩前伸 / 后缩走一步；
    - 纯数字：改每步的脉冲数；
    - 字母后跟数字（如 ``b5``）：连续走 5 步；
    - ``t``：把**当前读数**记成目标 ``(tx, ty)``；
    - ``q``：退出。
    """
    step = max(1, step_pulses)
    print("手动示教：把爪子挪到物块正上方，然后按 t 记录目标读数。")
    print("  a/b = 底座逆时针/顺时针    c/d = 伸缩前伸/后缩")
    print(f"  数字 = 改每步脉冲数（当前 {step}）；字母后跟数字 = 连走多步（如 b5）")
    print("  t = 记录当前读数为目标     q = 退出\n")

    while True:
        try:
            offset = measure_offset(source, selector, config)
            reading = f"({offset[0]:+7.2f}, {offset[1]:+6.2f}) mm"
        except ServoError as problem:
            reading = f"看不到物块（{problem}）"
        print(f"当前读数 {reading}   每步 {step} 脉冲")
        try:
            raw = input("命令：").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        if raw == "q":
            return 0
        if raw == "t":
            try:
                offset = measure_offset(source, selector, config)
            except ServoError as problem:
                print(f"  记不了：{problem}")
                continue
            target = ServoTarget(x_mm=offset[0], y_mm=offset[1])
            path = save_calibration(target=target)
            print(f"  已记录 {target.summary()}，写入 {path}")
            print("  接着跑 --calibrate 实测雅可比，再跑 --follow 跟随。")
            continue
        if raw.isdigit():
            step = max(1, int(raw))
            continue

        action = raw[0]
        if action not in "abcd":
            print("  不认识的命令；可用 a/b/c/d、数字、t、q")
            continue
        repeats = 1
        if len(raw) > 1:
            if not raw[1:].isdigit():
                print("  连走次数要是数字，例如 b5")
                continue
            repeats = max(1, int(raw[1:]))
        move = {
            "a": arm.rotate_ccw,
            "b": arm.rotate_cw,
            "c": arm.move_forward,
            "d": arm.move_backward,
        }[action]
        for _ in range(repeats):
            move(step, wait=True)
        time.sleep(0.3)  # 让机械臂停稳，读数才是停稳后的
        print(f"  已走 {repeats} 步 x {step} 脉冲")


def run_pick_ready(
    *,
    grasp: Grasp,
    z_pulses: Optional[int] = None,
    retract_pulses: Optional[int] = None,
) -> int:
    """**一键就位**：悬停高度 → 安全高度 → x/y 对准 → z 下降 → 伸缩后缩。

    主链由 :meth:`modules.grasp.runtime.Grasp.align_and_descend` 执行：x/y 走闭环，
    z 与后缩走配置里的两个脉冲数，每一段的耗时都会打出来——慢在哪一段一眼能看见
    （定位一次 178 ms、停稳测量一次约 537 ms，所以省"段数"比省"动作时间"更值钱）。
    """
    config = grasp._config
    print("一键就位：悬停高度 → 安全高度 → x/y 对准 → z 下降 → 伸缩后缩")
    print(f"  {grasp._selector.describe()}")
    print(f"  {grasp._target.summary()}；{grasp._jacobian.summary()}")
    print(
        f"  容差 {config.tolerance_mm:.1f} mm（粗调阶段按 "
        f"{config.coarse_settle_tolerance_mm:.1f} mm 判停稳）"
    )
    if config.hover_z_mm is not None:
        print(
            f"  先悬停到 z={config.hover_z_mm:.1f} mm（±{config.z_tolerance_mm:.1f}）"
            "——它决定下降的起点"
        )
    else:
        print("  没配 hover_z_mm：不主动调起始高度（只保证不低于 safe_lift_z_mm）")
    print(
        f"  下降 {config.z_descend_pulses} 脉冲（升降/6 号）、后缩 "
        f"{config.extension_retract_pulses} 脉冲（伸缩/7 号）"
        "——两个数在 modules/grasp/eye_in_hand.json，"
        "--z-pulses / --retract-pulses 只覆盖这一次"
    )
    print(f"  下降前要求平面误差 ≤ {config.z_center_tolerance_mm:.1f} mm\n")

    try:
        result = grasp.align_and_descend(
            z_pulses=z_pulses, retract_pulses=retract_pulses
        )
    except ServoError as problem:
        print(f"\n就位失败：{problem}")
        return 1
    print(f"\n{result.summary()}")
    print("下一步：合上夹爪完成抓取。")
    print(f"  要退回上方：--raise {result.descent.pulses}（升降）")
    return 0


def run_descend_z(
    *,
    arm: RobotArm,
    source: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    z_pulses: Optional[int] = None,
    retract_pulses: Optional[int] = None,
) -> int:
    """z 对准（脉冲版）：**确认平面已对准 → 升降下降 → 伸缩后缩**。

    两步都是开环的固定脉冲数，来自 ``modules/grasp/eye_in_hand.json``：

    - ``z_descend_pulses``：升降（6 号电机）下降多少脉冲到夹持高度；
    - ``extension_retract_pulses``：伸缩（7 号电机）再后缩多少脉冲。

    换物块高度、换夹爪就改这两个数，不用重标任何东西。``--z-pulses`` /
    ``--retract-pulses`` 只覆盖**这一次**，用来试数；试好了写回配置文件。
    """
    down = config.z_descend_pulses if z_pulses is None else z_pulses
    back = config.extension_retract_pulses if retract_pulses is None else retract_pulses
    print(f"z 对准（脉冲）：升降/6 号下降 {down} 脉冲，然后伸缩/7 号后缩 {back} 脉冲")
    print(
        "  两个数在 modules/grasp/eye_in_hand.json："
        "z_descend_pulses / extension_retract_pulses"
        "（--z-pulses / --retract-pulses 只覆盖这一次）"
    )
    print(
        f"  下降前要求平面误差 ≤ {config.z_center_tolerance_mm:.1f} mm"
        "——脉冲数只保证往下走了这么多，不保证下面正好是物块"
    )
    print(
        "  这是**相对**动作：跑过一次之后先用 --raise 抬回上方再跑第二次，"
        "否则会在原处再降这么多\n"
    )

    try:
        descent = descend_z(
            arm,
            source,
            selector,
            config,
            pulses=z_pulses,
            scale=load_lift_scale(),
            on_message=_say,
        )
    except ServoError as problem:
        # 三种失败（看不到物块、平面没对准、脉冲数不合法）都发生在发动作之前。
        print(f"\n下降失败（没有发出动作）：{problem}")
        return 1
    print(f"\n{descent.summary()}")

    try:
        moved = retract_extension(arm, config, pulses=retract_pulses, on_message=_say)
    except ServoError as problem:
        print(f"\n后缩失败：{problem}")
        return 1
    print(f"后缩 {moved} 脉冲完成。")

    print("下一步：合上夹爪完成抓取。")
    print(f"  要退回上方：--raise {down}（升降）；要伸回去：--teach 里按 c（伸缩）。")
    return 0


def run_align_z_visual(
    *,
    arm: RobotArm,
    source: FrameSource,
    selector: ObjectSelector,
    config: ServoConfig,
    recalibrate: bool = False,
) -> int:
    """z 对准（视觉伺服版）：驱动升降，让物块深度落到 ``z_target_mm``。

    现场用的是脉冲版 :func:`run_descend_z`；这一版留给"固定脉冲数不合适"的场合——
    它拿物块深度当反馈，代价是要先标定升降灵敏度，而且下降途中检测容易丢目标。

    起手式是**先找、找不到就抬起来**：物块太近时既模糊又掉出工作距离窗口，
    检测会直接失效；抬起来往往就看见了。
    """
    target = load_z_target()
    if target is None:
        print("还没设 z 目标；在 modules/grasp/eye_in_hand.json 里填 z_target_mm")
        return 1

    print(f"z 目标 {target:.1f} mm；下降护栏 z_min_mm={config.z_min_mm:.1f} mm")
    print(f"工作距离窗口 {config.z_min_mm:.0f}–{config.z_max_mm:.0f} mm")
    print(
        f"下降前要求平面误差 ≤ {config.z_center_tolerance_mm:.1f} mm"
        "（z 看不见爪子有没有对准，所以这一步先自己查一遍）\n"
    )

    # **先确认平面已经对准。** 顺序反了就是让爪子斜着压下去——现场因此差点压坏爪子。
    try:
        offset = measure_offset(source, selector, config)
    except ServoError as problem:
        print(f"看不到物块：{problem}")
        return 1
    planar = math.hypot(offset[0], offset[1])
    if planar > config.z_center_tolerance_mm:
        print(
            f"物块离光轴 {planar:.1f} mm，还没对准（要求 ≤ "
            f"{config.z_center_tolerance_mm:.1f} mm）。\n"
            "先做 x/y 对准，再回来做 z：\n"
            "  python -m modules.grasp.test.manual.align_xy            # 对准一次\n"
            "然后重跑 --align-z。"
        )
        return 1

    if not find_object(arm, source, selector, config, on_message=_say):
        print(
            f"抬升 {config.find_max_steps} 次仍然看不到物块；"
            "把物块往镜头下放一点、或确认它在画面里"
        )
        return 1
    print()

    scale = None if recalibrate else load_lift_scale()
    if scale is None:
        print("标定升降灵敏度（走两段同向抬升，然后退回原位）")
        try:
            scale = probe_lift_scale(arm, source, selector, config, on_message=_say)
        except ServoError as problem:
            print(f"标定失败：{problem}")
            return 1
        path = save_calibration(lift_scale=scale)
        print(f"已写入 {path}\n")

    try:
        result = align_z(
            arm, source, selector, target, scale, config, on_message=_say
        )
    except ServoError as problem:
        print(f"\nz 对准失败：{problem}")
        return 1
    print(f"\n{result.summary()}")
    print("下一步：合上夹爪完成抓取。")
    return 0


def run_raise(*, arm: RobotArm, pulses: int) -> int:
    """只驱动升降：正数抬升、负数下降。

    转底盘之前先用它把爪子抬到物块上方——**低位转底盘会撞到物块**。
    它不读摄像头、不转底盘，所以是这个阶段最安全的动作。
    """
    if pulses == 0:
        print("--raise 需要非零脉冲数（正=抬升，负=下降）")
        return 1
    direction = "抬升" if pulses > 0 else "下降"
    print(f"{direction} {abs(pulses)} 脉冲（不转底盘、不读摄像头）")
    seconds = _move_lift(arm, pulses)
    print(f"完成，估算耗时 {seconds:.2f} s")
    print("（升降的绝对高度没有标定，所以这里只报脉冲数）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="让爪子正下方对准物块（只动平面）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true", help="只读：实时打印物块读数，不开机械臂串口")
    mode.add_argument("--status", action="store_true", help="只读：打印读数、误差与打算走的脉冲")
    mode.add_argument("--set-target", action="store_true", help="把当前画面读数记成目标 (tx, ty)")
    mode.add_argument(
        "--target-centred",
        action="store_true",
        help="直接把目标设成 (0,0)：假设相机光轴正好穿过抓取点",
    )
    mode.add_argument(
        "--teach",
        action="store_true",
        help="手动示教：小步挪动机械臂，按 t 记录当前读数为目标（推荐）",
    )
    mode.add_argument("--calibrate", action="store_true", help="实测雅可比（两个轴各动两段）")
    mode.add_argument("--follow", action="store_true", help="跟随：物块移动时持续对准")
    mode.add_argument(
        "--align-and-descend",
        action="store_true",
        help="一键就位：安全高度 -> x/y 对准 -> z 下降 -> 伸缩后缩（每一步都报耗时）",
    )
    mode.add_argument(
        "--align-z",
        action="store_true",
        help="z 对准（脉冲版）：平面已对准后，升降下降 z_descend_pulses、伸缩后缩 "
        "extension_retract_pulses（都在 eye_in_hand.json 里）",
    )
    mode.add_argument(
        "--align-z-visual",
        action="store_true",
        help="z 对准（视觉伺服版）：驱动升降让物块深度到 z_target_mm；要先标定升降灵敏度",
    )
    parser.add_argument(
        "--z-pulses",
        type=int,
        default=None,
        metavar="PULSES",
        help="只覆盖 --align-z 这次的下降脉冲数（不改配置文件）",
    )
    parser.add_argument(
        "--hover-z",
        type=float,
        default=None,
        metavar="MM",
        help="只覆盖这次的悬停高度（镜头到物块的距离，毫米；不改配置文件）",
    )
    parser.add_argument(
        "--retract-pulses",
        type=int,
        default=None,
        metavar="PULSES",
        help="只覆盖 --align-z 这次的后缩脉冲数（不改配置文件）",
    )
    parser.add_argument(
        "--recalibrate-lift",
        action="store_true",
        help="--align-z-visual 时强制重标升降灵敏度（默认只在没标过时标）",
    )
    parser.add_argument(
        "--raise",
        dest="raise_pulses",
        type=int,
        default=None,
        metavar="PULSES",
        help="只驱动升降：正数抬升、负数下降。转底盘前先用它抬起来（低位转会撞到物块）",
    )
    parser.add_argument("--step", type=int, default=100, help="--teach 的默认步进脉冲数（默认 100）")
    parser.add_argument("--tcp", action="store_true", help="观测改从 TCP 远端镜头来（默认用本机摄像头）")
    parser.add_argument("--camera", type=int, default=None, help="摄像头序号；省略则读 modules/vision/config.json")
    parser.add_argument("--port", default=None, help="机械臂步进总线串口；省略则用 modules/motor/config.json")
    parser.add_argument("--listen-port", type=int, default=None, help="--tcp 时接收镜头端的本地端口")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """菜单和命令行共用的入口。

    ## 资源创建顺序（本文件的核心约束）

    普通动作（默认对准 / ``--align-and-descend``）走
    :meth:`modules.grasp.runtime.Grasp._open_settings` 这一条**唯一**的设备创建路径：
    它先校验 ``target`` / ``jacobian``，再取观测源、开机械臂、使能。所以这里
    **不能在进入会话之前提前** ``_open_source()`` 或建 ``RobotArm``——那等于把校验
    又挪到设备之后。

    其余工具模式（watch / status / set-target / target-centred / teach / calibrate /
    follow / align-z / align-z-visual / raise）各有自己的前置条件，仍然按原来的低层
    流程走：source 由该分支自己创建和关闭，机械臂由 ``RobotArm._from_settings``
    装配成一次就绪的会话（已经使能），收尾只调一次 ``arm.close()``。
    **不**给它们加统一的标定门禁——那会造出"缺标定所以不能跑标定"的循环。
    """
    arguments = build_parser().parse_args(argv)

    try:
        config = load_servo_config()
        selector = load_object_selector()
        target = load_target()
        jacobian = load_jacobian()
        # 升降灵敏度只用来"抬到安全高度时一步算到位"和打印预计深度；没标定就是 None。
        lift_scale = load_lift_scale()
    except ServoError as problem:
        print(f"配置有问题：{problem}")
        return 1

    if arguments.hover_z is not None:
        # 只覆盖这一次：配置文件不动。replace 会跑一遍 __post_init__，
        # 所以"悬停高度低于安全高度"这类错误在这里就会被拦下来。
        try:
            config = dataclasses.replace(config, hover_z_mm=arguments.hover_z)
        except ValueError as problem:
            print(f"--hover-z 不合法：{problem}")
            return 1
        print(f"本次悬停高度改为 z={config.hover_z_mm:.1f} mm（只覆盖这一次）")

    if config.min_step_mm > config.tolerance_mm:
        print(
            f"警告：min_step_mm={config.min_step_mm:.2f} 大于 "
            f"tolerance_mm={config.tolerance_mm:.2f}——背隙地板比容差还大，"
            "闭环永远报不了「对准」，只会在分辨率极限上收工。"
            "要么放宽 tolerance_mm，要么把 min_step_mm 调小。\n"
        )

    # 视觉快照只在**非 TCP 路线**读：TCP 用的是远端镜头，本机模型、摄像头和视觉配置
    # 都不参与，读它只会把"远端能跑"变成"本地配置写错就跑不了"。
    vision_settings = None
    if not arguments.tcp:
        from modules.vision.runtime import _load_settings, _override_settings

        try:
            vision_settings = _override_settings(
                _load_settings(), camera_id=arguments.camera
            )
        except (ValueError, FileNotFoundError) as problem:
            print(f"视觉配置有问题：{problem}")
            return 1

    if arguments.watch or arguments.status or arguments.set_target or arguments.target_centred:
        return _run_read_only_mode(arguments, vision_settings, selector, target, jacobian, config)

    if arguments.calibrate or arguments.teach or arguments.align_z or arguments.align_z_visual:
        return _run_low_level_mode(arguments, vision_settings, config)

    if arguments.raise_pulses is not None:
        return _run_raise_mode(arguments, vision_settings)

    # ---------------------------------------------------------- 普通动作
    #
    # 到这一步剩下的都是普通动作：默认对准、--align-and-descend、--follow。
    # 标定没做完就在**打开串口之前**说清楚：菜单里直接按 21 的人多半还没标定，
    # 这时候去开硬件毫无意义，还会让人以为程序坏了。
    if (target is None or jacobian is None) and not arguments.follow:
        print("还不能真动，先做两步标定（都很快）：\n")
        if target is None:
            print("  1) **手动示教**（推荐）：小步挪动机械臂，把爪子挪到物块正上方，")
            print("     程序实时显示读数，按 t 把当前读数记成目标：")
            print("     python -m modules.grasp.test.manual.align_xy --teach")
            print("     （如果相机是正装在抓取点上方的，也可以用 --target-centred 直接设成 (0,0)）")
        if jacobian is None:
            step = "  2)" if target is None else "  1)"
            print(f"{step} 让程序自己动两下，实测「每脉冲让画面挪多少毫米」：")
            print("     python -m modules.grasp.test.manual.align_xy --calibrate")
        print("\n标定完先跑 --status 核对数字（只看不动），再跑 --follow 跟随。")
        return 1

    if arguments.follow:
        return _run_follow_mode(arguments, vision_settings, config)

    # 普通动作统一进同一个会话：读好 arm_config / port，再把它们交给 _open_settings。
    port = arguments.port or load_motor_bus_config().port
    print(f"打开机械臂总线 {port}")
    try:
        grasp = Grasp._open_settings(
            config=config,
            selector=selector,
            target=target,
            jacobian=jacobian,
            lift_scale=lift_scale,
            arm_config=load_arm_config(),
            port=port,
            source_factory=lambda: _open_source(arguments, vision_settings=vision_settings),
            on_message=_say,
        )
    except (SystemExit, FileNotFoundError, RuntimeError, ServoError, ValueError) as problem:
        print(f"打不开会话：{problem}")
        return 1

    # with 拥有全部资源（机械臂 + 观测源），工具分支不再重复关一次。
    with grasp:
        if arguments.align_and_descend:
            return run_pick_ready(
                grasp=grasp,
                z_pulses=arguments.z_pulses,
                retract_pulses=arguments.retract_pulses,
            )
        return run_align(grasp=grasp)


# ------------------------------------------------------------------ 工具分支
#
# 下面几个模式各有前置条件，**不走** Grasp 的普通动作门禁；每个分支自己创建并关闭
# 它需要的资源，机械臂用 ``RobotArm._from_settings`` 装配成一次就绪的会话，
# 收尾只调一次 ``arm.close()``（停 / 失能 / 关串口都在它内部逐项试行）。


def _run_read_only_mode(arguments, vision_settings, selector, target, jacobian, config) -> int:
    """watch / status / set-target / target-centred：**不开机械臂串口**。"""
    try:
        source = _open_source(arguments, vision_settings=vision_settings)
    except (SystemExit, FileNotFoundError, RuntimeError, ServoError, ValueError) as problem:
        print(f"打不开观测源：{problem}")
        return 1
    try:
        if arguments.watch:
            return run_watch(source=source, selector=selector, config=config)
        if arguments.status:
            return run_status(
                source=source, selector=selector, target=target, jacobian=jacobian, config=config
            )
        return run_set_target(
            source=source,
            selector=selector,
            config=config,
            centred=arguments.target_centred,
        )
    finally:
        _close_quietly(source)


def _run_low_level_mode(arguments, vision_settings, config) -> int:
    """calibrate / teach / align-z / align-z-visual：需要机械臂，但各有自己的前提。"""
    try:
        source = _open_source(arguments, vision_settings=vision_settings)
    except (SystemExit, FileNotFoundError, RuntimeError, ServoError, ValueError) as problem:
        print(f"打不开观测源：{problem}")
        return 1

    selector = load_object_selector()
    port = arguments.port or load_motor_bus_config().port
    print(f"打开机械臂总线 {port}")
    # 统一生命周期：``_from_settings`` 读好配置快照、开串口并使能，是一次就绪的会话。
    arm = RobotArm._from_settings(config=load_arm_config(), port=port)
    try:
        if arguments.calibrate:
            return run_calibrate(
                arm=arm, source=source, selector=selector, config=config
            )
        if arguments.teach:
            return run_teach(
                arm=arm, source=source, selector=selector,
                config=config, step_pulses=arguments.step,
            )
        if arguments.align_z:
            return run_descend_z(
                arm=arm, source=source, selector=selector, config=config,
                z_pulses=arguments.z_pulses,
                retract_pulses=arguments.retract_pulses,
            )
        return run_align_z_visual(
            arm=arm, source=source, selector=selector, config=config,
            recalibrate=arguments.recalibrate_lift,
        )
    finally:
        # ``close`` 内部按 停 → 失能 → 关串口 逐项独立尝试（一项报错不跳过其余），
        # 所以这里只叫一次，不再自己重复 stop / disable。
        _attempt(arm.close)
        _close_quietly(source)


def _run_raise_mode(arguments, vision_settings) -> int:
    """``--raise``：只驱动升降，不读摄像头、不转底盘（原来就不开观测源）。

    这里同样不开观测源：``_move_lift`` 只需要机械臂。
    """
    port = arguments.port or load_motor_bus_config().port
    print(f"打开机械臂总线 {port}")
    arm = RobotArm._from_settings(config=load_arm_config(), port=port)
    try:
        return run_raise(arm=arm, pulses=arguments.raise_pulses)
    finally:
        # 收尾只有一处：``close`` 内部自行按 停 → 失能 → 关串口 逐项尝试。
        _attempt(arm.close)


def _run_follow_mode(arguments, vision_settings, config) -> int:
    """``--follow``：保留原来的连续跟随算法与循环。

    刻意**不**循环调用含悬停的 ``align_xy``（那会每轮重复悬停与安全检查，自己跟自己
    打架）；仍然走原来的 :func:`align_x_y` + ``quick_safe_lift=True``，并且保留它原有
    的 target / jacobian 闸门与动作行为。
    """
    target = load_target()
    jacobian = load_jacobian()
    if target is None or jacobian is None:
        print("跟随需要 target 与 jacobian；先跑 --teach / --calibrate。")
        return 1
    try:
        source = _open_source(arguments, vision_settings=vision_settings)
    except (SystemExit, FileNotFoundError, RuntimeError, ServoError, ValueError) as problem:
        print(f"打不开观测源：{problem}")
        return 1

    selector = load_object_selector()
    port = arguments.port or load_motor_bus_config().port
    print(f"打开机械臂总线 {port}")
    arm = RobotArm._from_settings(config=load_arm_config(), port=port)
    try:
        return run_follow(
            arm=arm, source=source, selector=selector,
            target=target, jacobian=jacobian, config=config,
            lift_scale=load_lift_scale(),
        )
    finally:
        # 收尾只有一处：``close`` 内部自行按 停 → 失能 → 关串口 逐项尝试。
        _attempt(arm.close)
        _close_quietly(source)


def _attempt(action) -> None:
    """尝试一项收尾；失败打印出来但不跳过后面的项。"""
    try:
        action()
    except Exception as problem:  # noqa: BLE001 - 收尾失败要看得见，也要让后面的项继续
        print(f"  收尾失败：{problem}")


def _close_quietly(resource) -> None:
    """按 ``with`` 协议关一个资源；失败打印出来，不让收尾盖住业务结果。"""
    close = getattr(resource, "close", None)
    if close is None:
        close = getattr(resource, "__exit__", None)
        if close is None:
            return
        _attempt(lambda: close(None, None, None))
        return
    _attempt(close)


if __name__ == "__main__":
    raise SystemExit(main())

"""真机入口：让爪子正下方对准物块（抓取第一步，只动平面）。

    python test.py            # 菜单选 21

## 观测从哪来

默认走**本机 USB 摄像头**：``modules/vision`` 的 ``MaterialLocator`` 直接读画面，
就是菜单 18 那条通路（你终端里看到的 `物料 1: @(543,382) …` 就是它打的）。
远端镜头（TCP 那种）用 ``--tcp``，两者对上层完全同形。

## 用法

```powershell
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

MENU_ENTRY = (21, "x/y 对准物块（对准后可 z 下降）")

import argparse
import math
import time
from typing import Optional, Sequence

from modules.arm.config import load_arm_config
from modules.arm.drive.robot_arm import RobotArm
from modules.grasp.drive.camera_source import CameraFrameSource
from modules.grasp.drive.eye_in_hand import (
    FrameSource,
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoTarget,
    _move_lift,
    align_x_y,
    align_z,
    descend_z,
    find_object,
    load_jacobian,
    load_lift_scale,
    load_object_selector,
    load_servo_config,
    load_target,
    load_z_target,
    measure_offset,
    probe_jacobian,
    probe_lift_scale,
    retract_extension,
    save_calibration,
)
from modules.motor.config import load_bus_config as load_motor_bus_config


def _say(message: str) -> None:
    print(f"  {message}")


def _open_source(arguments) -> FrameSource:
    """按参数开一个观测源；调用方负责 ``close()``。"""
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

    source = CameraFrameSource.from_vision_config(
        camera_id=arguments.camera, on_message=_say
    )
    print(f"已打开摄像头 {arguments.camera if arguments.camera is not None else '（配置里的）'}"
          f"，观测来自本机 MaterialLocator")
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


def run_align(
    *, arm: RobotArm, source: FrameSource, selector, target, jacobian, config
) -> int:
    """真动：闭环把物块伺服到目标读数（对准就停）。"""
    print("开始 x/y 对准（只动 底座旋转 与 伸缩，高度不动）")
    print(f"  {selector.describe()}")
    print(f"  {target.summary()}；{jacobian.summary()}")
    print(f"  容差 {config.tolerance_mm:.1f} mm，最多测量 {config.max_iterations} 次")
    print(f"  每轴脉冲预算 {config.max_total_pulses}（没有标定行程，这是唯一的兜底）\n")

    result = align_x_y(arm, source, selector, target, jacobian, config, on_message=_say)
    print(f"\n{result.summary()}")
    print(
        f"下一步：--align-z（升降下降 {config.z_descend_pulses} 脉冲 + 伸缩后缩 "
        f"{config.extension_retract_pulses} 脉冲），然后合上夹爪。"
    )
    return 0


def run_follow(
    *, arm: RobotArm, source: FrameSource, selector, target, jacobian, config
) -> int:
    """跟随：物块被移动时持续把它伺服回目标读数。

    每一轮都会重新测量并按当前误差算脉冲；物块不动时误差本来就在容差内，
    于是那一轮只花一次测量、不发任何动作帧。所以"静止"和"被推动"是同一段代码。
    """
    print("开始跟随（只动 底座旋转 与 伸缩，高度不动）")
    print(f"  {target.summary()}；{jacobian.summary()}")
    print(f"  容差 {config.tolerance_mm:.1f} mm；物块移动就会跟着动，Ctrl+C 退出\n")

    rounds = 0
    moved_rounds = 0
    limited_notice = False
    try:
        while True:
            rounds += 1
            started = time.monotonic()
            try:
                result = align_x_y(
                    arm, source, selector, target, jacobian, config
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
    """菜单和命令行共用的入口。"""
    arguments = build_parser().parse_args(argv)

    try:
        config = load_servo_config()
        selector = load_object_selector()
        target = load_target()
        jacobian = load_jacobian()
    except ServoError as problem:
        print(f"配置有问题：{problem}")
        return 1

    if config.min_step_mm > config.tolerance_mm:
        print(
            f"警告：min_step_mm={config.min_step_mm:.2f} 大于 "
            f"tolerance_mm={config.tolerance_mm:.2f}——背隙地板比容差还大，"
            "闭环永远报不了「对准」，只会在分辨率极限上收工。"
            "要么放宽 tolerance_mm，要么把 min_step_mm 调小。\n"
        )

    try:
        source = _open_source(arguments)
    except (SystemExit, FileNotFoundError, RuntimeError, ServoError) as problem:
        print(f"打不开观测源：{problem}")
        return 1

    try:
        if arguments.watch:
            return run_watch(source=source, selector=selector, config=config)
        if arguments.status:
            return run_status(
                source=source, selector=selector, target=target, jacobian=jacobian, config=config
            )
        if arguments.set_target or arguments.target_centred:
            return run_set_target(
                source=source,
                selector=selector,
                config=config,
                centred=arguments.target_centred,
            )

        # 标定没做完就在**打开串口之前**说清楚：菜单里直接按 21 的人多半还没标定，
        # 这时候去开硬件毫无意义，还会让人以为程序坏了。
        # 本身就是标定/对准手段的那几个模式不受这个闸限制。
        if not (
            arguments.calibrate
            or arguments.teach
            or arguments.align_z
            or arguments.align_z_visual
            or arguments.raise_pulses is not None
        ) and (target is None or jacobian is None):
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

        # 到这一步才打开机械臂串口。
        port = arguments.port or load_motor_bus_config().port
        print(f"打开机械臂总线 {port}")
        arm = RobotArm(port, config=load_arm_config())
        try:
            arm.initialize()
            arm.enable_all()
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
                if arguments.align_z_visual:
                    return run_align_z_visual(
                        arm=arm, source=source, selector=selector, config=config,
                        recalibrate=arguments.recalibrate_lift,
                    )
                if arguments.raise_pulses is not None:
                    return run_raise(arm=arm, pulses=arguments.raise_pulses)
                if arguments.follow:
                    return run_follow(
                        arm=arm, source=source, selector=selector,
                        target=target, jacobian=jacobian, config=config,
                    )
                return run_align(
                    arm=arm, source=source, selector=selector,
                    target=target, jacobian=jacobian, config=config,
                )
            finally:
                arm.stop_all()
                arm.disable_all()
        finally:
            arm.close()
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    raise SystemExit(main())

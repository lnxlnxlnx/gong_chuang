"""真机入口：让爪子正下方对准物块（抓取流程第一步，只动平面）。

    python test.py            # 菜单选 21

## 用法

```powershell
# 只看不动：打印每一帧收到的观测、算出的目标位姿、以及"要动多少"，不发任何动作帧
python -m modules.grasp.test.manual.align_xy --dry-run

# 真动：闭环对准，直到平面误差进容差
python -m modules.grasp.test.manual.align_xy

# 只验证链路：起监听、把收到的每一行原样打出来，不碰机械臂
python -m modules.grasp.test.manual.align_xy --listen-only

# 还没有平面标定时：转底座采几个点，拟合基座轴心与 θ 零点
python -m modules.grasp.test.manual.align_xy --calibrate-plane
```

## 跑之前必须准备好的三件事

1. **机械臂已经回过零**（或用别的方式拿到了绝对参考）。本臂只有相对位置模式，
   程序发的是"从当前位置转多少度、伸多少毫米"，它自己不知道绝对位置。
2. **``modules/arm/config.json`` 的 ``limits.radial_mm`` 已按实验 F 标定。**
   没有它这一维就没有任何软件保护——程序会在**打开串口之前**拒绝往下走。
3. **``modules/vision/sighting_plane.json`` 的 ``plane`` 三项已填**
   （``--calibrate-plane`` 就是干这个的）。

## 为什么不做回零

回零牵涉"零点设在哪、单圈会不会落到错误的圈"（见 ``modules/grasp/docs/抓取坐标链路.md``
第 4 节），那是操作纪律，不该由这个脚本替你做决定。它只提供 ``--park`` 之外的
一件事：把平面这一件事做对。
"""

from __future__ import annotations

MENU_ENTRY = (21, "x/y 对准物块")

import argparse
import json
import sys
from typing import List, Optional, Sequence, Tuple

from modules.arm.config import ArmConfig, load_arm_config
from modules.arm.drive.robot_arm import RobotArm
from modules.camera_link.drive.link import CameraLink, CameraLinkError, load_camera_link_config
from modules.grasp.config import GraspConfig, load_grasp_config
from modules.grasp.drive.run_xy import (
    GripperMarker,
    XYAligner,
    XYAlignConfig,
    XYAlignError,
    load_gripper_marker,
    load_xy_align_config,
)
from modules.motor.config import load_bus_config as load_motor_bus_config
from modules.vision.drive.calibration import PlaneCalibration, fit_plane_calibration, CalibrationSample
from modules.vision.drive.sighting import MaterialSighting, SightingParseError, parse_sighting
from modules.vision.drive.sighting_plane import SightingPlane, load_sighting_plane

#: 平面标定时底座每一步转多少度。取 30° 是因为转太小拟合出的圆心会很飘。
CALIBRATION_STEP_DEGREES = 30.0

#: 平面标定至少走这么多步（含起点）。三个点定一个圆，再留一个做交叉验证。
CALIBRATION_STEPS = 5


def missing_setup(
    *,
    arm: ArmConfig,
    plane: Optional[SightingPlane],
) -> List[str]:
    """列出还不能真动的原因；空列表表示可以开跑。

    扫描的顺序就是现场该做的顺序，而且**全部在打开串口之前**给出——
    缺一项都会变成一句难懂的运行时报错，不如一次说清。
    """
    problems: List[str] = []
    if arm.radial_limit is None:
        problems.append(
            "机械臂：limits.radial_mm 没填（实验 F）——伸缩轴没有任何软件保护，"
            "本脚本拒绝动它"
        )
    if plane is None:
        problems.append(
            "视觉：modules/vision/sighting_plane.json 的 plane 三项还没填；"
            "先跑 --calibrate-plane"
        )
    elif plane.plane.rms_mm > 5.0:
        problems.append(
            f"视觉：平面标定残差 RMS={plane.plane.rms_mm:.1f} mm 偏大（>5 mm），"
            "抓取精度会被它直接拖住；先重做 --calibrate-plane"
        )
    return problems


def _load_marker() -> GripperMarker:
    try:
        return load_gripper_marker()
    except XYAlignError as error:
        raise SystemExit(f"配置不完整：{error}") from error


def _load_plane_or_none() -> Optional[SightingPlane]:
    try:
        return load_sighting_plane()
    except ValueError:
        return None


def run_listen_only(*, link: CameraLink) -> int:
    """只验证链路：把收到的每一行原样打出来，并给出解析结果。"""
    print(f"正在监听 {link.host}:{link.port}（Ctrl+C 退出）")
    print("请让镜头端连上来并开始发观测。收到的原始行会原样打印。\n")
    try:
        while True:
            sighting = link.latest(timeout=5.0)
            if sighting is None:
                if link.error:
                    print(f"链路错误：{link.error}")
                    return 1
                print(
                    f"  … 5 秒内没有新观测（已收 {link.received} 条、"
                    f"拒绝 {link.rejected_count} 条、连接 {link.connections} 次）"
                )
                continue
            print(f"  收到：{sighting.summary()}")
            print(f"        原始：{sighting.raw}")
            for rejected in link.rejected[-3:]:
                print(f"  （解析不了的行）{rejected}")
    except KeyboardInterrupt:
        print(f"\n结束：共收到 {link.received} 条，拒绝 {link.rejected_count} 条。")
    return 0


def run_dry_run(
    *,
    link: CameraLink,
    plane: SightingPlane,
    marker: GripperMarker,
    align_config: XYAlignConfig,
    grasp_config: GraspConfig,
    arm_config: ArmConfig,
) -> int:
    """只算不动：打印"如果现在动，会怎么动"。

    这一档的价值在于**第一次上真机时先看数**：目标位姿对不对、要转的角度合不合理、
    有没有越过行程。看对了再去掉 ``--dry-run``。
    """
    from modules.grasp.drive.align import AlignmentError, solve_planar_alignment
    from modules.arm.drive.kinematics import CylindricalPose, check_reachable

    print("dry-run：只解算不发帧。Ctrl+C 退出。\n")
    try:
        while True:
            frame = link.frame(timeout=10.0, max_age_seconds=2.0)
            if frame is None:
                if link.error:
                    print(f"链路错误：{link.error}")
                    return 1
                print("  … 没有新观测")
                continue
            try:
                marker_sighting, material = _pick_marker_and_material(frame, marker)
            except XYAlignError as error:
                print(f"  {error}")
                continue

            problem = plane.depth_problem(material)
            if problem is not None:
                print(f"  警告：{problem}")

            base_now, radial_now = plane.polar(marker_sighting)
            base_want, radial_want = plane.polar(material)
            current = CylindricalPose(base_now, radial_now, 0.0)
            target = CylindricalPose(base_want, radial_want, 0.0)
            problems = check_reachable(
                target, base_limit=arm_config.base_limit, radial_limit=arm_config.radial_limit
            )
            print(f"\n物料 {material.material_id}（{material.color}）：{material.summary()}")
            print(f"  爪子现在：θ={base_now:+.2f}°，r={radial_now:.2f} mm")
            print(f"  物料要求：θ={base_want:+.2f}°，r={radial_want:.2f} mm")
            if problems:
                print(f"  越界：{'；'.join(problems)}")
                continue
            try:
                solution = solve_planar_alignment(current, target)
            except AlignmentError as error:
                print(f"  解不出来：{error}")
                continue
            print(f"  应当执行：{solution.summary()}")
            print(
                f"  （容差 {align_config.tolerance_mm:.1f} mm；"
                f"缩回半径 {grasp_config.retract_radius_mm}）"
            )
    except KeyboardInterrupt:
        print("\ndry-run 结束。")
    return 0


def _pick_marker_and_material(
    frame: Sequence[MaterialSighting], marker: GripperMarker
) -> Tuple[MaterialSighting, MaterialSighting]:
    """从一帧里挑出靶标与物料；复用 ``XYAligner`` 的判别规则。"""
    markers = [item for item in frame if marker.matches(item)]
    materials = [item for item in frame if not marker.matches(item)]
    if not markers:
        raise XYAlignError(
            f"这一帧里没有爪上靶标（{marker.describe()}）；"
            f"收到 {len(frame)} 条：{'、'.join(f'物料 {i.material_id}({i.color})' for i in frame) or '空'}"
        )
    if not materials:
        raise XYAlignError("这一帧里只有爪上靶标，没有物料")
    if len(materials) > 1:
        raise XYAlignError(
            "这一帧里有多个物料："
            + "、".join(f"物料 {i.material_id}({i.color})" for i in materials)
            + "；一次只对准一个"
        )
    return markers[-1], materials[0]


def run_align(
    *,
    arm: RobotArm,
    link: CameraLink,
    plane: SightingPlane,
    marker: GripperMarker,
    align_config: XYAlignConfig,
    grasp_config: GraspConfig,
    arm_config: ArmConfig,
) -> int:
    """真动：闭环对准一次。"""
    print("开始 x/y 对准（只动 底座旋转 与 伸缩，高度不动）")
    print(f"  {marker.describe()}")
    print(f"  缩回半径：{grasp_config.retract_radius_mm}")
    print(f"  容差：{align_config.tolerance_mm:.1f} mm，最多重新测量 {align_config.iterations} 次\n")

    arm.enable_all()
    aligner = XYAligner(
        arm=arm,
        link=link,
        plane=plane,
        marker=marker,
        config=align_config,
        grasp_config=grasp_config,
        arm_config=arm_config,
        on_message=lambda message: print(f"  {message}"),
    )
    try:
        result = aligner.align()
    except XYAlignError as error:
        print(f"\n对准失败：{error}")
        return 1
    finally:
        # 无论成败都松力矩：这一步之后机械臂不该保持通电堵着。
        arm.disable_all()

    print(f"\n{result.summary()}")
    print("下一步：z 对准（把爪子降到镜头正下方约 7 cm 处）与夹爪闭合。")
    return 0


def run_calibrate_plane(*, arm: RobotArm, link: CameraLink, marker: GripperMarker) -> int:
    """转底座采点，拟合基座轴心与 θ 零点，并把结果写回配置文件。

    原理：底座**纯旋转**时爪上靶标画出一个以基座轴心为圆心的圆，所以
    "从相机坐标里拟合圆心"就等于"找到基座旋转轴在镜头水平面上的位置"。
    θ 零点再用"命令的角度 − 测到的角度"的圆均值定出来。
    """
    from project_paths import project_path

    samples: List[CalibrationSample] = []
    radial_command = 0.0
    print(f"将转 {CALIBRATION_STEPS - 1} 步，每步 {CALIBRATION_STEP_DEGREES:.0f}°。")
    print("请确认机械臂前方没有障碍物，且爪子悬空不碰台面。\n")

    for step in range(CALIBRATION_STEPS):
        frame = link.frame(timeout=15.0, max_age_seconds=2.0)
        if frame is None:
            print("采点失败：等不到观测。请确认镜头端在发数据、且爪上靶标在画面里。")
            return 1
        sightings = [item for item in frame if marker.matches(item)]
        if not sightings:
            print(
                f"采点失败：第 {step + 1} 步的画面里没有爪上靶标。"
                f"收到的观测：{'、'.join(f'物料 {i.material_id}({i.color})' for i in frame)}"
            )
            return 1
        target = sightings[-1]
        samples.append(
            CalibrationSample(
                x_mm=target.x_mm,
                y_mm=target.y_mm,
                base_degrees=step * CALIBRATION_STEP_DEGREES,
                radial_mm=radial_command,
            )
        )
        print(
            f"  第 {step + 1} 步：命令 θ={step * CALIBRATION_STEP_DEGREES:+.0f}°，"
            f"靶标在镜头系 ({target.x_mm:+.2f}, {target.y_mm:+.2f}) mm"
        )
        if step == CALIBRATION_STEPS - 1:
            break
        arm.move_base_degrees(CALIBRATION_STEP_DEGREES, wait=True)
        radial_command = 0.0

    try:
        fitted = fit_plane_calibration(samples)
    except ValueError as error:
        print(f"拟合失败：{error}")
        return 1

    print(f"\n拟合结果：{fitted.summary()}")
    if fitted.rms_mm > 5.0:
        print(
            f"残差 RMS={fitted.rms_mm:.1f} mm 偏大；"
            "常见原因是靶标没装稳、画面抖动、或者转的角度太小。建议重做。"
        )

    path = project_path("modules", "vision", "sighting_plane.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["plane"] = {
        "center_x_mm": fitted.center_x_mm,
        "center_y_mm": fitted.center_y_mm,
        "rotation_degrees": fitted.rotation_degrees,
        "rms_mm": fitted.rms_mm,
        "samples": fitted.samples,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {path}")
    print("接着填 modules/vision/sighting_plane.json 的 table_depth_mm（实验 D），再跑 --dry-run。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="让爪子正下方对准物块（只动平面）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--listen-only",
        action="store_true",
        help="只起监听并打印收到的行，不打开机械臂串口",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="解算并打印要动多少，但不发任何动作帧（也不打开机械臂串口）",
    )
    mode.add_argument(
        "--calibrate-plane",
        action="store_true",
        help="转底座采点，拟合基座轴心与 θ 零点，写入 sighting_plane.json",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="机械臂步进总线串口；省略则用 modules/motor/config.json 的平台配置",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=None,
        help="接收镜头端的本地端口；省略则用 modules/camera_link/config.json",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=CALIBRATION_STEPS,
        help=f"平面标定采点数（默认 {CALIBRATION_STEPS}）",
    )
    parser.add_argument(
        "--step-degrees",
        type=float,
        default=CALIBRATION_STEP_DEGREES,
        help=f"平面标定每步转角（默认 {CALIBRATION_STEP_DEGREES:.0f}°）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """菜单和命令行共用的入口。"""
    global CALIBRATION_STEPS, CALIBRATION_STEP_DEGREES

    arguments = build_parser().parse_args(argv)
    CALIBRATION_STEPS = max(3, arguments.steps)
    CALIBRATION_STEP_DEGREES = arguments.step_degrees

    link_config = load_camera_link_config()
    try:
        link = CameraLink(
            config=link_config,
            port=link_config.port if arguments.listen_port is None else arguments.listen_port,
        ).start()
    except CameraLinkError as error:
        print(f"起不了接收链路：{error}")
        return 1
    print(f"接收链路已监听 {link.host}:{link.port}")

    try:
        if arguments.listen_only:
            return run_listen_only(link=link)

        align_config = load_xy_align_config()
        grasp_config = load_grasp_config()
        arm_config = load_arm_config()
        marker = _load_marker()

        if arguments.dry_run:
            plane = _load_plane_or_none()
            if plane is None:
                print(
                    "dry-run 需要平面标定：modules/vision/sighting_plane.json 的 plane 三项还没填。"
                    "先跑 --calibrate-plane。"
                )
                return 1
            return run_dry_run(
                link=link,
                plane=plane,
                marker=marker,
                align_config=align_config,
                grasp_config=grasp_config,
                arm_config=arm_config,
            )

        plane = _load_plane_or_none()
        if not arguments.calibrate_plane:
            problems = missing_setup(arm=arm_config, plane=plane)
            if problems:
                print("还不能开跑，缺这些：")
                for problem in problems:
                    print(f"  - {problem}")
                return 1
        if plane is None and not arguments.calibrate_plane:
            return 1

        # 到这一步才打开机械臂串口。
        port = arguments.port or load_motor_bus_config().port
        print(f"打开机械臂总线 {port}")
        arm = RobotArm(port, config=arm_config)
        try:
            arm.initialize()
            if arguments.calibrate_plane:
                return run_calibrate_plane(arm=arm, link=link, marker=marker)
            return run_align(
                arm=arm,
                link=link,
                plane=plane,
                marker=marker,
                align_config=align_config,
                grasp_config=grasp_config,
                arm_config=arm_config,
            )
        finally:
            arm.close()
    finally:
        link.stop()


if __name__ == "__main__":
    raise SystemExit(main())

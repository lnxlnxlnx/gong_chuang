"""真机抓取流程入口。

启动前会把"还没标定"的东西一次列全并拒绝运行——抓取流程牵涉标定最多，缺一项都会变成
一句难懂的运行时报错，所以在打开硬件之前先说清楚。

    python test.py            # 菜单选 19

只想核对标定、不想让机械臂动：``--observe-only``。它会打开摄像头、打印每一帧测到的
机械臂位姿与物料位姿（圆柱坐标），不做任何动作。
"""

from __future__ import annotations

MENU_ENTRY = (19, "抓取流程")

import argparse
import os
import sys
from typing import Callable, List, Optional

from modules.arm.config import ArmConfig, load_arm_config
from modules.arm.drive.kinematics import CylindricalPose
from modules.grasp.config import GraspConfig, load_grasp_config
from modules.grasp.drive.perception import CellPerception
from modules.grasp.drive.pipeline import GraspError, GraspPipeline
from modules.motor.config import load_bus_config as load_motor_bus_config
from modules.servo.config import GripperConfig, load_gripper_config
from modules.servo.drive.bus_servo import load_bus_config as load_servo_bus_config
from modules.servo.drive.gripper import GripperServo
from modules.vision.config import VisionConfig, load_vision_config
from modules.arm.drive.robot_arm import RobotArm

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


def calibration_report(
    *,
    vision: VisionConfig,
    arm: ArmConfig,
    grasp: GraspConfig,
) -> List[str]:
    """列出还缺哪些标定；空列表表示可以开跑。

    顺序按"实验 A → B/C → 靶标/夹爪"排，照这个顺序做下来每一步都能单独验收。
    """
    missing: List[str] = []
    if vision.plane is None:
        missing.append("视觉：还没做平面标定（实验 A），calibration.json 的 plane 全空")
    elif vision.plane.rms_mm > 5.0:
        missing.append(
            f"视觉：平面标定残差 {vision.plane.rms_mm:.1f} mm 偏大，"
            "抓取精度会被它直接拖住，建议重标（实验 A）"
        )
    if vision.focal_px is None:
        missing.append("视觉：calibration.json 的 focal_px 还是空的（真实安装高度下重标焦距）")
    if vision.table_depth_mm is None:
        missing.append("视觉：calibration.json 的 table_depth_mm 还是空的（镜头到台面的深度）")
    if vision.targets.marker_diameter_mm is None:
        missing.append("视觉：config.json 的 targets.marker_diameter_mm 还没填")
    if not vision.targets.can_split_by_color:
        missing.append(
            "视觉：targets.material_bgr / marker_bgr 没配对，无法在同一帧里区分物料与靶标"
        )
    for axis, label in (("lift", "升降"), ("extension", "伸缩")):
        if arm.mm_per_motor_rev(axis) is None:
            missing.append(f"机械臂：{label}轴 ({axis}) 的 mm_per_motor_rev 还没标（实验 B/C）")
    if arm.safe_pose is None:
        missing.append("机械臂：safe_pose 还没配置（转角度前后的安全位）")
    if arm.base_limit is None or arm.radial_limit is None or arm.height_limit is None:
        missing.append("机械臂：limits 还有空的，等于这几个轴没有软件行程保护")
    if grasp.marker_to_grasp_mm == 0.0:
        missing.append(
            "抓取：marker_to_grasp_mm 还是 0；除非靶标正好在夹持点高度，否则必须实测后填上"
        )
    return missing


def describe(observation) -> str:
    """把一帧观测打印成一行。"""
    return observation.summary()


def run_observe_only(
    *,
    perception: CellPerception,
    frames: Callable[[], object],
    prompt: Callable[[str], str] = input,
    print_line: Callable[[str], None] = print,
) -> None:
    """只观测不动：用来核对平面标定与靶标检出。"""
    while True:
        observation = perception.observe(frames())
        print_line(describe(observation))
        answer = prompt("回车再看一帧，输入 q 结束：").strip().lower()
        if answer == "q":
            return


def run_grasp_once(
    *,
    pipeline: GraspPipeline,
    gripper: GripperServo,
    prompt: Callable[[str], str] = input,
    print_line: Callable[[str], None] = print,
) -> bool:
    """走一次完整抓取；返回是否真的抓起并退回了安全位。"""
    observation = pipeline.observe()
    print_line(describe(observation))
    if observation.material is None:
        print_line("画面里没有物料，先放一个再试。")
        return False
    if observation.marker is None:
        print_line("画面里看不到夹爪靶标；检查靶标是否装好、颜色是否配对。")
        return False
    if prompt("准备抓取；确认工作区内无人，输入大写 YES 执行：").strip() != "YES":
        print_line("已取消。")
        return False

    result = pipeline.pick_up()
    print_line(result.summary())
    if not result.held_something:
        print_line("[!] 夹爪走到闭合位 = 空夹；多半是标定或夹持高度的问题，先别继续。")
    return result.held_something


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="摄像头识别 + 机械臂抓取")
    parser.add_argument("--observe-only", action="store_true", help="只看不动：打印观测后退出")
    parser.add_argument("--port", default=None, help="覆盖 Emm42 总线串口")
    parser.add_argument("--servo-port", default=None, help="覆盖舵机总线串口")
    parser.add_argument("--camera", type=int, default=None, help="覆盖摄像头编号")
    return parser.parse_args()


def main() -> None:
    from modules.vision import USBCamera, close_previews, open_preview

    args = parse_args()
    platform = "windows" if os.name == "nt" else "jetson"
    vision = load_vision_config()
    arm_config = load_arm_config()
    grasp_config = load_grasp_config()

    missing = calibration_report(vision=vision, arm=arm_config, grasp=grasp_config)
    if missing:
        print("还缺以下标定，先补齐再跑（详见 docs/抓取坐标链路.md）：", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(2)

    camera_id = args.camera if args.camera is not None else vision.camera_id(platform)
    window = None
    camera = USBCamera(camera_id, width=CAMERA_WIDTH, height=CAMERA_HEIGHT).initialize()
    try:
        frame = camera.read()
        if frame is None:
            raise RuntimeError("摄像头没有返回图像")
        perception = CellPerception.from_vision_config(vision, frame.shape)
        frames = camera.read

        if args.observe_only:
            run_observe_only(perception=perception, frames=frames)
            return

        window = open_preview()
        # 机械臂在步进总线，夹爪在舵机总线：两条串口，两个 loader，不要混。
        gripper_config: GripperConfig = load_gripper_config()
        servo_bus = load_servo_bus_config(platform)
        arm = RobotArm(args.port or load_motor_bus_config(platform).port, config=arm_config)
        gripper = GripperServo(
            args.servo_port or servo_bus.port,
            config=gripper_config,
            bus_config=servo_bus,
        )
        pipeline = GraspPipeline(
            arm=arm,
            gripper=gripper,
            observer=perception,
            frames=frames,
            config=grasp_config,
            arm_config=arm_config,
        )
        try:
            arm.initialize()
            arm.enable_all()
            gripper.initialize()
            print(f"摄像头 {camera_id}；窗口 {window!r}，按 q 关闭预览")
            run_grasp_once(pipeline=pipeline, gripper=gripper)
        finally:
            pipeline.abort()
            _close_quietly(arm, gripper)
    finally:
        camera.release()
        if window is not None:
            close_previews()


def _close_quietly(arm: RobotArm, gripper: GripperServo) -> None:
    """两条串口都要关；任何一个失败都不能盖住真正的错误。"""
    for name, action in (("机械臂串口", arm.close), ("舵机串口", gripper.close)):
        try:
            action()
        except Exception as error:  # noqa: BLE001 - 收尾失败不能掩盖原始异常
            print(f"[!] 关闭{name}失败：{error!r}", file=sys.stderr)


if __name__ == "__main__":
    main()

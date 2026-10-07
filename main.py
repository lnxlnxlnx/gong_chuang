"""项目应用入口：默认预览物料定位结果。"""

from __future__ import annotations

import os
from pathlib import Path

from motor_drivers import BusServo
from robot_arm import RobotArm
from robot_arm_config import load_arm_config
from vision import default_model_path, load_camera_id


def _confirmed(label: str) -> bool:
    """Only execute a hardware action after the operator confirms it."""
    return input(f"准备{label}，回车执行，输入 q 跳过：").strip().lower() != "q"


def demo_arm(port: str) -> None:
    """Run one bounded position command in each arm direction."""
    config = load_arm_config()
    arm = RobotArm(port, config=config)
    try:
        arm.initialize()
        arm.enable_all()
        motions = (
            ("底座顺时针", arm.rotate_cw),
            ("底座逆时针", arm.rotate_ccw),
            ("机械臂向上", arm.move_up),
            ("机械臂向下", arm.move_down),
            ("机械臂前伸", arm.move_forward),
            ("机械臂后缩", arm.move_backward),
        )
        for label, action in motions:
            if _confirmed(label):
                action(config.default_pulses)
    finally:
        arm.stop_all()
        arm.disable_all()
        arm.close()


def arm_grab(port: str) -> None:
    """Run one bounded position command in each arm direction."""
    config = load_arm_config()
    arm = RobotArm(port, config=config)
    try:
        arm.initialize()
        arm.enable_all()
        motions = (
            # ("底座顺时针", arm.rotate_cw, 5 * config.default_pulses),  # 差不多90度
            # ("底座逆时针", arm.rotate_ccw, 5 * config.default_pulses),
            # ("机械臂前伸", arm.move_forward, 3 * config.default_pulses),
            # ("机械臂后缩", arm.move_backward, 3 * config.default_pulses),
            # 默认差不多2cm->3个默认是5cm
            ("机械臂向上", arm.move_up, 3 * config.default_pulses),
            ("机械臂向下", arm.move_down, 3 * config.default_pulses),
            ("机械臂前伸", arm.move_forward, config.default_pulses),
        )
        for label, action, pulses in motions:
            if _confirmed(label):
                action(pulses)
    finally:
        arm.stop_all()
        arm.disable_all()
        arm.close()


def demo_vision(
    model_path: str | Path | None = None, camera_id: int | None = None
) -> None:
    """Preview detections until q is pressed in the image window."""
    model_path = Path(model_path) if model_path is not None else default_model_path()
    camera_id = (
        camera_id
        if camera_id is not None
        else load_camera_id("windows" if os.name == "nt" else "jetson")
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    import cv2

    from vision import (
        BLANK_FRAME_WARNING,
        USBCamera,
        close_previews,
        create_detector,
        looks_blank,
        open_preview,
        warn_blank_preview,
    )

    # 窗口名必须是 ASCII，且要先显式建窗，见 vision/preview.py 顶部说明。
    window = open_preview()
    blank_frames = 0
    warned_blank = False
    try:
        # 反斜杠续行而不是括号化 with：括号化多管理器是 3.9+ 语法，Jetson 的
        # Python 3.8 会直接 SyntaxError（见 tests/test_python38_compatibility.py）。
        with USBCamera(camera_id).initialize() as camera, \
                create_detector(model_path) as detector:
            print(f"摄像头编号 {camera_id}，模型 {model_path.name}；预览窗口 {window!r}，按 q 退出")
            while True:
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("摄像头未返回图像")
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)
                hits = detector.detect(frame)
                cv2.imshow(window, detector.draw(frame, hits))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        close_previews()


def demo_material(
    model_path: str | Path | None = None,
    camera_id: int | None = None,
    *,
    focal_px: float | None = None,
) -> None:
    """预览物料的像素位置和相机坐标；按 p 打印，按 q 退出。

    可用 ``python -m vision.hough_lock`` 单独检查圆面定位；这里使用
    :class:`vision.MaterialLocator` 跑完整流程。
    """
    model_path = Path(model_path) if model_path is not None else default_model_path()
    camera_id = (
        camera_id
        if camera_id is not None
        else load_camera_id("windows" if os.name == "nt" else "jetson")
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    import cv2

    from vision import (
        BLANK_FRAME_WARNING,
        MaterialLocator,
        USBCamera,
        close_previews,
        looks_blank,
        open_preview,
        warn_blank_preview,
    )

    window = open_preview()
    blank_frames = 0
    warned_blank = False
    try:
        with MaterialLocator(
            model_path=model_path,
            focal_px=focal_px,
        ) as locator, USBCamera(camera_id, width=1280, height=720).initialize() as camera:
            print(f"摄像头编号 {camera_id}，模型 {model_path.name}；预览窗口 {window!r}，按 q 退出")
            print("画面里只画 ASCII；按 p 打印所有物料的完整结果，按 q 退出。")
            while True:
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("摄像头未返回图像")
                blank_frames = blank_frames + 1 if looks_blank(frame) else 0
                if blank_frames >= BLANK_FRAME_WARNING and not warned_blank:
                    warned_blank = True
                    warn_blank_preview(camera_id)

                report = locator.locate(frame)
                annotated = locator.draw(frame, report)
                cv2.putText(
                    annotated, f"present={report.present} count={len(report.regions)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                for index, item in enumerate(report.regions, start=1):
                    confidence = f"{item.confidence:.2f}" if item.confidence is not None else "--"
                    if item.name is None:
                        continue  # 只画有名字的物料，避免干扰
                    line = (
                        f"{index}: {item.name or 'unknown'} confidence={confidence} "
                        f"xyz=({item.x_mm:.0f},{item.y_mm:.0f},{item.z_mm:.0f})mm "
                        f"range={item.range_mm:.0f}mm"
                    )
                    cv2.putText(
                        annotated, line, (10, 24 + 26 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    )
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    print(report.summary())
                if key == ord("q"):
                    break
    finally:
        close_previews()


def demo_servo(port: str, servo_id: int = 1) -> None:
    """Read a bus servo's position without commanding movement."""
    servo = BusServo(port, servo_id).initialize()
    try:
        print(f"舵机 {servo_id} 当前的位置：{servo.read_position()}")
    finally:
        servo.close()


def main() -> None:
    """Run the material preview; wheel routes live in ``app.wheel``."""
    demo_material()


if __name__ == "__main__":
    main()

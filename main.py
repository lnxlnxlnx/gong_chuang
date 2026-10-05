"""项目应用入口：按顺序演示底盘、机械臂和视觉功能。

在 ``main()`` 里注释暂时不用的演示调用即可。所有运动都需要现场回车确认。
"""

from __future__ import annotations

import os
from pathlib import Path

from mecanum_config import HardwareProfile, create_chassis, load_profile
from motor_drivers import BusServo
from robot_arm import RobotArm
from robot_arm_config import load_arm_config
from vision import default_model_path, load_camera_id


def _confirmed(label: str) -> bool:
    """Only execute a hardware action after the operator confirms it."""
    return input(f"准备{label}，回车执行，输入 q 跳过：").strip().lower() != "q"


def demo_chassis(profile: HardwareProfile) -> None:
    """Run two short whole-vehicle motions using the configured calibration."""
    car = create_chassis(profile)
    try:
        car.initialize()
        car.enable_all()
        if _confirmed("底盘前进"):
            car.forward(rpm=profile.chassis_rpm, duration=10 * profile.chassis_duration)
        # if _confirmed("底盘顺时针旋转"):
        # car.rotate_cw(rpm=profile.chassis_rpm, duration=profile.chassis_duration)
        # if _confirmed("底盘左移"):
        #     car.left(rpm=profile.chassis_rpm, duration=profile.chassis_duration)
        # if _confirmed("底盘右移"):
        #     car.right(rpm=profile.chassis_rpm, duration=profile.chassis_duration)
        # if _confirmed("底盘左前移"):
        #     car.forward_left(rpm=profile.chassis_rpm, duration=profile.chassis_duration)
        # if _confirmed("底盘右前移"):
        #     car.forward_right(rpm=profile.chassis_rpm, duration=profile.chassis_duration)
    finally:
        car.stop()
        car.disable_all()
        car.close()

def demo_run_one_map(profile: HardwareProfile) -> None:
    """Run one map file with the configured calibration."""
    car = create_chassis(profile)
    try:
        car.initialize()
        car.enable_all()
        # if _confirmed(f"运行地图 {map_path}"):
        if _confirmed("运行地图"):
            car.forward_left(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)
            car.forward(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
            car.left(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
        # if _confirmed("往返"):
            car.backward(rpm=profile.chassis_rpm, duration=15 * profile.chassis_duration)
            car.right(rpm=profile.chassis_rpm, duration=16 * profile.chassis_duration)
            car.backward_right(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)
        
    finally:
        car.stop()
        car.disable_all()
        car.close()
        
def func_run_circle_map(car, profile):
    """Run one map file with the configured calibration."""
    if _confirmed("运行地图"):
        car.forward_left(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)
        car.forward(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
        car.left(rpm=profile.chassis_rpm, duration=15.5 * profile.chassis_duration)
        car.backward_right(rpm=profile.chassis_rpm, duration= profile.chassis_duration)
        car.backward(rpm=profile.chassis_rpm, duration=15 * profile.chassis_duration)
        car.right(rpm=profile.chassis_rpm, duration=16 * profile.chassis_duration)
        car.backward_right(rpm=profile.chassis_rpm, duration=3 * profile.chassis_duration)
        
def run_map_by_func(profile: HardwareProfile, func) -> None:
    """Run one map file with the configured calibration."""
    car = create_chassis(profile)
    try:
        car.initialize()
        car.enable_all()
        if _confirmed("运行地图"):
            func(car, profile)
    finally:
        car.stop()
        car.disable_all()
        car.close()

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


def demo_vision(model_path: str | Path | None = None, camera_id: int | None = None) -> None:
    """Preview detections until q is pressed in the image window."""
    model_path = Path(model_path) if model_path is not None else default_model_path()
    camera_id = camera_id if camera_id is not None else load_camera_id(
        "windows" if os.name == "nt" else "jetson")
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    import cv2
    from vision import USBCamera, create_detector

    try:
        with USBCamera(camera_id).initialize() as camera, create_detector(model_path) as detector:
            while True:
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("摄像头未返回图像")
                hits = detector.detect(frame)
                cv2.imshow("项目视觉演示", detector.draw(frame, hits))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()


def demo_servo(port: str, servo_id: int = 1) -> None:
    """Read a bus servo's position without commanding movement."""
    servo = BusServo(port, servo_id).initialize()
    try:
        print(f"舵机 {servo_id} 当前的位置：{servo.read_position()}")
    finally:
        servo.close()


def main() -> None:
    profile = load_profile("windows" if os.name == "nt" else "jetson")

    # 按需注释以下调用；机械臂和底盘使用同一平台配置中的步进总线端口。
    # demo_chassis(profile)
    run_map_by_func(profile, func_run_circle_map)
    # demo_arm(profile.port)
    # demo_vision()
    # 总线舵机使用独立串口；确认设备名后取消注释并填写端口。
    # demo_servo("<舵机串口>")


if __name__ == "__main__":
    main()

"""Bus-servo demo: read one servo's position without commanding movement."""

from __future__ import annotations

from modules.servo.drive.bus_servo import BusServo


def demo_servo(port: str, servo_id: int) -> None:
    """Read a bus servo's position without commanding movement.

    ``servo_id`` 没有默认值：舵机 ID 是硬件事实，调用方要么从
    ``modules/servo/hardware.json`` 读，要么自己明确给一个——不在这里猜。
    """
    servo = BusServo(port, servo_id).initialize()
    try:
        print(f"舵机 {servo_id} 当前的位置：{servo.read_position()}")
    finally:
        servo.close()

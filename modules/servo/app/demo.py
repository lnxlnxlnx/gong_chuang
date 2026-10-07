"""Bus-servo demo: read one servo's position without commanding movement."""

from __future__ import annotations

from modules.servo.drive.bus_servo import BusServo


def demo_servo(port: str, servo_id: int = 1) -> None:
    """Read a bus servo's position without commanding movement."""
    servo = BusServo(port, servo_id).initialize()
    try:
        print(f"舵机 {servo_id} 当前的位置：{servo.read_position()}")
    finally:
        servo.close()

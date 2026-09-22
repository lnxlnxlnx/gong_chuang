"""Diagnostic motor test: torque-lock check + 3x full 360 deg sweeps."""

import sys
import time

sys.path.insert(0, "/home/pi/gong_chuang")

from motor_drivers.emm42 import Direction, Emm42Driver


def step(msg: str) -> None:
    print(f"\n== {msg}")
    sys.stdout.flush()


def main() -> None:
    motor = Emm42Driver("/dev/ttyUSB0", motor_id=1).initialize()
    print("[*] serial opened OK")
    try:
        step("enable torque - SHAFT SHOULD LOCK, try to turn it by hand for 5s")
        motor.enable(True)
        time.sleep(5.0)

        step("full 360 deg CW @ 30rpm (8s) - rotation 1/3")
        motor.move_relative(degrees=360.0, rpm=30, acceleration=5)
        time.sleep(10.0)

        step("full 360 deg CCW @ 30rpm (8s) - rotation 2/3")
        motor.move_relative(degrees=-360.0, rpm=30, acceleration=5)
        time.sleep(10.0)

        step("full 360 deg CW @ 30rpm (8s) - rotation 3/3")
        motor.set_speed(Direction.CW, rpm=30, acceleration=5)
        time.sleep(6.0)
        motor.stop()
        time.sleep(1.0)

        step("continuous CCW @ 15rpm for 8s, then immediate stop")
        motor.set_speed(Direction.CCW, rpm=15, acceleration=3)
        time.sleep(8.0)
        motor.stop()

        step("diagnostic test commands completed")
    except Exception as error:
        print(f"[!] FAILED: {error!r}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            motor.stop()
        except Exception:
            pass
        motor.enable(False)
        motor.close()
        print("[*] torque released, port closed")


if __name__ == "__main__":
    main()
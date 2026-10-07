"""Diagnose why the Emm42 motor does not stop.

Reads the driver ACK for every command and polls realtime speed (0x35)
to see whether the stop command is accepted and takes effect.

用来排查 **EMM42 步进电机驱动器无法停机** 的问题
"""

MENU_ENTRY = (7, "单电机停机诊断")

import os
import sys
import time

import serial
from modules.motor.config import load_bus_config

BAUD = 115200


def read_ack(ser: serial.Serial, window: float = 0.6) -> bytes:
    ser.timeout = window
    data = ser.read(500)
    return data


def send(ser: serial.Serial, frame: bytes) -> bytes:
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    return read_ack(ser)


def main() -> None:
    addr = b"\x01"
    CSUM = b"\x6B"
    port = load_bus_config().port

    with serial.Serial(port, baudrate=BAUD, timeout=0.2) as ser:
        ser.reset_input_buffer()

        print("[1] enable (F3 AB 01 00)")
        ack = send(ser, addr + b"\xF3\xAB\x01\x00" + CSUM)
        print(f"    ack: {ack.hex(' ')}")

        print("[2] start velocity CW @ 120 rpm (F6 00 0078 05 00)")
        ack = send(ser, addr + b"\xF6\x00\x00\x78\x05\x00" + CSUM)
        print(f"    ack: {ack.hex(' ')}")
        time.sleep(3.0)

        print("[3] GET realtime speed while running (35)")
        ack = send(ser, addr + b"\x35" + CSUM)
        print(f"    ack: {ack.hex(' ')}")

        print("[4] EStop (FE 98 00) -- wait for ack right after send")
        ack = send(ser, addr + b"\xFE\x98\x00" + CSUM)
        print(f"    ack after stop: {ack.hex(' ')}")

        print("[5] poll realtime speed x8 to see if motor slowed")
        for i in range(8):
            ack = send(ser, addr + b"\x35" + CSUM)
            print(f"    poll {i}: {ack.hex(' ')}")
            time.sleep(0.2)

        print("[6] try velocity-stop: F6 speed=0 (00 0000 05)")
        ack = send(ser, addr + b"\xF6\x00\x00\x00\x05\x00" + CSUM)
        print(f"    ack: {ack.hex(' ')}")
        time.sleep(0.5)

        print("[7] poll realtime speed x4 after velocity-stop")
        for i in range(4):
            ack = send(ser, addr + b"\x35" + CSUM)
            print(f"    poll {i}: {ack.hex(' ')}")
            time.sleep(0.2)

        print("[8] disable torque (F3 AB 00 00)")
        ack = send(ser, addr + b"\xF3\xAB\x00\x00" + CSUM)
        print(f"    ack: {ack.hex(' ')}")


if __name__ == "__main__":
    main()

"""Bench probe: which Emm42 addresses actually answer on this bus?  (read-only)

Sweeps a range of addresses with ``0x3A`` (read status flags), the cheapest query
the driver answers.  Use it when the three arm axes read as "no reply": either the
arm is on a different adapter than the one ``modules/motor/config.json`` names, or
two drivers share an address and their replies interleave.

The chassis wheels are 1-4 and the arm axes are 5-7, all on one bus.  This script
sends only read commands, so nothing moves and no torque is enabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if TYPE_CHECKING:
    from modules.motor.drive.emm42 import SerialPort


STATUS_COMMAND = 0x3A
STATUS_REPLY_LENGTH = 4

#: 总线上的角色约定，用来核对"读到的东西符不符合预期"。
KNOWN_ROLES = {
    1: "chassis FL",
    2: "chassis FR",
    3: "chassis RL",
    4: "chassis RR",
    5: "arm BASE",
    6: "arm LIFT",
    7: "arm EXT",
}


def open_bus(timeout: float = 0.15) -> Tuple[str, "SerialPort"]:
    import serial

    from modules.motor.config import load_bus_config
    from modules.motor.drive.emm42 import Emm42Driver

    port = load_bus_config().port
    return port, serial.Serial(port, baudrate=Emm42Driver.BAUDRATE, timeout=timeout)


def probe(bus: "SerialPort", motor_id: int) -> Tuple[str, bytes]:
    """发一条 ``0x3A`` 并分类应答：``ok`` / ``rejected`` / ``silent`` / ``odd``。"""
    bus.reset_input_buffer()
    bus.write(bytes((motor_id, STATUS_COMMAND, 0x6B)))
    bus.flush()
    reply = bus.read(STATUS_REPLY_LENGTH)
    if len(reply) == STATUS_REPLY_LENGTH and reply[0] == motor_id and reply[-1] == 0x6B:
        if reply[1] == STATUS_COMMAND:
            return "ok", reply
        if reply[1] == 0x00 and reply[2] == 0xEE:
            return "rejected", reply
    return ("silent", reply) if not reply else ("odd", reply)


def describe_flags(flags: int) -> str:
    parts = []
    if flags & 0x01:
        parts.append("enabled")
    if flags & 0x02:
        parts.append("in_position")
    if flags & 0x04:
        parts.append("STALLED")
    if flags & 0x08:
        parts.append("stall_protect")
    return ",".join(parts) if parts else "none"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="扫一遍 Emm42 地址，看总线上到底挂着谁（只读，不使能不运动）。"
    )
    parser.add_argument("--first", type=int, default=1, help="起始地址，默认 1")
    parser.add_argument("--last", type=int, default=16, help="结束地址，默认 16")
    parser.add_argument("--timeout", type=float, default=0.15, help="串口读超时（秒）")
    arguments = parser.parse_args()

    if not 1 <= arguments.first <= arguments.last <= 255:
        raise SystemExit("地址范围必须是 1 <= first <= last <= 255")

    port, bus = open_bus(arguments.timeout)
    found: Dict[int, int] = {}
    odd: List[Tuple[int, bytes]] = []
    try:
        print("[*] port = %s   （只读：不使能、不发任何运动帧）" % port)
        print("[*] 扫描 %d..%d，每地址超时 %.2f s"
              % (arguments.first, arguments.last, arguments.timeout))
        print("")
        for motor_id in range(arguments.first, arguments.last + 1):
            outcome, reply = probe(bus, motor_id)
            role = KNOWN_ROLES.get(motor_id, "")
            suffix = ("  <- %s" % role) if role else ""
            if outcome == "ok":
                flags = reply[2]
                found[motor_id] = flags
                print("  addr %-3d OK        0x3A = 0x%02X  [%s]%s"
                      % (motor_id, flags, describe_flags(flags), suffix))
            elif outcome == "rejected":
                print("  addr %-3d rejected  %s%s" % (motor_id, reply.hex(" "), suffix))
            elif outcome == "odd":
                odd.append((motor_id, reply))
                print("  addr %-3d ODD REPLY %s%s" % (motor_id, reply.hex(" "), suffix))
    finally:
        bus.close()

    print("")
    print("应答的地址: %s" % (", ".join(str(item) for item in sorted(found)) or "(无)"))
    if odd:
        print("")
        print("[!] 有地址回了不成帧的数据。最常见的原因是**两个驱动器共用一个地址**，")
        print("    两条应答在总线上叠在一起。逐个拔掉再扫一遍可以定位。")
    missing = [item for item in (5, 6, 7) if item not in found]
    if missing:
        print("")
        print("[!] 机械臂地址 %s 没有应答。先确认线插在配置里写的那条总线上"
              % ", ".join(str(item) for item in missing))
        print("    （模块里的端口来自 modules/motor/config.json）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

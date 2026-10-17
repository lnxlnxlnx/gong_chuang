"""Bench probe: read each driver's configuration frame (``0x42 6C``).  (read-only)

Manuaul section 6.3.4 documents the reply as a fixed 33-byte frame::

    [addr] [0x42] [frame length = 33] [param count = 21] [21 params, 28B] [0x6B]

Four of the 21 fields decide how the arm code must behave, so this probe exists to
read them off the real drivers instead of assuming:

* **MSTEP** (offset 6) -- if this is not 16, then ``pulses_per_revolution: 3200``
  in ``modules/arm/config.json`` is wrong and every pulse<->mm conversion is off.
* **RESPONSE** (offset 19) -- whether control commands reply at all.  (Read
  commands answer regardless; this probe working at all is the proof.)
* **STALL_PROTECT** (offset 20) -- the manual says an enabled stall protection
  *switches the driver off* when the motor stalls, i.e. the axis drops torque.
  Know this before deliberately bumping into anything.
* **in_position_window** (offset 27) -- how tight the "reached" flag is.

The driver class does not wrap ``0x42`` yet, so this script speaks the frame
directly on the shared bus.  It never enables torque and never sends motion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if TYPE_CHECKING:
    from modules.motor.drive.emm42 import SerialPort


CONFIG_COMMAND = 0x42
CONFIG_SUBCOMMAND = 0x6C

#: addr + 0x42 + frame length + param count + 28 parameter bytes + checksum.
REPLY_LENGTH = 33

#: Where the 28 parameter bytes start inside the reply frame.
PARAMS_AT = 4
PARAMS_LENGTH = 28

ARM_AXES: Tuple[Tuple[int, str], ...] = ((5, "BASE"), (6, "LIFT"), (7, "EXT"))

#: ``(offset, length, name, note)`` -- offsets are 1-based inside the parameter block,
#: straight out of manual table 11.1 / the ``0x42 6C`` example frame.
FIELDS: Tuple[Tuple[int, int, str, str], ...] = (
    (1, 1, "motor_type", "25 = 1.8 deg motor, 50 = 0.9 deg"),
    (2, 1, "pulse_port_mode", "PUL_FOC / OPEN / ESI_RCO ..."),
    (3, 1, "comm_port_mode", "UART_FUN = serial/RS232/RS485"),
    (4, 1, "en_pin_level", "Hold / L / H"),
    (5, 1, "dir_pin_level", "CW / CCW"),
    (6, 1, "MSTEP_raw", "0x10 = 16 microsteps, 0x00 = 256"),
    (7, 1, "mstep_interpolation", "Enable / Disable"),
    (8, 1, "auto_screen_off", "Enable / Disable"),
    (9, 2, "open_loop_current_mA", ""),
    (11, 2, "stall_max_current_mA", "closed-loop ceiling"),
    (13, 2, "max_output_voltage_mV", ""),
    (15, 1, "uart_baud_option", "menu option index, not the baud value"),
    (16, 1, "can_rate_option", "menu option index"),
    (17, 1, "id_addr", ""),
    (18, 1, "checksum_mode", "0 = 0x6B fixed tail"),
    (19, 1, "RESPONSE_raw", "0=None 1=Receive 2=Reached 3=Both 4=Other"),
    (20, 1, "STALL_PROTECT", "1 = stall protection on -> a stall kills the axis"),
    (21, 2, "stall_rpm_threshold", "RPM"),
    (23, 2, "stall_current_threshold_mA", "mA"),
    (25, 2, "stall_time_threshold_ms", "ms"),
    (27, 2, "in_position_window", "1 = 0.1 deg (manual 8.1 says 0.3 deg)"),
)

RESPONSE_NAMES = {0: "None", 1: "Receive", 2: "Reached", 3: "Both", 4: "Other"}


def open_bus(timeout: float = 0.3) -> Tuple[str, "SerialPort"]:
    import serial

    from modules.motor.config import load_bus_config
    from modules.motor.drive.emm42 import Emm42Driver

    port = load_bus_config().port
    return port, serial.Serial(port, baudrate=Emm42Driver.BAUDRATE, timeout=timeout)


def read_config(bus: "SerialPort", motor_id: int) -> Tuple[bytes, Dict[str, int]]:
    """发一条 ``0x42 6C`` 并解析 21 个字段；帧不对就抛错，不猜。"""
    bus.reset_input_buffer()
    bus.write(bytes((motor_id, CONFIG_COMMAND, CONFIG_SUBCOMMAND, 0x6B)))
    bus.flush()
    reply = bus.read(REPLY_LENGTH)
    if len(reply) != REPLY_LENGTH:
        raise RuntimeError(
            "expected %d bytes, got %d: %s"
            % (REPLY_LENGTH, len(reply), reply.hex(" ") or "(nothing)")
        )
    if reply[0] != motor_id or reply[1] != CONFIG_COMMAND or reply[-1] != 0x6B:
        raise RuntimeError("unexpected frame: %s" % reply.hex(" "))
    params = reply[PARAMS_AT:PARAMS_AT + PARAMS_LENGTH]
    values: Dict[str, int] = {}
    for offset, length, name, _note in FIELDS:
        chunk = params[offset - 1:offset - 1 + length]
        values[name] = int.from_bytes(chunk, "big")
    return reply, values


def print_axis(motor_id: int, name: str, reply: bytes, values: Dict[str, int]) -> None:
    print("--- motor %d  %s" % (motor_id, name))
    print("    frame length field = %d   param count field = %d"
          % (reply[2], reply[3]))
    print("    raw: %s" % reply.hex(" "))
    print("    id_addr          = %d" % values["id_addr"])
    print("    MSTEP_raw        = 0x%02X  %s"
          % (values["MSTEP_raw"],
             "(16 microsteps)" if values["MSTEP_raw"] == 0x10
             else "(256 microsteps)" if values["MSTEP_raw"] == 0x00
             else "(NOT 16 -- pulses_per_revolution=3200 is wrong!)"))
    print("    RESPONSE_raw     = 0x%02X  -> %s"
          % (values["RESPONSE_raw"],
             RESPONSE_NAMES.get(values["RESPONSE_raw"], "unknown option")))
    print("    STALL_PROTECT    = %d  -> %s"
          % (values["STALL_PROTECT"],
             "ENABLED (a stall switches the axis OFF)" if values["STALL_PROTECT"]
             else "disabled"))
    print("    stall thresholds = %d RPM / %d mA / %d ms"
          % (values["stall_rpm_threshold"],
             values["stall_current_threshold_mA"],
             values["stall_time_threshold_ms"]))
    print("    in_position_win  = %d  = %.1f deg"
          % (values["in_position_window"], values["in_position_window"] * 0.1))
    print("    open_loop_mA     = %d   stall_max_mA = %d   max_mV = %d"
          % (values["open_loop_current_mA"],
             values["stall_max_current_mA"],
             values["max_output_voltage_mV"]))


def print_summary(collected: Dict[str, Dict[str, int]]) -> None:
    print("")
    print("axis  name   MSTEP  RESPONSE  STALL_PROTECT  in_pos_win")
    for motor_id, name in ARM_AXES:
        values = collected.get(str(motor_id))
        if values is None:
            print("%4d  %-6s  READ FAILED" % (motor_id, name))
            continue
        print("%4d  %-6s  0x%02X   %-8s  %-13s  %d (%.1f deg)"
              % (motor_id, name, values["MSTEP_raw"],
                 RESPONSE_NAMES.get(values["RESPONSE_raw"], "?"),
                 "ON" if values["STALL_PROTECT"] else "off",
                 values["in_position_window"],
                 values["in_position_window"] * 0.1))
    print("")
    print("怎么读这张表：")
    print("  MSTEP 不是 0x10 -> pulses_per_revolution 就得改，换算全废。")
    print("                    否则所有 脉冲<->毫米 换算都是错的。")
    print("  RESPONSE        -> 读命令不受它影响（这个脚本能读出来就是证据）；")
    print("                    它只决定控制动作命令要不要回确认包/到位包。")
    print("  STALL_PROTECT   -> ON 表示撞到东西时该轴会**直接掉力矩**。")
    print("                    做任何碰撞实验之前必须先看到这一列。")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="读三轴 0x42 6C 配置帧并逐字段解析（只读，不使能不运动）。"
    )
    parser.add_argument("--timeout", type=float, default=0.3, help="串口读超时（秒）")
    arguments = parser.parse_args()

    port, bus = open_bus(arguments.timeout)
    collected: Dict[str, Dict[str, int]] = {}
    try:
        print("[*] port = %s   （只读：不使能、不发任何运动帧）" % port)
        print("")
        for motor_id, name in ARM_AXES:
            try:
                reply, values = read_config(bus, motor_id)
            except Exception as error:
                print("--- motor %d  %s" % (motor_id, name))
                print("    READ FAILED: %r" % (error,))
                continue
            collected[str(motor_id)] = values
            print_axis(motor_id, name, reply, values)
        print_summary(collected)
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Bench probe: where are the three arm axes right now?  (read-only)

Sends only read commands -- 0x31 (single-turn encoder), 0x36 (real-time position),
0x3A (status flags), 0x3B (homing flags).  It NEVER enables torque and NEVER sends
a motion frame, so the arm neither moves nor gets energised.

Do not use the menu entries under ``modules/*/test/manual`` for this:
:meth:`modules.arm.runtime.RobotArm.open` calls ``enable_all()`` while assembling,
which puts torque on all three axes.  This script talks to ``Emm42Driver`` directly
to avoid that -- it opens the bus itself and builds one driver per address.

Intended for the power-cycle experiment described in README.md::

    python only_for_test/read_axes.py
    python only_for_test/read_axes.py --save before.json
    python only_for_test/read_axes.py --compare before.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# 这个脚本不在包根目录下，直接 ``python only_for_test/read_axes.py`` 时
# ``sys.path[0]`` 是脚本自己所在的目录，``import modules...`` 会找不到包。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if TYPE_CHECKING:  # 只为类型检查导入：真正要用时在函数里再导，避免无谓的依赖。
    from modules.motor.drive.emm42 import SerialPort


#: ``0x36`` 与 ``0x31`` 共用的量纲：一个电机圈 = 65536。
UNITS_PER_REV = 65536

#: 共享步进总线上的三个机械臂轴（1-4 是麦轮）。
ARM_AXES: Tuple[Tuple[int, str], ...] = ((5, "BASE"), (6, "LIFT"), (7, "EXT"))


def open_bus(timeout: float = 0.2) -> Tuple[str, "SerialPort"]:
    """打开配置里的步进总线。光是打开端口不会让机械臂有任何变化。"""
    import serial

    from modules.motor.config import load_bus_config
    from modules.motor.drive.emm42 import Emm42Driver

    port = load_bus_config().port
    return port, serial.Serial(port, baudrate=Emm42Driver.BAUDRATE, timeout=timeout)


def driver_for(bus: "SerialPort", port: str, motor_id: int) -> Any:
    """共享一条总线上的一个逻辑驱动器；``expect_ack=False``，与 5/6/7 号轴一致。"""
    from modules.motor.drive.emm42 import Emm42Driver

    return Emm42Driver(port, motor_id=motor_id, expect_ack=False, serial_port=bus)


def read_axis(bus: "SerialPort", port: str, motor_id: int) -> Dict[str, Any]:
    """读一根轴在不运动的前提下能回答的一切。"""
    driver = driver_for(bus, port, motor_id)
    position = driver.read_position()
    status = driver.read_status()
    homing = driver.read_homing_status()
    return {
        "position": position,
        "position_revs": position / UNITS_PER_REV,
        "position_mod": position % UNITS_PER_REV,
        "position_rev_index": position // UNITS_PER_REV,
        "encoder": driver.read_encoder(),
        "enabled": status.enabled,
        "in_position": status.in_position,
        "stalled": status.stalled,
        "stall_protected": status.stall_protected,
        "encoder_ready": homing.encoder_ready,
        "calibration_ready": homing.calibration_ready,
        "homing": homing.homing,
        "homing_failed": homing.failed,
    }


def read_all(
    bus: "SerialPort", port: str, axes: Tuple[Tuple[int, str], ...]
) -> Dict[str, Any]:
    """逐轴读取；一根轴读失败不影响其余轴继续报告。"""
    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for motor_id, _name in axes:
        try:
            bus.reset_input_buffer()
            results[str(motor_id)] = read_axis(bus, port, motor_id)
        except Exception as error:
            errors[str(motor_id)] = repr(error)
    return {"axes": results, "errors": errors}


def signed_delta(position_mod: int, encoder: int) -> int:
    """``0x36 mod 65536`` 与 ``0x31`` 的差，折算到 ±半圈以内。"""
    delta = (position_mod - encoder) % UNITS_PER_REV
    return delta - UNITS_PER_REV if delta > UNITS_PER_REV // 2 else delta


def print_report(payload: Dict[str, Any]) -> None:
    """先逐轴明细，再一张汇总表。"""
    by_id: Dict[str, Any] = payload["axes"]
    errors: Dict[str, str] = payload["errors"]
    for motor_id, name in ARM_AXES:
        key = str(motor_id)
        print("--- motor %d  %s" % (motor_id, name))
        if key not in by_id:
            print("    READ FAILED: %s" % errors.get(key, "?"))
            continue
        data = by_id[key]
        print(
            "    0x36 position = %d   (%.4f motor rev, rev index %d)   mod 65536 = %d"
            % (
                data["position"],
                data["position_revs"],
                data["position_rev_index"],
                data["position_mod"],
            )
        )
        print("    0x31 encoder  = %d" % data["encoder"])
        print(
            "    0x3A status   = enabled=%s in_position=%s stalled=%s "
            "stall_protected=%s"
            % (
                data["enabled"],
                data["in_position"],
                data["stalled"],
                data["stall_protected"],
            )
        )
        print(
            "    0x3B homing   = encoder_ready=%s calibration_ready=%s "
            "busy=%s failed=%s"
            % (
                data["encoder_ready"],
                data["calibration_ready"],
                data["homing"],
                data["homing_failed"],
            )
        )
        if not data["encoder_ready"]:
            print("    [!] encoder_ready=False（屏幕上是不是 Not Cal？）读数还不可信，")
            print("        先做编码器校准再往下走。")

    print("")
    print("axis  name   0x36 position   rev_idx      mod 65536   0x31 encoder   delta")
    for motor_id, name in ARM_AXES:
        key = str(motor_id)
        if key not in by_id:
            print("%4d  %-6s  READ FAILED" % (motor_id, name))
            continue
        data = by_id[key]
        print(
            "%4d  %-6s  %13d   %7d   %10d   %12d   %5d"
            % (
                motor_id,
                name,
                data["position"],
                data["position_rev_index"],
                data["position_mod"],
                data["encoder"],
                signed_delta(data["position_mod"], data["encoder"]),
            )
        )
    print("")
    print("delta = 0x36 mod 65536 减去 0x31。两者都是电机轴角度，正常情况下应该很小。")
    print("三根轴都是同一个较大的固定值 = 只是零点参考不同；各轴不一样才需要查。")


def save_snapshot(path: str, payload: Dict[str, Any]) -> None:
    document = dict(payload)
    document["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    Path(path).write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[*] 快照已写入 %s" % path)


def load_snapshot(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verdict(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[str, str]:
    """判定一根轴在掉电前后的行为。

    有判别力的量是**圈数索引**，不是原始值：驱动器若把多圈计数存了 flash，上电时
    必须把小数部分重新对齐到当前编码器读数，所以小数部分会随"断电期间手转的角度"
    改变，而整数部分（圈数索引）保留下来。
    """
    before_index = before["position_rev_index"]
    after_index = after["position_rev_index"]
    if before_index == 0 and after_index == 0:
        return "INCONCLUSIVE", (
            "两边的圈数索引都是 0，分不出来。请在**掉电之前**先把这根轴用手转过"
            "2 圈以上，让索引不为 0，再做一次"
        )
    if after_index == before_index:
        return "HELD", (
            "圈数索引 %d -> %d，扛过了掉电：多圈计数是存下来的，"
            "每次上电锚定一次就够" % (before_index, after_index)
        )
    if after_index == 0:
        return "RESET", (
            "圈数索引 %d -> 0：计数在上电时从单圈角重新开始，"
            "**每次上电都要重新锚定**" % before_index
        )
    return "UNKNOWN", "圈数索引 %d -> %d，把原始数字记下来人工判断" % (
        before_index,
        after_index,
    )


def print_comparison(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    print("=== 掉电前后对照")
    print("断电前快照: %s" % before.get("saved_at", "?"))
    print("现在      : %s" % after.get("saved_at", "?"))
    print("")
    for motor_id, name in ARM_AXES:
        key = str(motor_id)
        print("--- motor %d  %s" % (motor_id, name))
        old = before["axes"].get(key)
        new = after["axes"].get(key)
        if old is None or new is None:
            print("    两侧缺一边，无法比较")
            continue
        print(
            "    0x36 断电前 = %d   (圈数索引 %d)"
            % (old["position"], old["position_rev_index"])
        )
        print(
            "    0x36 现在   = %d   (圈数索引 %d)"
            % (new["position"], new["position_rev_index"])
        )
        outcome, explanation = verdict(old, new)
        print("    判定: %s -- %s" % (outcome, explanation))


def parse_axes(text: str) -> Optional[Tuple[Tuple[int, str], ...]]:
    """``--axes 5,7`` -> 只保留点名的轴；空串表示三个轴全读。"""
    if not text.strip():
        return None
    wanted = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            wanted.add(int(item))
        except ValueError:
            raise SystemExit("--axes 只接受逗号分隔的电机地址，例如 5,6,7")
    return tuple(item for item in ARM_AXES if item[0] in wanted)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只读地读三根机械臂轴的位置与状态，不使能、不运动。"
    )
    parser.add_argument("--save", metavar="PATH", help="把这次读数写成 JSON 快照")
    parser.add_argument(
        "--compare", metavar="PATH", help="与之前 --save 的快照做掉电前后对照"
    )
    parser.add_argument(
        "--axes", metavar="IDS", default="", help="逗号分隔的电机地址，默认 5,6,7"
    )
    parser.add_argument("--timeout", type=float, default=0.2, help="串口读超时（秒）")
    arguments = parser.parse_args()

    if arguments.compare:
        # 存下来的是"断电前"，这次现场读的是"现在"。
        before = load_snapshot(arguments.compare)
        port, bus = open_bus(arguments.timeout)
        try:
            print("[*] port = %s" % port)
            after = read_all(bus, port, ARM_AXES)
        finally:
            bus.close()
        after["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        print_comparison(before, after)
        return 0

    port, bus = open_bus(arguments.timeout)
    payload: Dict[str, Any] = {"axes": {}, "errors": {}}
    try:
        print("[*] port = %s   （只读：不使能、不发任何运动帧）" % port)
        print("")
        selected = parse_axes(arguments.axes)
        payload = read_all(bus, port, ARM_AXES if selected is None else selected)
        print_report(payload)
    finally:
        bus.close()

    if arguments.save:
        save_snapshot(arguments.save, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

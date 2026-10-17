"""Confirmed station-by-station test for the turntable on the shared bus servo.

One turntable, three material stations.  Every station needs its own confirmation:
a turntable carrying material will hit whatever the arm left in its path, so the
operator confirms each index instead of the script sweeping all three.
"""

from __future__ import annotations

MENU_ENTRY = (20, "转盘测试")

import argparse
import sys
from collections.abc import Callable

from modules.servo.config import HOME_STATION, TurntableConfig, load_turntable_config
from modules.servo.drive.bus_servo import BusServo, BusServoConfig, load_bus_config
from modules.servo.drive.turntable import DEGREES_PER_UNIT, TurntableServo


def station_order(config: TurntableConfig) -> tuple[str, ...]:
    """Return the stations with ``home`` first, then the rest in file order.

    回零排在第一位不是为了好看：三工位转盘每次上电时的朝向我们并不知道，先回零把
    "现在在哪个工位"这件事定下来，后面两个工位才是从已知位置出发的。
    """
    return (HOME_STATION,) + tuple(
        name for name in config.stations if name != HOME_STATION
    )


def run_manual(
    turntable: TurntableServo,
    config: TurntableConfig,
    *,
    prompt: Callable[[str], str] = input,
) -> None:
    """Read first; turn to each station on its own confirmation, then read back.

    ``回车``就是确认——真机调试要连着走好几个工位，每次都敲 ``YES`` 是纯粹的负担。
    ``q`` 结束剩下的动作，其他任何输入跳过这一项；两者都保留，是为了在"手已经放在
    回车上了"和"发现不对想停下来"之间还留得住区别。
    """
    print(f"转盘舵机 {config.servo_id} 当前位置：{turntable.read_position()}")
    for name in station_order(config):
        position = config.station(name)
        answer = (
            prompt(
                f"转盘将转到工位 {name}（位置 {position}，"
                f"约 {position * DEGREES_PER_UNIT:.1f}°，耗时 {config.duration_ms} 毫秒）。"
                "确认转盘有活动空间、机械臂已让开之后按回车执行，输入 q 结束，其他输入跳过："
            )
            .strip()
            .lower()
        )
        if answer == "q":
            print("已结束后续转盘动作")
            return
        if answer:
            print(f"已跳过工位 {name}")
            continue
        print(turntable.go_to_station(name).summary())


def parse_port(argv: list[str] | None = None) -> str | None:
    """Return the optional ``--port`` override for one-off wiring checks."""
    parser = argparse.ArgumentParser(description="转盘测试（总线舵机直连）")
    parser.add_argument(
        "--port", default=None, help="覆盖 modules/servo/config.json 中的平台串口"
    )
    return parser.parse_args(argv).port


def _cleanup(turntable: TurntableServo) -> None:
    """Attempt every cleanup action without hiding the command that originally failed."""
    for name, action in (("停止", turntable.stop), ("关闭串口", turntable.close)):
        try:
            action()
        except Exception as error:
            print(f"[!] 清理{name}失败：{error!r}", file=sys.stderr)


def main() -> None:
    """Open the bus, walk the stations on confirmation, and always close the UART."""
    port_override = parse_port()
    config = load_turntable_config()
    bus = load_bus_config()
    port = port_override if port_override is not None else bus.port
    print(f"转盘舵机 {config.servo_id} @ {port} {bus.baudrate} bps")
    turntable = TurntableServo(
        port,
        config=config,
        bus_config=bus,
        # 手动测试用比生产路径更宽的读回超时：接线接触不良时先看清"是没回包"，
        # 而不是把一个时序问题误报成舵机不动。
        servo=BusServo(port, config.servo_id, baudrate=bus.baudrate, timeout=0.5),
    )
    try:
        turntable.initialize()
    except Exception as error:
        # 端口被占用或设备名不对时直接说明，不再抛 traceback。
        # 这里只关串口：端口没打开时 close 是空操作，而 stop 会真的去发帧。
        print(f"无法打开 {port}：{error}")
        turntable.close()
        return
    try:
        try:
            run_manual(turntable, config)
        except (RuntimeError, TimeoutError) as error:
            print(f"转盘测试未通过：{error}")
    finally:
        _cleanup(turntable)


if __name__ == "__main__":
    main()

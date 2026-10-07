"""Display live STP-23L distances while the scan motor remains untouched."""

from __future__ import annotations

MENU_ENTRY = (11, "激光固定测距")

import argparse
from pathlib import Path
from statistics import median
from typing import Callable

from modules.laser_scan.drive.laser_scanner import DEFAULT_CONFIG_PATH, ScanConfig, load_scan_config
from modules.laser_scan.drive.stp23l import Stp23lReader


def display_distances(
    reader: Stp23lReader, config: ScanConfig, output: Callable[[str], None] = print
) -> None:
    """Print each checked frame's valid median distance until interrupted."""
    while True:
        distances = reader.read_distances(config.sample_timeout_seconds)
        valid = [distance for distance in distances
                 if config.min_distance_mm <= distance <= config.max_distance_mm]
        if valid:
            distance = int(median(valid))
            status = "是" if distance <= config.obstacle_distance_mm else "否"
            result = f"{distance} mm"
        else:
            status = "未知"
            result = "无有效数据"
        output(f"距离: {result} | 有效点: {len(valid)}/{len(distances)} | 障碍物: {status}")


def main() -> None:
    """Open only the lidar serial port; Ctrl+C closes it and returns to the menu."""
    parser = argparse.ArgumentParser(description="STP-23L 固定位置实时测距")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_scan_config(args.config)

    try:
        import serial
    except ImportError as error:
        raise RuntimeError("固定测距需要当前 Python 环境提供 pyserial") from error

    print(f"固定测距：{config.lidar_port}；按 Ctrl+C 返回菜单。")
    with serial.Serial(config.lidar_port, baudrate=230400, timeout=0.1) as lidar_serial:
        try:
            display_distances(Stp23lReader(lidar_serial), config)
        except KeyboardInterrupt:
            print("固定测距结束。")


if __name__ == "__main__":
    main()

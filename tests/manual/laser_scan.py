"""Run one manually confirmed STP-23L sector sweep on motor 8."""

from __future__ import annotations

MENU_ENTRY = (10, "激光障碍物扫描")

import argparse
from datetime import datetime
from pathlib import Path

from laser_scanner import DEFAULT_CONFIG_PATH, load_scan_config, save_scan, scan_sector
from motor_drivers.emm42 import Emm42Driver
from motor_drivers.stp23l import Stp23lReader


def main() -> None:
    parser = argparse.ArgumentParser(description="STP-23L 90-degree obstacle scan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-prefix", type=Path, help="CSV/SVG 文件前缀")
    args = parser.parse_args()
    config = load_scan_config(args.config)
    prefix = args.output_prefix or (
        Path("output/laser_scan") / datetime.now().strftime("scan-%Y%m%d-%H%M%S-%f")
    )
    if prefix.with_suffix(".csv").exists() or prefix.with_suffix(".svg").exists():
        parser.error(f"输出文件已存在：{prefix}")

    try:
        import serial
    except ImportError as error:
        raise RuntimeError("真机扫描需要当前 Python 环境提供 pyserial") from error

    with serial.Serial(config.motor_port, baudrate=Emm42Driver.BAUDRATE, timeout=0.1) as motor_serial:
        with serial.Serial(config.lidar_port, baudrate=230400, timeout=0.1) as lidar_serial:
            reader = Stp23lReader(lidar_serial)
            reader.read_distances(config.sample_timeout_seconds)
            answer = input(
                f"确认扫描头处于顺时针起点，逆时针方向有 {config.max_ccw_pulses} "
                "脉冲的空余行程；回车开始，输入 q 取消："
            ).strip().lower()
            if answer == "q":
                return
            motor = Emm42Driver(
                config.motor_port,
                config.motor_id,
                pulses_per_revolution=config.pulses_per_revolution,
                expect_ack=config.motor_expect_ack,
                serial_port=motor_serial,
            )
            try:
                points = scan_sector(
                    motor, reader, config,
                    confirm_return=lambda: input("确认扫描头已回到起点，按回车停止并失能："),
                )
            except BaseException:
                print("扫描中断；扫描头当前位置未确认，请重新核对起点后再运行。")
                raise

    csv_path, svg_path = save_scan(points, config, prefix)
    obstacles = sum(point.obstacle for point in points)
    print(f"采样 {len(points)} 个角度，检测到 {obstacles} 个近距点。")
    print(f"数据：{csv_path.resolve()}")
    print(f"雷达图：{svg_path.resolve()}")


if __name__ == "__main__":
    main()

"""Capture a COIN-D6 point cloud from the test menu and draw it.

Menu 16.  Without ``--simulate`` the script opens the serial port from
``modules/lidar/config.json``, starts the sensor, collects whole revolutions and
writes both a CSV of raw points and a PNG point-cloud figure.
"""

from __future__ import annotations

MENU_ENTRY = (16, "D6 激光雷达点云")

import argparse
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

from modules.lidar import (
    DEFAULT_CONFIG_PATH,
    CoinD6,
    LidarConfig,
    LidarError,
    PointCloud,
    build_figure,
    describe_config,
    load_lidar_config,
    save_figure,
    show_figure,
)
from modules.lidar.drive.protocol import (
    START_COMMAND,
    STOP_COMMAND,
    Ack,
    DeviceInfo,
    PacketReader,
)
from modules.lidar.app.simulate import SimulatedLidar


from project_paths import output_path

OUTPUT_ROOT = output_path("lidar_cloud")
DIAGNOSE_SECONDS = 3.0


def scan_output_prefix(now: Optional[datetime] = None) -> Path:
    """Return a fresh timestamped output prefix, one per capture."""
    return OUTPUT_ROOT / (now or datetime.now()).strftime("cloud-%Y%m%d-%H%M%S-%f")


def create_lidar(config: LidarConfig, *, simulate: bool = False) -> CoinD6:
    """Build the real driver, or a driver wired to the built-in fake sensor."""
    if simulate:
        return CoinD6(config.device.port, serial_port=SimulatedLidar())
    return CoinD6(
        config.device.port,
        baudrate=config.device.baudrate,
        timeout=config.device.read_timeout_seconds,
        ack_timeout=config.device.ack_timeout_seconds,
        strict_checksum=config.device.strict_checksum,
        require_ack=config.device.require_ack,
    )


def capture(
    lidar: CoinD6,
    config: LidarConfig,
    *,
    output: Callable[[str], None] = print,
) -> PointCloud:
    """Read the device description when available, then run one scan."""
    try:
        lidar.initialize()
        device = None
        try:
            device = lidar.read_device_info(0.5)
            direction = "顺时针" if device.clockwise else "逆时针"
            output(f"设备信息：{device.model} | 旋转方向 {direction} | 软件版本 Rev{device.software_revision}")
        except LidarError:
            output("未读到上电设备信息包（正常现象，它只在雷达上电时发送一次）。")
        cloud = lidar.scan(
            config.scan,
            on_revolution=lambda done: output(f"已采集完整 {done} 圈"),
        )
        return replace(cloud, device=device) if device is not None else cloud
    finally:
        lidar.close()


def save_cloud(
    cloud: PointCloud,
    config: LidarConfig,
    prefix: Path,
    *,
    output: Callable[[str], None] = print,
    show_window: Optional[bool] = None,
) -> Tuple[Path, Path]:
    """Write the CSV and the PNG point-cloud figure; returns both paths."""
    csv_path = cloud.save_csv(prefix.with_suffix(".csv"))
    png_path = prefix.with_suffix(".png")
    figure = build_figure(cloud, config.view)
    save_figure(figure, png_path)
    show = config.view.show_window if show_window is None else show_window
    if show:
        output("已保存图片；关闭图片窗口后继续。")
        show_figure(figure)
    return csv_path, png_path


def report(cloud: PointCloud, config: LidarConfig, output: Callable[[str], None] = print) -> None:
    """Print what was captured and where it landed."""
    if cloud.revolutions < config.scan.revolutions:
        output(
            f"注意：只采到 {cloud.revolutions}/{config.scan.revolutions} 圈（超时或中断），"
            "已按现有数据出图。"
        )
    output(
        f"共 {len(cloud)} 个点 | 距离 {cloud.nearest_mm}~{cloud.farthest_mm} mm | "
        f"角度窗口 {config.scan.angle_min_degrees:g}~{config.scan.angle_max_degrees:g}°"
    )


def diagnose(
    link,
    config: LidarConfig,
    *,
    output: Callable[[str], None] = print,
    seconds: float = DIAGNOSE_SECONDS,
) -> dict:
    """Send the start command and report exactly what the sensor sends back.

    This is the first thing to run on unknown hardware: it prints the raw bytes
    that were sent, the raw bytes that came back and how they decoded, which
    separates a wiring/baud problem from a protocol assumption.  The input
    buffer is deliberately not flushed, so anything already streaming shows up.
    """
    reader = PacketReader(link, strict_checksum=config.device.strict_checksum)
    link.write(START_COMMAND)
    output(f"已发送启动指令（原始字节，非文本）：{START_COMMAND.hex(' ').upper()}")
    counts = {"ack": 0, "info": 0, "point": 0, "bytes": 0, "skipped": 0, "rejected": 0}
    raw = bytearray()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        packet = reader.next_packet()
        if packet is None:
            chunk = link.read(256)
            if chunk:
                raw.extend(chunk)
                counts["bytes"] += len(chunk)
                reader.feed(chunk)
            continue
        if isinstance(packet, Ack):
            counts["ack"] += 1
            verdict = "校验通过" if packet.checksum_valid else "末字节与异或规则不符"
            output(f"收到指令应答：{packet.raw.hex(' ').upper()}（状态 0x{packet.status:02X}，{verdict}）")
        elif isinstance(packet, DeviceInfo):
            counts["info"] += 1
            output(f"收到设备信息包：{packet.model} | 软件版本 Rev{packet.software_revision}")
        else:
            counts["point"] += 1
            if counts["point"] == 1:
                speed = f"转速 {packet.speed_hz} Hz | " if packet.is_start else ""
                output(
                    f"收到第一个点云包：LSN={len(packet.points)} "
                    f"FSA={packet.start_angle_degrees:.2f}° LSA={packet.end_angle_degrees:.2f}° | "
                    f"{speed}校验分组 {packet.checksum_variant} "
                    f"| 起始包 {'是' if packet.is_start else '否'}"
                )
    try:
        link.write(STOP_COMMAND)
    except Exception as error:  # the port may already be gone; diagnostics still stand
        output(f"停止指令发送失败：{error}")
    counts["skipped"] = reader.skipped_bytes
    counts["rejected"] = reader.rejected_packets
    shown = raw[:48].hex(" ").upper()
    output(f"回传原始字节 {counts['bytes']} 个：{shown}{' ...' if len(raw) > 48 else ''}")
    output(
        f"统计：指令应答 {counts['ack']} | 设备信息 {counts['info']} | 点云包 {counts['point']} | "
        f"噪声 {counts['skipped']} 字节 | 校验不过的包 {counts['rejected']} 个"
    )
    if counts["point"] == 0 and counts["ack"] == 0:
        output(
            "没收到任何可解析的数据：先核对波特率（D6 出厂 230400）与接线；"
            "原始字节若是乱码，基本就是波特率不对。"
        )
    elif counts["point"] == 0:
        output("有应答但没解出点云包：把上面那行原始字节发出来即可定位。")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="COIN-D6 激光雷达点云采集")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profile", default=None, help="配置档名，默认按当前系统选择")
    parser.add_argument("--output-prefix", type=Path, help="CSV/PNG 文件前缀")
    parser.add_argument("--simulate", action="store_true", help="不接硬件，用内置虚拟雷达数据出图")
    parser.add_argument("--no-window", action="store_true", help="只保存图片，不弹出窗口")
    parser.add_argument(
        "--diagnose", action="store_true",
        help="只发启动指令并打印雷达回传的原始字节与解析统计，用来排查接线/波特率",
    )
    args = parser.parse_args()

    config = load_lidar_config(args.profile, args.config)
    print("COIN-D6 点云采集")
    print(describe_config(config))
    prefix = args.output_prefix or scan_output_prefix()
    if not args.diagnose and (
        prefix.with_suffix(".csv").exists() or prefix.with_suffix(".png").exists()
    ):
        parser.error(f"输出文件已存在：{prefix}")

    if args.simulate:
        print("模拟模式：不打开串口，使用内置虚拟雷达数据验证整条链路。")
    else:
        try:
            import serial  # noqa: F401  (fail before touching hardware config)
        except ImportError as error:
            raise RuntimeError("真机采集需要当前 Python 环境提供 pyserial") from error
        answer = input(
            f"确认 {config.device.port} 已接好激光雷达，且雷达前方无遮挡；回车开始，输入 q 取消："
        ).strip().lower()
        if answer == "q":
            return

    if args.diagnose:
        if args.simulate:
            diagnose(SimulatedLidar(), config)
        else:
            import serial

            try:
                with serial.Serial(
                    config.device.port,
                    baudrate=config.device.baudrate,
                    timeout=config.device.read_timeout_seconds,
                ) as link:
                    diagnose(link, config)
            except OSError as error:
                print(f"无法打开串口 {config.device.port}：{error}")
        return

    lidar = create_lidar(config, simulate=args.simulate)
    try:
        cloud = capture(lidar, config)
    except (LidarError, OSError) as error:
        print(f"采集失败：{error}")
        return

    report(cloud, config)
    csv_path, png_path = save_cloud(cloud, config, prefix, show_window=not args.no_window)
    print(f"点云数据：{csv_path.resolve()}")
    print(f"点云图片：{png_path.resolve()}")


if __name__ == "__main__":
    main()

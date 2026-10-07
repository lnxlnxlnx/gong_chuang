"""COIN-D6 scanning-lidar support.

Everything an operator retunes between machines lives in ``config/coin_d6.json``:
the serial port per platform, the angle and distance window of a scan, the
checksum/acknowledgement policy and the point-cloud rendering options.

    from lidar import CoinD6, load_lidar_config, render_point_cloud
    from project_paths import output_path

    config = load_lidar_config()
    with CoinD6(config.device.port, baudrate=config.device.baudrate) as lidar:
        cloud = lidar.scan(config.scan)
    render_point_cloud(cloud, config.view, output_path("lidar_cloud", "scan.png"))
"""

from .cloud import CloudPoint, PointCloud
from .coin_d6 import CoinD6, LidarAckError, LidarError, LidarTimeoutError, correct_angle
from .config import (
    DEFAULT_CONFIG_PATH,
    CloudView,
    DeviceConfig,
    LidarConfig,
    ScanConfig,
    default_profile_name,
    describe_config,
    load_lidar_config,
)
from .protocol import (
    START_COMMAND,
    STOP_COMMAND,
    Ack,
    DeviceInfo,
    PointCloudPacket,
    SensorPoint,
)
from .visualize import build_figure, render_point_cloud, save_figure, show_figure

__all__ = [
    "Ack",
    "CloudPoint",
    "CloudView",
    "CoinD6",
    "DEFAULT_CONFIG_PATH",
    "DeviceConfig",
    "DeviceInfo",
    "LidarAckError",
    "LidarConfig",
    "LidarError",
    "LidarTimeoutError",
    "PointCloud",
    "PointCloudPacket",
    "START_COMMAND",
    "STOP_COMMAND",
    "ScanConfig",
    "SensorPoint",
    "build_figure",
    "correct_angle",
    "default_profile_name",
    "describe_config",
    "load_lidar_config",
    "render_point_cloud",
    "save_figure",
    "show_figure",
]

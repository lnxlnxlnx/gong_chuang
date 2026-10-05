"""Live COIN-D6 point-cloud window with distance and angle controls.

Menu 17.  A small operator console: the sensor is read in a background thread
and the cloud is redrawn continuously, while the distance and angle windows can
be typed or dragged and take effect immediately.  ``--simulate`` runs the same
window against the built-in fake sensor, so the UI can be tried without
hardware.
"""

from __future__ import annotations

MENU_ENTRY = (17, "D6 点云实时显示")

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from lidar import (
    DEFAULT_CONFIG_PATH,
    CoinD6,
    LidarConfig,
    describe_config,
    load_lidar_config,
    save_figure,
)
from lidar.live import LiveCloud
from lidar.simulate import SimulatedLidar
from lidar.visualize import use_cjk_font

OUTPUT_ROOT = Path("output") / "lidar_cloud"
DISTANCE_LIMIT_MM = 16383
ANGLE_LIMIT_DEGREES = 180.0
REVOLUTIONS_KEPT = 2
COLOR_CHOICES = (("距离", "distance"), ("强度", "intensity"), ("角度", "angle"))
COLOR_LABELS = {"distance": "距离 / mm", "intensity": "强度", "angle": "角度 / °"}


class LiveCloudApp:
    """The operator window: sliders, buttons and the redrawn cloud.

    ``step()`` performs one refresh and is also what the animation timer calls,
    so tests can drive the window without ever showing it.
    """

    def __init__(
        self,
        reader: LiveCloud,
        config: LidarConfig,
        *,
        interval_ms: int = 100,
        mode: Optional[str] = None,
        output: Callable[[str], None] = print,
        output_root: Path = OUTPUT_ROOT,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

        self.reader = reader
        self.config = config
        self.interval_ms = interval_ms
        self.mode = mode or config.device.port
        self.output = output
        self.output_root = Path(output_root)
        self.labels = use_cjk_font()
        self._paused = False
        self._closed = False
        self._color_by = config.view.color_by
        self._saved: Optional[Path] = None
        self._cloud = None
        self._boxes = {}

        self.figure = plt.figure(figsize=(12.5, 7.2), dpi=100)
        self.axes = self.figure.add_axes((0.05, 0.09, 0.46, 0.83))
        self.figure.text(0.05, 0.955, self._title(), fontsize=12)

        self.scatter = self.axes.scatter([], [], c=[], s=config.view.point_size,
                                         cmap=config.view.colormap, linewidths=0)
        self._rings = self._build_grid()
        self.colorbar = self.figure.colorbar(self.scatter, ax=self.axes, fraction=0.046, pad=0.02)

        scan = reader.scan
        self.distance_min = self._slider_row(0.86, "距离下限 / mm", 0.0, DISTANCE_LIMIT_MM,
                                             scan.min_distance_mm, 10.0, self._on_distance)
        self.distance_max = self._slider_row(0.79, "距离上限 / mm", 1.0, DISTANCE_LIMIT_MM,
                                             scan.max_distance_mm, 10.0, self._on_distance)
        self.angle_min = self._slider_row(0.72, "角度下限 / °", -ANGLE_LIMIT_DEGREES,
                                          ANGLE_LIMIT_DEGREES, scan.angle_min_degrees, 1.0,
                                          self._on_angle)
        self.angle_max = self._slider_row(0.65, "角度上限 / °", -ANGLE_LIMIT_DEGREES,
                                          ANGLE_LIMIT_DEGREES, scan.angle_max_degrees, 1.0,
                                          self._on_angle)
        self.revolutions = self._slider_row(0.58, "保留圈数", 1.0, 5.0,
                                            float(reader.revolutions_kept), 1.0,
                                            self._on_revolutions)

        self.figure.text(0.73, 0.51, "着色字段", fontsize=10)
        color_axes = self.figure.add_axes((0.73, 0.395, 0.16, 0.105))
        color_axes.set_title("")
        names = tuple(name for name, _ in COLOR_CHOICES)
        active = next((index for index, (_, key) in enumerate(COLOR_CHOICES)
                       if key == self._color_by), 0)
        self.color_buttons = RadioButtons(color_axes, names, active=active)
        self.color_buttons.on_clicked(self._on_color)

        self.pause_button = Button(self.figure.add_axes((0.73, 0.30, 0.25, 0.05)), "暂停显示")
        self.pause_button.on_clicked(self._on_pause)
        self.clear_button = Button(self.figure.add_axes((0.73, 0.225, 0.25, 0.05)), "清空缓存")
        self.clear_button.on_clicked(self._on_clear)
        self.save_button = Button(self.figure.add_axes((0.73, 0.15, 0.25, 0.05)), "保存快照")
        self.save_button.on_clicked(self._on_save)

        self.status = self.figure.text(0.73, 0.10, "", fontsize=9, va="top", wrap=True)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._animation = None
        self.step()

    # -- public API -------------------------------------------------------
    def step(self, _frame=None):
        """Refresh the cloud, the rings and the status line."""
        if not self._paused:
            self._cloud = self.reader.snapshot()
        cloud = self._cloud
        if cloud is not None:
            self._draw_cloud(cloud)
        self._draw_grid_scale()
        self.status.set_text(self._status_text())
        return self.scatter, self.status

    def show(self) -> None:
        """Open the window and run until the operator closes it."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self._animation = FuncAnimation(
            self.figure, self.step, interval=self.interval_ms, blit=False,
            cache_frame_data=False,
        )
        try:
            plt.show()
        finally:
            self.close()

    def close(self) -> None:
        """Stop the sensor reader; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self.reader.stop()

    def save_snapshot(self) -> Optional[Path]:
        """Write the current view to CSV + PNG under ``output/lidar_cloud``."""
        cloud = self.reader.snapshot()
        if cloud.empty:
            self.output("当前没有点云数据，未保存。")
            return None
        prefix = self.output_root / datetime.now().strftime("live-%Y%m%d-%H%M%S-%f")
        csv_path = cloud.save_csv(prefix.with_suffix(".csv"))
        png_path = save_figure(self.figure, prefix.with_suffix(".png"))
        self._saved = png_path
        self.output(f"已保存快照：{csv_path.resolve()}")
        self.output(f"已保存图片：{png_path.resolve()}")
        return png_path

    # -- drawing ----------------------------------------------------------
    def _draw_cloud(self, cloud) -> None:
        import numpy as np

        if cloud.empty:
            self.scatter.set_offsets(np.empty((0, 2)))
            self.scatter.set_array(np.empty(0))
            return
        self.scatter.set_offsets(np.array([[point.x_mm, point.y_mm] for point in cloud]))
        self.scatter.set_array(np.array(self._color_values(cloud)))
        self.scatter.set_clim(*self._color_limits())

    def _build_grid(self):
        from matplotlib.patches import Circle

        rings = []
        for _ in range(4):
            circle = Circle((0.0, 0.0), 1.0, fill=False, color="#dde5e6", linewidth=0.8, zorder=0)
            self.axes.add_patch(circle)
            label = self.axes.text(0.0, 0.0, "", color="#8a9a9c", fontsize=8, ha="left", va="bottom")
            rings.append((circle, label))
        self.axes.axhline(0.0, color="#c8d2d3", linewidth=0.8, zorder=0)
        self.axes.axvline(0.0, color="#c8d2d3", linewidth=0.8, zorder=0)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True, color="#eef2f2", linewidth=0.6)
        self.axes.set_axisbelow(True)
        self.axes.set_xlabel(self.labels["x"])
        self.axes.set_ylabel(self.labels["y"])
        return rings

    def _draw_grid_scale(self) -> None:
        import math

        limit = self._view_limit()
        self.axes.set_xlim(-limit, limit)
        self.axes.set_ylim(-limit, limit)
        for index, (circle, label) in enumerate(self._rings, start=1):
            radius = limit * index / len(self._rings)
            circle.set_radius(radius)
            label.set_position((radius * math.cos(math.radians(35)),
                                radius * math.sin(math.radians(35))))
            label.set_text(f"{radius:g} mm")
        self.colorbar.set_label(COLOR_LABELS.get(self._color_by, self._color_by))

    def _color_values(self, cloud):
        if self._color_by == "intensity":
            return [point.intensity for point in cloud]
        if self._color_by == "angle":
            return [point.angle_degrees for point in cloud]
        return [point.distance_mm for point in cloud]

    def _color_limits(self):
        if self._color_by == "intensity":
            return 0.0, 255.0
        if self._color_by == "angle":
            return -ANGLE_LIMIT_DEGREES, ANGLE_LIMIT_DEGREES
        return float(self.reader.scan.min_distance_mm), float(self.reader.scan.max_distance_mm)

    def _view_limit(self) -> float:
        return max(500.0, float(self.reader.scan.max_distance_mm))

    # -- widgets ----------------------------------------------------------
    def _slider_row(self, y: float, label: str, minimum: float, maximum: float,
                    initial: float, step: float, handler):
        """One labelled slider with a matching text box for typed values."""
        from matplotlib.widgets import Slider, TextBox

        self.figure.text(0.72, y + 0.011, label, fontsize=9, ha="right", va="center")
        slider = Slider(
            self.figure.add_axes((0.73, y, 0.16, 0.022)), "", minimum, maximum,
            valinit=initial, valstep=step, valfmt=None,
        )
        box = TextBox(self.figure.add_axes((0.90, y, 0.08, 0.022)), "", initial=f"{initial:g}")
        self._boxes[slider] = box
        slider.on_changed(handler)
        slider.on_changed(lambda value: box.set_val(f"{value:g}"))
        box.on_submit(lambda text: self._submit(slider, text))
        return slider

    def _submit(self, slider, text: str) -> None:
        """Accept a typed value, clamp it into the slider's range and apply it."""
        box = self._boxes[slider]
        try:
            value = float(text)
        except ValueError:
            box.set_val(f"{slider.val:g}")
            return
        value = min(max(value, slider.valmin), slider.valmax)
        box.set_val(f"{value:g}")
        slider.set_val(value)

    def _on_distance(self, _value) -> None:
        self.reader.set_filters(
            min_distance_mm=int(self.distance_min.val),
            max_distance_mm=int(self.distance_max.val),
        )

    def _on_angle(self, _value) -> None:
        self.reader.set_filters(
            angle_min_degrees=float(self.angle_min.val),
            angle_max_degrees=float(self.angle_max.val),
        )

    def _on_revolutions(self, value) -> None:
        self.reader.set_revolutions_kept(int(value))

    def _on_color(self, label: str) -> None:
        self._color_by = dict(COLOR_CHOICES)[label]

    def _on_pause(self, _event) -> None:
        self._paused = not self._paused
        self.pause_button.label.set_text("继续显示" if self._paused else "暂停显示")

    def _on_clear(self, _event) -> None:
        self.reader.clear()
        self._cloud = None

    def _on_save(self, _event) -> None:
        self.save_snapshot()

    def _on_close(self, _event) -> None:
        self.close()

    # -- text -------------------------------------------------------------
    def _title(self) -> str:
        return f"COIN-D6 实时点云 | {self.mode} | {self.config.device.baudrate} bps"

    def _status_text(self) -> str:
        stats = self.reader.stats
        scan = self.reader.scan
        lines = [
            f"点 {stats.points} | 圈 {stats.revolutions} | 包 {stats.packets} | "
            f"丢弃包 {stats.rejected} | 噪声 {stats.skipped_bytes} 字节",
            f"显示窗口：{scan.min_distance_mm}~{scan.max_distance_mm} mm | "
            f"{scan.angle_min_degrees:g}~{scan.angle_max_degrees:g}°"
            + ("　[已暂停]" if self._paused else ""),
        ]
        if stats.error:
            lines.append(f"错误：{stats.error}")
        elif stats.idle_seconds is None:
            lines.append("正在等待雷达数据…")
        elif stats.idle_seconds > 2.0:
            lines.append(f"已 {stats.idle_seconds:.1f} 秒没有收到数据")
        if self._saved is not None:
            lines.append(f"上次保存：{self._saved.name}")
        return "\n".join(lines)


def create_reader(config: LidarConfig, *, simulate: bool = False,
                  output: Callable[[str], None] = print) -> LiveCloud:
    """Build the background reader for real hardware or the fake sensor."""
    if simulate:
        lidar = CoinD6(config.device.port, serial_port=SimulatedLidar())
    else:
        lidar = CoinD6(
            config.device.port,
            baudrate=config.device.baudrate,
            timeout=config.device.read_timeout_seconds,
            ack_timeout=config.device.ack_timeout_seconds,
            strict_checksum=config.device.strict_checksum,
            require_ack=config.device.require_ack,
        )
    return LiveCloud(lidar, config.scan, revolutions_kept=REVOLUTIONS_KEPT, on_note=output)


def main() -> None:
    parser = argparse.ArgumentParser(description="COIN-D6 实时点云显示")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profile", default=None, help="配置档名，默认按当前系统选择")
    parser.add_argument("--simulate", action="store_true", help="不接硬件，用内置虚拟雷达驱动界面")
    parser.add_argument("--interval", type=int, default=100, help="刷新间隔（毫秒）")
    args = parser.parse_args()

    config = load_lidar_config(args.profile, args.config)
    print("COIN-D6 实时点云显示")
    print(describe_config(config))
    if args.simulate:
        print("模拟模式：不打开串口，使用内置虚拟雷达数据。")
    else:
        print(f"正在连接 {config.device.port}；关闭窗口即停止雷达并释放串口。")

    reader = create_reader(config, simulate=args.simulate)
    app = LiveCloudApp(
        reader, config, interval_ms=args.interval,
        mode="模拟数据" if args.simulate else config.device.port,
    )
    reader.start()
    try:
        app.show()
    finally:
        reader.stop()
        print("已停止采集并释放串口。")


if __name__ == "__main__":
    main()

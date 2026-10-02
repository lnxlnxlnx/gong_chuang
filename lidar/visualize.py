"""Draw a COIN-D6 point cloud as a top-down figure and save it as a PNG."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from .cloud import PointCloud
from .config import CloudView


#: Preferred CJK fonts, in order.  When none is installed the figure falls back
#: to English labels instead of printing missing-glyph boxes.
CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
)

LABELS = {
    "cn": {
        "title": "COIN-D6 点云",
        "x": "x / mm（前方）",
        "y": "y / mm（左方）",
        "distance": "距离 / mm",
        "intensity": "强度",
        "angle": "角度 / °",
        "empty": "无有效点",
        "unknown": "未知型号",
        "points": "{count} 点",
        "turns": "{count} 圈",
    },
    "en": {
        "title": "COIN-D6 point cloud",
        "x": "x / mm (forward)",
        "y": "y / mm (left)",
        "distance": "distance / mm",
        "intensity": "intensity",
        "angle": "angle / deg",
        "empty": "no points",
        "unknown": "unknown model",
        "points": "{count} points",
        "turns": "{count} revolutions",
    },
}


def render_point_cloud(
    cloud: PointCloud,
    view: CloudView,
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> Path:
    """Draw the cloud and save it; returns the written PNG path."""
    figure = build_figure(cloud, view, title=title)
    try:
        return save_figure(figure, output_path)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def build_figure(cloud: PointCloud, view: CloudView, *, title: Optional[str] = None):
    """Build the matplotlib figure without displaying it."""
    import matplotlib.pyplot as plt

    labels = use_cjk_font()
    figure, axes = plt.subplots(figsize=(7.6, 7.2), dpi=110)
    _draw_grid(axes, view)
    if not cloud.empty:
        scatter = axes.scatter(
            [point.x_mm for point in cloud],
            [point.y_mm for point in cloud],
            c=_color_values(cloud, view.color_by),
            s=view.point_size,
            cmap=view.colormap,
            linewidths=0,
        )
        bar = figure.colorbar(scatter, ax=axes, fraction=0.046, pad=0.03)
        bar.set_label(labels.get(view.color_by, view.color_by))
    axes.set_title(f"{title or labels['title']}\n{_subtitle(cloud, labels)}", fontsize=12)
    axes.set_xlabel(labels["x"])
    axes.set_ylabel(labels["y"])
    figure.tight_layout()
    return figure


def save_figure(figure, output_path: Path) -> Path:
    """Write the figure to ``output_path``, creating its directory first."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    return output_path


def show_figure(figure) -> None:
    """Open the interactive window; the caller closes it by hand.

    Headless backends (``Agg``) have nothing to display, so this returns
    immediately instead of warning that the canvas is non-interactive.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() == "agg":
        return
    plt.show()


def use_cjk_font() -> dict:
    """Select the label set and, when installed, a Chinese-capable font."""
    import matplotlib
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name] + list(
                matplotlib.rcParams["font.sans-serif"]
            )
            matplotlib.rcParams["axes.unicode_minus"] = False
            return LABELS["cn"]
    return LABELS["en"]


def _draw_grid(axes, view: CloudView) -> None:
    """Draw the equal-aspect grid, range rings and origin cross."""
    from matplotlib.patches import Circle

    limit = view.range_mm
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.axhline(0.0, color="#c8d2d3", linewidth=0.8, zorder=0)
    axes.axvline(0.0, color="#c8d2d3", linewidth=0.8, zorder=0)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        radius = limit * fraction
        axes.add_patch(Circle(
            (0.0, 0.0), radius, fill=False, color="#dde5e6", linewidth=0.8, zorder=0
        ))
        axes.text(radius * math.cos(math.radians(35)), radius * math.sin(math.radians(35)),
                  f"{radius:g} mm", color="#8a9a9c", fontsize=8, ha="left", va="bottom")
    axes.grid(True, color="#eef2f2", linewidth=0.6)
    axes.set_axisbelow(True)


def _color_values(cloud: PointCloud, color_by: str):
    if color_by == "intensity":
        return [point.intensity for point in cloud]
    if color_by == "angle":
        return [point.angle_degrees for point in cloud]
    return [point.distance_mm for point in cloud]


def _subtitle(cloud: PointCloud, labels: dict) -> str:
    if cloud.empty:
        return labels["empty"]
    model = cloud.device.model if cloud.device is not None else labels["unknown"]
    return " | ".join((
        model,
        labels["points"].format(count=len(cloud)),
        labels["turns"].format(count=cloud.revolutions),
        f"{cloud.nearest_mm}~{cloud.farthest_mm} mm",
    ))

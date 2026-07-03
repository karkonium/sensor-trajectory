"""Sliding-experiment plotting helpers for per-window frame rendering."""

import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, TwoSlopeNorm, to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from experiments.common.plot_style import (
    apply_axis_style,
    apply_dark_axis_style,
    color_for_method,
    display_label,
    finalize_dark_legend,
    finalize_legend,
    paper_dark_plot_context,
    paper_plot_context,
    pretty_flow_name,
    SPINE_COLOR_DARK,
    TEXT_COLOR_DARK,
)


METRIC_LABEL = r"Relative $L_h^2$ Error"
TRAJECTORY_FONT_SIZE = 11
TRAJECTORY_TITLE_FONTSIZE = TRAJECTORY_FONT_SIZE
TRAJECTORY_LEGEND_FONTSIZE = TRAJECTORY_FONT_SIZE
TRAJECTORY_LEGEND_TITLE_FONTSIZE = TRAJECTORY_FONT_SIZE
TRAJECTORY_INITIAL_EDGE_COLOR = "#98A2B3"
FLOW_SNAPSHOT_TITLE_FONTSIZE = TRAJECTORY_FONT_SIZE
FLOW_SNAPSHOT_WIDE_DOMAIN_ASPECT = 3.0
FLOW_SNAPSHOT_TARGET_SHORT_ARROWS = 6
FLOW_SNAPSHOT_MAX_LONG_ARROWS = 24
FLOW_SNAPSHOT_VORTICITY_CMAP = "RdBu_r"
FLOW_SNAPSHOT_VORTICITY_ALPHA = 0.70
FLOW_SNAPSHOT_TARGET_ARROW_LENGTH_FRACTION = 0.075
FLOW_SNAPSHOT_QUIVER_WIDTH = 0.0021
CYLINDER_OBSTACLE_CENTER = (0.0, 0.0)
CYLINDER_OBSTACLE_RADIUS = 0.0625
VORTICITY_SYMLOG_LINTHRESH_FRACTION = 0.012

PLACEMENT_COLORS = {
    "Fixed": color_for_method("Fixed"),
    "Eulerian": color_for_method("Eulerian"),
    "Lagrangian": color_for_method("Lagrangian"),
    "QR teleport": color_for_method("QR teleport"),
    "Moving POD-QR": color_for_method("Moving POD-QR"),
}

TRAJECTORY_STYLE = {
    "Lagrangian": {
        "palette": ("#14B8A6", "#0891B2", "#2563EB", "#4F46E5", "#7C3AED"),
        "marker": "o",
    },
    "Moving POD-QR": {
        "palette": ("#F59E0B", "#F97316", "#EF4444", "#DC2626", "#DB2777"),
        "marker": "o",
    },
}

ANIMATION_CMAP = "magma"
ANIMATION_QUIVER_COLOR = "#E2E8F0"
ANIMATION_CONTOUR_COLOR = "#F8FAFC"
ANIMATION_SENSOR_STYLE = {
    "Lagrangian": {
        "color": "#43D3C2",
        "marker": "o",
    },
    "Moving POD-QR": {
        "color": "#7CC4FF",
        "marker": "o",
    },
}


class _ColorStrip:
    """Legend proxy for a row of sensor-family colors."""

    def __init__(self, colors):
        self.colors = list(colors)


class _HandlerColorStrip(HandlerBase):
    """Draw a compact multi-color bar inside a Matplotlib legend handle."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        colors = orig_handle.colors
        if not colors:
            colors = ["#6B7280"]

        strip_height = 0.56 * height
        strip_y = ydescent + 0.22 * height
        strip_width = width / len(colors)
        return [
            Rectangle(
                (xdescent + color_idx * strip_width, strip_y),
                strip_width,
                strip_height,
                transform=trans,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
            )
            for color_idx, color in enumerate(colors)
        ]


def _plot_l2h_history(axis, l2h_records, total_windows, r_norm_history=None):
    """Plot relative L2_h error history and optional residual norm history."""
    if total_windows is None or total_windows <= 0:
        axis.set_xlim(0, 1)
        axis.set_xlabel("window")
        axis.set_ylabel(METRIC_LABEL)
        axis.set_title(f"{METRIC_LABEL} Over Window")
        apply_axis_style(axis, x_grid=True, y_grid=True)
        return

    l2h_records = l2h_records or []
    basis_names = {
        str(record.get("basis", "")).strip()
        for record in l2h_records
        if str(record.get("basis", "")).strip()
    }
    show_basis = len(basis_names) > 1

    labels = []
    series_by_label = {}
    for record in l2h_records:
        placement_name = str(record["placement"]).strip()
        basis_name = str(record.get("basis", "")).strip()
        display_name = display_label(placement_name)
        label = f"{display_name} - {basis_name}" if show_basis and basis_name else display_name

        if label not in series_by_label:
            series_by_label[label] = np.full(int(total_windows), np.nan, dtype=float)
            labels.append(label)

        window_idx = int(record["window"])
        if 0 <= window_idx < total_windows:
            series_by_label[label][window_idx] = float(record["L2_h"])

    window_axis = np.arange(int(total_windows))
    for label in labels:
        placement_name = label.split(" - ", 1)[0]
        axis.plot(
            window_axis,
            series_by_label[label],
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            color=PLACEMENT_COLORS.get(placement_name),
            label=label,
        )

    if r_norm_history:
        r_norm_series = np.full(int(total_windows), np.nan, dtype=float)
        valid_count = min(len(r_norm_history), int(total_windows))
        r_values = np.asarray(r_norm_history[:valid_count], dtype=float)
        r_values[r_values <= 0.0] = np.nan
        r_norm_series[:valid_count] = r_values
        axis.plot(
            window_axis,
            r_norm_series,
            color="#111827",
            linestyle="--",
            linewidth=2.0,
            marker="^",
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=r"Window Relative $\|r\|_h$",
        )

    axis.set_xlim(0, max(int(total_windows) - 1, 1))
    axis.set_yscale("log")
    axis.set_xlabel("Window")
    axis.set_ylabel(METRIC_LABEL)
    axis.set_title(rf"{METRIC_LABEL} and Window $\|r\|_h$")
    apply_axis_style(axis, x_grid=True, y_grid=True)
    legend_cols = 1 if len(labels) <= 4 else 2
    finalize_legend(axis, loc="upper right", ncol=legend_cols)


def _offset_points(points, x_origin=0.0, y_origin=0.0):
    """Return display-coordinate points shifted by the plot origin."""
    display_points = np.asarray(points, dtype=float).copy()
    display_points[..., 0] += float(x_origin)
    display_points[..., 1] += float(y_origin)
    return display_points


def _flow_mesh(u_grid, v_grid, lx, ly, x_origin=0.0, y_origin=0.0):
    """Build physical-coordinate grids and speed magnitude for one flow snapshot."""
    nx, ny = u_grid.shape
    x_coords = np.linspace(float(x_origin), float(x_origin) + float(lx), nx)
    y_coords = np.linspace(float(y_origin), float(y_origin) + float(ly), ny)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
    speed = np.hypot(u_grid, v_grid)
    return x_grid, y_grid, speed


def _flow_vorticity(u_grid, v_grid, lx, ly):
    """Compute scalar vorticity dv/dx - du/dy on the physical grid."""
    dx = float(lx) / max(int(u_grid.shape[0]) - 1, 1)
    dy = float(ly) / max(int(u_grid.shape[1]) - 1, 1)
    dv_dx = np.gradient(v_grid, dx, axis=0)
    du_dy = np.gradient(u_grid, dy, axis=1)
    return dv_dx - du_dy


def _flow_snapshot_vorticity_abs_max(flow_snapshots, lx, ly):
    """Return a shared symmetric vorticity scale for the compact flow snapshots."""
    maxima = []
    for snapshot in flow_snapshots:
        u_grid = np.asarray(snapshot["u_grid"], dtype=float)
        v_grid = np.asarray(snapshot["v_grid"], dtype=float)
        vorticity = _flow_vorticity(u_grid, v_grid, lx, ly)
        finite_vorticity = vorticity[np.isfinite(vorticity)]
        if finite_vorticity.size:
            maxima.append(float(np.max(np.abs(finite_vorticity))))

    if not maxima:
        return 1.0
    return max(max(maxima), float(np.finfo(float).eps))


def _is_cylinder_wake_plot(run_name, lx, ly):
    """Return whether the trajectory summary should use cylinder-wake layout/styling."""
    del lx, ly
    if run_name and "cylinder" in str(run_name).lower():
        return True
    return False


def _vorticity_norm(vorticity_abs_max, use_symlog=False):
    """Build the shared vorticity normalization for flow snapshot panels."""
    vorticity_abs_max = max(float(vorticity_abs_max), float(np.finfo(float).eps))
    if use_symlog:
        linthresh = max(
            VORTICITY_SYMLOG_LINTHRESH_FRACTION * vorticity_abs_max,
            float(np.finfo(float).eps),
        )
        return SymLogNorm(
            linthresh=linthresh,
            linscale=0.8,
            vmin=-vorticity_abs_max,
            vmax=vorticity_abs_max,
            base=10.0,
        )

    return TwoSlopeNorm(vmin=-vorticity_abs_max, vcenter=0.0, vmax=vorticity_abs_max)


def _vorticity_colorbar_ticks(vorticity_norm, vorticity_abs_max):
    """Return compact signed vorticity ticks shared by all snapshot panels."""
    del vorticity_norm
    vorticity_abs_max = max(float(vorticity_abs_max), float(np.finfo(float).eps))
    return [-vorticity_abs_max, 0.0, vorticity_abs_max]


def _vorticity_colorbar_tick_labels(ticks):
    """Format vorticity colorbar ticks without noisy decimals."""
    max_abs_tick = max((abs(float(tick)) for tick in ticks), default=0.0)
    if max_abs_tick >= 1.0:
        return [str(int(round(float(tick)))) for tick in ticks]

    decimals = 1 if max_abs_tick >= 0.05 else 2
    labels = []
    for tick in ticks:
        rounded_tick = round(float(tick), decimals)
        if rounded_tick == 0.0:
            labels.append("0")
        else:
            labels.append(f"{rounded_tick:.{decimals}f}")
    return labels


def _flow_snapshot_speed_reference(speed):
    """Use a local high-percentile speed so quiver arrows remain visible across flows."""
    finite_speed = np.asarray(speed, dtype=float)
    finite_speed = finite_speed[np.isfinite(finite_speed)]
    finite_speed = finite_speed[finite_speed > float(np.finfo(float).eps)]
    if finite_speed.size == 0:
        return float(np.finfo(float).eps)
    return max(float(np.percentile(finite_speed, 90.0)), float(np.finfo(float).eps))


def _flow_snapshot_quiver_steps(u_shape, lx, ly, quiver_step):
    """Choose quiver strides for compact flow snapshots, with extra density for long domains."""
    nx, ny = u_shape
    min_step = max(int(quiver_step), 1)
    default_step = max(min_step, int(np.ceil(max(nx, ny) / 12.0)), 1)
    lx = max(float(lx), float(np.finfo(float).eps))
    ly = max(float(ly), float(np.finfo(float).eps))
    domain_aspect = max(lx / ly, ly / lx)
    if domain_aspect < FLOW_SNAPSHOT_WIDE_DOMAIN_ASPECT:
        return default_step, default_step

    if lx >= ly:
        target_x = min(
            FLOW_SNAPSHOT_MAX_LONG_ARROWS,
            max(FLOW_SNAPSHOT_TARGET_SHORT_ARROWS, int(np.ceil(FLOW_SNAPSHOT_TARGET_SHORT_ARROWS * lx / ly))),
        )
        target_y = FLOW_SNAPSHOT_TARGET_SHORT_ARROWS
    else:
        target_x = FLOW_SNAPSHOT_TARGET_SHORT_ARROWS
        target_y = min(
            FLOW_SNAPSHOT_MAX_LONG_ARROWS,
            max(FLOW_SNAPSHOT_TARGET_SHORT_ARROWS, int(np.ceil(FLOW_SNAPSHOT_TARGET_SHORT_ARROWS * ly / lx))),
        )

    x_step = max(min_step, int(np.ceil(nx / target_x)), 1)
    y_step = max(min_step, int(np.ceil(ny / target_y)), 1)
    return x_step, y_step


def _sensor_palette_colors(label, n_sensors):
    """Return distinguishable colors within the requested trajectory family."""
    style = TRAJECTORY_STYLE[label]
    sensor_count = max(int(n_sensors), 1)
    cmap = LinearSegmentedColormap.from_list(
        f"{label.lower().replace(' ', '_')}_sensor_palette",
        style["palette"],
    )
    if sensor_count == 1:
        color_positions = [0.5]
    else:
        color_positions = np.linspace(0.08, 0.92, sensor_count)

    return [cmap(position) for position in color_positions]


def _style_dark_colorbar(colorbar, label):
    """Apply dark-theme styling to a Matplotlib colorbar."""
    colorbar.set_label(label, color=TEXT_COLOR_DARK)
    colorbar.ax.tick_params(colors=TEXT_COLOR_DARK, width=0.85)
    colorbar.outline.set_edgecolor(SPINE_COLOR_DARK)
    colorbar.outline.set_linewidth(0.85)


def _draw_dark_flow_field(axis, x_grid, y_grid, u_grid, v_grid, speed, speed_max, quiver_step):
    """Draw a dark themed flow-field background with a stable speed color scale."""
    resolved_speed_max = max(float(speed_max), float(np.finfo(float).eps))
    levels = np.linspace(0.0, resolved_speed_max, 18)

    contour = axis.contourf(
        x_grid,
        y_grid,
        speed,
        levels=levels,
        cmap=ANIMATION_CMAP,
        vmin=0.0,
        vmax=resolved_speed_max,
        extend="max",
    )
    axis.contour(
        x_grid,
        y_grid,
        speed,
        levels=levels[::3],
        colors=ANIMATION_CONTOUR_COLOR,
        linewidths=0.35,
        alpha=0.12,
    )
    axis.quiver(
        x_grid[::quiver_step, ::quiver_step],
        y_grid[::quiver_step, ::quiver_step],
        u_grid[::quiver_step, ::quiver_step],
        v_grid[::quiver_step, ::quiver_step],
        color=ANIMATION_QUIVER_COLOR,
        alpha=0.30,
        scale_units="xy",
        scale=None,
        width=0.0025,
        pivot="mid",
    )
    return contour


def _unwrap_periodic_history(history, lx, ly):
    """Unwrap periodic trajectories so boundary crossings stay visually continuous."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 3 or history.shape[2] != 2:
        raise ValueError("history must have shape (n_steps, n_sensors, 2)")

    unwrapped = history.copy()
    domain_lengths = (float(lx), float(ly))
    for axis_idx, domain_length in enumerate(domain_lengths):
        if domain_length <= 0.0:
            continue

        deltas = np.diff(history[:, :, axis_idx], axis=0)
        jumps = np.zeros_like(deltas)
        jumps[deltas > 0.5 * domain_length] = -domain_length
        jumps[deltas < -0.5 * domain_length] = domain_length
        cumulative_shift = np.vstack([np.zeros((1, history.shape[1])), np.cumsum(jumps, axis=0)])
        unwrapped[:, :, axis_idx] += cumulative_shift

    return unwrapped


def _plot_trajectory_family(
    axis,
    history,
    label,
    lx=None,
    ly=None,
    periodic=False,
    x_origin=0.0,
    y_origin=0.0,
):
    """Plot one family of sensor trajectories on a shared axis."""
    style = TRAJECTORY_STYLE[label]
    sensor_colors = _sensor_palette_colors(label, history.shape[1])

    if history.shape[0] >= 2:
        num_segments = history.shape[0] - 1
        segment_fade = np.linspace(0.16, 0.92, num_segments)
        segment_widths = np.linspace(0.75, 2.15, num_segments)
        for sensor_idx, sensor_color in enumerate(sensor_colors):
            path = history[:, sensor_idx, :]
            display_path = _offset_points(path, x_origin=x_origin, y_origin=y_origin)
            segments = np.stack([display_path[:-1], display_path[1:]], axis=1)
            segment_colors = np.tile(np.asarray(to_rgba(sensor_color)), (num_segments, 1))
            segment_colors[:, 3] = segment_fade
            segment_widths_use = segment_widths
            if periodic:
                keep_segments = np.ones(num_segments, dtype=bool)
                if lx is not None and float(lx) > 0.0:
                    keep_segments &= np.abs(np.diff(path[:, 0])) <= 0.5 * float(lx)
                if ly is not None and float(ly) > 0.0:
                    keep_segments &= np.abs(np.diff(path[:, 1])) <= 0.5 * float(ly)
                segments = segments[keep_segments]
                segment_colors = segment_colors[keep_segments]
                segment_widths_use = segment_widths[keep_segments]
                if segments.shape[0] == 0:
                    continue

            line_collection = LineCollection(
                segments,
                colors=segment_colors,
                linewidths=segment_widths_use,
                capstyle="round",
                joinstyle="round",
                zorder=3,
            )
            axis.add_collection(line_collection)

    initial_positions = _offset_points(history[0], x_origin=x_origin, y_origin=y_origin)
    end_positions = _offset_points(history[-1], x_origin=x_origin, y_origin=y_origin)
    axis.scatter(
        initial_positions[:, 0],
        initial_positions[:, 1],
        s=54,
        marker="o",
        facecolors="none",
        edgecolors=TRAJECTORY_INITIAL_EDGE_COLOR,
        linewidths=1.45,
        alpha=0.98,
        zorder=6,
    )
    axis.scatter(
        end_positions[:, 0],
        end_positions[:, 1],
        s=48,
        marker=style["marker"],
        facecolors=sensor_colors,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.98,
        zorder=5,
    )

    return sensor_colors


def _finalize_trajectory_legend(axis, lagrangian_colors, moving_colors):
    """Create the custom trajectory legend with marker and palette keys."""
    handles, labels, handler_map = _trajectory_legend_parts(lagrangian_colors, moving_colors)
    finalize_legend(
        axis,
        handles=handles,
        labels=labels,
        handler_map=handler_map,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="Sensors",
        fontsize=TRAJECTORY_LEGEND_FONTSIZE,
        title_fontsize=TRAJECTORY_LEGEND_TITLE_FONTSIZE,
        handlelength=2.4,
        handletextpad=0.8,
    )


def _trajectory_legend_parts(lagrangian_colors, moving_colors):
    """Return reusable trajectory legend handles, labels, and handler map."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=TRAJECTORY_INITIAL_EDGE_COLOR,
            markeredgewidth=1.45,
            markersize=7.2,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#111827",
            markeredgecolor="#111827",
            markersize=7.2,
        ),
        _ColorStrip(lagrangian_colors),
        _ColorStrip(moving_colors),
    ]
    labels = ["Initial", "Current", "Lagrangian colors", "Moving POD-QR colors"]
    return handles, labels, {_ColorStrip: _HandlerColorStrip()}


def _trajectory_limits(histories, lx, ly, periodic, x_origin=0.0, y_origin=0.0):
    """Choose axis limits for wrapped or non-wrapped trajectory plots."""
    x_origin = float(x_origin)
    y_origin = float(y_origin)
    if periodic:
        return (x_origin, x_origin + float(lx)), (y_origin, y_origin + float(ly))

    if not histories:
        return (x_origin, x_origin + float(lx)), (y_origin, y_origin + float(ly))

    if not periodic:
        has_plot_origin = abs(x_origin) > 0.0 or abs(y_origin) > 0.0
        x_pad = 0.0 if has_plot_origin else 0.04 * float(lx)
        y_pad = 0.0 if has_plot_origin else 0.04 * float(ly)
        return (
            x_origin - x_pad,
            x_origin + float(lx) + x_pad,
        ), (
            y_origin - y_pad,
            y_origin + float(ly) + y_pad,
        )

    stacked_points = np.concatenate([history.reshape(-1, 2) for history in histories], axis=0)
    x_min = float(np.min(stacked_points[:, 0]))
    x_max = float(np.max(stacked_points[:, 0]))
    y_min = float(np.min(stacked_points[:, 1]))
    y_max = float(np.max(stacked_points[:, 1]))
    x_pad = 0.08 * float(lx)
    y_pad = 0.08 * float(ly)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _draw_periodic_domain_guides(axis, x_limits, y_limits, lx, ly, x_origin=0.0, y_origin=0.0):
    """Draw faint dashed guides marking periodic domain copies."""
    if lx > 0.0:
        x_start = int(np.floor((x_limits[0] - x_origin) / lx))
        x_end = int(np.ceil((x_limits[1] - x_origin) / lx))
        for k_idx in range(x_start, x_end + 1):
            axis.axvline(
                x_origin + k_idx * lx,
                color="#98A2B3",
                linestyle=(0, (3, 3)),
                linewidth=0.9,
                alpha=0.45,
                zorder=0,
            )

    if ly > 0.0:
        y_start = int(np.floor((y_limits[0] - y_origin) / ly))
        y_end = int(np.ceil((y_limits[1] - y_origin) / ly))
        for k_idx in range(y_start, y_end + 1):
            axis.axhline(
                y_origin + k_idx * ly,
                color="#98A2B3",
                linestyle=(0, (3, 3)),
                linewidth=0.9,
                alpha=0.45,
                zorder=0,
            )


def _crosses_periodic_boundary(point_a, point_b, lx, ly):
    """Return whether a wrapped-domain segment crosses a periodic boundary."""
    if lx > 0.0 and abs(float(point_b[0]) - float(point_a[0])) > 0.5 * float(lx):
        return True
    if ly > 0.0 and abs(float(point_b[1]) - float(point_a[1])) > 0.5 * float(ly):
        return True
    return False


def _plot_fading_sensor_tail(
    axis,
    history,
    label,
    lx,
    ly,
    periodic=False,
    tail_length=48,
    x_origin=0.0,
    y_origin=0.0,
):
    """Plot one sensor family with a fading trajectory tail and current-position dots."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 3 or history.shape[2] != 2 or history.shape[0] == 0:
        return

    style = ANIMATION_SENSOR_STYLE[label]
    tail_steps = max(int(tail_length), 1)
    recent_history = history[-min(history.shape[0], tail_steps + 1) :]
    num_segments = max(recent_history.shape[0] - 1, 0)

    if num_segments > 0:
        for sensor_idx in range(recent_history.shape[1]):
            sensor_path = recent_history[:, sensor_idx, :]
            for segment_idx in range(num_segments):
                point_a = sensor_path[segment_idx]
                point_b = sensor_path[segment_idx + 1]
                if periodic and _crosses_periodic_boundary(point_a, point_b, lx, ly):
                    continue

                age_fraction = float(segment_idx + 1) / float(num_segments)
                axis.plot(
                    [point_a[0] + x_origin, point_b[0] + x_origin],
                    [point_a[1] + y_origin, point_b[1] + y_origin],
                    color=style["color"],
                    linewidth=0.8 + 1.3 * age_fraction,
                    alpha=0.08 + 0.78 * age_fraction,
                    solid_capstyle="round",
                    zorder=4,
                )

    current_positions = _offset_points(recent_history[-1], x_origin=x_origin, y_origin=y_origin)
    axis.scatter(
        current_positions[:, 0],
        current_positions[:, 1],
        s=20,
        marker=style["marker"],
        color=style["color"],
        edgecolors="white",
        linewidths=0.4,
        alpha=0.98,
        label=label,
        zorder=6,
    )


def _plot_flow_snapshot_sensor_dots(axis, positions, label, zorder, x_origin=0.0, y_origin=0.0):
    """Overlay sensor positions as compact dots on a flow snapshot."""
    if positions is None:
        return

    positions = _offset_points(positions, x_origin=x_origin, y_origin=y_origin)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        return

    sensor_colors = _sensor_palette_colors(label, positions.shape[0])
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=32,
        marker="o",
        facecolors=sensor_colors,
        edgecolors="white",
        linewidths=0.45,
        alpha=0.98,
        zorder=zorder,
    )


def _plot_snapshot_trajectory_family(
    axis,
    history,
    label,
    lx,
    ly,
    periodic=False,
    x_origin=0.0,
    y_origin=0.0,
):
    """Overlay one trajectory family up to the current snapshot."""
    history = np.asarray(history, dtype=float)
    if history.ndim == 2 and history.shape[1] == 2:
        history = history[None, ...]
    if history.ndim != 3 or history.shape[2] != 2 or history.shape[0] == 0:
        return []

    style = TRAJECTORY_STYLE[label]
    sensor_colors = _sensor_palette_colors(label, history.shape[1])
    if history.shape[0] >= 2:
        num_segments = history.shape[0] - 1
        segment_alpha = np.linspace(0.14, 0.92, num_segments)
        segment_widths = np.linspace(0.55, 1.65, num_segments)
        for sensor_idx, sensor_color in enumerate(sensor_colors):
            path = history[:, sensor_idx, :]
            display_path = _offset_points(path, x_origin=x_origin, y_origin=y_origin)
            segments = np.stack([display_path[:-1], display_path[1:]], axis=1)
            segment_colors = np.tile(np.asarray(to_rgba(sensor_color)), (num_segments, 1))
            segment_colors[:, 3] = segment_alpha
            segment_widths_use = segment_widths
            if periodic:
                keep_segments = np.ones(num_segments, dtype=bool)
                if lx > 0.0:
                    keep_segments &= np.abs(np.diff(path[:, 0])) <= 0.5 * float(lx)
                if ly > 0.0:
                    keep_segments &= np.abs(np.diff(path[:, 1])) <= 0.5 * float(ly)
                segments = segments[keep_segments]
                segment_colors = segment_colors[keep_segments]
                segment_widths_use = segment_widths[keep_segments]
                if segments.shape[0] == 0:
                    continue

            line_collection = LineCollection(
                segments,
                colors=segment_colors,
                linewidths=segment_widths_use,
                capstyle="round",
                joinstyle="round",
                zorder=6,
            )
            axis.add_collection(line_collection)

    initial_positions = _offset_points(history[0], x_origin=x_origin, y_origin=y_origin)
    current_positions = _offset_points(history[-1], x_origin=x_origin, y_origin=y_origin)
    axis.scatter(
        initial_positions[:, 0],
        initial_positions[:, 1],
        s=34,
        marker="o",
        facecolors="none",
        edgecolors=TRAJECTORY_INITIAL_EDGE_COLOR,
        linewidths=1.1,
        alpha=0.96,
        zorder=7,
    )
    axis.scatter(
        current_positions[:, 0],
        current_positions[:, 1],
        s=34,
        marker=style["marker"],
        facecolors=sensor_colors,
        edgecolors="white",
        linewidths=0.55,
        alpha=0.98,
        zorder=8,
    )
    return sensor_colors


def _draw_cylinder_obstacle(axis):
    """Draw the cylinder obstacle in physical coordinates below sensor paths."""
    obstacle = Circle(
        CYLINDER_OBSTACLE_CENTER,
        CYLINDER_OBSTACLE_RADIUS,
        facecolor="#111827",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.96,
        zorder=5,
    )
    axis.add_patch(obstacle)


def _plot_flow_snapshot(
    axis,
    snapshot,
    lx,
    ly,
    speed_max,
    quiver_step,
    vorticity_norm,
    vorticity_abs_max=None,
    x_origin=0.0,
    y_origin=0.0,
    periodic=False,
    draw_cylinder=False,
    show_ylabel=True,
    snapshot_idx=0,
    snapshot_count=1,
):
    """Plot one compact flow-field snapshot for the trajectory summary."""
    u_grid = np.asarray(snapshot["u_grid"], dtype=float)
    v_grid = np.asarray(snapshot["v_grid"], dtype=float)
    x_grid, y_grid, _ = _flow_mesh(u_grid, v_grid, lx, ly, x_origin=x_origin, y_origin=y_origin)

    speed = np.hypot(u_grid, v_grid)
    del speed_max, vorticity_abs_max, show_ylabel
    speed_ref = _flow_snapshot_speed_reference(speed)
    target_arrow_length = FLOW_SNAPSHOT_TARGET_ARROW_LENGTH_FRACTION * min(float(lx), float(ly))
    quiver_scale = speed_ref / max(target_arrow_length, float(np.finfo(float).eps))
    vorticity = _flow_vorticity(u_grid, v_grid, lx, ly)
    x_quiver_step, y_quiver_step = _flow_snapshot_quiver_steps(u_grid.shape, lx, ly, quiver_step)

    vorticity_mesh = axis.pcolormesh(
        x_grid,
        y_grid,
        vorticity,
        cmap=FLOW_SNAPSHOT_VORTICITY_CMAP,
        norm=vorticity_norm,
        alpha=FLOW_SNAPSHOT_VORTICITY_ALPHA,
        shading="auto",
        zorder=1,
    )
    axis.quiver(
        x_grid[::x_quiver_step, ::y_quiver_step],
        y_grid[::x_quiver_step, ::y_quiver_step],
        u_grid[::x_quiver_step, ::y_quiver_step],
        v_grid[::x_quiver_step, ::y_quiver_step],
        color="#111827",
        alpha=0.88,
        scale_units="xy",
        scale=quiver_scale,
        width=FLOW_SNAPSHOT_QUIVER_WIDTH,
        headwidth=3.4,
        headlength=4.2,
        headaxislength=3.8,
        pivot="mid",
        zorder=2,
    )

    if draw_cylinder:
        _draw_cylinder_obstacle(axis)

    lagrangian_colors = _plot_snapshot_trajectory_family(
        axis,
        snapshot.get("lagrangian_history", snapshot.get("lagrangian_positions")),
        "Lagrangian",
        lx=lx,
        ly=ly,
        periodic=periodic,
        x_origin=x_origin,
        y_origin=y_origin,
    )
    moving_colors = _plot_snapshot_trajectory_family(
        axis,
        snapshot.get("moving_history", snapshot.get("moving_positions")),
        "Moving POD-QR",
        lx=lx,
        ly=ly,
        periodic=periodic,
        x_origin=x_origin,
        y_origin=y_origin,
    )

    label = str(snapshot.get("label", "Flow"))
    t_label = snapshot.get("t_idx", snapshot.get("history_t_idx", None))
    if t_label is None:
        title = f"{label} Flow"
    else:
        title = f"{label} ($t={int(t_label)}$)"

    x_min = float(x_origin)
    x_max = x_min + float(lx)
    y_min = float(y_origin)
    y_max = y_min + float(ly)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    domain_aspect = max(float(lx) / float(ly), float(ly) / float(lx))
    if domain_aspect < 1.25 and snapshot_count > 1:
        anchor = ("NE", "N", "NW")[min(snapshot_idx, 2)]
    else:
        anchor = "N"
    axis.set_aspect("equal", adjustable="box", anchor=anchor)
    x_ticks = [x_min, x_min + 0.5 * float(lx), x_max]
    x_tick_labels = [f"{tick:g}" for tick in x_ticks]
    y_ticks = [y_min, y_min + 0.5 * float(ly), y_max]
    y_tick_labels = [f"{tick:g}" for tick in y_ticks]
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_tick_labels)
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(y_tick_labels)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="both", which="major", pad=4)
    axis.set_title(title, fontsize=FLOW_SNAPSHOT_TITLE_FONTSIZE)
    apply_axis_style(axis, x_grid=False, y_grid=False)
    return vorticity_mesh, lagrangian_colors, moving_colors


def _trajectory_summary_figure_size(lx, ly, snapshot_count, vertical_layout=False):
    """Choose figure size from fixed font scale and panel geometry."""
    domain_aspect = max(float(lx) / float(ly), float(ly) / float(lx))
    if vertical_layout:
        panel_width = 7.2
        panel_height = panel_width / domain_aspect
        return (8.9, max(4.8, snapshot_count * (panel_height + 0.78) + 1.0))

    panel_width = 2.65 if domain_aspect < 1.25 else 3.05
    panel_height = panel_width / domain_aspect
    return (snapshot_count * panel_width + 1.45, max(3.55, panel_height + 1.12))


def _add_trajectory_figure_legend(fig, lagrangian_colors, moving_colors, *, y_anchor=0.985):
    """Place one shared trajectory legend on the figure."""
    handles, labels, handler_map = _trajectory_legend_parts(lagrangian_colors, moving_colors)
    fig.legend(
        handles=handles,
        labels=labels,
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fancybox=False,
        fontsize=TRAJECTORY_LEGEND_FONTSIZE,
        handlelength=2.4,
        handletextpad=0.8,
        columnspacing=1.15,
    )


def save_trajectory_plot(
    lagrangian_history,
    moving_history,
    lx,
    ly,
    out_path,
    run_name=None,
    periodic=False,
    flow_snapshots=None,
    flow_speed_max=None,
    quiver_step=4,
    x_origin=0.0,
    y_origin=0.0,
    dpi=220,
):
    """Render a polished end-of-run sensor trajectory summary plot.

    Args:
        lagrangian_history: Sensor positions shaped (n_steps, n_sensors, 2).
        moving_history: Sensor positions shaped (n_steps, n_sensors, 2).
        lx: Domain length in x.
        ly: Domain length in y.
        out_path: Output PNG path.
        run_name: Optional run label for the figure title.
        periodic: Whether the flow domain is periodic and should be shown in wrapped coordinates.
        flow_snapshots: Optional list of dicts with u_grid, v_grid, label, and t_idx.
        flow_speed_max: Optional shared speed scale for the flow snapshots.
        quiver_step: Quiver decimation step for the flow snapshots.
        x_origin: Physical lower x-coordinate used for display.
        y_origin: Physical lower y-coordinate used for display.
        dpi: Figure DPI.

    Returns:
        None.
    """
    lagrangian_history = np.asarray(lagrangian_history, dtype=float)
    moving_history = np.asarray(moving_history, dtype=float)
    if lagrangian_history.shape != moving_history.shape:
        raise ValueError("lagrangian_history and moving_history must have identical shapes")

    if flow_snapshots is None:
        flow_snapshots = []
    else:
        flow_snapshots = list(flow_snapshots)[:3]

    flow_label = pretty_flow_name(run_name) if run_name else "Sliding"
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with paper_plot_context():
        if flow_snapshots:
            snapshot_count = len(flow_snapshots)
            vertical_layout = _is_cylinder_wake_plot(run_name, lx, ly)
            fig = plt.figure(
                figsize=_trajectory_summary_figure_size(lx, ly, snapshot_count, vertical_layout=vertical_layout),
                constrained_layout=False,
            )
            if vertical_layout:
                grid_spec = fig.add_gridspec(
                    snapshot_count,
                    2,
                    width_ratios=[1.0, 0.035],
                    left=0.10,
                    right=0.88,
                    bottom=0.10,
                    top=0.84,
                    hspace=0.92,
                    wspace=0.050,
                )
                flow_axes = [fig.add_subplot(grid_spec[idx, 0]) for idx in range(snapshot_count)]
                flow_colorbar_axis = fig.add_subplot(grid_spec[:, 1])
            else:
                grid_spec = fig.add_gridspec(
                    1,
                    snapshot_count + 1,
                    width_ratios=[1.0] * snapshot_count + [0.045],
                    left=0.08,
                    right=0.92,
                    bottom=0.17,
                    top=0.80,
                    wspace=0.16,
                )
                flow_axes = [fig.add_subplot(grid_spec[0, idx]) for idx in range(snapshot_count)]
                flow_colorbar_axis = fig.add_subplot(grid_spec[0, snapshot_count])

            vorticity_abs_max = _flow_snapshot_vorticity_abs_max(flow_snapshots, lx, ly)
            vorticity_norm = _vorticity_norm(vorticity_abs_max, use_symlog=vertical_layout)
            vorticity_meshes = []
            lagrangian_colors = []
            moving_colors = []
            for snapshot_idx, (flow_axis, snapshot) in enumerate(zip(flow_axes, flow_snapshots)):
                vorticity_mesh, snapshot_lagrangian_colors, snapshot_moving_colors = _plot_flow_snapshot(
                    flow_axis,
                    snapshot,
                    lx=lx,
                    ly=ly,
                    speed_max=flow_speed_max,
                    quiver_step=quiver_step,
                    vorticity_norm=vorticity_norm,
                    vorticity_abs_max=vorticity_abs_max,
                    x_origin=x_origin,
                    y_origin=y_origin,
                    periodic=periodic,
                    draw_cylinder=vertical_layout,
                    show_ylabel=snapshot_idx == 0 or vertical_layout,
                    snapshot_idx=snapshot_idx,
                    snapshot_count=snapshot_count,
                )
                vorticity_meshes.append(vorticity_mesh)
                if snapshot_lagrangian_colors:
                    lagrangian_colors = snapshot_lagrangian_colors
                if snapshot_moving_colors:
                    moving_colors = snapshot_moving_colors

            if not lagrangian_colors:
                lagrangian_colors = _sensor_palette_colors("Lagrangian", lagrangian_history.shape[1])
            if not moving_colors:
                moving_colors = _sensor_palette_colors("Moving POD-QR", moving_history.shape[1])

            if vorticity_meshes:
                if not vertical_layout:
                    fig.canvas.draw()
                    reference_bbox = flow_axes[-1].get_position()
                    colorbar_bbox = flow_colorbar_axis.get_position()
                    flow_colorbar_axis.set_position(
                        [
                            colorbar_bbox.x0,
                            reference_bbox.y0,
                            colorbar_bbox.width,
                            reference_bbox.height,
                        ]
                    )
                colorbar = fig.colorbar(vorticity_meshes[0], cax=flow_colorbar_axis)
                colorbar_ticks = _vorticity_colorbar_ticks(vorticity_norm, vorticity_abs_max)
                colorbar.set_ticks(colorbar_ticks)
                colorbar.set_ticklabels(_vorticity_colorbar_tick_labels(colorbar_ticks))
                colorbar.set_label(r"Vorticity $\omega$")
            _add_trajectory_figure_legend(
                fig,
                lagrangian_colors,
                moving_colors,
                y_anchor=0.985 if vertical_layout else 0.975,
            )
        else:
            x_limits, y_limits = _trajectory_limits(
                [lagrangian_history, moving_history],
                lx,
                ly,
                periodic,
                x_origin=x_origin,
                y_origin=y_origin,
            )
            fig, axis = plt.subplots(1, 1, figsize=(9.4, 7.2), constrained_layout=True)
            lagrangian_colors = _plot_trajectory_family(
                axis,
                lagrangian_history,
                "Lagrangian",
                lx=lx,
                ly=ly,
                periodic=periodic,
                x_origin=x_origin,
                y_origin=y_origin,
            )
            moving_colors = _plot_trajectory_family(
                axis,
                moving_history,
                "Moving POD-QR",
                lx=lx,
                ly=ly,
                periodic=periodic,
                x_origin=x_origin,
                y_origin=y_origin,
            )
            if periodic:
                _draw_periodic_domain_guides(axis, x_limits, y_limits, lx, ly, x_origin=x_origin, y_origin=y_origin)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            title_suffix = "Periodic Sensor Trajectories" if periodic else "Sensor Trajectories"
            axis.set_title(f"{flow_label}: {title_suffix}", fontsize=TRAJECTORY_TITLE_FONTSIZE)
            apply_axis_style(axis, x_grid=True, y_grid=True)
            _finalize_trajectory_legend(axis, lagrangian_colors, moving_colors)

        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def save_flow_field_frame(
    u_grid,
    v_grid,
    lx,
    ly,
    t_idx,
    out_path,
    run_name=None,
    speed_max=None,
    quiver_step=4,
    x_origin=0.0,
    y_origin=0.0,
    dpi=180,
):
    """Render one dark-theme flow-field frame for animation."""
    x_grid, y_grid, speed = _flow_mesh(u_grid, v_grid, lx, ly, x_origin=x_origin, y_origin=y_origin)
    resolved_speed_max = float(np.max(speed)) if speed_max is None else float(speed_max)
    flow_label = pretty_flow_name(run_name) if run_name else "Sliding"
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with paper_dark_plot_context():
        fig, axis = plt.subplots(1, 1, figsize=(8.2, 6.4), constrained_layout=True)
        contour = _draw_dark_flow_field(
            axis,
            x_grid,
            y_grid,
            u_grid,
            v_grid,
            speed,
            speed_max=resolved_speed_max,
            quiver_step=quiver_step,
        )
        colorbar = fig.colorbar(contour, ax=axis, fraction=0.045, pad=0.03)
        _style_dark_colorbar(colorbar, r"Speed $\|\mathbf{u}\|_2$")

        axis.set_xlim(float(x_origin), float(x_origin) + float(lx))
        axis.set_ylim(float(y_origin), float(y_origin) + float(ly))
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(f"{flow_label}: Flow Field at $t={int(t_idx)}$")
        apply_dark_axis_style(axis, x_grid=False, y_grid=False)

        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def save_sensor_motion_frame(
    u_grid,
    v_grid,
    lx,
    ly,
    moving_history,
    t_idx,
    out_path,
    run_name=None,
    periodic=False,
    speed_max=None,
    quiver_step=4,
    tail_length=48,
    x_origin=0.0,
    y_origin=0.0,
    dpi=180,
):
    """Render one dark-theme flow frame with a fading Moving POD-QR trail."""
    x_grid, y_grid, speed = _flow_mesh(u_grid, v_grid, lx, ly, x_origin=x_origin, y_origin=y_origin)
    resolved_speed_max = float(np.max(speed)) if speed_max is None else float(speed_max)
    flow_label = pretty_flow_name(run_name) if run_name else "Sliding"
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with paper_dark_plot_context():
        fig, axis = plt.subplots(1, 1, figsize=(8.2, 6.4), constrained_layout=True)
        contour = _draw_dark_flow_field(
            axis,
            x_grid,
            y_grid,
            u_grid,
            v_grid,
            speed,
            speed_max=resolved_speed_max,
            quiver_step=quiver_step,
        )
        _plot_fading_sensor_tail(
            axis,
            moving_history,
            "Moving POD-QR",
            lx=lx,
            ly=ly,
            periodic=periodic,
            tail_length=tail_length,
            x_origin=x_origin,
            y_origin=y_origin,
        )

        colorbar = fig.colorbar(contour, ax=axis, fraction=0.045, pad=0.03)
        _style_dark_colorbar(colorbar, r"Speed $\|\mathbf{u}\|_2$")

        axis.set_xlim(float(x_origin), float(x_origin) + float(lx))
        axis.set_ylim(float(y_origin), float(y_origin) + float(ly))
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(f"{flow_label}: Moving POD-QR Motion at $t={int(t_idx)}$")
        apply_dark_axis_style(axis, x_grid=False, y_grid=False)
        finalize_dark_legend(axis, loc="upper right", title="Trajectory")

        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def save_window_frame(
    u_grid,
    v_grid,
    lx,
    ly,
    fixed_sensor_positions,
    lagrangian_sensor_positions,
    window_qr_target_positions,
    moving_sensor_positions,
    window_idx,
    start_idx,
    end_idx,
    t_mid,
    out_path,
    l2h_records=None,
    r_norm_history=None,
    total_windows=None,
    quiver_step=4,
    x_origin=0.0,
    y_origin=0.0,
    dpi=180,
):
    """Render and save one sliding-window frame with flow and sensor overlays.

    Args:
        u_grid: u velocity component for one snapshot, shape (nx, ny).
        v_grid: v velocity component for one snapshot, shape (nx, ny).
        lx: Domain length in x.
        ly: Domain length in y.
        fixed_sensor_positions: Fixed sensor coordinates, shape (n_sensors, 2).
        lagrangian_sensor_positions: Lagrangian coordinates, shape (n_sensors, 2).
        window_qr_target_positions: Window QR target coordinates, shape (n_sensors, 2).
        moving_sensor_positions: Moving POD-QR coordinates, shape (n_sensors, 2).
        window_idx: Zero-based window index.
        start_idx: Window start index.
        end_idx: Window end index.
        t_mid: Midpoint index used for quiver snapshot.
        out_path: Output PNG path.
        l2h_records: Running L2_h records collected up to the current window.
        r_norm_history: Running residual norms (one value per window).
        total_windows: Total number of windows in the full run.
        quiver_step: Quiver decimation step.
        dpi: Figure DPI.

    Returns:
        None.
    """
    nx, ny = u_grid.shape

    x_coords = np.linspace(float(x_origin), float(x_origin) + float(lx), nx)
    y_coords = np.linspace(float(y_origin), float(y_origin) + float(ly), ny)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")

    speed = np.hypot(u_grid, v_grid)

    with paper_plot_context():
        fig, (flow_axis, l2h_axis) = plt.subplots(1, 2, figsize=(13.6, 5.8), constrained_layout=True)

        flow_axis.contourf(
            x_grid,
            y_grid,
            speed,
            levels=12,
            cmap="Greys",
            alpha=0.18,
        )
        flow_axis.quiver(
            x_grid[::quiver_step, ::quiver_step],
            y_grid[::quiver_step, ::quiver_step],
            u_grid[::quiver_step, ::quiver_step],
            v_grid[::quiver_step, ::quiver_step],
            color="#344054",
            alpha=0.88,
            scale_units="xy",
            scale=None,
            width=0.0028,
            pivot="mid",
        )

        flow_axis.scatter(
            fixed_sensor_positions[:, 0] + float(x_origin),
            fixed_sensor_positions[:, 1] + float(y_origin),
            color=PLACEMENT_COLORS["Fixed"],
            s=54,
            marker="s",
            edgecolors="white",
            linewidths=0.85,
            label=display_label("Fixed"),
            zorder=4,
        )
        flow_axis.scatter(
            lagrangian_sensor_positions[:, 0] + float(x_origin),
            lagrangian_sensor_positions[:, 1] + float(y_origin),
            color=PLACEMENT_COLORS["Lagrangian"],
            s=56,
            marker="o",
            edgecolors="white",
            linewidths=0.85,
            label="Lagrangian",
            zorder=4,
        )
        flow_axis.scatter(
            window_qr_target_positions[:, 0] + float(x_origin),
            window_qr_target_positions[:, 1] + float(y_origin),
            color=PLACEMENT_COLORS["QR teleport"],
            s=60,
            marker="X",
            edgecolors="white",
            linewidths=0.7,
            label="QR teleport",
            zorder=4,
        )
        flow_axis.scatter(
            moving_sensor_positions[:, 0] + float(x_origin),
            moving_sensor_positions[:, 1] + float(y_origin),
            color=PLACEMENT_COLORS["Moving POD-QR"],
            s=56,
            marker="o",
            edgecolors="white",
            linewidths=0.85,
            label="Moving POD-QR",
            zorder=4,
        )

        flow_axis.set_xlim(float(x_origin), float(x_origin) + float(lx))
        flow_axis.set_ylim(float(y_origin), float(y_origin) + float(ly))
        flow_axis.set_aspect("equal", adjustable="box")
        flow_axis.set_xlabel("x")
        flow_axis.set_ylabel("y")
        flow_axis.set_title(f"Window {window_idx}: t in [{start_idx}, {end_idx})  midpoint={t_mid}")
        apply_axis_style(flow_axis, x_grid=False, y_grid=False)
        finalize_legend(flow_axis, loc="upper right")

        _plot_l2h_history(
            l2h_axis,
            l2h_records=l2h_records,
            total_windows=total_windows,
            r_norm_history=r_norm_history,
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def make_window_gif(frames_dir, gif_path, duration=0.10):
    """Create GIF from saved frame images.

    Args:
        frames_dir: Directory containing sequential frame images.
        gif_path: Output GIF file path.
        duration: Frame duration in seconds.

    Returns:
        None.
    """
    output_dir = os.path.dirname(gif_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(
        file_name
        for file_name in os.listdir(frames_dir)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not frame_files:
        print(f"[GIF] No image files found in {frames_dir}, skipping GIF.")
        return

    frames = [imageio.imread(os.path.join(frames_dir, file_name)) for file_name in frame_files]
    imageio.mimsave(gif_path, frames, duration=duration)

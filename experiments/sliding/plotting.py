"""Sliding-experiment plotting helpers for per-window frame rendering."""

import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
TRAJECTORY_COLORBAR_LABEL = "time step"

PLACEMENT_COLORS = {
    "Fixed": color_for_method("Fixed"),
    "Eulerian": color_for_method("Eulerian"),
    "Lagrangian": color_for_method("Lagrangian"),
    "QR teleport": color_for_method("QR teleport"),
    "Moving POD-QR": color_for_method("Moving POD-QR"),
}

TRAJECTORY_STYLE = {
    "Lagrangian": {
        "light": "#D9F3EC",
        "dark": color_for_method("Lagrangian"),
        "marker": "o",
    },
    "Moving POD-QR": {
        "light": "#DCE7F6",
        "dark": color_for_method("Moving POD-QR"),
        "marker": "s",
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


def _flow_mesh(u_grid, v_grid, lx, ly):
    """Build physical-coordinate grids and speed magnitude for one flow snapshot."""
    nx, ny = u_grid.shape
    x_coords = np.linspace(0.0, lx, nx)
    y_coords = np.linspace(0.0, ly, ny)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
    speed = np.hypot(u_grid, v_grid)
    return x_grid, y_grid, speed


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


def _build_time_colormap(light_color, dark_color, cmap_name):
    """Create a sequential colormap that darkens with time."""
    return LinearSegmentedColormap.from_list(cmap_name, [light_color, dark_color])


def _trajectory_ticks(max_value):
    """Return a small set of clean ticks for the trajectory color scale."""
    if max_value <= 0:
        return [0]

    tick_count = min(4, int(max_value) + 1)
    ticks = np.linspace(0.0, float(max_value), num=tick_count)
    ticks = np.unique(np.round(ticks).astype(int))
    return ticks.tolist()


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


def _plot_trajectory_family(axis, history, label):
    """Plot one family of sensor trajectories on a shared axis."""
    style = TRAJECTORY_STYLE[label]
    cmap = _build_time_colormap(style["light"], style["dark"], f"{label.lower().replace(' ', '_')}_time")
    max_step = max(int(history.shape[0]) - 2, 0)
    norm = Normalize(vmin=0, vmax=max(max_step, 1))

    if history.shape[0] >= 2:
        segment_values = np.arange(history.shape[0] - 1, dtype=float)
        for sensor_idx in range(history.shape[1]):
            path = history[:, sensor_idx, :]
            segments = np.stack([path[:-1], path[1:]], axis=1)
            line_collection = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=2.15,
                capstyle="round",
                joinstyle="round",
                alpha=0.96,
            )
            line_collection.set_array(segment_values)
            axis.add_collection(line_collection)

    end_positions = history[-1]
    axis.scatter(
        end_positions[:, 0],
        end_positions[:, 1],
        s=48,
        marker=style["marker"],
        facecolors=style["dark"],
        edgecolors="white",
        linewidths=0.8,
        alpha=0.98,
        label=label,
        zorder=5,
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    return scalar_mappable


def _trajectory_limits(histories, lx, ly, periodic):
    """Choose axis limits for wrapped or non-wrapped trajectory plots."""
    if not histories:
        return (0.0, lx), (0.0, ly)

    if not periodic:
        x_pad = 0.04 * float(lx)
        y_pad = 0.04 * float(ly)
        return (-x_pad, float(lx) + x_pad), (-y_pad, float(ly) + y_pad)

    stacked_points = np.concatenate([history.reshape(-1, 2) for history in histories], axis=0)
    x_min = float(np.min(stacked_points[:, 0]))
    x_max = float(np.max(stacked_points[:, 0]))
    y_min = float(np.min(stacked_points[:, 1]))
    y_max = float(np.max(stacked_points[:, 1]))
    x_pad = 0.08 * float(lx)
    y_pad = 0.08 * float(ly)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _draw_periodic_domain_guides(axis, x_limits, y_limits, lx, ly):
    """Draw faint dashed guides marking periodic domain copies."""
    if lx > 0.0:
        x_start = int(np.floor(x_limits[0] / lx))
        x_end = int(np.ceil(x_limits[1] / lx))
        for k_idx in range(x_start, x_end + 1):
            axis.axvline(
                k_idx * lx,
                color="#98A2B3",
                linestyle=(0, (3, 3)),
                linewidth=0.9,
                alpha=0.45,
                zorder=0,
            )

    if ly > 0.0:
        y_start = int(np.floor(y_limits[0] / ly))
        y_end = int(np.ceil(y_limits[1] / ly))
        for k_idx in range(y_start, y_end + 1):
            axis.axhline(
                k_idx * ly,
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


def _plot_fading_sensor_tail(axis, history, label, lx, ly, periodic=False, tail_length=48):
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
                    [point_a[0], point_b[0]],
                    [point_a[1], point_b[1]],
                    color=style["color"],
                    linewidth=0.8 + 1.3 * age_fraction,
                    alpha=0.08 + 0.78 * age_fraction,
                    solid_capstyle="round",
                    zorder=4,
                )

    current_positions = recent_history[-1]
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


def save_trajectory_plot(
    lagrangian_history,
    moving_history,
    lx,
    ly,
    out_path,
    run_name=None,
    periodic=False,
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
        periodic: Whether the flow domain is periodic and should be unwrapped.
        dpi: Figure DPI.

    Returns:
        None.
    """
    lagrangian_history = np.asarray(lagrangian_history, dtype=float)
    moving_history = np.asarray(moving_history, dtype=float)
    if lagrangian_history.shape != moving_history.shape:
        raise ValueError("lagrangian_history and moving_history must have identical shapes")

    if periodic:
        lagrangian_history = _unwrap_periodic_history(lagrangian_history, lx, ly)
        moving_history = _unwrap_periodic_history(moving_history, lx, ly)

    flow_label = pretty_flow_name(run_name) if run_name else "Sliding"
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    x_limits, y_limits = _trajectory_limits([lagrangian_history, moving_history], lx, ly, periodic)

    with paper_plot_context():
        fig, axis = plt.subplots(1, 1, figsize=(9.4, 7.2), constrained_layout=True)
        lagrangian_mappable = _plot_trajectory_family(axis, lagrangian_history, "Lagrangian")
        moving_mappable = _plot_trajectory_family(axis, moving_history, "Moving POD-QR")

        if periodic:
            _draw_periodic_domain_guides(axis, x_limits, y_limits, lx, ly)

        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        title_suffix = "Periodic Sensor Trajectories" if periodic else "Sensor Trajectories"
        axis.set_title(f"{flow_label}: {title_suffix}")
        apply_axis_style(axis, x_grid=True, y_grid=True)
        finalize_legend(axis, loc="lower right", title="Final Position")

        lagrangian_cb_axis = inset_axes(
            axis,
            width="3.0%",
            height="30%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.12, 1.0, 1.0),
            bbox_transform=axis.transAxes,
            borderpad=0,
        )
        moving_cb_axis = inset_axes(
            axis,
            width="3.0%",
            height="30%",
            loc="lower left",
            bbox_to_anchor=(1.10, 0.12, 1.0, 1.0),
            bbox_transform=axis.transAxes,
            borderpad=0,
        )

        for colorbar_axis, scalar_mappable, label in (
            (lagrangian_cb_axis, lagrangian_mappable, "Lagrangian"),
            (moving_cb_axis, moving_mappable, "Moving POD-QR"),
        ):
            colorbar = fig.colorbar(scalar_mappable, cax=colorbar_axis)
            ticks = _trajectory_ticks(max(int(lagrangian_history.shape[0]) - 2, 0))
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels([str(int(tick)) for tick in ticks])
            colorbar.set_label(f"{label} {TRAJECTORY_COLORBAR_LABEL}")

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
    dpi=180,
):
    """Render one dark-theme flow-field frame for animation."""
    x_grid, y_grid, speed = _flow_mesh(u_grid, v_grid, lx, ly)
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

        axis.set_xlim(0.0, lx)
        axis.set_ylim(0.0, ly)
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
    dpi=180,
):
    """Render one dark-theme flow frame with a fading Moving POD-QR trail."""
    x_grid, y_grid, speed = _flow_mesh(u_grid, v_grid, lx, ly)
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
        )

        colorbar = fig.colorbar(contour, ax=axis, fraction=0.045, pad=0.03)
        _style_dark_colorbar(colorbar, r"Speed $\|\mathbf{u}\|_2$")

        axis.set_xlim(0.0, lx)
        axis.set_ylim(0.0, ly)
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

    x_coords = np.linspace(0.0, lx, nx)
    y_coords = np.linspace(0.0, ly, ny)
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
            fixed_sensor_positions[:, 0],
            fixed_sensor_positions[:, 1],
            color=PLACEMENT_COLORS["Fixed"],
            s=54,
            marker="s",
            edgecolors="white",
            linewidths=0.85,
            label=display_label("Fixed"),
            zorder=4,
        )
        flow_axis.scatter(
            lagrangian_sensor_positions[:, 0],
            lagrangian_sensor_positions[:, 1],
            color=PLACEMENT_COLORS["Lagrangian"],
            s=56,
            marker="o",
            edgecolors="white",
            linewidths=0.85,
            label="Lagrangian",
            zorder=4,
        )
        flow_axis.scatter(
            window_qr_target_positions[:, 0],
            window_qr_target_positions[:, 1],
            color=PLACEMENT_COLORS["QR teleport"],
            s=60,
            marker="X",
            edgecolors="white",
            linewidths=0.7,
            label="QR teleport",
            zorder=4,
        )
        flow_axis.scatter(
            moving_sensor_positions[:, 0],
            moving_sensor_positions[:, 1],
            color=PLACEMENT_COLORS["Moving POD-QR"],
            s=56,
            marker="o",
            edgecolors="white",
            linewidths=0.85,
            label="Moving POD-QR",
            zorder=4,
        )

        flow_axis.set_xlim(0.0, lx)
        flow_axis.set_ylim(0.0, ly)
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

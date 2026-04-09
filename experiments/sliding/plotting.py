"""Sliding-experiment plotting helpers for per-window frame rendering."""

import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.common.plot_style import (
    apply_axis_style,
    color_for_method,
    display_label,
    finalize_legend,
    paper_plot_context,
)
from regional_piv.plotting import make_gif_from_dir


METRIC_LABEL = r"Relative $L_h^2$ Error"

PLACEMENT_COLORS = {
    "Fixed": color_for_method("Fixed"),
    "Eulerian": color_for_method("Eulerian"),
    "Lagrangian": color_for_method("Lagrangian"),
    "QR teleport": color_for_method("QR teleport"),
    "Moving POD-QR": color_for_method("Moving POD-QR"),
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
    make_gif_from_dir(frames_dir, gif_path, duration=duration)

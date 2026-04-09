"""Sliding-experiment plotting helpers for per-window frame rendering."""

import os

import matplotlib.pyplot as plt
import numpy as np

from regional_piv.plotting import make_gif_from_dir


PLACEMENT_COLORS = {
    "Fixed": "orange",
    "Lagrangian": "green",
    "QR teleport": "red",
    "Moving POD-QR": "blue",
}


def _plot_l2h_history(axis, l2h_records, total_windows, r_norm_history=None):
    """Plot relative L2_h error history and optional residual norm history."""
    if total_windows is None or total_windows <= 0:
        axis.set_xlim(0, 1)
        axis.set_xlabel("window")
        axis.set_ylabel("Relative L2_h Error")
        axis.set_title("Relative Sensor Error Over Window")
        axis.grid(True, which="both", alpha=0.25)
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
        label = f"{placement_name} - {basis_name}" if show_basis and basis_name else placement_name

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
            linewidth=1.5,
            markersize=3.5,
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
            color="black",
            linestyle="--",
            linewidth=1.8,
            marker="^",
            markersize=3.5,
            label="Window relative ||r||_h",
        )

    axis.set_xlim(0, max(int(total_windows) - 1, 1))
    axis.set_yscale("log")
    axis.set_xlabel("window")
    axis.set_ylabel("Relative L2_h Error")
    axis.set_title("Relative Sensor Error + Window ||r||_h")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)


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
    dpi=150,
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

    fig, (flow_axis, l2h_axis) = plt.subplots(1, 2, figsize=(13.4, 5.6), constrained_layout=True)

    flow_axis.quiver(
        x_grid[::quiver_step, ::quiver_step],
        y_grid[::quiver_step, ::quiver_step],
        u_grid[::quiver_step, ::quiver_step],
        v_grid[::quiver_step, ::quiver_step],
        color="black",
        scale_units="xy",
        scale=None,
        width=0.0028,
        pivot="mid",
    )

    flow_axis.scatter(
        fixed_sensor_positions[:, 0],
        fixed_sensor_positions[:, 1],
        color=PLACEMENT_COLORS["Fixed"],
        s=50,
        marker="s",
        label="Fixed",
    )
    flow_axis.scatter(
        lagrangian_sensor_positions[:, 0],
        lagrangian_sensor_positions[:, 1],
        color=PLACEMENT_COLORS["Lagrangian"],
        s=50,
        marker="o",
        label="Lagrangian",
    )
    flow_axis.scatter(
        window_qr_target_positions[:, 0],
        window_qr_target_positions[:, 1],
        color=PLACEMENT_COLORS["QR teleport"],
        s=55,
        marker="x",
        label="QR teleport",
    )
    flow_axis.scatter(
        moving_sensor_positions[:, 0],
        moving_sensor_positions[:, 1],
        color=PLACEMENT_COLORS["Moving POD-QR"],
        s=50,
        marker="o",
        label="Moving POD-QR",
    )

    flow_axis.set_xlim(0.0, lx)
    flow_axis.set_ylim(0.0, ly)
    flow_axis.set_aspect("equal", adjustable="box")
    flow_axis.set_xlabel("x")
    flow_axis.set_ylabel("y")
    flow_axis.set_title(f"Window {window_idx}  t in [{start_idx},{end_idx})  mid={t_mid}")
    flow_axis.legend(loc="upper right")

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

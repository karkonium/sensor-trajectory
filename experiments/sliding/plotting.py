"""Sliding-experiment plotting helpers for per-window frame rendering."""

import os

import matplotlib.pyplot as plt
import numpy as np

from regional_piv.plotting import make_gif_from_dir


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
        quiver_step: Quiver decimation step.
        dpi: Figure DPI.

    Returns:
        None.
    """
    nx, ny = u_grid.shape

    x_coords = np.linspace(0.0, lx, nx)
    y_coords = np.linspace(0.0, ly, ny)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")

    fig, axis = plt.subplots(1, 1, figsize=(7.2, 5.6), constrained_layout=True)

    axis.quiver(
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

    axis.scatter(
        fixed_sensor_positions[:, 0],
        fixed_sensor_positions[:, 1],
        color="orange",
        s=50,
        marker="s",
        label="Fixed",
    )
    axis.scatter(
        lagrangian_sensor_positions[:, 0],
        lagrangian_sensor_positions[:, 1],
        color="green",
        s=50,
        marker="o",
        label="Lagrangian",
    )
    axis.scatter(
        window_qr_target_positions[:, 0],
        window_qr_target_positions[:, 1],
        color="red",
        s=55,
        marker="x",
        label="QR teleport",
    )
    axis.scatter(
        moving_sensor_positions[:, 0],
        moving_sensor_positions[:, 1],
        color="blue",
        s=50,
        marker="o",
        label="Moving POD-QR",
    )

    axis.set_xlim(0.0, lx)
    axis.set_ylim(0.0, ly)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"Window {window_idx}  t in [{start_idx},{end_idx})  mid={t_mid}")
    axis.legend(loc="upper right")

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

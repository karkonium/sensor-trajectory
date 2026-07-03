"""Plotting helpers for online reconstructed-window experiments."""

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.common.plot_style import (
    PAPER_RC_PARAMS,
    apply_axis_style,
    color_for_method,
    finalize_legend,
    pretty_flow_name,
)
from experiments.sliding.plotting import make_window_gif

from .pipeline import METHOD_LAGRANGIAN, METHOD_ONLINE, METHOD_ORDER, METHOD_STATIC


METHOD_COLORS = {
    METHOD_STATIC: color_for_method("Static QR"),
    METHOD_LAGRANGIAN: color_for_method("Lagrangian"),
    METHOD_ONLINE: color_for_method("Moving POD-QR"),
}
METRIC_LABEL = r"Relative $L_h^2$ Error"
CUMULATIVE_METRIC_LABEL = r"Running Mean Relative $L_h^2$ Error"
PLOT_RC_PARAMS = PAPER_RC_PARAMS.copy()
PLOT_RC_PARAMS.update(
    {
        "text.usetex": False,
        "text.latex.preamble": "",
        "font.family": "DejaVu Sans",
    }
)


def _flow_mesh(u_grid, v_grid, domain):
    """Build physical plotting grids and speed magnitude for one flow snapshot."""
    nx, ny = u_grid.shape
    x_coords = np.linspace(domain.x_min, domain.x_max, nx)
    y_coords = np.linspace(domain.y_min, domain.y_max, ny)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
    return x_grid, y_grid, np.hypot(u_grid, v_grid)


def _plot_sensor_history(
    axis,
    history,
    *,
    label,
    color,
    lx,
    ly,
    periodic,
    x_origin=0.0,
    y_origin=0.0,
):
    """Plot trajectories and current positions for one moving sensor family."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 3 or history.shape[0] == 0:
        return

    if history.shape[0] >= 2:
        for sensor_idx in range(history.shape[1]):
            path = history[:, sensor_idx, :]
            for segment_idx in range(path.shape[0] - 1):
                point_a = path[segment_idx]
                point_b = path[segment_idx + 1]
                if periodic:
                    crosses_x = lx > 0.0 and abs(float(point_b[0]) - float(point_a[0])) > 0.5 * lx
                    crosses_y = ly > 0.0 and abs(float(point_b[1]) - float(point_a[1])) > 0.5 * ly
                    if crosses_x or crosses_y:
                        continue

                age = float(segment_idx + 1) / float(path.shape[0] - 1)
                axis.plot(
                    [point_a[0] + x_origin, point_b[0] + x_origin],
                    [point_a[1] + y_origin, point_b[1] + y_origin],
                    color=color,
                    linewidth=0.65 + 1.25 * age,
                    alpha=0.12 + 0.66 * age,
                    solid_capstyle="round",
                    zorder=5,
                )

    initial_positions = history[0]
    current_positions = history[-1]
    axis.scatter(
        initial_positions[:, 0] + x_origin,
        initial_positions[:, 1] + y_origin,
        s=36,
        marker="o",
        facecolors="none",
        edgecolors="#98A2B3",
        linewidths=1.1,
        alpha=0.95,
        zorder=6,
    )
    axis.scatter(
        current_positions[:, 0] + x_origin,
        current_positions[:, 1] + y_origin,
        s=42,
        marker="o",
        color=color,
        edgecolors="white",
        linewidths=0.6,
        alpha=0.98,
        label=label,
        zorder=7,
    )


def _error_y_limits(raw_df):
    """Return stable log-scale limits for one flow's error traces."""
    values = pd.to_numeric(raw_df["cumulative_L2_h"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return (1e-8, 1.0)
    lower = max(float(np.min(values)) * 0.72, 1e-12)
    upper = max(float(np.max(values)) * 1.35, lower * 10.0)
    return lower, upper


def _plot_error_axis(axis, raw_df, current_t, *, is_final_frame=False, y_limits=None):
    """Plot method errors over time up to the requested snapshot."""
    current_t = int(current_t)
    current_df = raw_df[raw_df["t"] <= current_t].copy()

    final_lines = []
    for method_name in METHOD_ORDER:
        method_df = current_df[current_df["method"] == method_name].sort_values("t")
        if method_df.empty:
            continue

        final_cumulative = float(method_df["cumulative_L2_h"].iloc[-1])
        label = method_name
        if is_final_frame:
            label = f"{method_name} ({final_cumulative:.3e})"
            final_lines.append(f"{method_name}: {final_cumulative:.3e}")

        axis.plot(
            method_df["t"],
            method_df["cumulative_L2_h"].where(method_df["cumulative_L2_h"] > 0.0, np.nan),
            color=METHOD_COLORS[method_name],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            markersize=3.8,
            label=label,
        )

    t_values = np.sort(raw_df["t"].unique())
    axis.axvline(current_t, color="#667085", linewidth=1.0, alpha=0.55)
    if int(t_values[0]) == int(t_values[-1]):
        axis.set_xlim(float(t_values[0]) - 0.5, float(t_values[-1]) + 0.5)
    else:
        axis.set_xlim(int(t_values[0]), int(t_values[-1]))
    axis.set_yscale("log")
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    axis.set_xlabel("Time Index")
    axis.set_ylabel(CUMULATIVE_METRIC_LABEL)
    title = f"Errors Through t={current_t}"
    if is_final_frame:
        title = "Running Mean Errors Over Full Test Segment"
    axis.set_title(title)
    apply_axis_style(axis, x_grid=True, y_grid=True)
    finalize_legend(axis, loc="upper right")

    if is_final_frame and final_lines:
        axis.text(
            0.03,
            0.04,
            "Final running mean\n" + "\n".join(final_lines),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.34",
                "facecolor": "white",
                "edgecolor": "#D0D7E2",
                "alpha": 0.94,
            },
        )


def _save_online_frame(
    *,
    flow_name,
    u_grid,
    v_grid,
    domain,
    static_positions,
    lagrangian_history,
    online_history,
    raw_df,
    current_t,
    out_path,
    speed_max,
    quiver_step,
    periodic=False,
    is_final_frame=False,
    y_limits=None,
    dpi=180,
):
    """Save one side-by-side trajectory/error frame."""
    x_grid, y_grid, speed = _flow_mesh(u_grid, v_grid, domain)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PLOT_RC_PARAMS):
        fig, (flow_axis, error_axis) = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)

        speed_max = max(float(speed_max), float(np.finfo(float).eps))
        flow_axis.contourf(
            x_grid,
            y_grid,
            speed,
            levels=np.linspace(0.0, speed_max, 16),
            cmap="Greys",
            alpha=0.28,
            extend="max",
        )
        flow_axis.quiver(
            x_grid[::quiver_step, ::quiver_step],
            y_grid[::quiver_step, ::quiver_step],
            u_grid[::quiver_step, ::quiver_step],
            v_grid[::quiver_step, ::quiver_step],
            color="#344054",
            alpha=0.78,
            scale_units="xy",
            scale=None,
            width=0.0026,
            pivot="mid",
        )

        x_origin = domain.x_min
        y_origin = domain.y_min
        flow_axis.scatter(
            static_positions[:, 0] + x_origin,
            static_positions[:, 1] + y_origin,
            s=42,
            marker="s",
            color=METHOD_COLORS[METHOD_STATIC],
            edgecolors="white",
            linewidths=0.65,
            alpha=0.84,
            label=METHOD_STATIC,
            zorder=6,
        )
        _plot_sensor_history(
            flow_axis,
            lagrangian_history,
            label=METHOD_LAGRANGIAN,
            color=METHOD_COLORS[METHOD_LAGRANGIAN],
            lx=domain.lx,
            ly=domain.ly,
            periodic=periodic,
            x_origin=x_origin,
            y_origin=y_origin,
        )
        _plot_sensor_history(
            flow_axis,
            online_history,
            label=METHOD_ONLINE,
            color=METHOD_COLORS[METHOD_ONLINE],
            lx=domain.lx,
            ly=domain.ly,
            periodic=periodic,
            x_origin=x_origin,
            y_origin=y_origin,
        )

        flow_axis.set_xlim(domain.x_min, domain.x_max)
        flow_axis.set_ylim(domain.y_min, domain.y_max)
        flow_axis.set_aspect("equal", adjustable="box")
        flow_axis.set_xlabel("x")
        flow_axis.set_ylabel("y")
        flow_axis.set_title(f"{pretty_flow_name(flow_name)}: Sensors at t={int(current_t)}")
        apply_axis_style(flow_axis, x_grid=False, y_grid=False)
        finalize_legend(flow_axis, loc="upper right")

        _plot_error_axis(
            error_axis,
            raw_df,
            current_t,
            is_final_frame=is_final_frame,
            y_limits=y_limits,
        )

        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)


def save_full_error_plot(flow_name, raw_df, out_path):
    """Save a full-time error plot for one flow."""
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_t = int(np.max(raw_df["t"]))

    with plt.rc_context(PLOT_RC_PARAMS):
        fig, axis = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
        _plot_error_axis(
            axis,
            raw_df,
            final_t,
            is_final_frame=True,
            y_limits=_error_y_limits(raw_df),
        )
        axis.set_title(f"{pretty_flow_name(flow_name)}: Full Test Running Mean Error")
        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def _frame_indices(test_indices, frame_stride):
    """Return test-history offsets that should be rendered as animation frames."""
    frame_stride = max(1, int(frame_stride))
    indices = list(range(0, len(test_indices), frame_stride))
    final_idx = len(test_indices) - 1
    if indices[-1] != final_idx:
        indices.append(final_idx)
    return indices


def save_flow_artifacts(
    flow_case,
    result,
    artifact_paths,
    *,
    run_label,
    quiver_step=4,
    frame_stride=1,
    make_gifs=True,
    gif_duration=0.10,
    show_progress=True,
):
    """Save per-flow frames, GIF, final frame, and full error plot."""
    flow_name = flow_case.flow_name
    raw_df = result.raw_records
    y_limits = _error_y_limits(raw_df)

    full_error_path = artifact_paths.plots_dir / f"{flow_name}_{run_label}_errors.png"
    final_frame_path = artifact_paths.plots_dir / f"{flow_name}_{run_label}_final_frame.png"
    save_full_error_plot(flow_name, raw_df, full_error_path)

    if artifact_paths.frames_dir is None:
        history_idx = len(result.test_indices) - 1
        t_idx = int(result.test_indices[history_idx])
        _save_online_frame(
            flow_name=flow_name,
            u_grid=flow_case.u[t_idx],
            v_grid=flow_case.v[t_idx],
            domain=flow_case.domain_config,
            static_positions=result.static_positions,
            lagrangian_history=result.lagrangian_history[: history_idx + 1],
            online_history=result.online_history[: history_idx + 1],
            raw_df=raw_df,
            current_t=t_idx,
            out_path=final_frame_path,
            speed_max=result.speed_max,
            quiver_step=quiver_step,
            periodic=flow_case.is_periodic,
            is_final_frame=True,
            y_limits=y_limits,
        )
        return

    frames_dir = artifact_paths.frames_dir / run_label / flow_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    final_frame_in_sequence = None
    iterator = _frame_indices(result.test_indices, frame_stride)
    frame_iterator = tqdm(iterator, desc=f"{flow_name} frames") if show_progress else iterator
    for frame_number, history_idx in enumerate(frame_iterator):
        t_idx = int(result.test_indices[history_idx])
        is_final = history_idx == len(result.test_indices) - 1
        frame_path = frames_dir / f"frame_{frame_number:04d}.png"
        _save_online_frame(
            flow_name=flow_name,
            u_grid=flow_case.u[t_idx],
            v_grid=flow_case.v[t_idx],
            domain=flow_case.domain_config,
            static_positions=result.static_positions,
            lagrangian_history=result.lagrangian_history[: history_idx + 1],
            online_history=result.online_history[: history_idx + 1],
            raw_df=raw_df,
            current_t=t_idx,
            out_path=frame_path,
            speed_max=result.speed_max,
            quiver_step=quiver_step,
            periodic=flow_case.is_periodic,
            is_final_frame=is_final,
            y_limits=y_limits,
        )
        if is_final:
            final_frame_in_sequence = frame_path

    if final_frame_in_sequence is not None:
        shutil.copyfile(final_frame_in_sequence, final_frame_path)

    if make_gifs:
        gif_path = artifact_paths.frames_dir / f"{flow_name}_{run_label}.gif"
        make_window_gif(str(frames_dir), str(gif_path), duration=gif_duration)

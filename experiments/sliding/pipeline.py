"""Sliding-window experiment pipeline for moving-sensor placement strategies."""

import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import svdvals
from tqdm.auto import tqdm

from experiments.common.config import config_from_arrays
from experiments.common.sensor_motion import advect, advect_hungarian, bounce_apart
from experiments.common.spatial_utils import coords_to_linear_index, grid_to_phys, seed_sensor_grid, seed_uniform_random
from experiments.common.state_reconstruction import (
    flatten_state,
    fit_sspor_model,
    l2h_norm,
    relative_l2h_error_with_basis_matrix,
    selected_nodes_from_uv,
)
from experiments.common.windowing import get_sliding_intervals

from .plotting import (
    make_window_gif,
    save_flow_field_frame,
    save_sensor_motion_frame,
    save_trajectory_plot,
)


def _sample_flow_vectors(points, u_grid, v_grid, lx, ly):
    """Sample velocity vectors at arbitrary physical coordinates.

    Args:
        points: Coordinates shaped (n_points, 2).
        u_grid: u velocity field shaped (nx, ny).
        v_grid: v velocity field shaped (nx, ny).
        lx: Domain length in x.
        ly: Domain length in y.

    Returns:
        Array shaped (n_points, 2) with sampled vectors.
    """
    grid_x = np.linspace(0.0, lx, u_grid.shape[0])
    grid_y = np.linspace(0.0, ly, u_grid.shape[1])

    u_interpolator = RegularGridInterpolator((grid_x, grid_y), u_grid, bounds_error=False, fill_value=None)
    v_interpolator = RegularGridInterpolator((grid_x, grid_y), v_grid, bounds_error=False, fill_value=None)

    point_array = np.asarray(points, dtype=float)
    return np.stack([u_interpolator(point_array), v_interpolator(point_array)], axis=1)


def _resolve_animation_paths(
    plot_windows,
    save_window_frames,
    run_name,
    frames_dir,
    make_flow_gif,
    flow_gif_path,
    make_sensor_motion_gif,
    sensor_motion_gif_path,
):
    """Resolve sliding animation frame and GIF output paths.

    Args:
        plot_windows: Whether animation frame rendering is enabled.
        save_window_frames: Whether frame directories should be materialized.
        run_name: Name used for default output naming.
        frames_dir: Optional user-provided frame directory.
        make_flow_gif: Whether the flow-only GIF is requested.
        flow_gif_path: Optional output path for the flow-only GIF.
        make_sensor_motion_gif: Whether the sensor-motion GIF is requested.
        sensor_motion_gif_path: Optional output path for the sensor-motion GIF.

    Returns:
        Tuple of resolved flow-frame dir, sensor-frame dir, flow GIF path,
        and sensor-motion GIF path.
    """
    render_flow_frames = plot_windows or make_flow_gif
    render_sensor_motion_frames = plot_windows or make_sensor_motion_gif
    should_materialize_frames = save_window_frames or render_flow_frames or render_sensor_motion_frames

    if not should_materialize_frames and not make_flow_gif and not make_sensor_motion_gif:
        return None, None, None, None

    base_frames_dir = frames_dir or os.path.join("experiments", "sliding", "artifacts", "frames", run_name)
    resolved_flow_frames_dir = (
        os.path.join(base_frames_dir, "flow_field")
        if should_materialize_frames and render_flow_frames
        else None
    )
    resolved_sensor_motion_frames_dir = (
        os.path.join(base_frames_dir, "sensor_motion")
        if should_materialize_frames and render_sensor_motion_frames
        else None
    )

    resolved_flow_gif_path = None
    if make_flow_gif:
        resolved_flow_gif_path = flow_gif_path or os.path.join(base_frames_dir, f"{run_name}_flow.gif")

    resolved_sensor_motion_gif_path = None
    if make_sensor_motion_gif:
        resolved_sensor_motion_gif_path = sensor_motion_gif_path or os.path.join(
            base_frames_dir,
            f"{run_name}_sensor_motion.gif",
        )

    return (
        resolved_flow_frames_dir,
        resolved_sensor_motion_frames_dir,
        resolved_flow_gif_path,
        resolved_sensor_motion_gif_path,
    )

def _state_at_t(full_state_matrix, t_idx, total_steps):
    """Return flattened state vector x(t_idx) regardless of matrix orientation."""
    if full_state_matrix.shape[0] == total_steps:
        return np.asarray(full_state_matrix[t_idx], dtype=float).ravel()
    if full_state_matrix.shape[1] == total_steps:
        return np.asarray(full_state_matrix[:, t_idx], dtype=float).ravel()
    raise ValueError(
        f"Could not infer time axis from full_state_matrix shape {full_state_matrix.shape} "
        f"and total_steps={total_steps}"
    )


def _cumulative_variance_from_direct_svd(state_matrix, basis_rank):
    """Compute cumulative captured energy from an independent SVD-based check."""
    if basis_rank <= 0:
        return 0.0

    temporal_gram = np.asarray(state_matrix, dtype=float) @ np.asarray(state_matrix, dtype=float).T
    singular_values_sq = np.sort(svdvals(temporal_gram))[::-1]
    total_energy = float(np.sum(singular_values_sq))
    if total_energy <= 0.0:
        return 0.0

    return float(np.sum(singular_values_sq[:basis_rank]) / total_energy)


def _history_with_final_position(history, final_positions):
    """Append the final sensor positions for end-of-run trajectory plotting."""
    final_positions = np.asarray(final_positions, dtype=float)
    if not history:
        return final_positions[None, ...]
    return np.concatenate([np.stack(history), final_positions[None, ...]], axis=0)


def _trajectory_flow_snapshots(
    u,
    v,
    lagrangian_history=None,
    moving_history=None,
    history_time_indices=None,
    snapshot_time_indices=None,
):
    """Collect flow snapshots and matching sensor positions for the trajectory summary."""
    total_steps = int(u.shape[0])
    if snapshot_time_indices is None:
        snapshot_specs = (
            ("Start", 0),
            ("Middle", total_steps // 2),
            ("End", total_steps - 1),
        )
    else:
        snapshot_time_indices = [int(t_idx) for t_idx in snapshot_time_indices]
        if len(snapshot_time_indices) != 3:
            raise ValueError("trajectory_snapshot_indices must contain exactly three time indices")
        for t_idx in snapshot_time_indices:
            if t_idx < 0 or t_idx >= total_steps:
                raise ValueError(
                    "trajectory_snapshot_indices must be within the available time range "
                    f"[0, {total_steps - 1}]"
                )
        snapshot_specs = tuple(
            (f"Snapshot {snapshot_idx + 1}", t_idx)
            for snapshot_idx, t_idx in enumerate(snapshot_time_indices)
        )

    has_sensor_positions = (
        lagrangian_history is not None
        and moving_history is not None
        and history_time_indices is not None
    )
    if has_sensor_positions:
        lagrangian_history = np.asarray(lagrangian_history, dtype=float)
        moving_history = np.asarray(moving_history, dtype=float)
        history_time_indices = np.asarray(history_time_indices, dtype=int)
        if (
            lagrangian_history.shape[0] != moving_history.shape[0]
            or lagrangian_history.shape[0] != history_time_indices.shape[0]
        ):
            raise ValueError("Trajectory histories and time indices must have matching lengths")

    snapshots = []
    for label, t_idx in snapshot_specs:
        snapshot = {
            "label": label,
            "t_idx": int(t_idx),
            "u_grid": np.asarray(u[t_idx], dtype=float).copy(),
            "v_grid": np.asarray(v[t_idx], dtype=float).copy(),
        }
        if has_sensor_positions:
            history_idx = int(np.argmin(np.abs(history_time_indices - int(t_idx))))
            snapshot["history_idx"] = history_idx
            snapshot["history_t_idx"] = int(history_time_indices[history_idx])
            snapshot["lagrangian_history"] = lagrangian_history[: history_idx + 1].copy()
            snapshot["moving_history"] = moving_history[: history_idx + 1].copy()
            snapshot["lagrangian_positions"] = lagrangian_history[history_idx].copy()
            snapshot["moving_positions"] = moving_history[history_idx].copy()

        snapshots.append(snapshot)

    return snapshots


def run_experiment_sliding(
    u,
    v,
    window_len=30,
    step_size=1,
    min_dist_pct=0.05,
    dt=1.0,
    periodic=False,
    return_paths=False,
    config=None,
    plot_windows=True,
    save_window_frames=True,
    frames_dir=None,
    make_flow_gif=False,
    flow_gif_path=None,
    make_sensor_motion_gif=False,
    sensor_motion_gif_path=None,
    sensor_tail_length=48,
    plot_trajectories=False,
    trajectory_plot_path=None,
    trajectory_snapshot_indices=None,
    gif_duration=0.10,
    run_name="run",
    show_progress=True,
):
    """Run the sliding-window Window-POD experiment.

    Args:
        u: Velocity component shaped (T, nx, ny).
        v: Velocity component shaped (T, nx, ny).
        window_len: Sliding POD window length.
        step_size: Shift between windows.
        min_dist_pct: Minimum spacing between sensors as a fraction of lx.
        dt: Time step used for advection updates.
        periodic: Whether to apply periodic boundaries in motion updates.
        return_paths: Whether to return trajectory/history arrays.
        config: Optional ExperimentConfig; inferred from arrays if omitted.
        plot_windows: Whether animation frames should be rendered.
        save_window_frames: Whether animation frame directories are emitted.
        frames_dir: Optional base output directory for animation frames.
        make_flow_gif: Whether to create a flow-only GIF.
        flow_gif_path: Optional output GIF path for the flow-only animation.
        make_sensor_motion_gif: Whether to create a flow-plus-sensors GIF.
        sensor_motion_gif_path: Optional output GIF path for the sensor-motion animation.
        sensor_tail_length: Number of recent steps to retain in the fading sensor tail.
        plot_trajectories: Whether to save a final sensor-trajectory summary plot.
        trajectory_plot_path: Optional output path for the trajectory plot PNG.
        trajectory_snapshot_indices: Optional three time indices to show in the trajectory summary.
        gif_duration: GIF frame duration in seconds.
        run_name: Run label used in default output paths.
        show_progress: Whether to show tqdm progress.

    Returns:
        DataFrame of L2_h records, or tuple with path histories when return_paths=True.
    """
    if u.shape != v.shape:
        raise ValueError("u and v must have identical shape (T, nx, ny)")
    if u.ndim != 3:
        raise ValueError("u and v must be 3D arrays with shape (T, nx, ny)")

    total_steps, nx, ny = u.shape

    if config is None:
        experiment_config = config_from_arrays(u.shape)
    else:
        experiment_config = config
        if experiment_config.domain.nx != nx or experiment_config.domain.ny != ny:
            raise ValueError(
                f"Config domain (nx, ny)=({experiment_config.domain.nx}, {experiment_config.domain.ny}) "
                f"does not match data shape ({nx}, {ny})"
            )

    grid_n = nx * ny
    intervals = get_sliding_intervals(total_steps, window_len + 1, step_size)
    if not intervals:
        raise ValueError("No sliding intervals produced; adjust window_len or step_size")
    total_windows = len(intervals)

    # lagrangian_sensor_positions = seed_sensor_grid(
    #     experiment_config.num_sensors,
    #     experiment_config.domain.lx,
    #     experiment_config.domain.ly,
    # )

    lagrangian_sensor_positions = seed_uniform_random(
        experiment_config.num_sensors,
        experiment_config.domain.lx,
        experiment_config.domain.ly,
        seed=experiment_config.seed,
    )

    moving_pod_qr_sensor_positions = lagrangian_sensor_positions.copy()
    fixed_sensor_positions = lagrangian_sensor_positions.copy()

    lagrangian_history = []
    moving_history = []
    moving_frame_history = []
    moving_heading_history = []
    moving_flow_history = []
    trajectory_history_times = []

    records = []
    r_norm_history = []
    variance_captured_attr_history = []
    variance_captured_svd_history = []

    (
        resolved_flow_frames_dir,
        resolved_sensor_motion_frames_dir,
        resolved_flow_gif_path,
        resolved_sensor_motion_gif_path,
    ) = _resolve_animation_paths(
        plot_windows=plot_windows,
        save_window_frames=save_window_frames,
        run_name=run_name,
        frames_dir=frames_dir,
        make_flow_gif=make_flow_gif,
        flow_gif_path=flow_gif_path,
        make_sensor_motion_gif=make_sensor_motion_gif,
        sensor_motion_gif_path=sensor_motion_gif_path,
    )

    render_sensor_motion_frames = resolved_sensor_motion_frames_dir is not None
    track_sensor_paths = return_paths or plot_trajectories

    for animation_frames_dir in (resolved_flow_frames_dir, resolved_sensor_motion_frames_dir):
        if animation_frames_dir is not None:
            os.makedirs(animation_frames_dir, exist_ok=True)

    resolved_trajectory_plot_path = trajectory_plot_path
    if plot_trajectories and resolved_trajectory_plot_path is None:
        resolved_trajectory_plot_path = os.path.join(
            "experiments",
            "sliding",
            "artifacts",
            "plots",
            f"{run_name}_sensor_trajectories.png",
        )

    max_sensor_speed = float(np.max(np.hypot(u, v)))

    full_state_matrix = flatten_state(u, v)
    dx = experiment_config.domain.lx / nx
    dy = experiment_config.domain.ly / ny
    max_flow_speed = max_sensor_speed
    cfl_advect = max_flow_speed * dt / min(dx, dy)
    print("max speed:", max_flow_speed)
    print(
        "advection CFL:",
        cfl_advect,
        f"(max_flow_speed={max_flow_speed}, dx={dx}, dy={dy}, dt={dt})",
    )

    fixed_nodes = coords_to_linear_index(
        fixed_sensor_positions,
        nx,
        ny,
        experiment_config.domain.lx,
        experiment_config.domain.ly,
    )

    iterator = tqdm(intervals, desc="windows") if show_progress else intervals
    for window_idx, (start_idx, end_idx) in enumerate(iterator):
        # Midpoint snapshot used for this window's reconstruction diagnostics.
        t_eval = (start_idx + (end_idx - 1)) // 2
        if track_sensor_paths:
            lagrangian_history.append(lagrangian_sensor_positions.copy())
            moving_history.append(moving_pod_qr_sensor_positions.copy())
            trajectory_history_times.append(t_eval)

        # Fit POD/QR on the current window; keep midpoint diagnostics while
        # scoring relative L2_h error at the midpoint snapshot.
        window_state_matrix = flatten_state(u[start_idx : end_idx - 1], v[start_idx : end_idx - 1])

        window_sspor_model = fit_sspor_model(
            window_state_matrix,
            num_sensors=experiment_config.num_sensors,
            max_basis_dim=experiment_config.max_basis_dim,
            seed=experiment_config.seed,
        )

        window_qr_nodes = selected_nodes_from_uv(window_sspor_model.selected_sensors, nx, ny)
        window_basis_matrix = window_sspor_model.basis_matrix_
        basis_rank = int(window_basis_matrix.shape[1])
        explained_variance_ratio = np.asarray(
            window_sspor_model.basis.explained_variance_ratio_,
            dtype=float,
        )
        variance_captured_attr_history.append(float(np.sum(explained_variance_ratio)))
        variance_captured_svd_history.append(
            _cumulative_variance_from_direct_svd(window_state_matrix, basis_rank)
        )

        x_t = _state_at_t(full_state_matrix, t_eval, total_steps)
        Psi = np.asarray(window_basis_matrix, dtype=float)

        a_proj, *_ = np.linalg.lstsq(Psi, x_t, rcond=None)
        r_vec = x_t - Psi @ a_proj
        x_t_norm = l2h_norm(x_t, dx, dy)
        r_norm = l2h_norm(r_vec, dx, dy) / x_t_norm if x_t_norm > 0.0 else 0.0
        r_norm_history.append(r_norm)

        window_qr_index_pairs = np.column_stack(np.unravel_index(window_qr_nodes, (nx, ny)))
        window_qr_target_positions = grid_to_phys(
            window_qr_index_pairs,
            nx,
            ny,
            experiment_config.domain.lx,
            experiment_config.domain.ly,
        )

        lagrangian_nodes = coords_to_linear_index(
            lagrangian_sensor_positions,
            nx,
            ny,
            experiment_config.domain.lx,
            experiment_config.domain.ly,
        )
        

        # TODO: think about moving this back, right now, im moving it before we compute error
        moving_next_positions = advect_hungarian(
            curr_pts=moving_pod_qr_sensor_positions,
            opt_pts=window_qr_target_positions,
            lx=experiment_config.domain.lx,
            ly=experiment_config.domain.ly,
            v_max=max_sensor_speed,
            dt=dt,
            periodic=periodic,
        )

        # moving_pod_qr_sensor_positions = bounce_apart(
        #     moving_pod_qr_sensor_positions,
        #     min_dist_pct * experiment_config.domain.lx,
        #     experiment_config.domain.lx,
        #     experiment_config.domain.ly,
        # )

        moving_pod_qr_sensor_positions = moving_next_positions
        moving_pod_qr_nodes = coords_to_linear_index(
            moving_pod_qr_sensor_positions,
            nx,
            ny,
            experiment_config.domain.lx,
            experiment_config.domain.ly,
        )

        placement_nodes = {
            "QR teleport": window_qr_nodes,
            "Fixed": fixed_nodes,
            "Lagrangian": lagrangian_nodes,
            "Moving POD-QR": moving_pod_qr_nodes,
        }

        for placement_name, placement_node_idx in placement_nodes.items():
            # # Average over whole window:
            # l2h_samples = [
            #     relative_l2h_error_with_basis_matrix(
            #         full_state_matrix,
            #         t_idx=frame_idx,
            #         node_idx=placement_node_idx,
            #         basis_matrix=window_basis_matrix,
            #         grid_n=grid_n,
            #         dx=dx,
            #         dy=dy,
            #     )
            #     for frame_idx in range(start_idx, end_idx - 1)
            # ]
            # l2h_value = float(np.mean(l2h_samples))

            # Midpoint only evaluation:
            l2h_value = relative_l2h_error_with_basis_matrix(
                full_state_matrix,
                t_idx=t_eval,
                node_idx=placement_node_idx,
                basis_matrix=window_basis_matrix,
                grid_n=grid_n,
                dx=dx,
                dy=dy,
            )

            records.append(
                {
                    "window": window_idx,
                    "placement": placement_name,
                    "basis": "Window POD",
                    "L2_h": float(l2h_value),
                }
            )

        if resolved_flow_frames_dir is not None:
            flow_out_png = os.path.join(resolved_flow_frames_dir, f"frame_{window_idx:04d}.png")
            save_flow_field_frame(
                u_grid=u[t_eval],
                v_grid=v[t_eval],
                lx=experiment_config.domain.lx,
                ly=experiment_config.domain.ly,
                t_idx=t_eval,
                out_path=flow_out_png,
                run_name=run_name,
                speed_max=max_sensor_speed,
                quiver_step=experiment_config.quiver_step,
                x_origin=experiment_config.domain.x_min,
                y_origin=experiment_config.domain.y_min,
            )

        if resolved_sensor_motion_frames_dir is not None:
            moving_frame_history.append(moving_pod_qr_sensor_positions.copy())
            sensor_motion_out_png = os.path.join(
                resolved_sensor_motion_frames_dir,
                f"frame_{window_idx:04d}.png",
            )
            save_sensor_motion_frame(
                u_grid=u[t_eval],
                v_grid=v[t_eval],
                lx=experiment_config.domain.lx,
                ly=experiment_config.domain.ly,
                moving_history=moving_frame_history,
                t_idx=t_eval,
                out_path=sensor_motion_out_png,
                run_name=run_name,
                periodic=periodic,
                speed_max=max_sensor_speed,
                quiver_step=experiment_config.quiver_step,
                tail_length=sensor_tail_length,
                x_origin=experiment_config.domain.x_min,
                y_origin=experiment_config.domain.y_min,
            )

        # Match random_trials movement timing at the midpoint snapshot.
        lagrangian_sensor_positions = advect(
            lagrangian_sensor_positions,
            u[t_eval],
            v[t_eval],
            experiment_config.domain.lx,
            experiment_config.domain.ly,
            dt=dt,
            periodic=periodic,
        )
        lagrangian_sensor_positions = bounce_apart(
            lagrangian_sensor_positions,
            min_dist_pct * experiment_config.domain.lx,
            experiment_config.domain.lx,
            experiment_config.domain.ly,
        )
        
        moving_flow_vectors = _sample_flow_vectors(
            moving_pod_qr_sensor_positions,
            u[t_eval],
            v[t_eval],
            experiment_config.domain.lx,
            experiment_config.domain.ly,
        )



        moving_heading_velocity = (moving_next_positions - moving_pod_qr_sensor_positions) / dt
        if return_paths:
            moving_heading_history.append(moving_heading_velocity.copy())
            moving_flow_history.append(moving_flow_vectors.copy())


    if make_flow_gif and resolved_flow_frames_dir is not None and resolved_flow_gif_path is not None:
        make_window_gif(resolved_flow_frames_dir, resolved_flow_gif_path, duration=gif_duration)

    if (
        make_sensor_motion_gif
        and resolved_sensor_motion_frames_dir is not None
        and resolved_sensor_motion_gif_path is not None
    ):
        make_window_gif(
            resolved_sensor_motion_frames_dir,
            resolved_sensor_motion_gif_path,
            duration=gif_duration,
        )

    if plot_trajectories and resolved_trajectory_plot_path is not None:
        lagrangian_plot_history = _history_with_final_position(lagrangian_history, lagrangian_sensor_positions)
        moving_plot_history = _history_with_final_position(moving_history, moving_pod_qr_sensor_positions)
        trajectory_plot_times = np.asarray(trajectory_history_times + [total_steps - 1], dtype=int)
        save_trajectory_plot(
            lagrangian_plot_history,
            moving_plot_history,
            lx=experiment_config.domain.lx,
            ly=experiment_config.domain.ly,
            out_path=resolved_trajectory_plot_path,
            run_name=run_name,
            periodic=periodic,
            flow_snapshots=_trajectory_flow_snapshots(
                u,
                v,
                lagrangian_history=lagrangian_plot_history,
                moving_history=moving_plot_history,
                history_time_indices=trajectory_plot_times,
                snapshot_time_indices=trajectory_snapshot_indices,
            ),
            flow_speed_max=max_sensor_speed,
            quiver_step=experiment_config.quiver_step,
            x_origin=experiment_config.domain.x_min,
            y_origin=experiment_config.domain.y_min,
        )

    if r_norm_history:
        r_norm_array = np.asarray(r_norm_history, dtype=float)
        print(
            "\nrelative r_norm summary (L2_h): "
            f"mean={r_norm_array.mean():.6e} "
            f"variance={r_norm_array.var(ddof=0):.6e}"
        )

    if variance_captured_attr_history:
        variance_captured_attr_array = np.asarray(variance_captured_attr_history, dtype=float)
        print(
            "cumulative variance captured summary (basis attr): "
            f"mean={variance_captured_attr_array.mean():.6e} "
            f"variance={variance_captured_attr_array.var(ddof=0):.6e}"
        )

    if variance_captured_svd_history:
        variance_captured_svd_array = np.asarray(variance_captured_svd_history, dtype=float)
        print(
            "cumulative variance captured summary (direct SVD): "
            f"mean={variance_captured_svd_array.mean():.6e} "
            f"variance={variance_captured_svd_array.var(ddof=0):.6e}"
        )

    results = pd.DataFrame(records)

    if return_paths:
        return (
            results,
            np.stack(lagrangian_history),
            np.stack(moving_history),
            np.stack(moving_heading_history),
            np.stack(moving_flow_history),
        )

    return results

"""Sliding-window experiment pipeline for moving-sensor placement strategies."""

import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

from experiments.common.config import config_from_arrays
from experiments.common.sensor_motion import advect, advect_hungarian, bounce_apart
from experiments.common.spatial_utils import coords_to_linear_index, grid_to_phys, seed_sensor_grid
from experiments.common.state_reconstruction import (
    flatten_state,
    fit_sspor_model,
    rmse_with_basis_matrix,
    selected_nodes_from_uv,
)
from experiments.common.windowing import get_sliding_intervals

from .plotting import make_window_gif, save_window_frame


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


def _resolve_window_visual_paths(plot_windows, save_window_frames, run_name, frames_dir, make_gif, gif_path):
    """Resolve frame and GIF output paths.

    Args:
        plot_windows: Whether plotting is enabled.
        save_window_frames: Whether PNG frames are written.
        run_name: Name used for default output naming.
        frames_dir: Optional user-provided frame directory.
        make_gif: Whether GIF creation is requested.
        gif_path: Optional user-provided GIF output path.

    Returns:
        Tuple (resolved_frames_dir, resolved_gif_path).
    """
    if not plot_windows:
        return None, None

    resolved_frames_dir = None
    resolved_gif_path = None

    should_render_frames = save_window_frames or make_gif
    if should_render_frames:
        resolved_frames_dir = frames_dir or os.path.join("experiments", "sliding", "artifacts", "frames", run_name)

    if make_gif:
        resolved_gif_path = gif_path or os.path.join("experiments", "sliding", "artifacts", "frames", f"{run_name}.gif")

    return resolved_frames_dir, resolved_gif_path

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
    make_gif=False,
    gif_path=None,
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
        plot_windows: Whether per-window rendering is enabled.
        save_window_frames: Whether to save individual window PNGs.
        frames_dir: Optional output directory for saved frames.
        make_gif: Whether to create GIF from window PNGs.
        gif_path: Optional output GIF path.
        gif_duration: GIF frame duration in seconds.
        run_name: Run label used in default output paths.
        show_progress: Whether to show tqdm progress.

    Returns:
        DataFrame of RMSE records, or tuple with path histories when return_paths=True.
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

    lagrangian_sensor_positions = seed_sensor_grid(
        experiment_config.num_sensors,
        experiment_config.domain.lx,
        experiment_config.domain.ly,
    )
    moving_pod_qr_sensor_positions = lagrangian_sensor_positions.copy()
    fixed_sensor_positions = lagrangian_sensor_positions.copy()

    lagrangian_history = []
    moving_history = []
    moving_heading_history = []
    moving_flow_history = []

    records = []
    r_norm_history = []

    resolved_frames_dir, resolved_gif_path = _resolve_window_visual_paths(
        plot_windows=plot_windows,
        save_window_frames=save_window_frames,
        run_name=run_name,
        frames_dir=frames_dir,
        make_gif=make_gif,
        gif_path=gif_path,
    )

    if resolved_frames_dir is not None:
        os.makedirs(resolved_frames_dir, exist_ok=True)

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
        if return_paths:
            lagrangian_history.append(lagrangian_sensor_positions.copy())
            moving_history.append(moving_pod_qr_sensor_positions.copy())

        # Fit POD/QR on the current window; keep midpoint diagnostics, but score using the
        # mean reconstruction RMSE across every frame in the fitted window.
        t_eval = (start_idx + (end_idx - 1)) // 2
        window_state_matrix = flatten_state(u[start_idx : end_idx - 1], v[start_idx : end_idx - 1])
        window_frame_indices = range(start_idx, end_idx - 1)

        window_sspor_model = fit_sspor_model(
            window_state_matrix,
            num_sensors=experiment_config.num_sensors,
            max_basis_dim=experiment_config.max_basis_dim,
            seed=experiment_config.seed,
        )

        window_qr_nodes = selected_nodes_from_uv(window_sspor_model.selected_sensors, nx, ny)
        window_basis_matrix = window_sspor_model.basis_matrix_

        x_t = _state_at_t(full_state_matrix, t_eval, total_steps)
        Psi = np.asarray(window_basis_matrix, dtype=float)

        a_proj, *_ = np.linalg.lstsq(Psi, x_t, rcond=None)
        r_vec = x_t - Psi @ a_proj
        r_norm = float(np.linalg.norm(r_vec))
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
            # rmse_samples = [
            #     rmse_with_basis_matrix(
            #         full_state_matrix,
            #         t_idx=frame_idx,
            #         node_idx=placement_node_idx,
            #         basis_matrix=window_basis_matrix,
            #         grid_n=grid_n,
            #         nx=nx,
            #         ny=ny,
            #     )
            #     for frame_idx in window_frame_indices
            # ]
            # rmse_value = float(np.mean(rmse_samples))

            # Midpoint only evaluation:
            rmse_value = rmse_with_basis_matrix(
                full_state_matrix,
                t_idx=t_eval,
                node_idx=placement_node_idx,
                basis_matrix=window_basis_matrix,
                grid_n=grid_n,
                nx=nx,
                ny=ny,
            )

            records.append(
                {
                    "window": window_idx,
                    "placement": placement_name,
                    "basis": "Window POD",
                    "RMSE": float(rmse_value),
                }
            )

        if plot_windows and resolved_frames_dir is not None:
            out_png = os.path.join(resolved_frames_dir, f"frame_{window_idx:04d}.png")
            save_window_frame(
                u_grid=u[t_eval],
                v_grid=v[t_eval],
                lx=experiment_config.domain.lx,
                ly=experiment_config.domain.ly,
                fixed_sensor_positions=fixed_sensor_positions,
                lagrangian_sensor_positions=lagrangian_sensor_positions,
                window_qr_target_positions=window_qr_target_positions,
                moving_sensor_positions=moving_pod_qr_sensor_positions,
                window_idx=window_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                t_mid=t_eval,
                out_path=out_png,
                rmse_records=records,
                r_norm_history=r_norm_history,
                total_windows=total_windows,
                quiver_step=experiment_config.quiver_step,
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


    if make_gif and resolved_frames_dir is not None and resolved_gif_path is not None:
        make_window_gif(resolved_frames_dir, resolved_gif_path, duration=gif_duration)

    if r_norm_history:
        r_norm_array = np.asarray(r_norm_history, dtype=float)
        print(
            "\nr_norm summary: "
            f"mean={r_norm_array.mean():.6e} "
            f"variance={r_norm_array.var(ddof=0):.6e}"
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

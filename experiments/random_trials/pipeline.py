"""Pipeline for random-initialization Window-POD placement trials."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.common.config import config_from_arrays
from experiments.common.sensor_motion import advect, advect_hungarian, bounce_apart
from experiments.common.spatial_utils import coords_to_linear_index, grid_to_phys
from experiments.common.state_reconstruction import (
    fit_sspor_model,
    flatten_state,
    relative_l2h_error_with_basis_matrix,
    selected_nodes_from_uv,
)
from experiments.common.windowing import get_sliding_intervals


def _seed_uniform_random(num_sensors, lx, ly, rng):
    """Sample uniform random sensor coordinates in the physical domain.

    Args:
        num_sensors: Number of sensors.
        lx: Domain length in x.
        ly: Domain length in y.
        rng: NumPy random generator instance.

    Returns:
        Array shaped (num_sensors, 2) of sampled coordinates.
    """
    return np.column_stack(
        [
            rng.uniform(0.0, lx, int(num_sensors)),
            rng.uniform(0.0, ly, int(num_sensors)),
        ]
    )


def run_random_trials_window_pod(
    u,
    v,
    window_len,
    step_size=1,
    min_dist_pct=0.05,
    n_trials=50,
    seed=42,
    dt=1.0,
    periodic=False,
    config=None,
    show_progress=True,
    flow=None,
):
    """Run random-initialization Window-POD trials across placement strategies.

    Args:
        u: Velocity u snapshots shaped (T, nx, ny).
        v: Velocity v snapshots shaped (T, nx, ny).
        window_len: Sliding planning window length.
        step_size: Shift between consecutive windows.
        min_dist_pct: Minimum spacing as fraction of lx.
        n_trials: Number of random initialization trials.
        seed: Random seed for trial initialization.
        dt: Time step used for movement updates.
        periodic: Whether to apply periodic boundaries during movement.
        config: Optional ExperimentConfig; inferred from arrays if omitted.
        show_progress: Whether to show progress bars.
        flow: Optional flow label attached to output records.

    Returns:
        DataFrame with columns flow, num_sensors, trial, window, t, placement, L2_h.
    """
    if u.shape != v.shape:
        raise ValueError("u and v must have identical shape (T, nx, ny)")
    if u.ndim != 3:
        raise ValueError("u and v must be 3D arrays with shape (T, nx, ny)")
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")

    total_steps, nx, ny = u.shape

    if config is None:
        experiment_config = config_from_arrays(u.shape, seed=seed)
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

    full_state_matrix = flatten_state(u, v)
    dx = experiment_config.domain.lx / nx
    dy = experiment_config.domain.ly / ny
    max_sensor_speed = float(np.max(np.hypot(u, v)))

    rng = np.random.default_rng(seed)
    records = []

    trial_iterator = tqdm(range(int(n_trials)), desc="random-trials") if show_progress else range(int(n_trials))
    for trial_idx in trial_iterator:
        initial_sensor_positions = _seed_uniform_random(
            experiment_config.num_sensors,
            experiment_config.domain.lx,
            experiment_config.domain.ly,
            rng,
        )

        fixed_sensor_positions = initial_sensor_positions.copy()
        lagrangian_sensor_positions = initial_sensor_positions.copy()
        moving_pod_qr_sensor_positions = initial_sensor_positions.copy()

        for window_idx, (start_idx, end_idx) in enumerate(intervals):
            t_eval = (start_idx + (end_idx - 1)) // 2

            window_state_matrix = flatten_state(u[start_idx : end_idx - 1], v[start_idx : end_idx - 1])
            window_sspor_model = fit_sspor_model(
                window_state_matrix,
                num_sensors=experiment_config.num_sensors,
                max_basis_dim=experiment_config.max_basis_dim,
                seed=experiment_config.seed,
            )

            window_qr_nodes = selected_nodes_from_uv(window_sspor_model.selected_sensors, nx, ny)
            window_basis_matrix = window_sspor_model.basis_matrix_

            window_qr_index_pairs = np.column_stack(np.unravel_index(window_qr_nodes, (nx, ny)))
            window_qr_target_positions = grid_to_phys(
                window_qr_index_pairs,
                nx,
                ny,
                experiment_config.domain.lx,
                experiment_config.domain.ly,
            )

            fixed_nodes = coords_to_linear_index(
                fixed_sensor_positions,
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
            moving_pod_qr_nodes = coords_to_linear_index(
                moving_pod_qr_sensor_positions,
                nx,
                ny,
                experiment_config.domain.lx,
                experiment_config.domain.ly,
            )

            placement_nodes = {
                "Fixed": fixed_nodes,
                "Lagrangian": lagrangian_nodes,
                "Moving POD-QR": moving_pod_qr_nodes,
                "QR teleport": window_qr_nodes,
            }

            for placement_name, placement_node_idx in placement_nodes.items():
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
                        "flow": flow,
                        "num_sensors": experiment_config.num_sensors,
                        "trial": trial_idx,
                        "window": window_idx,
                        "t": t_eval,
                        "placement": placement_name,
                        "L2_h": float(l2h_value),
                    }
                )

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

            moving_pod_qr_sensor_positions = advect_hungarian(
                curr_pts=moving_pod_qr_sensor_positions,
                opt_pts=window_qr_target_positions,
                lx=experiment_config.domain.lx,
                ly=experiment_config.domain.ly,
                v_max=max_sensor_speed,
                dt=dt,
                periodic=periodic,
            )
            moving_pod_qr_sensor_positions = bounce_apart(
                moving_pod_qr_sensor_positions,
                min_dist_pct * experiment_config.domain.lx,
                experiment_config.domain.lx,
                experiment_config.domain.ly,
            )

    return pd.DataFrame(records)

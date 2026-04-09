"""Pipeline for Global-POD versus Window-POD sensor-strategy comparison."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.common.config import config_from_arrays
from experiments.common.sensor_motion import advect, advect_hungarian, bounce_apart
from experiments.common.spatial_utils import coords_to_linear_index, grid_to_phys, seed_sensor_grid
from experiments.common.state_reconstruction import (
    fit_pod_basis,
    fit_sspor_model,
    flatten_state,
    relative_l2h_error_with_basis_matrix,
    selected_nodes_from_uv,
)
from experiments.common.windowing import get_sliding_intervals


def run_pod_basis_comparison(
    u,
    v,
    window_len=30,
    step_size=1,
    min_dist_pct=0.05,
    dt=1.0,
    periodic=False,
    config=None,
    show_progress=True,
    flow=None,
):
    """Compare reconstruction errors under Global POD and Window POD.

    Args:
        u: Velocity u snapshots shaped (T, nx, ny).
        v: Velocity v snapshots shaped (T, nx, ny).
        window_len: Sliding planning window length.
        step_size: Shift between consecutive windows.
        min_dist_pct: Minimum spacing as fraction of lx.
        dt: Time step used for movement updates.
        periodic: Whether to apply periodic boundaries during movement.
        config: Optional ExperimentConfig; inferred from arrays if omitted.
        show_progress: Whether to show a tqdm progress bar.
        flow: Optional flow label stored in output records.

    Returns:
        DataFrame with columns flow, num_sensors, window, t, basis, method, L2_h.
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

    full_state_matrix = flatten_state(u, v)
    dx = experiment_config.domain.lx / nx
    dy = experiment_config.domain.ly / ny

    global_basis_matrix = fit_pod_basis(
        full_state_matrix,
        max_basis_dim=experiment_config.max_basis_dim,
        seed=experiment_config.seed,
    )

    static_sspor_model = fit_sspor_model(
        full_state_matrix,
        num_sensors=experiment_config.num_sensors,
        max_basis_dim=experiment_config.max_basis_dim,
        seed=experiment_config.seed,
    )
    static_qr_nodes = selected_nodes_from_uv(static_sspor_model.selected_sensors, nx, ny)

    lagrangian_sensor_positions = seed_sensor_grid(
        experiment_config.num_sensors,
        experiment_config.domain.lx,
        experiment_config.domain.ly,
    )
    moving_pod_qr_sensor_positions = lagrangian_sensor_positions.copy()

    max_sensor_speed = float(np.max(np.hypot(u, v)))

    records = []

    iterator = tqdm(intervals, desc="global-sensors") if show_progress else intervals
    for window_idx, (start_idx, end_idx) in enumerate(iterator):
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

        method_nodes = {
            "Static QR": static_qr_nodes,
            "Teleport QR": window_qr_nodes,
            "Lagrangian": lagrangian_nodes,
            "Moving QR": moving_pod_qr_nodes,
        }

        for method_name, method_node_idx in method_nodes.items():
            global_l2h = relative_l2h_error_with_basis_matrix(
                full_state_matrix,
                t_idx=t_eval,
                node_idx=method_node_idx,
                basis_matrix=global_basis_matrix,
                grid_n=grid_n,
                dx=dx,
                dy=dy,
            )
            records.append(
                {
                    "window": window_idx,
                    "t": t_eval,
                    "basis": "Global POD",
                    "method": method_name,
                    "L2_h": float(global_l2h),
                    "flow": flow,
                    "num_sensors": experiment_config.num_sensors,
                }
            )

            window_l2h = relative_l2h_error_with_basis_matrix(
                full_state_matrix,
                t_idx=t_eval,
                node_idx=method_node_idx,
                basis_matrix=window_basis_matrix,
                grid_n=grid_n,
                dx=dx,
                dy=dy,
            )
            records.append(
                {
                    "window": window_idx,
                    "t": t_eval,
                    "basis": "Window POD",
                    "method": method_name,
                    "L2_h": float(window_l2h),
                    "flow": flow,
                    "num_sensors": experiment_config.num_sensors,
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

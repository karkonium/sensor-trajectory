"""Pipeline for causal reconstructed-window Moving POD-QR."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.common.config import config_from_arrays
from experiments.common.sensor_motion import advect, advect_hungarian, bounce_apart
from experiments.common.spatial_utils import coords_to_linear_index, grid_to_phys
from experiments.common.state_reconstruction import (
    fit_pod_basis,
    fit_sspor_model,
    flatten_state,
    l2h_norm,
    reconstruct_state_with_basis_matrix,
    selected_nodes_from_uv,
)


METHOD_STATIC = "Static QR"
METHOD_LAGRANGIAN = "Lagrangian"
METHOD_ONLINE = "Online Moving POD-QR"
METHOD_ORDER = [METHOD_STATIC, METHOD_LAGRANGIAN, METHOD_ONLINE]


@dataclass
class OnlineReconstructedWindowResult:
    """Outputs needed for records and per-flow visualizations."""

    raw_records: pd.DataFrame
    static_positions: np.ndarray
    lagrangian_history: np.ndarray
    online_history: np.ndarray
    test_indices: np.ndarray
    speed_max: float


def _nodes_to_positions(node_idx, nx, ny, lx, ly):
    """Map scalar node indices to physical coordinates."""
    node_idx = np.asarray(node_idx, dtype=int)
    index_pairs = np.column_stack(np.unravel_index(node_idx, (nx, ny)))
    return grid_to_phys(index_pairs, nx, ny, lx, ly)


def _relative_l2h_error(true_state, reconstructed_state, dx, dy):
    """Score one reconstructed flattened state against its full-state truth."""
    true_norm = l2h_norm(true_state, dx, dy)
    if true_norm <= 0.0:
        return 0.0
    return l2h_norm(true_state - reconstructed_state, dx, dy) / true_norm


def _append_error_record(
    records,
    cumulative_sums,
    cumulative_counts,
    *,
    flow,
    method,
    t_idx,
    l2h_value,
    num_sensors,
    window_len,
    max_basis_dim,
    train_fraction,
):
    """Append one raw record with a running mean error."""
    cumulative_sums[method] += float(l2h_value)
    cumulative_counts[method] += 1
    records.append(
        {
            "flow": flow,
            "method": method,
            "t": int(t_idx),
            "L2_h": float(l2h_value),
            "cumulative_L2_h": float(cumulative_sums[method] / cumulative_counts[method]),
            "num_sensors": int(num_sensors),
            "window_len": int(window_len),
            "max_basis_dim": int(max_basis_dim),
            "train_fraction": float(train_fraction),
        }
    )


def _reconstruct_and_score(full_state_matrix, t_idx, node_idx, basis_matrix, grid_n, dx, dy):
    """Reconstruct one state from sensor nodes and return reconstruction plus error."""
    true_state = np.asarray(full_state_matrix[int(t_idx)], dtype=float).ravel()
    reconstructed_state = reconstruct_state_with_basis_matrix(
        true_state,
        node_idx=node_idx,
        basis_matrix=basis_matrix,
        grid_n=grid_n,
    )
    return reconstructed_state, _relative_l2h_error(true_state, reconstructed_state, dx, dy)


def run_online_reconstructed_window(
    u,
    v,
    *,
    train_fraction=0.5,
    window_len=13,
    min_dist_pct=0.05,
    dt=1.0,
    periodic=False,
    config=None,
    num_sensors=10,
    max_basis_dim=10,
    seed=90,
    flow=None,
    show_progress=True,
):
    """Run causal reconstructed-window Moving POD-QR on one flow case.

    Args:
        u: Velocity component shaped (T, nx, ny).
        v: Velocity component shaped (T, nx, ny).
        train_fraction: Fraction of snapshots reserved as full-state training data.
        window_len: Rolling reconstructed-state window length.
        min_dist_pct: Minimum spacing between moving sensors as a fraction of lx.
        dt: Time step for sensor advection/motion.
        periodic: Whether to apply periodic boundary handling in motion updates.
        config: Optional ExperimentConfig; inferred from arrays if omitted.
        num_sensors: Number of sensor locations.
        max_basis_dim: Maximum POD basis dimension.
        seed: Random seed passed to POD/SSPOR fitting.
        flow: Optional flow label stored in records.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        OnlineReconstructedWindowResult for one flow.
    """
    if u.shape != v.shape:
        raise ValueError("u and v must have identical shape (T, nx, ny)")
    if u.ndim != 3:
        raise ValueError("u and v must be 3D arrays with shape (T, nx, ny)")

    total_steps, nx, ny = u.shape
    if config is None:
        experiment_config = config_from_arrays(
            u.shape,
            num_sensors=num_sensors,
            max_basis_dim=max_basis_dim,
            seed=seed,
        )
    else:
        experiment_config = config
        if experiment_config.domain.nx != nx or experiment_config.domain.ny != ny:
            raise ValueError(
                f"Config domain (nx, ny)=({experiment_config.domain.nx}, {experiment_config.domain.ny}) "
                f"does not match data shape ({nx}, {ny})"
            )

    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    train_steps = int(np.floor(total_steps * float(train_fraction)))
    if train_steps < 2:
        raise ValueError("Training segment must contain at least two snapshots")
    if train_steps >= total_steps:
        raise ValueError("Test segment must contain at least one snapshot")
    if int(window_len) <= 0:
        raise ValueError("window_len must be > 0")

    grid_n = nx * ny
    full_state_matrix = flatten_state(u, v)
    train_state_matrix = full_state_matrix[:train_steps]
    test_indices = np.arange(train_steps, total_steps, dtype=int)

    dx = experiment_config.domain.lx / nx
    dy = experiment_config.domain.ly / ny
    lx = experiment_config.domain.lx
    ly = experiment_config.domain.ly
    min_sensor_dist = float(min_dist_pct) * lx
    max_sensor_speed = float(np.max(np.hypot(u, v)))

    print(
        f"Training snapshots: 0..{train_steps - 1}; "
        f"test snapshots: {train_steps}..{total_steps - 1}"
    )
    print("max speed:", max_sensor_speed)

    global_basis_matrix = fit_pod_basis(
        train_state_matrix,
        max_basis_dim=experiment_config.max_basis_dim,
        seed=experiment_config.seed,
    )
    static_sspor_model = fit_sspor_model(
        train_state_matrix,
        num_sensors=experiment_config.num_sensors,
        max_basis_dim=experiment_config.max_basis_dim,
        seed=experiment_config.seed,
    )

    static_nodes = selected_nodes_from_uv(static_sspor_model.selected_sensors, nx, ny)
    static_positions = _nodes_to_positions(static_nodes, nx, ny, lx, ly)
    lagrangian_positions = static_positions.copy()
    online_positions = static_positions.copy()
    online_basis_matrix = np.asarray(global_basis_matrix, dtype=float)

    reconstructed_window = deque(maxlen=int(window_len))
    records = []
    cumulative_sums = {method: 0.0 for method in METHOD_ORDER}
    cumulative_counts = {method: 0 for method in METHOD_ORDER}
    lagrangian_history = []
    online_history = []

    iterator = tqdm(test_indices, desc=f"{flow or 'flow'} online-window") if show_progress else test_indices
    last_test_t = int(test_indices[-1])
    for t_idx in iterator:
        t_idx = int(t_idx)
        lagrangian_history.append(lagrangian_positions.copy())
        online_history.append(online_positions.copy())

        lagrangian_nodes = coords_to_linear_index(lagrangian_positions, nx, ny, lx, ly)
        online_nodes = coords_to_linear_index(online_positions, nx, ny, lx, ly)

        _, static_l2h = _reconstruct_and_score(
            full_state_matrix,
            t_idx,
            static_nodes,
            global_basis_matrix,
            grid_n,
            dx,
            dy,
        )
        _, lagrangian_l2h = _reconstruct_and_score(
            full_state_matrix,
            t_idx,
            lagrangian_nodes,
            global_basis_matrix,
            grid_n,
            dx,
            dy,
        )
        online_reconstruction, online_l2h = _reconstruct_and_score(
            full_state_matrix,
            t_idx,
            online_nodes,
            online_basis_matrix,
            grid_n,
            dx,
            dy,
        )

        for method_name, l2h_value in (
            (METHOD_STATIC, static_l2h),
            (METHOD_LAGRANGIAN, lagrangian_l2h),
            (METHOD_ONLINE, online_l2h),
        ):
            _append_error_record(
                records,
                cumulative_sums,
                cumulative_counts,
                flow=flow,
                method=method_name,
                t_idx=t_idx,
                l2h_value=l2h_value,
                num_sensors=experiment_config.num_sensors,
                window_len=window_len,
                max_basis_dim=experiment_config.max_basis_dim,
                train_fraction=train_fraction,
            )

        reconstructed_window.append(online_reconstruction)
        if len(reconstructed_window) == int(window_len):
            reconstructed_window_matrix = np.stack(reconstructed_window)
            window_sspor_model = fit_sspor_model(
                reconstructed_window_matrix,
                num_sensors=experiment_config.num_sensors,
                max_basis_dim=experiment_config.max_basis_dim,
                seed=experiment_config.seed,
            )
            online_basis_matrix = window_sspor_model.basis_matrix_

            if t_idx != last_test_t:
                target_nodes = selected_nodes_from_uv(window_sspor_model.selected_sensors, nx, ny)
                target_positions = _nodes_to_positions(target_nodes, nx, ny, lx, ly)
                online_positions = advect_hungarian(
                    curr_pts=online_positions,
                    opt_pts=target_positions,
                    lx=lx,
                    ly=ly,
                    v_max=max_sensor_speed,
                    dt=dt,
                    periodic=periodic,
                )
                online_positions = bounce_apart(online_positions, min_sensor_dist, lx, ly)

        if t_idx != last_test_t:
            lagrangian_positions = advect(
                lagrangian_positions,
                u[t_idx],
                v[t_idx],
                lx,
                ly,
                dt=dt,
                periodic=periodic,
            )
            lagrangian_positions = bounce_apart(lagrangian_positions, min_sensor_dist, lx, ly)

    raw_df = pd.DataFrame(records)
    raw_df = raw_df[
        [
            "flow",
            "method",
            "t",
            "L2_h",
            "cumulative_L2_h",
            "num_sensors",
            "window_len",
            "max_basis_dim",
            "train_fraction",
        ]
    ]
    return OnlineReconstructedWindowResult(
        raw_records=raw_df,
        static_positions=static_positions,
        lagrangian_history=np.stack(lagrangian_history),
        online_history=np.stack(online_history),
        test_indices=test_indices,
        speed_max=max_sensor_speed,
    )

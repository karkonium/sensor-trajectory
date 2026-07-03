"""Pipeline for causal reconstructed-window Moving POD-QR."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
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

BASIS_STATIC_TRAIN = "static_train"
BASIS_GLOBAL_TRAIN = "global_train"
BASIS_RECONSTRUCTED_WINDOW = "reconstructed_window"

DIAGNOSTIC_COLUMNS = [
    "flow",
    "t",
    "target_overlap_previous",
    "target_overlap_static",
    "mean_distance_to_targets",
    "mean_step_distance",
    "num_sensors",
    "global_basis_dim",
    "window_len",
    "window_basis_dim",
    "basis_used",
]


@dataclass
class OnlineReconstructedWindowResult:
    """Outputs needed for records, diagnostics, and per-flow visualizations."""

    raw_records: pd.DataFrame
    diagnostic_records: pd.DataFrame
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


def _validate_rank_settings(num_sensors, global_basis_dim, window_len, window_basis_dim):
    """Validate rank choices against paired u/v sensor measurements."""
    num_measurements = 2 * int(num_sensors)
    if int(global_basis_dim) <= 0:
        raise ValueError("global_basis_dim must be > 0")
    if int(window_basis_dim) <= 0:
        raise ValueError("window_basis_dim must be > 0")
    if int(window_len) <= 0:
        raise ValueError("window_len must be > 0")
    if int(global_basis_dim) > num_measurements:
        raise ValueError(
            "global_basis_dim must be <= 2 * num_sensors when the global basis is used "
            f"for bootstrap reconstruction; got global_basis_dim={global_basis_dim}, "
            f"2*num_sensors={num_measurements}"
        )
    if int(window_basis_dim) > num_measurements:
        raise ValueError(
            "window_basis_dim must be <= 2 * num_sensors to avoid underdetermined "
            f"reconstruction; got window_basis_dim={window_basis_dim}, "
            f"2*num_sensors={num_measurements}"
        )
    if int(window_basis_dim) > int(window_len):
        raise ValueError(
            "window_basis_dim must be <= window_len for reconstructed-window POD; "
            f"got window_basis_dim={window_basis_dim}, window_len={window_len}"
        )
    if int(window_basis_dim) > int(global_basis_dim):
        print(
            "[online_reconstruction warning] window_basis_dim > global_basis_dim. "
            "This is allowed, but window_basis_dim <= global_basis_dim is preferred "
            "when reconstructed states are bootstrapped from the global basis."
        )


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
    global_basis_dim,
    window_len,
    window_basis_dim,
    train_fraction,
    basis_used,
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
            "global_basis_dim": int(global_basis_dim),
            "window_len": int(window_len),
            "window_basis_dim": int(window_basis_dim),
            "train_fraction": float(train_fraction),
            "basis_used": str(basis_used),
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


def _as_basis_matrix_or_raise(basis_matrix, grid_n, basis_name):
    """Validate fitted basis shape and return a NumPy basis matrix."""
    basis_matrix = np.asarray(basis_matrix, dtype=float)
    expected_features = 2 * int(grid_n)
    if basis_matrix.ndim != 2:
        raise ValueError(f"{basis_name} basis must be 2D, got shape {basis_matrix.shape}")
    if basis_matrix.shape[0] != expected_features and basis_matrix.shape[1] == expected_features:
        basis_matrix = basis_matrix.T
    if basis_matrix.shape[0] != expected_features:
        raise ValueError(
            f"{basis_name} basis has unexpected shape {basis_matrix.shape}; "
            f"expected ({expected_features}, r) or (r, {expected_features})"
        )
    if basis_matrix.shape[1] < 1:
        raise ValueError(f"{basis_name} basis has no active modes")
    return basis_matrix


def _periodic_delta(p, q, length):
    """Return shortest signed displacement from p to q on one periodic axis."""
    delta = q - p
    delta[delta > +length / 2.0] -= length
    delta[delta < -length / 2.0] += length
    return delta


def _assignment_distances(curr_pts, target_pts, lx, ly, periodic=False):
    """Return Hungarian-assigned distances from current positions to targets."""
    curr_pts = np.asarray(curr_pts, dtype=float)
    target_pts = np.asarray(target_pts, dtype=float)
    if periodic:
        dx = np.abs(curr_pts[:, None, 0] - target_pts[None, :, 0])
        dy = np.abs(curr_pts[:, None, 1] - target_pts[None, :, 1])
        dx = np.minimum(dx, float(lx) - dx)
        dy = np.minimum(dy, float(ly) - dy)
        cost = np.hypot(dx, dy)
    else:
        cost = np.linalg.norm(curr_pts[:, None, :] - target_pts[None, :, :], axis=2)

    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols]


def _mean_step_distance(old_pts, new_pts, lx, ly, periodic=False):
    """Return mean realized sensor step length."""
    old_pts = np.asarray(old_pts, dtype=float)
    new_pts = np.asarray(new_pts, dtype=float)
    step = new_pts - old_pts
    if periodic:
        step[:, 0] = _periodic_delta(old_pts[:, 0], new_pts[:, 0], float(lx))
        step[:, 1] = _periodic_delta(old_pts[:, 1], new_pts[:, 1], float(ly))
    return float(np.mean(np.linalg.norm(step, axis=1)))


def _target_overlap(nodes_a, nodes_b):
    """Return fraction of unique target nodes in nodes_a that overlap nodes_b."""
    if nodes_a is None or nodes_b is None:
        return np.nan
    set_a = set(np.asarray(nodes_a, dtype=int).ravel().tolist())
    set_b = set(np.asarray(nodes_b, dtype=int).ravel().tolist())
    if not set_a:
        return np.nan
    return float(len(set_a.intersection(set_b)) / len(set_a))


def _append_target_diagnostic(
    diagnostic_records,
    *,
    flow,
    t_idx,
    target_nodes,
    previous_target_nodes,
    static_nodes,
    online_positions,
    target_positions,
    next_online_positions,
    lx,
    ly,
    periodic,
    num_sensors,
    global_basis_dim,
    window_len,
    window_basis_dim,
    basis_used,
):
    """Append one online target/movement diagnostic record."""
    assigned_distances = _assignment_distances(
        online_positions,
        target_positions,
        lx=lx,
        ly=ly,
        periodic=periodic,
    )
    diagnostic_records.append(
        {
            "flow": flow,
            "t": int(t_idx),
            "target_overlap_previous": _target_overlap(target_nodes, previous_target_nodes),
            "target_overlap_static": _target_overlap(target_nodes, static_nodes),
            "mean_distance_to_targets": float(np.mean(assigned_distances)),
            "mean_step_distance": _mean_step_distance(
                online_positions,
                next_online_positions,
                lx=lx,
                ly=ly,
                periodic=periodic,
            ),
            "num_sensors": int(num_sensors),
            "global_basis_dim": int(global_basis_dim),
            "window_len": int(window_len),
            "window_basis_dim": int(window_basis_dim),
            "basis_used": str(basis_used),
        }
    )


def _fit_reconstructed_window_model(
    reconstructed_window_matrix,
    *,
    num_sensors,
    window_basis_dim,
    seed,
    grid_n,
):
    """Fit reconstructed-window SSPOR and validate the local basis."""
    window_sspor_model = fit_sspor_model(
        reconstructed_window_matrix,
        num_sensors=num_sensors,
        max_basis_dim=window_basis_dim,
        seed=seed,
    )
    window_basis_matrix = _as_basis_matrix_or_raise(
        window_sspor_model.basis_matrix_,
        grid_n=grid_n,
        basis_name="reconstructed-window",
    )
    return window_sspor_model, window_basis_matrix


def _summarize_diagnostics(flow, diagnostic_df):
    """Print concise per-flow target/motion diagnostics."""
    if diagnostic_df.empty:
        print(f"{flow}: no online target diagnostics were recorded.")
        return

    summary = diagnostic_df.agg(
        {
            "target_overlap_previous": "mean",
            "target_overlap_static": "mean",
            "mean_distance_to_targets": "mean",
            "mean_step_distance": "mean",
        }
    )
    print(
        f"{flow}: online target diagnostics "
        f"mean overlap(prev)={summary['target_overlap_previous']:.3f}, "
        f"mean overlap(static)={summary['target_overlap_static']:.3f}, "
        f"mean distance-to-target={summary['mean_distance_to_targets']:.6e}, "
        f"mean step={summary['mean_step_distance']:.6e}"
    )


def run_online_reconstructed_window(
    u,
    v,
    *,
    train_fraction=0.5,
    window_len=25,
    min_dist_pct=0.05,
    dt=1.0,
    periodic=False,
    config=None,
    num_sensors=10,
    global_basis_dim=15,
    window_basis_dim=8,
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
        global_basis_dim: Richer training rank used to bootstrap online reconstructions.
        window_basis_dim: Smaller local rank used by reconstructed-window POD-QR.
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
            max_basis_dim=max(global_basis_dim, window_basis_dim),
            seed=seed,
        )
    else:
        experiment_config = config
        if experiment_config.domain.nx != nx or experiment_config.domain.ny != ny:
            raise ValueError(
                f"Config domain (nx, ny)=({experiment_config.domain.nx}, {experiment_config.domain.ny}) "
                f"does not match data shape ({nx}, {ny})"
            )

    _validate_rank_settings(
        num_sensors=experiment_config.num_sensors,
        global_basis_dim=global_basis_dim,
        window_len=window_len,
        window_basis_dim=window_basis_dim,
    )

    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    train_steps = int(np.floor(total_steps * float(train_fraction)))
    if train_steps < 2:
        raise ValueError("Training segment must contain at least two snapshots")
    if train_steps >= total_steps:
        raise ValueError("Test segment must contain at least one snapshot")

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

    # GLOBAL_BASIS_DIM is the richer training representation used only to
    # bootstrap online reconstructed states before a local window basis exists.
    global_basis_matrix = fit_pod_basis(
        train_state_matrix,
        max_basis_dim=global_basis_dim,
        seed=experiment_config.seed,
    )

    # WINDOW_BASIS_DIM is the smaller local rank used by reconstructed-window
    # POD-QR. Static QR uses the same rank for a fair fixed-sensor baseline.
    static_sspor_model = fit_sspor_model(
        train_state_matrix,
        num_sensors=experiment_config.num_sensors,
        max_basis_dim=window_basis_dim,
        seed=experiment_config.seed,
    )
    static_basis_matrix = _as_basis_matrix_or_raise(
        static_sspor_model.basis_matrix_,
        grid_n=grid_n,
        basis_name="static-training",
    )

    static_nodes = selected_nodes_from_uv(static_sspor_model.selected_sensors, nx, ny)
    static_positions = _nodes_to_positions(static_nodes, nx, ny, lx, ly)
    lagrangian_positions = static_positions.copy()
    online_positions = static_positions.copy()
    online_basis_matrix = np.asarray(global_basis_matrix, dtype=float)
    online_basis_used = BASIS_GLOBAL_TRAIN

    reconstructed_window = deque(maxlen=int(window_len))
    previous_target_nodes = None
    records = []
    diagnostic_records = []
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
            static_basis_matrix,
            grid_n,
            dx,
            dy,
        )
        _, lagrangian_l2h = _reconstruct_and_score(
            full_state_matrix,
            t_idx,
            lagrangian_nodes,
            static_basis_matrix,
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

        for method_name, l2h_value, basis_used in (
            (METHOD_STATIC, static_l2h, BASIS_STATIC_TRAIN),
            (METHOD_LAGRANGIAN, lagrangian_l2h, BASIS_STATIC_TRAIN),
            (METHOD_ONLINE, online_l2h, online_basis_used),
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
                global_basis_dim=global_basis_dim,
                window_len=window_len,
                window_basis_dim=window_basis_dim,
                train_fraction=train_fraction,
                basis_used=basis_used,
            )

        # This is the causal step: target updates use only reconstructions
        # accumulated up to the current time, never true full-state test windows.
        reconstructed_window.append(online_reconstruction)
        if len(reconstructed_window) == int(window_len):
            reconstructed_window_matrix = np.stack(reconstructed_window)
            try:
                window_sspor_model, window_basis_matrix = _fit_reconstructed_window_model(
                    reconstructed_window_matrix,
                    num_sensors=experiment_config.num_sensors,
                    window_basis_dim=window_basis_dim,
                    seed=experiment_config.seed,
                    grid_n=grid_n,
                )
                online_basis_matrix = window_basis_matrix
                online_basis_used = BASIS_RECONSTRUCTED_WINDOW
                target_nodes = selected_nodes_from_uv(window_sspor_model.selected_sensors, nx, ny)
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(
                    f"[online_reconstruction warning] {flow or 'flow'} t={t_idx}: "
                    f"reconstructed-window fit failed ({exc}); falling back to static_train basis/targets."
                )
                online_basis_matrix = static_basis_matrix
                online_basis_used = BASIS_STATIC_TRAIN
                target_nodes = static_nodes

            target_positions = _nodes_to_positions(target_nodes, nx, ny, lx, ly)
            next_online_positions = online_positions.copy()
            if t_idx != last_test_t:
                next_online_positions = advect_hungarian(
                    curr_pts=online_positions,
                    opt_pts=target_positions,
                    lx=lx,
                    ly=ly,
                    v_max=max_sensor_speed,
                    dt=dt,
                    periodic=periodic,
                )
                next_online_positions = bounce_apart(next_online_positions, min_sensor_dist, lx, ly)

            _append_target_diagnostic(
                diagnostic_records,
                flow=flow,
                t_idx=t_idx,
                target_nodes=target_nodes,
                previous_target_nodes=previous_target_nodes,
                static_nodes=static_nodes,
                online_positions=online_positions,
                target_positions=target_positions,
                next_online_positions=next_online_positions,
                lx=lx,
                ly=ly,
                periodic=periodic,
                num_sensors=experiment_config.num_sensors,
                global_basis_dim=global_basis_dim,
                window_len=window_len,
                window_basis_dim=window_basis_dim,
                basis_used=online_basis_used,
            )
            previous_target_nodes = np.asarray(target_nodes, dtype=int).copy()
            online_positions = next_online_positions

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
            "global_basis_dim",
            "window_len",
            "window_basis_dim",
            "train_fraction",
            "basis_used",
        ]
    ]
    diagnostic_df = pd.DataFrame(diagnostic_records)
    if not diagnostic_df.empty:
        diagnostic_df = diagnostic_df[DIAGNOSTIC_COLUMNS]
    else:
        diagnostic_df = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
    _summarize_diagnostics(flow or "flow", diagnostic_df)

    return OnlineReconstructedWindowResult(
        raw_records=raw_df,
        diagnostic_records=diagnostic_df,
        static_positions=static_positions,
        lagrangian_history=np.stack(lagrangian_history),
        online_history=np.stack(online_history),
        test_indices=test_indices,
        speed_max=max_sensor_speed,
    )

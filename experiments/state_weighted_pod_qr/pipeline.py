"""Pipeline for state-weighted POD-QR sensor-selection diagnostics.

Standard POD-QR is basis-only and fixed in time: it computes one global POD
basis and one QR sensor set. Instant weighted POD-QR uses the current state's
POD coefficient magnitudes to reweight modes before QR sensor selection.
Reconstructing with PsiW_i is kept as a numerical diagnostic because PsiW_i
typically spans the same subspace as Psi when all weights are nonzero.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.common.config import config_from_arrays
from experiments.common.spatial_utils import grid_to_phys

try:
    from scipy.linalg import qr as scipy_qr
except ImportError:
    scipy_qr = None


STANDARD_METHOD = "Standard POD-QR / recon Psi"
WEIGHTED_PSI_METHOD = "Instant weighted / recon Psi"
WEIGHTED_PSIW_METHOD = "Instant weighted / recon PsiW"
METHOD_ORDER = [STANDARD_METHOD, WEIGHTED_PSI_METHOD, WEIGHTED_PSIW_METHOD]

PSI_LABEL = "Psi"
PSIW_LABEL = "PsiW_i"


@dataclass
class StateWeightedPodQrResult:
    """Experiment outputs for one flow case."""

    raw_records: pd.DataFrame
    condition_records: pd.DataFrame
    sensor_records: pd.DataFrame
    overlap_records: pd.DataFrame


def flatten_state(u, v):
    """Flatten (u, v) fields into concatenated state matrix (T, 2 * grid_n)."""
    if u.shape != v.shape:
        raise ValueError("u and v must have identical shape (T, nx, ny)")

    total_steps = u.shape[0]
    u_flat = u.reshape(total_steps, -1)
    v_flat = v.reshape(total_steps, -1)
    return np.concatenate([u_flat, v_flat], axis=1)


def _safe_basis_dim(state_matrix, requested):
    """Pick a valid direct-SVD basis rank from matrix shape and requested rank."""
    if state_matrix.ndim != 2:
        raise ValueError("Expected 2D matrix for basis fitting")

    cap = min(state_matrix.shape)
    if cap < 1:
        raise ValueError(f"Cannot fit POD basis for matrix with shape {state_matrix.shape}")

    return max(1, min(int(requested), int(cap)))


def fit_pod_basis_snapshot_svd(state_matrix, max_basis_dim):
    """Fit an orthonormal POD basis using the method of snapshots.

    This avoids a hard dependency on pysensors for the new experiment while
    preserving the same state-matrix orientation used by the older pipelines.
    """
    state_matrix = np.asarray(state_matrix, dtype=float)
    basis_dim = _safe_basis_dim(state_matrix, max_basis_dim)

    gram_matrix = state_matrix @ state_matrix.T
    eigenvalues, temporal_modes = np.linalg.eigh(gram_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    temporal_modes = temporal_modes[:, order]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    positive = singular_values > (np.finfo(float).eps * max(state_matrix.shape))
    if not np.any(positive):
        raise ValueError("Cannot fit POD basis because the state matrix has zero energy")

    temporal_modes = temporal_modes[:, positive][:, :basis_dim]
    singular_values = singular_values[positive][:basis_dim]
    basis_matrix = state_matrix.T @ (temporal_modes / singular_values[None, :])

    basis_matrix, _ = np.linalg.qr(basis_matrix)
    return basis_matrix[:, : len(singular_values)]


def measurement_indices_uv(node_idx, grid_n):
    """Expand scalar-node indices to augmented [u|v] measurement indices."""
    node_idx = np.asarray(node_idx, dtype=int)
    node_idx = np.unique(np.clip(node_idx, 0, int(grid_n) - 1))
    return np.sort(np.unique(np.concatenate([node_idx, node_idx + int(grid_n)])))


def l2h_norm(state_vector, dx, dy):
    """Compute the discrete L2_h norm for a flattened [u, v] state."""
    return float(np.sqrt(dx * dy) * np.linalg.norm(np.asarray(state_vector, dtype=float).ravel()))


def relative_l2h_error_with_basis_matrix(full_state_matrix, t_idx, node_idx, basis_matrix, grid_n, dx, dy):
    """Reconstruct one state at a time index and score relative L2_h error."""
    sensor_indices_uv = measurement_indices_uv(node_idx, grid_n)

    true_state = np.asarray(full_state_matrix[t_idx : t_idx + 1], dtype=float)
    sampled_state = true_state[:, sensor_indices_uv]

    basis_matrix_use = _as_feature_by_mode_basis(basis_matrix, grid_n)
    sampled_basis = basis_matrix_use[sensor_indices_uv, :]
    sampled_basis_pinv = np.linalg.pinv(sampled_basis)
    reconstructed_state = (sampled_state @ sampled_basis_pinv.T) @ basis_matrix_use.T

    error_norm = l2h_norm(true_state - reconstructed_state, dx, dy)
    true_norm = l2h_norm(true_state, dx, dy)
    return error_norm / true_norm if true_norm > 0.0 else 0.0


def resolve_eval_indices(total_steps, start=None, end=None, stride=1):
    """Resolve a snapshot segment for scoring and weighted sensor selection."""
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    start_idx = 0 if start is None else int(start)
    end_idx = int(total_steps) if end is None else int(end)
    start_idx = max(0, min(start_idx, int(total_steps) - 1))
    end_idx = max(start_idx + 1, min(end_idx, int(total_steps)))

    return np.arange(start_idx, end_idx, int(stride), dtype=int)


def _as_feature_by_mode_basis(basis_matrix, grid_n):
    """Orient a POD basis as (2 * grid_n, rank)."""
    basis_matrix_use = np.asarray(basis_matrix, dtype=float)
    expected_features = 2 * int(grid_n)
    if basis_matrix_use.shape[0] != expected_features and basis_matrix_use.shape[1] == expected_features:
        basis_matrix_use = basis_matrix_use.T
    if basis_matrix_use.shape[0] != expected_features:
        raise ValueError(
            f"basis_matrix has unexpected shape {basis_matrix.shape}; "
            f"expected ({expected_features}, r) or (r, {expected_features})"
        )
    return basis_matrix_use


def _pivoted_qr_feature_order(feature_by_mode_basis, tol=1e-12):
    """Return a QRCP ordering of feature rows.

    SciPy's pivoted QR is used when available. If it is not installed, the
    fallback is modified Gram-Schmidt with column pivoting on ``basis.T``. The
    basis rank is small in these experiments, while the feature count can be
    large.
    """
    basis_matrix = np.asarray(feature_by_mode_basis, dtype=float)
    if basis_matrix.ndim != 2:
        raise ValueError("feature_by_mode_basis must be 2D")

    if scipy_qr is not None:
        _q_matrix, _r_matrix, pivots = scipy_qr(
            basis_matrix.T,
            pivoting=True,
            mode="economic",
            check_finite=False,
        )
        return [int(feature_idx) for feature_idx in pivots]

    qr_matrix = basis_matrix.T.copy()
    rank, n_features = qr_matrix.shape
    if rank <= 0 or n_features <= 0:
        raise ValueError("feature_by_mode_basis must have nonzero dimensions")

    original_norms = np.einsum("ij,ij->j", qr_matrix, qr_matrix)
    residual_norms = original_norms.copy()
    selected = np.zeros(n_features, dtype=bool)
    pivots = []
    norm_tol = float(tol) * max(float(np.max(original_norms)), 1.0)

    for _ in range(min(rank, n_features)):
        residual_norms[selected] = -np.inf
        pivot = int(np.argmax(residual_norms))
        pivot_norm_sq = float(residual_norms[pivot])
        if not np.isfinite(pivot_norm_sq) or pivot_norm_sq <= norm_tol:
            break

        pivots.append(pivot)
        selected[pivot] = True

        q_vec = qr_matrix[:, pivot] / np.sqrt(pivot_norm_sq)
        qr_matrix -= np.outer(q_vec, q_vec @ qr_matrix)
        residual_norms = np.einsum("ij,ij->j", qr_matrix, qr_matrix)

    fallback_order = np.argsort(original_norms)[::-1]
    fallback_order = [int(feature_idx) for feature_idx in fallback_order if not selected[feature_idx]]
    return pivots + fallback_order


def _select_nodes_from_basis_qr(feature_by_mode_basis, num_sensors, grid_n):
    """Select scalar grid nodes from QRCP feature pivots on a POD-like basis."""
    if num_sensors <= 0:
        raise ValueError("num_sensors must be > 0")

    feature_order = _pivoted_qr_feature_order(feature_by_mode_basis)
    selected_nodes = []
    seen_nodes = set()
    for feature_idx in feature_order:
        node_idx = int(feature_idx % int(grid_n))
        if node_idx in seen_nodes:
            continue
        selected_nodes.append(node_idx)
        seen_nodes.add(node_idx)
        if len(selected_nodes) == int(num_sensors):
            break

    if len(selected_nodes) < int(num_sensors):
        raise ValueError(
            f"Only selected {len(selected_nodes)} unique nodes; requested {num_sensors}"
        )
    return np.asarray(selected_nodes, dtype=int)


def _condition_number_for_nodes(basis_matrix, node_idx, grid_n):
    """Compute cond(C basis) for paired u/v measurements at scalar nodes."""
    sensor_indices_uv = measurement_indices_uv(node_idx, grid_n)
    sampled_basis = _as_feature_by_mode_basis(basis_matrix, grid_n)[sensor_indices_uv, :]
    return float(np.linalg.cond(sampled_basis))


def _node_indices_to_phys(node_indices, nx, ny, lx, ly):
    """Convert scalar-grid node indices to physical coordinates."""
    index_pairs = np.column_stack(np.unravel_index(np.asarray(node_indices, dtype=int), (nx, ny)))
    return grid_to_phys(index_pairs, nx, ny, lx, ly)


def _append_sensor_records(records, flow, t_idx, method, node_idx, nx, ny, lx, ly, num_sensors, max_basis_dim):
    """Append one row per selected sensor for plotting and auditability."""
    positions = _node_indices_to_phys(node_idx, nx, ny, lx, ly)
    for sensor_id, (node, position) in enumerate(zip(node_idx, positions)):
        records.append(
            {
                "flow": flow,
                "t": int(t_idx),
                "method": method,
                "sensor_id": int(sensor_id),
                "node": int(node),
                "x": float(position[0]),
                "y": float(position[1]),
                "num_sensors": int(num_sensors),
                "max_basis_dim": int(max_basis_dim),
            }
        )


def _append_raw_record(
    records,
    flow,
    t_idx,
    method,
    selection_basis,
    reconstruction_basis,
    num_sensors,
    max_basis_dim,
    l2h_value,
    cond_selection_basis,
    cond_reconstruction_basis,
):
    """Append one raw reconstruction diagnostic row."""
    records.append(
        {
            "flow": flow,
            "t": int(t_idx),
            "method": method,
            "selection_basis": selection_basis,
            "reconstruction_basis": reconstruction_basis,
            "num_sensors": int(num_sensors),
            "max_basis_dim": int(max_basis_dim),
            "L2_h": float(l2h_value),
            "cond_selection_basis": float(cond_selection_basis),
            "cond_reconstruction_basis": float(cond_reconstruction_basis),
        }
    )


def run_state_weighted_pod_qr(
    u,
    v,
    num_sensors,
    max_basis_dim,
    seed=90,
    eps=1e-8,
    eval_start=None,
    eval_end=None,
    eval_stride=1,
    config=None,
    show_progress=True,
    flow=None,
):
    """Compare fixed global POD-QR against instantaneous weighted POD-QR.

    Args:
        u: Velocity u snapshots shaped (T, nx, ny).
        v: Velocity v snapshots shaped (T, nx, ny).
        num_sensors: Number of scalar grid-node sensors.
        max_basis_dim: Maximum global POD basis rank K.
        seed: Random seed passed to POD fitting.
        eps: Positive floor added to abs(POD coefficients).
        eval_start: Optional first snapshot index to score.
        eval_end: Optional exclusive final snapshot index to score.
        eval_stride: Snapshot stride for the scored/test segment.
        config: Optional ExperimentConfig; inferred from arrays if omitted.
        show_progress: Whether to show a tqdm progress bar.
        flow: Optional flow label stored in output records.

    Returns:
        StateWeightedPodQrResult with raw, condition, sensor, and overlap records.
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

    grid_n = nx * ny
    full_state_matrix = flatten_state(u, v)
    dx = experiment_config.domain.lx / nx
    dy = experiment_config.domain.ly / ny
    eval_indices = resolve_eval_indices(total_steps, eval_start, eval_end, eval_stride)

    # Standard POD-QR is basis-only and fixed in time.
    global_basis = fit_pod_basis_snapshot_svd(
        full_state_matrix,
        max_basis_dim=max_basis_dim,
    )
    global_basis = _as_feature_by_mode_basis(global_basis, grid_n)

    standard_nodes = _select_nodes_from_basis_qr(global_basis, num_sensors, grid_n)
    standard_condition = _condition_number_for_nodes(global_basis, standard_nodes, grid_n)

    raw_records = []
    condition_records = []
    sensor_records = []
    overlap_records = []
    previous_weighted_nodes = None
    previous_t_idx = None

    iterator = tqdm(eval_indices, desc=f"{flow or 'flow'} weighted-POD-QR") if show_progress else eval_indices
    for t_idx in iterator:
        t_idx = int(t_idx)
        x_t = np.asarray(full_state_matrix[int(t_idx)], dtype=float).ravel()

        # Instant weighted POD-QR uses current-state coefficients to prioritize
        # active modes before QR sensor selection.
        coefficients = global_basis.T @ x_t
        weights = np.abs(coefficients) + float(eps)
        weighted_basis = global_basis * weights[None, :]
        weighted_nodes = _select_nodes_from_basis_qr(weighted_basis, num_sensors, grid_n)

        cond_weighted_selection = _condition_number_for_nodes(weighted_basis, weighted_nodes, grid_n)
        cond_weighted_recon_psi = _condition_number_for_nodes(global_basis, weighted_nodes, grid_n)

        method_specs = [
            (
                STANDARD_METHOD,
                PSI_LABEL,
                PSI_LABEL,
                standard_nodes,
                global_basis,
                standard_condition,
                standard_condition,
            ),
            (
                WEIGHTED_PSI_METHOD,
                PSIW_LABEL,
                PSI_LABEL,
                weighted_nodes,
                global_basis,
                cond_weighted_selection,
                cond_weighted_recon_psi,
            ),
            (
                WEIGHTED_PSIW_METHOD,
                PSIW_LABEL,
                PSIW_LABEL,
                weighted_nodes,
                weighted_basis,
                cond_weighted_selection,
                cond_weighted_selection,
            ),
        ]

        for (
            method_name,
            selection_basis,
            reconstruction_basis,
            node_idx,
            reconstruction_basis_matrix,
            cond_selection_basis,
            cond_reconstruction_basis,
        ) in method_specs:
            l2h_value = relative_l2h_error_with_basis_matrix(
                full_state_matrix,
                t_idx=t_idx,
                node_idx=node_idx,
                basis_matrix=reconstruction_basis_matrix,
                grid_n=grid_n,
                dx=dx,
                dy=dy,
            )
            _append_raw_record(
                raw_records,
                flow=flow,
                t_idx=t_idx,
                method=method_name,
                selection_basis=selection_basis,
                reconstruction_basis=reconstruction_basis,
                num_sensors=num_sensors,
                max_basis_dim=max_basis_dim,
                l2h_value=l2h_value,
                cond_selection_basis=cond_selection_basis,
                cond_reconstruction_basis=cond_reconstruction_basis,
            )

            condition_records.append(
                {
                    "flow": flow,
                    "t": t_idx,
                    "method": method_name,
                    "num_sensors": int(num_sensors),
                    "max_basis_dim": int(max_basis_dim),
                    "condition_number": float(cond_reconstruction_basis),
                    "cond_selection_basis": float(cond_selection_basis),
                    "cond_reconstruction_basis": float(cond_reconstruction_basis),
                }
            )
            _append_sensor_records(
                sensor_records,
                flow=flow,
                t_idx=t_idx,
                method=method_name,
                node_idx=node_idx,
                nx=nx,
                ny=ny,
                lx=experiment_config.domain.lx,
                ly=experiment_config.domain.ly,
                num_sensors=num_sensors,
                max_basis_dim=max_basis_dim,
            )

        if previous_weighted_nodes is not None:
            overlap = len(set(weighted_nodes).intersection(set(previous_weighted_nodes))) / float(num_sensors)
            overlap_records.append(
                {
                    "flow": flow,
                    "t": t_idx,
                    "previous_t": int(previous_t_idx),
                    "method": "Instant weighted POD-QR",
                    "num_sensors": int(num_sensors),
                    "max_basis_dim": int(max_basis_dim),
                    "overlap": float(overlap),
                }
            )
        previous_weighted_nodes = weighted_nodes.copy()
        previous_t_idx = t_idx

    raw_df = pd.DataFrame(raw_records)
    raw_df = raw_df[
        [
            "flow",
            "t",
            "method",
            "selection_basis",
            "reconstruction_basis",
            "num_sensors",
            "max_basis_dim",
            "L2_h",
            "cond_selection_basis",
            "cond_reconstruction_basis",
        ]
    ]
    condition_df = pd.DataFrame(condition_records)
    sensor_df = pd.DataFrame(sensor_records)
    overlap_df = pd.DataFrame(overlap_records)
    return StateWeightedPodQrResult(raw_df, condition_df, sensor_df, overlap_df)

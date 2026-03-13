"""State reconstruction and POD/QR fitting utilities for sensor evaluation."""

import numpy as np
import pysensors as ps

from state_concatenation.split_state import split_state

from .spatial_utils import expand_to_uv


def rmse(a, b):
    """Compute root-mean-square error.

    Args:
        a: Reference array.
        b: Predicted array.

    Returns:
        Scalar RMSE value.
    """
    return float(np.sqrt(np.mean((a - b) ** 2)))


def flatten_state(u, v):
    """Flatten (u, v) fields into concatenated state matrix (T, 2 * grid_n).

    Args:
        u: Velocity component array shaped (T, nx, ny).
        v: Velocity component array shaped (T, nx, ny).

    Returns:
        Concatenated state matrix shaped (T, 2 * grid_n).
    """
    if u.shape != v.shape:
        raise ValueError("u and v must have identical shape (T, nx, ny)")

    total_steps = u.shape[0]
    u_flat = u.reshape(total_steps, -1)
    v_flat = v.reshape(total_steps, -1)
    return np.concatenate([u_flat, v_flat], axis=1)


def _safe_basis_dim(state_matrix, requested):
    """Pick an ARPACK-safe basis rank from matrix shape and requested rank.

    Args:
        state_matrix: 2D snapshot matrix.
        requested: User-requested maximum basis rank.

    Returns:
        Safe integer basis rank.
    """
    if state_matrix.ndim != 2:
        raise ValueError("Expected 2D matrix for basis fitting")

    cap = min(state_matrix.shape) - 1
    if cap < 1:
        raise ValueError(
            f"Cannot fit ARPACK basis for matrix with shape {state_matrix.shape}; need min(shape) >= 2"
        )

    return max(1, min(int(requested), int(cap)))


def fit_pod_basis(state_matrix, max_basis_dim, seed):
    """Fit POD basis matrix using pysensors SVD backend.

    Args:
        state_matrix: Snapshot matrix (n_samples, n_features).
        max_basis_dim: Requested maximum basis rank.
        seed: Random seed.

    Returns:
        Basis matrix from fitted POD model.
    """
    basis_dim = _safe_basis_dim(state_matrix, max_basis_dim)
    pod_basis = ps.basis.SVD(
        n_basis_modes=basis_dim,
        algorithm="arpack",
        random_state=seed,
    )
    pod_basis.fit(state_matrix)
    return pod_basis.basis_matrix_

def fit_sspor_model(state_matrix, num_sensors, max_basis_dim, seed):
    """Fit SSPOR model on a snapshot matrix.

    Args:
        state_matrix: Snapshot matrix (n_samples, n_features).
        num_sensors: Number of sensors to select.
        max_basis_dim: Requested maximum basis rank.
        seed: Random seed.

    Returns:
        Trained pysensors SSPOR model.
    """
    basis_dim = _safe_basis_dim(state_matrix, max_basis_dim)
    pod_basis = ps.basis.SVD(
        n_basis_modes=basis_dim,
        algorithm="arpack",
        random_state=seed,
    )
    sspor_model = ps.SSPOR(n_sensors=num_sensors, basis=pod_basis)
    sspor_model.fit(state_matrix)
    return sspor_model


def selected_nodes_from_uv(selected_sensors, nx, ny):
    """Collapse [u|v] feature indices to scalar-node indices.

    Args:
        selected_sensors: Selected feature indices from SSPOR.
        nx: Number of grid points in x.
        ny: Number of grid points in y.

    Returns:
        Node indices in scalar-grid indexing.
    """
    grid_n = int(nx) * int(ny)
    selected_sensors = np.asarray(selected_sensors, dtype=int)
    return np.mod(selected_sensors, grid_n)


def measurement_indices_uv(node_idx, grid_n):
    """Expand scalar-node indices to augmented [u|v] measurement indices.

    Args:
        node_idx: Scalar-node indices in [0, grid_n-1].
        grid_n: Number of scalar nodes per velocity component.

    Returns:
        Sorted unique measurement indices that sample both u and v components.
    """
    node_idx = np.asarray(node_idx, dtype=int)
    node_idx = np.unique(np.clip(node_idx, 0, grid_n - 1))
    return expand_to_uv(node_idx, grid_n)


def rmse_with_basis_matrix(full_state_matrix, t_idx, node_idx, basis_matrix, grid_n, nx, ny):
    """Reconstruct state at one time index and score mean u/v RMSE.

    Args:
        full_state_matrix: Full state matrix shaped (T, 2 * grid_n).
        t_idx: Time index to evaluate.
        node_idx: Scalar-node sensor indices.
        basis_matrix: Basis matrix shaped (2*grid_n, r) or transposed.
        grid_n: Number of scalar nodes per velocity component.
        nx: Number of grid points in x.
        ny: Number of grid points in y.

    Returns:
        Scalar RMSE score averaged across u and v components.
    """
    sensor_indices_uv = measurement_indices_uv(node_idx, grid_n)

    true_state = full_state_matrix[t_idx : t_idx + 1]
    sampled_state = true_state[:, sensor_indices_uv]

    basis_matrix_use = basis_matrix
    if basis_matrix_use.shape[0] != 2 * grid_n and basis_matrix_use.shape[1] == 2 * grid_n:
        basis_matrix_use = basis_matrix_use.T
    if basis_matrix_use.shape[0] != 2 * grid_n:
        raise ValueError(
            f"basis_matrix has unexpected shape {basis_matrix.shape}; "
            f"expected (2*grid_n, r) or (r, 2*grid_n)"
        )

    sampled_basis = basis_matrix_use[sensor_indices_uv, :]
    sampled_basis_pinv = np.linalg.pinv(sampled_basis)

    reconstructed_state = (sampled_state @ sampled_basis_pinv.T) @ basis_matrix_use.T

    u_true, v_true = split_state(true_state, 2 * nx, ny, horizontal_concat=False)
    u_recon, v_recon = split_state(reconstructed_state, 2 * nx, ny, horizontal_concat=False)
    return float(0.5 * (rmse(u_true, u_recon) + rmse(v_true, v_recon)))

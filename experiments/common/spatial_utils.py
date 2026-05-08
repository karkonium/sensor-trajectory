"""Geometry and index-conversion helpers for sensor placement experiments."""

from random import seed

import numpy as np


def linspace_interior(length, n, pad_frac=0.15):
    """Generate interior points with symmetric edge padding.

    Args:
        length: Domain length.
        n: Number of points.
        pad_frac: Fractional padding from both boundaries.

    Returns:
        NumPy array of evenly spaced points.
    """
    return np.linspace(pad_frac * length, (1.0 - pad_frac) * length, n)


def seed_uniform_random(num_sensors, lx, ly, seed=42):
    """Sample uniform random sensor coordinates in the physical domain.

    Args:
        num_sensors: Number of sensors.
        lx: Domain length in x.
        ly: Domain length in y.
        rng: NumPy random generator instance.

    Returns:
        Array shaped (num_sensors, 2) of sampled coordinates.
    """
    rng = np.random.default_rng(seed)

    return np.column_stack(
        [
            rng.uniform(0.0, lx, int(num_sensors)),
            rng.uniform(0.0, ly, int(num_sensors)),
        ]
    )


def seed_sensor_grid(n, lx, ly):
    """Seed approximately-uniform sensor coordinates in physical space.

    Args:
        n: Number of sensors.
        lx: Domain length in x.
        ly: Domain length in y.

    Returns:
        Array of shape (n, 2) with seeded physical coordinates.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    gx = int(np.sqrt(n * lx / ly))
    gx = max(gx, 1)
    gy = int(np.ceil(n / gx))

    xs = linspace_interior(lx, gx)
    ys = linspace_interior(ly, gy)
    xv, yv = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xv.ravel(), yv.ravel()])[:n]


def expand_to_uv(node_indices, grid_n):
    """Expand scalar node indices to [u|v] concatenated-state indices.

    Args:
        node_indices: 1D array of scalar node indices.
        grid_n: Number of scalar nodes in one field component.

    Returns:
        Sorted unique indices that include u and v positions.
    """
    node_indices = np.asarray(node_indices, dtype=int)
    return np.sort(np.unique(np.concatenate([node_indices, node_indices + grid_n])))


def coords_to_linear_index(points, nx, ny, lx, ly):
    """Map physical coordinates to nearest linearized grid-node indices.

    Args:
        points: Array shaped (n_points, 2) of physical coordinates.
        nx: Number of grid points in x.
        ny: Number of grid points in y.
        lx: Domain length in x.
        ly: Domain length in y.

    Returns:
        Integer array of linear node indices.
    """
    points = np.asarray(points, dtype=float)

    i = np.clip(np.round(points[:, 0] / lx * (nx - 1)).astype(int), 0, nx - 1)
    j = np.clip(np.round(points[:, 1] / ly * (ny - 1)).astype(int), 0, ny - 1)
    return np.ravel_multi_index((i, j), (nx, ny))


def grid_to_phys(coords, nx, ny, lx, ly):
    """Map integer grid indices (i, j) to physical coordinates (x, y).

    Args:
        coords: Array shaped (n_points, 2) of integer grid indices.
        nx: Number of grid points in x.
        ny: Number of grid points in y.
        lx: Domain length in x.
        ly: Domain length in y.

    Returns:
        Array shaped (n_points, 2) with physical coordinates.
    """
    grid_x = np.linspace(0.0, lx, nx)
    grid_y = np.linspace(0.0, ly, ny)
    coords = np.asarray(coords, dtype=int)

    i_idx, j_idx = coords[:, 0], coords[:, 1]
    x_phys = grid_x[i_idx]
    y_phys = grid_y[j_idx]
    return np.column_stack([x_phys, y_phys])

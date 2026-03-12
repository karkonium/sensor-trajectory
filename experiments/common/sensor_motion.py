"""Sensor motion utilities: advection, assignment-based movement, and spacing control."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment


def bounce_apart(coords, min_dist, lx, ly):
    """Push sensor pairs apart when they violate minimum separation.

    Args:
        coords: Array shaped (n_sensors, 2) of physical coordinates.
        min_dist: Minimum allowed pairwise distance.
        lx: Domain length in x.
        ly: Domain length in y.

    Returns:
        Updated coordinate array with spacing enforcement applied.
    """
    coords = np.asarray(coords, dtype=float).copy()
    p = len(coords)

    for i in range(p):
        for j in range(i + 1, p):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]

            dx = (dx + lx / 2.0) % lx - lx / 2.0
            dy = (dy + ly / 2.0) % ly - ly / 2.0
            dist = np.hypot(dx, dy)

            if dist < 1e-12:
                dx, dy, dist = 1e-6, 0.0, 1e-6

            if dist < min_dist:
                overlap = 0.5 * (min_dist - dist)
                ux, uy = dx / dist, dy / dist
                coords[i, 0] += overlap * ux
                coords[i, 1] += overlap * uy
                coords[j, 0] -= overlap * ux
                coords[j, 1] -= overlap * uy

    coords[:, 0] = np.clip(coords[:, 0], 0.0, lx)
    coords[:, 1] = np.clip(coords[:, 1], 0.0, ly)
    return coords


def advect(points, u, v, lx, ly, dt=1.0, periodic=False):
    """Forward-Euler advection of points through a velocity field.

    Args:
        points: Array shaped (n_points, 2) of physical coordinates.
        u: Velocity component on grid (nx, ny).
        v: Velocity component on grid (nx, ny).
        lx: Domain length in x.
        ly: Domain length in y.
        dt: Time step.
        periodic: Whether to enforce periodic boundaries.

    Returns:
        Updated point coordinates after advection.
    """
    if periodic:
        grid_x = np.linspace(0.0, lx, u.shape[0], endpoint=False)
        grid_y = np.linspace(0.0, ly, u.shape[1], endpoint=False)
    else:
        grid_x = np.linspace(0.0, lx, u.shape[0])
        grid_y = np.linspace(0.0, ly, u.shape[1])

    u_interpolator = RegularGridInterpolator((grid_x, grid_y), u, bounds_error=False, fill_value=None)
    v_interpolator = RegularGridInterpolator((grid_x, grid_y), v, bounds_error=False, fill_value=None)

    point_array = np.asarray(points, dtype=float).copy()

    if periodic:
        point_array[:, 0] = np.mod(point_array[:, 0], lx)
        point_array[:, 1] = np.mod(point_array[:, 1], ly)

    sampled_velocity = np.stack([u_interpolator(point_array), v_interpolator(point_array)], axis=1)
    updated_points = point_array + dt * sampled_velocity

    if periodic:
        updated_points[:, 0] = np.mod(updated_points[:, 0], lx)
        updated_points[:, 1] = np.mod(updated_points[:, 1], ly)
    else:
        updated_points[:, 0] = np.clip(updated_points[:, 0], 0.0, lx)
        updated_points[:, 1] = np.clip(updated_points[:, 1], 0.0, ly)

    return updated_points


def _periodic_delta(p, q, length):
    """Compute signed shortest displacement from p to q on a periodic axis.

    Args:
        p: Current coordinates.
        q: Target coordinates.
        length: Domain length for the axis.

    Returns:
        Shortest signed displacement values.
    """
    d = q - p
    d[d > +length / 2.0] -= length
    d[d < -length / 2.0] += length
    return d


def advect_hungarian(curr_pts, opt_pts, lx, ly, v_max, dt=1.0, periodic=False, alpha=1.0):
    """Move sensors toward targets via Hungarian assignment with speed capping.

    Args:
        curr_pts: Current sensor coordinates, shape (n_sensors, 2).
        opt_pts: Target coordinates, shape (n_sensors, 2).
        lx: Domain length in x.
        ly: Domain length in y.
        v_max: Maximum speed magnitude.
        dt: Time step.
        periodic: Whether domain is periodic.
        alpha: Fraction of target displacement to apply before speed cap.

    Returns:
        Updated sensor coordinates.
    """
    curr_pts = np.asarray(curr_pts, dtype=float)
    opt_pts = np.asarray(opt_pts, dtype=float)

    if periodic:
        dx = np.abs(curr_pts[:, None, 0] - opt_pts[None, :, 0])
        dy = np.abs(curr_pts[:, None, 1] - opt_pts[None, :, 1])
        dx = np.minimum(dx, lx - dx)
        dy = np.minimum(dy, ly - dy)
        cost = np.hypot(dx, dy)
    else:
        cost = np.linalg.norm(curr_pts[:, None, :] - opt_pts[None, :, :], axis=2)

    rows, cols = linear_sum_assignment(cost)
    displacement = opt_pts[cols] - curr_pts[rows]

    if periodic:
        displacement[:, 0] = _periodic_delta(curr_pts[rows, 0], opt_pts[cols, 0], lx)
        displacement[:, 1] = _periodic_delta(curr_pts[rows, 1], opt_pts[cols, 1], ly)

    remaining_distance = np.linalg.norm(displacement, axis=1)
    step_length = np.minimum(alpha * remaining_distance, v_max * dt)
    step_vector = displacement * (step_length / (remaining_distance + 1e-12))[:, None]

    updated_points = curr_pts.copy()
    updated_points[rows] += step_vector

    if periodic:
        updated_points[:, 0] = np.mod(updated_points[:, 0], lx)
        updated_points[:, 1] = np.mod(updated_points[:, 1], ly)
    else:
        updated_points[:, 0] = np.clip(updated_points[:, 0], 0.0, lx)
        updated_points[:, 1] = np.clip(updated_points[:, 1], 0.0, ly)

    return updated_points

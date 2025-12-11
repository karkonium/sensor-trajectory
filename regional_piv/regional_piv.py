import os
from joblib import Parallel, delayed

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap

import pysensors as ps


def _tile_bounds_from_center(ci, cj, win_nx, win_ny, nx, ny):
    """Clamp a (win_nx × win_ny) tile centered at (ci,cj) to image bounds."""
    hi = win_nx // 2
    hj = win_ny // 2
    i0 = int(np.clip(ci - hi, 0, max(0, nx - win_nx)))
    j0 = int(np.clip(cj - hj, 0, max(0, ny - win_ny)))
    return i0, i0 + win_nx, j0, j0 + win_ny


def _auto_rank_energy(svals, tau=0.99):
    """Smallest k s.t. cumulative energy ≥ tau."""
    e = svals**2
    cum = np.cumsum(e)
    k = int(np.searchsorted(cum / cum[-1], tau) + 1)
    return k


def _local_pod_qr_sensors(u_blk, v_blk):
    """
    u_blk, v_blk: (W, nx_t, ny_t) local time-space block.
    Returns selected *feature* indices from SSPOR (length = n_sensors_eff) and tile size.
    """
    W, nx_t, ny_t = u_blk.shape
    X = np.concatenate([u_blk, v_blk], axis=2).reshape(W, nx_t * 2 * ny_t)
    X = X - X.mean(axis=0, keepdims=True)
    
    # automatically compute rank
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    energy_tau = 0.99
    r_auto = _auto_rank_energy(s, tau=energy_tau)
    n_basis_modes = r_auto
    n_sensors = r_auto
    
    basis = ps.basis.SVD(n_basis_modes=n_basis_modes, algorithm='arpack', random_state=90)
    # POD–QR: default optimizer is QR when you pass n_sensors into SSPOR directly
    n_sensors_eff = int(min(n_sensors, n_basis_modes))
    model = ps.SSPOR(n_sensors=n_sensors_eff, basis=basis)
    model.fit(X)
    return np.asarray(model.selected_sensors, dtype=int), nx_t, ny_t


def _tile_sensor_coords_global(sensor_idx, nx_t, ny_t, i0, j0):
    """
    Map selected *feature* indices (u|v concatenation) to physical grid (i,j),
    then shift into global coords by adding tile origin.
    """
    grid_N = nx_t * ny_t
    grid_idx = np.mod(sensor_idx, grid_N)               # collapse [u|v] duplication
    ii, jj = np.unravel_index(grid_idx, (nx_t, ny_t))   # tile-local (i,j)
    return np.column_stack([ii + i0, jj + j0])          # global (i,j)


def _sliding_intervals(T, window_len, step=1):
    out = []
    s = 0
    while s + window_len <= T:
        out.append((s, s + window_len))
        s += step
    return out


import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# assumes you already have:
# _tile_bounds_from_center
# _auto_rank_energy
# _local_pod_qr_sensors
# _tile_sensor_coords_global
# _sliding_intervals
# and pysindy imported as ps somewhere


def regional_local_optimal_direction_series(
    u, v, lx, ly, dt,
    phys_window,          # (wx, wy) physical units
    time_window,          # W frames
    out_nx, out_ny,       # output grid resolution (centers)
    time_step=1,          # slide by this many frames
    t_start=0,
    t_end=None,
    scale_mode="mean_radius",  # "mean_radius" or "fixed"
    fixed_scale=None,          # if scale_mode="fixed", arrow length in physical units
    plot_every=1,
    show=True,
    save_plots=False,
    plot_dir="window_plots",
    parallel=False,
    n_jobs=None,
):
    """
    For each sliding time window [s,e), tile the domain with overlapping physical windows.
    In each tile, run POD–QR on (u,v) restricted to that tile/time window, select local sensors,
    and put an arrow at the tile center pointing toward the *average direction* from center
    to those sensors. No cross-window matching; each window is independent.

    Parallelization:
      - If parallel=True, time windows are processed in parallel with joblib (threading backend).
      - Plotting is done *after* all windows are computed, and works in both serial + parallel modes.

    Plotting:
      - If show=True, figures are displayed with plt.show().
      - If save_plots=True, each plotted window is saved to `plot_dir/window_XXXX.png`.
    """
    T, nx, ny = u.shape
    if t_end is None:
        t_end = T
    t_start = int(max(0, t_start))
    t_end   = int(min(T, t_end))
    if t_end - t_start < time_window:
        raise ValueError("time_window does not fit into [t_start, t_end).")

    # grid spacing (physical per index)
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    # convert physical window size → index window size (use odd sizes)
    wx_phys, wy_phys = phys_window
    win_nx = max(3, int(round(wx_phys / dx)))
    win_ny = max(3, int(round(wy_phys / dy)))
    if win_nx % 2 == 0: win_nx += 1
    if win_ny % 2 == 0: win_ny += 1

    # centers placed so tiles stay in-bounds
    i_min = win_nx // 2
    j_min = win_ny // 2
    i_max = nx - 1 - i_min
    j_max = ny - 1 - j_min
    centers_i = np.linspace(i_min, i_max, out_nx).astype(int)
    centers_j = np.linspace(j_min, j_max, out_ny).astype(int)

    # fixed center coordinates (physical), same for all time windows
    centers_xy = np.array(
        [[ci * dx, cj * dy] for ci in centers_i for cj in centers_j],
        dtype=float
    )  # (M,2)
    M = centers_xy.shape[0]

    # build time windows over requested range
    raw_intervals = _sliding_intervals(t_end - t_start, time_window, time_step)
    intervals = [(s + t_start, e + t_start) for (s, e) in raw_intervals]
    K = len(intervals)

    move_series = np.zeros((K, M, 2), dtype=float)

    # ---------- worker for ONE window (no plotting here) ----------
    def _compute_window_vectors(w_idx, s, e):
        vecs_this = []
        for ci in centers_i:
            for cj in centers_j:
                i0, i1, j0, j1 = _tile_bounds_from_center(ci, cj, win_nx, win_ny, nx, ny)

                # local time-space block
                u_blk = u[s:e, i0:i1, j0:j1]
                v_blk = v[s:e, i0:i1, j0:j1]

                # local POD–QR sensors
                idx, nx_t, ny_t = _local_pod_qr_sensors(u_blk, v_blk)
                coords = _tile_sensor_coords_global(idx, nx_t, ny_t, i0, j0)  # (m,2) global (i,j)

                # center (physical)
                xc, yc = ci * dx, cj * dy

                if coords.size == 0:
                    vec = np.array([0.0, 0.0], dtype=float)
                else:
                    xs = coords[:, 0] * dx
                    ys = coords[:, 1] * dy
                    d  = np.column_stack([xs - xc, ys - yc])  # (m,2) vectors center→sensor
                    r  = np.linalg.norm(d, axis=1) + 1e-12
                    dirs = d / r[:, None]                     # unit directions
                    mean_dir = dirs.mean(axis=0)
                    norm = np.linalg.norm(mean_dir) + 1e-12
                    mean_dir = mean_dir / norm

                    if scale_mode == "fixed" and (fixed_scale is not None):
                        mag = float(fixed_scale)
                    else:
                        # default: scale by mean radius, clipped to half-window
                        r_mean = float(np.mean(r))
                        r_cap  = 0.5 * min(wx_phys, wy_phys)
                        mag    = min(r_mean, r_cap)

                    vec = mean_dir * mag

                vecs_this.append(vec)

        return w_idx, np.asarray(vecs_this, float)

    # ------------------ compute all windows (parallel or serial) ------------------
    if parallel:
        # infer n_jobs from env if not explicitly provided
        if n_jobs is None:
            n_jobs = int(
                os.environ.get(
                    "PYTHON_THREADS",
                    os.environ.get("OMP_NUM_THREADS", "1"),
                )
            )

        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_compute_window_vectors)(w_idx, s, e)
            for w_idx, (s, e) in enumerate(intervals)
        )
        for w_idx, vecs_this in results:
            move_series[w_idx, :, :] = vecs_this

    else:
        for w_idx, (s, e) in enumerate(intervals):
            _, vecs_this = _compute_window_vectors(w_idx, s, e)
            move_series[w_idx, :, :] = vecs_this

    # ------------------ plotting pass (works for both modes) ------------------
    if show or save_plots:
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)

        for w_idx, (s, e) in enumerate(intervals):
            if w_idx % max(1, plot_every) != 0:
                continue

            mv = move_series[w_idx, :, :]  # (M,2)

            fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
            ax.set_title(f"Regional local optimal dir — window {w_idx}  (t ∈ [{s},{e}))")
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(0, lx)
            ax.set_ylim(0, ly)

            ax.quiver(
                centers_xy[:, 0], centers_xy[:, 1],
                mv[:, 0], mv[:, 1],
                color='crimson', angles='xy', scale_units='xy', scale=None, width=0.003,
                label='sensor direction'
            )
            ax.legend(loc='upper right')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            if save_plots:
                fname = f"window_{w_idx:04d}.png"
                fig.savefig(os.path.join(plot_dir, fname), dpi=150)

            if show:
                plt.show()
            else:
                plt.close(fig)

    # reshape to grid for convenience
    move_grid = move_series.reshape(K, out_nx, out_ny, 2)

    return dict(
        centers_xy=centers_xy,      # (M,2) physical coords of tile centers
        move_series=move_series,    # (K,M,2) vectors per time window (phys units)
        move_grid=move_grid,        # (K,out_nx,out_ny,2)
        intervals=intervals,        # list of (s,e)
        meta=dict(
            phys_window=phys_window,
            win_nx=win_nx, win_ny=win_ny,
            centers_nx=out_nx, centers_ny=out_ny,
            time_window=time_window, time_step=time_step,
            scale_mode=scale_mode, fixed_scale=fixed_scale
        )
    )

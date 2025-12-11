import os
import numpy as np
import matplotlib.pyplot as plt

from numbacs.flows import get_interp_arrays_2D, get_flow_2D
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_grid_2D


def quiver_row(
    res, dt, N=6, outdir="figures", basename="quiver_row",
    dpi=150, show=False
):
    ms = np.asarray(res["move_series"])            # (K, M, 2)
    xy = np.asarray(res["centers_xy"])             # (M, 2)
    K = ms.shape[0]
    idxs = np.unique(np.linspace(0, K-1, N).round().astype(int))
    cols = len(idxs)

    # one global scale so arrows are comparable across panels
    mags = np.hypot(ms[idxs, :, 0].ravel(), ms[idxs, :, 1].ravel())
    x, y = xy[:, 0], xy[:, 1]
    diag = np.hypot(x.max()-x.min(), y.max()-y.min())
    scale = np.quantile(mags, 0.90) / max(0.12*diag, 1e-12)

    fig, axes = plt.subplots(1, cols, figsize=(4.5*cols, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    xpad = 0.05*(x.max()-x.min()); ypad = 0.05*(y.max()-y.min())
    xlim = (x.min()-xpad, x.max()+xpad); ylim = (y.min()-ypad, y.max()+ypad)

    for ax, k in zip(axes, idxs):
        v = ms[k] / dt
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.quiver(x, y, v[:, 0], v[:, 1], angles='xy', scale_units='xy', scale=None, width=0.003)
        if "intervals" in res:
            s, e = res["intervals"][k]; ax.set_title(f"w{k}  t∈[{s},{e})")
        else:
            ax.set_title(f"w{k}")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # --- save instead of show ---
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fname = f"{basename}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

def velocity_from_optimal_direction(result_dict, time_window, dt, tau=None):
    """
    Convert the 'move_series' from regional_local_optimal_direction_series into a
    velocity field suitable for FTLE.

    result_dict : output of regional_local_optimal_direction_series(...)
                  must contain keys: move_series (K,M,2), meta with centers_nx/ny
    time_window : W (frames) used in the regional function
    dt          : time per frame
    tau         : relaxation time (seconds); default = time_window * dt

    Returns:
      V_series : (K, M, 2) velocity snapshots
      out_nx, out_ny : grid shape
      lx_ly        : convenience tuple if you want to re-use extents
    """
    move_series = result_dict["move_series"]  # (K, M, 2), "distance-to-sensors" vectors per window
    meta        = result_dict["meta"]
    out_nx, out_ny = meta["centers_nx"], meta["centers_ny"]

    # pick a timescale τ (how quickly the field would move along those headings)
    if tau is None:
        tau = float(time_window) * float(dt)   # natural: relax over one window duration

    # velocity = distance / τ
    V_series = move_series / dt # / max(tau, 1e-12)   # (K, M, 2)

    return V_series, out_nx, out_ny


def _ftle_from_velocity_series(V_series_slice, out_nx, out_ny, lx, ly, dt_snap, direction="forward"):
    """
    Compute FTLE over the *entire* V_series_slice.

    V_series_slice : (nt, M, 2) or (nt, out_nx, out_ny, 2)
                     velocity snapshots covering exactly the time window you want
    out_nx, out_ny : grid dims
    lx, ly         : physical extents
    dt_snap        : time between snapshots (e.g., time_step*dt)
    direction      : "forward" (repelling) or "backward" (attracting via time reversal)

    Returns:
      ftle : (out_nx, out_ny)
      x, y: 1D grids
    """
    V = V_series_slice.reshape(-1, out_nx, out_ny, 2).astype(np.float64)
    nt = V.shape[0]
    if nt < 2:
        raise ValueError("Need at least 2 snapshots in the slice to compute FTLE.")

    # Backward FTLE via time reversal → forward integration on reversed/negated flow
    if direction.lower().startswith("back"):
        V = -V[::-1].copy()

    u = V[..., 0]
    v = V[..., 1]

    # time/space axes
    t = np.arange(nt, dtype=np.float64) * float(dt_snap)
    x = np.linspace(0.0, lx, out_nx, dtype=np.float64)
    y = np.linspace(0.0, ly, out_ny, dtype=np.float64)
    dx, dy = x[1]-x[0], y[1]-y[0]

    # build interpolants and integrate forward over the slice span
    grid_vel, C_eval_u, C_eval_v = get_interp_arrays_2D(t, x, y, u, v)
    funcptr = get_flow_2D(grid_vel, C_eval_u, C_eval_v)

    T = t[-1] - t[0]                       # full slice duration
    params = np.array([1.0], dtype=np.float64)
    flowmap = flowmap_grid_2D(funcptr, t[0], T, x, y, params)
    ftle = ftle_grid_2D(flowmap, T, dx, dy)

    return ftle, x, y


def compute_ftle_from_optimal_direction(result_dict, u, v, lx, ly,
                                        time_window, dt, time_step,
                                        tau=None, t0_idx=0, ):
    """
    Full pipeline:
      - convert 'move_series' -> velocity snapshots using tau (default W*dt),
      - compute forward & backward FTLE.

    dt_snap = time between snapshots = time_step * dt
    """
    # 1) distance-to-sensor vectors -> velocity
    V_series, out_nx, out_ny = velocity_from_optimal_direction(
        result_dict, time_window=time_window, dt=dt, tau=tau
    )

    # 2) FTLE on that velocity field
    dt_snap = float(time_step) * float(dt)    # spacing between snapshots
    ftle_fwd, x, y = _ftle_from_velocity_series(
        V_series, out_nx, out_ny, lx, ly, dt_snap, "forward"
    )
    ftle_bwd, _, _ = _ftle_from_velocity_series(
        V_series, out_nx, out_ny, lx, ly, dt_snap,  "backward"
    )
    return ftle_fwd, ftle_bwd, x, y, V_series.reshape(-1, out_nx, out_ny, 2)

def plot_ftle(
    ftle_fwd, ftle_bwd, lx, ly, pad_frac=0.05, ridge_pct=90,
    outdir="figures", basename="ftle", dpi=150, show=False
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    extent = (0, lx, 0, ly)
    px, py = pad_frac*lx, pad_frac*ly

    im0 = axes[0].imshow(ftle_fwd.T, origin='lower', extent=extent, aspect='equal', cmap='magma')
    axes[0].set_title("Forward FTLE (repelling)")
    axes[0].set_xlim(-px, lx+px); axes[0].set_ylim(-py, ly+py)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='FTLE')

    im1 = axes[1].imshow(ftle_bwd.T, origin='lower', extent=extent, aspect='equal', cmap='magma')
    axes[1].set_title("Backward FTLE (attracting)")
    axes[1].set_xlim(-px, lx+px); axes[1].set_ylim(-py, ly+py)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='FTLE')

    # simple ridge overlay
    th_f = np.percentile(ftle_fwd, ridge_pct)
    th_b = np.percentile(ftle_bwd, ridge_pct)
    X, Y = np.meshgrid(
        np.linspace(0, lx, ftle_fwd.shape[0]),
        np.linspace(0, ly, ftle_fwd.shape[1]),
        indexing='ij'
    )
    axes[0].contour(X, Y, ftle_fwd, levels=[th_f], colors='cyan', linewidths=1.2)
    axes[1].contour(X, Y, ftle_bwd, levels=[th_b], colors='cyan', linewidths=1.2)

    # --- save instead of show ---
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fname = f"{basename}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

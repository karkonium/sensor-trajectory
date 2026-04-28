import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from numbacs.flows import get_interp_arrays_2D, get_flow_2D
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_grid_2D
from plot_style import (
    FTLE_CMAP,
    INFO_VECTOR_COLOR,
    RIDGE_COLOR,
    WIDE_PANEL_FIGSIZE,
    add_frame_badge,
    presentation_plot_context,
    set_panel_title,
    style_colorbar,
    style_spatial_axis,
)


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

    xpad = 0.05*(x.max()-x.min()); ypad = 0.05*(y.max()-y.min())
    xlim = (x.min()-xpad, x.max()+xpad); ylim = (y.min()-ypad, y.max()+ypad)
    tau = res["meta"]["time_window"] * dt

    with presentation_plot_context():
        fig, axes = plt.subplots(1, cols, figsize=(4.6 * cols, 4.8), constrained_layout=True)
        axes = np.atleast_1d(axes)

        for ax, k in zip(axes, idxs):
            v = ms[k] / tau
            ax.quiver(
                x,
                y,
                v[:, 0],
                v[:, 1],
                angles="xy",
                scale_units="xy",
                scale=None,
                width=0.0034,
                color=INFO_VECTOR_COLOR,
                alpha=0.95,
            )
            style_spatial_axis(ax, xlim=xlim, ylim=ylim)
            set_panel_title(ax, "Info flow snapshot")

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            fname = f"{basename}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)


def velocity_from_optimal_direction(result_dict, time_window, dt):
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
    """
    move_series = result_dict["move_series"]  # (K, M, 2), "distance-to-sensors" vectors per window
    meta        = result_dict["meta"]
    out_nx, out_ny = meta["centers_nx"], meta["centers_ny"]

    # pick a timescale τ (how quickly the field would move along those headings)
    
    # velocity = distance / τ
    tau = dt # ideally dt * time_step, but time_step == 1 so we good

    V_series = move_series / tau # / max(tau, 1e-12)   # (K, M, 2)

    return V_series, out_nx, out_ny


def _run_ftle_windows(k_starts, worker, parallel=False, n_jobs=1):
    """One place where joblib vs serial execution is decided."""
    if parallel and len(k_starts) > 1:
        return Parallel(
            n_jobs=n_jobs,
            backend="threading",
            verbose=0,
        )(
            delayed(worker)(idx, k0)
            for idx, k0 in enumerate(k_starts)
        )
    else:
        return [
            worker(idx, k0)
            for idx, k0 in enumerate(k_starts)
        ]


def _compute_ftle_series_core(
    V_series,
    out_nx, out_ny,
    xlim, ylim,
    dt_snap,
    ftle_len=None,
    stride=1,
    parallel=False,
    n_jobs=None,
    center_time_fn=None,
):
    """
    Shared engine for FTLE time series. Both compute_ftle_series_from_optimal_direction
    and compute_ftle_series_from_velocity_series call this.
    """
    V_series = np.asarray(V_series, dtype=np.float64)
    K = V_series.shape[0]    # number of velocity snapshots

    if ftle_len is None:
        ftle_len = K  # use full series (old behaviour)

    if ftle_len < 2:
        raise ValueError("ftle_len must be at least 2 snapshots.")

    # FTLE windows (start indices)
    k_starts = list(range(0, K - ftle_len + 1, stride))
    N_ftle = len(k_starts)

    ftle_fwd_series = np.zeros((N_ftle, out_nx, out_ny), dtype=float)
    ftle_bwd_series = np.zeros((N_ftle, out_nx, out_ny), dtype=float)
    t_centers       = np.zeros(N_ftle, dtype=float)

    if center_time_fn is None:
        # default if caller doesn't care about physical frame mapping
        def center_time_fn(k0):
            return (k0 + 0.5 * ftle_len) * float(dt_snap)

    # worker for one FTLE window (by index in k_starts)
    def _ftle_window_worker(idx, k0):
        k1 = k0 + ftle_len
        print(f"[LCS] FTLE window {idx+1}/{N_ftle}: snapshots [{k0}, {k1})", flush=True)

        V_slice = V_series[k0:k1]   # (ftle_len, M, 2) or (ftle_len, out_nx, out_ny, 2)
        ftle_fwd_k, x, y = _ftle_from_velocity_series(
            V_slice, out_nx, out_ny, xlim, ylim, dt_snap, direction="forward"
        )
        ftle_bwd_k, _, _ = _ftle_from_velocity_series(
            V_slice, out_nx, out_ny, xlim, ylim, dt_snap, direction="backward"
        )
        t_center = float(center_time_fn(k0))
        return idx, ftle_fwd_k, ftle_bwd_k, t_center, x, y

    # infer n_jobs if needed
    if n_jobs is None:
        n_jobs = int(
            os.environ.get(
                "PYTHON_THREADS",
                os.environ.get("OMP_NUM_THREADS", "1"),
            )
        )

    # run all FTLE windows (shared parallelization path)
    results = _run_ftle_windows(
        k_starts,
        _ftle_window_worker,
        parallel=parallel,
        n_jobs=n_jobs,
    )

    # gather results in the right order
    x = y = None
    for idx, ftle_fwd_k, ftle_bwd_k, t_center, x_k, y_k in results:
        ftle_fwd_series[idx, :, :] = ftle_fwd_k
        ftle_bwd_series[idx, :, :] = ftle_bwd_k
        t_centers[idx] = t_center
        x, y = x_k, y_k

    return ftle_fwd_series, ftle_bwd_series, x, y, t_centers


def _ftle_from_velocity_series(V_series_slice, out_nx, out_ny, xlim, ylim, dt_snap, direction="forward"):
    """
    Compute FTLE over the *entire* V_series_slice.

    V_series_slice : (nt, M, 2) or (nt, out_nx, out_ny, 2)
                     velocity snapshots covering exactly the time window you want
    out_nx, out_ny : grid dims
    xlim, ylim     : physical extents (x0, x1), (y0, y1)
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
    (x0, x1), (y0, y1) = xlim, ylim
    x = np.linspace(x0, x1, out_nx, dtype=np.float64)
    y = np.linspace(y0, y1, out_ny, dtype=np.float64)
    dx, dy = x[1]-x[0], y[1]-y[0]

    # build interpolants and integrate forward over the slice span
    grid_vel, C_eval_u, C_eval_v = get_interp_arrays_2D(t, x, y, u, v)
    funcptr = get_flow_2D(grid_vel, C_eval_u, C_eval_v)

    T = t[-1] - t[0]                       # full slice duration
    params = np.array([1.0], dtype=np.float64)
    flowmap = flowmap_grid_2D(funcptr, t[0], T, x, y, params)
    ftle = ftle_grid_2D(flowmap, T, dx, dy)

    return ftle, x, y


def compute_ftle_series_from_velocity_series(
    V_series,        # (K, out_nx, out_ny, 2)
    lx, ly,
    dt_snap,
    ftle_len=None,
    stride=1,
    parallel=False,
    n_jobs=None,
    xlim=None,
    ylim=None,
):
    """
    Compute a TIME SERIES of FTLE fields from ANY velocity series on a uniform grid.

    V_series : (K, out_nx, out_ny, 2)
    lx, ly   : physical domain size (used for default extents if xlim/ylim not provided)
    dt_snap  : time between snapshots in V_series

    Returns:
      ftle_fwd_series : (N_ftle, out_nx, out_ny)
      ftle_bwd_series : (N_ftle, out_nx, out_ny)
      x, y            : 1D coordinates
      t_centers       : physical time associated with each FTLE field
    """
    V_series = np.asarray(V_series, dtype=np.float64)
    if V_series.ndim != 4 or V_series.shape[-1] != 2:
        raise ValueError("V_series must have shape (K, out_nx, out_ny, 2).")

    K, out_nx, out_ny, _ = V_series.shape

    if xlim is None:
        xlim = (0.0, float(lx))
    if ylim is None:
        ylim = (0.0, float(ly))

    return _compute_ftle_series_core(
        V_series=V_series,
        out_nx=out_nx,
        out_ny=out_ny,
        xlim=xlim,
        ylim=ylim,
        dt_snap=float(dt_snap),
        ftle_len=ftle_len,
        stride=stride,
        parallel=parallel,
        n_jobs=n_jobs,
        center_time_fn=None,
    )


def compute_ftle_series_from_optimal_direction(
    result_dict,
    lx, ly,
    time_window,   # W used in regional PIV
    dt,            # time per original frame
    time_step,     # step between windows in regional PIV
    ftle_len=None, # number of V-series snapshots per FTLE; default = all
    stride=1,      # slide this many snapshots between FTLE fields
    parallel=False,
    n_jobs=None,
):
    """
    Compute a TIME SERIES of FTLE fields from the regional PIV result.

    result_dict : output of regional_local_optimal_direction_series(...)
    lx, ly      : physical domain size
    time_window : W (frames) used in the regional function
    dt          : time per frame (original data)
    time_step   : how many frames between regional windows
    ftle_len    : number of V-series snapshots in each FTLE integration window.
                  If None, use all snapshots (i.e., one FTLE like before).
    stride      : how many snapshots to advance between FTLE windows.
                  e.g. stride=1 => max temporal resolution; stride=2 => every other.
    parallel    : if True, parallelize over FTLE windows with joblib.
    n_jobs      : number of parallel workers (default from env: PYTHON_THREADS or OMP_NUM_THREADS).

    Returns:
      ftle_fwd_series : (N_ftle, out_nx, out_ny)
      ftle_bwd_series : (N_ftle, out_nx, out_ny)
      x, y            : 1D coordinates
      t_centers       : physical time (in same units as dt) associated with each FTLE field
    """
    # 1) distance-to-sensor vectors -> velocity snapshots
    V_series, out_nx, out_ny = velocity_from_optimal_direction(
        result_dict, time_window=time_window, dt=dt
    )
    K = V_series.shape[0]    # number of regional windows / velocity snapshots

    if ftle_len is None:
        ftle_len = K  # use full series (old behaviour)

    if ftle_len < 2:
        raise ValueError("ftle_len must be at least 2 snapshots.")

    # dt between V-series snapshots (regional windows)
    dt_snap = float(time_step) * float(dt)

    intervals = result_dict.get("intervals", None)
    centers_xy = result_dict.get("centers_xy", None)
    x0 = float(centers_xy[:, 0].min())
    x1 = float(centers_xy[:, 0].max())
    y0 = float(centers_xy[:, 1].min())
    y1 = float(centers_xy[:, 1].max())

    # helper to infer center time for one FTLE window
    def _center_time(k0):
        if intervals is not None:
            k_center = k0 + ftle_len // 2
            k_center = min(k_center, len(intervals) - 1)
            s_frame, e_frame = intervals[k_center]
            return 0.5 * (s_frame + e_frame) * dt
        else:
            return (k0 + 0.5 * ftle_len) * dt_snap

    return _compute_ftle_series_core(
        V_series=V_series,
        out_nx=out_nx,
        out_ny=out_ny,
        xlim=(x0, x1),
        ylim=(y0, y1),
        dt_snap=dt_snap,
        ftle_len=ftle_len,
        stride=stride,
        parallel=parallel,
        n_jobs=n_jobs,
        center_time_fn=_center_time,
    )


def save_ftle_series_plots(
    ftle_fwd_series,
    ftle_bwd_series,
    x, y,
    t_centers,
    lx, ly,
    outdir="ftle_series",
    basename="ftle",
    pad_frac=0.05,
    ridge_pct=90,
    dpi=150,
    show=False,
):
    """
    Save a time series of FTLE fields as PNG files.

    ftle_fwd_series : (N, out_nx, out_ny)
    ftle_bwd_series : (N, out_nx, out_ny)
    x, y            : 1D coordinates (from _ftle_from_velocity_series)
    t_centers       : array of physical times associated with each FTLE field
    lx, ly          : domain size (for plotting extents)
    """
    os.makedirs(outdir, exist_ok=True)

    N, out_nx, out_ny = ftle_fwd_series.shape
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    px, py = pad_frac * lx, pad_frac * ly
    ftle_values = np.concatenate([ftle_fwd_series.ravel(), ftle_bwd_series.ravel()])
    ftle_lo = float(np.percentile(ftle_values, 5))
    ftle_hi = float(np.percentile(ftle_values, 98))
    if (not np.isfinite(ftle_lo)) or (not np.isfinite(ftle_hi)) or (ftle_hi <= ftle_lo):
        ftle_lo = float(np.nanmin(ftle_values))
        ftle_hi = float(np.nanmax(ftle_values) + 1e-12)

    with presentation_plot_context():
        for idx in range(N):
            ftle_fwd = ftle_fwd_series[idx]
            ftle_bwd = ftle_bwd_series[idx]
            fig, axes = plt.subplots(1, 2, figsize=WIDE_PANEL_FIGSIZE, constrained_layout=True)

            im0 = axes[0].imshow(
                ftle_fwd.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=FTLE_CMAP,
                vmin=ftle_lo,
                vmax=ftle_hi,
            )
            style_spatial_axis(axes[0], xlim=(-px, lx + px), ylim=(-py, ly + py))
            set_panel_title(axes[0], "Forward FTLE")
            cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            style_colorbar(cbar0, "FTLE")

            im1 = axes[1].imshow(
                ftle_bwd.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=FTLE_CMAP,
                vmin=ftle_lo,
                vmax=ftle_hi,
            )
            style_spatial_axis(axes[1], xlim=(-px, lx + px), ylim=(-py, ly + py))
            set_panel_title(axes[1], "Backward FTLE")
            cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            style_colorbar(cbar1, "FTLE")

            th_f = np.percentile(ftle_fwd, ridge_pct)
            th_b = np.percentile(ftle_bwd, ridge_pct)
            Xg, Yg = np.meshgrid(x, y, indexing="ij")
            axes[0].contour(Xg, Yg, ftle_fwd, levels=[th_f], colors=RIDGE_COLOR, linewidths=1.4)
            axes[1].contour(Xg, Yg, ftle_bwd, levels=[th_b], colors=RIDGE_COLOR, linewidths=1.4)
            add_frame_badge(axes[0], "White contour: LCS ridge")

            fname = f"{basename}_{idx:04d}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=dpi)

            if show:
                plt.show()
            else:
                plt.close(fig)

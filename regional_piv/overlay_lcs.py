import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Rectangle

from numbacs.flows import get_interp_arrays_2D, get_flow_2D
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_grid_2D

from plotting import make_gif_from_dir, load_pickles, _k_starts_from
from data_generation import *
from plot_style import (
    INFO_LINE_COLOR,
    SCALAR_OVERLAY_CMAP,
    SINGLE_PANEL_FIGSIZE,
    add_frame_badge,
    presentation_plot_context,
    set_panel_title,
    style_spatial_axis,
)


def _norm01(A, lo, hi, gamma=0.85):
    """
    Normalize a scalar field A to [0, 1] for visualization.

    Parameters
    ----------
    A : array-like
    lo, hi : float
        Lower/upper reference values.
    gamma : float
        Gamma correction applied after normalization.
        - gamma < 1 makes mid-range values brighter 
        - gamma > 1 makes mid-range values darker
        - gamma = 1 disables gamma correction
    """
    B = (A - lo) / max(hi - lo, 1e-12)
    B = np.clip(B, 0.0, 1.0)
    B = B ** gamma
    B[~np.isfinite(B)] = 0.0
    return B


def _finite_percentile_limits(field, low=5, high=95):
    """Return robust percentile limits using only finite values."""
    values = np.asarray(field, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def _temporal_mean_field(field_slice):
    """Compute a time mean while tolerating NaNs in scalar fields."""
    data = np.asarray(field_slice, dtype=np.float64)
    if not np.isnan(data).any():
        return data.mean(axis=0)

    valid = np.isfinite(data)
    counts = valid.sum(axis=0)
    summed = np.where(valid, data, 0.0).sum(axis=0)
    out = np.full(data.shape[1:], np.nan, dtype=np.float64)
    np.divide(summed, counts, out=out, where=counts > 0)
    return out


def save_ftle_overlap_series_plots(
    ftle_info_series, background_series,
    x_info, y_info,          # info-grid coordinates
    LX, LY,                  # full fluid domain size
    outdir, basename,
    dpi=180,
    gamma=0.85,
    alpha_max=0.90,
    background_label="fluid FTLE",
    background_cmap="Blues",
    title="Info FTLE overlap",
):
    """
    Draw overlap WITHOUT resampling:
      - Background field on the full domain extent (0..LX, 0..LY)
      - Info FTLE on its own extent (min/max of x_info/y_info) using Reds
      - Alpha is field-dependent so low values are transparent and hotspots pop.
    """
    os.makedirs(outdir, exist_ok=True)

    ftle_info_series  = np.asarray(ftle_info_series)
    background_series = np.asarray(background_series)

    if ftle_info_series.shape[0] != background_series.shape[0]:
        raise ValueError("Time length mismatch: info FTLE series vs background series")

    # global normalization (stable brightness across frames)
    info_lo, info_hi = _finite_percentile_limits(ftle_info_series, 5, 95)
    background_lo, background_hi = _finite_percentile_limits(background_series, 5, 95)

    x_info = np.asarray(x_info)
    y_info = np.asarray(y_info)

    info_extent  = (float(x_info.min()), float(x_info.max()), float(y_info.min()), float(y_info.max()))
    fluid_extent = (0.0, float(LX), 0.0, float(LY))

    N = ftle_info_series.shape[0]
    with presentation_plot_context():
        for i in range(N):
            A = ftle_info_series[i]   # (nx_info, ny_info)
            B = background_series[i]  # (NX, NY)

            A01 = _norm01(A, info_lo, info_hi, gamma=gamma)
            B01 = _norm01(B, background_lo, background_hi, gamma=gamma)

            fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_FIGSIZE, constrained_layout=True)

            ax.imshow(
                B01.T,
                origin="lower",
                extent=fluid_extent,
                aspect="equal",
                cmap=background_cmap,
                alpha=np.clip(B01.T * alpha_max, 0.0, alpha_max),
                interpolation="nearest",
            )

            ax.imshow(
                A01.T,
                origin="lower",
                extent=info_extent,
                aspect="equal",
                cmap="Reds",
                alpha=np.clip(A01.T * alpha_max, 0.0, alpha_max),
                interpolation="nearest",
            )

            ax.add_patch(
                Rectangle(
                    (info_extent[0], info_extent[2]),
                    info_extent[1] - info_extent[0],
                    info_extent[3] - info_extent[2],
                    fill=False,
                    linewidth=1.4,
                    linestyle=(0, (4, 2)),
                    edgecolor=INFO_LINE_COLOR,
                    alpha=0.9,
                )
            )

            style_spatial_axis(ax, xlim=(0.0, float(LX)), ylim=(0.0, float(LY)))
            set_panel_title(ax, title)
            add_frame_badge(ax, f"Background: {background_label}\nRed: info FTLE")
            add_frame_badge(ax, "Dashed box: info subdomain", loc="upper right")

            fig.savefig(os.path.join(outdir, f"{basename}_{i:04d}.png"), dpi=dpi)
            plt.close(fig)


def ftle_from_uv_series_on_grid(u_t, v_t, x, y, dt, direction="forward"):
    u_t = np.asarray(u_t, dtype=np.float64)
    v_t = np.asarray(v_t, dtype=np.float64)
    nt = u_t.shape[0]
    if nt < 2:
        raise ValueError("Need >=2 snapshots for FTLE")

    if direction.lower().startswith("back"):
        u_t = -u_t[::-1].copy()
        v_t = -v_t[::-1].copy()

    t = np.arange(nt, dtype=np.float64) * float(dt)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    grid_vel, C_u, C_v = get_interp_arrays_2D(t, x, y, u_t, v_t)
    funcptr = get_flow_2D(grid_vel, C_u, C_v)

    T = float(t[-1] - t[0])
    params = np.array([1.0], dtype=np.float64)
    flowmap = flowmap_grid_2D(funcptr, t[0], T, x, y, params)
    return ftle_grid_2D(flowmap, T, dx, dy)


def compute_fluid_ftle_series_matched(u, v, LX, LY, reg_piv, ftle, which="backward"):
    """
    For each info-FTLE frame:
      - regional windows [k0:k1) -> fluid frames [f1:f2) using reg_piv["intervals"]
      - compute fluid FTLE on the FULL fluid grid (NX,NY) over u[f1:f2], v[f1:f2]
    """
    dt = float(ftle["dt"])  # must exist
    intervals = reg_piv["intervals"]
    k_starts  = _k_starts_from(reg_piv, ftle)
    ftle_len  = int(ftle["ftle_len"])

    NX, NY = u.shape[1], u.shape[2]
    x_full = np.linspace(0.0, LX, NX, dtype=np.float64)
    y_full = np.linspace(0.0, LY, NY, dtype=np.float64)

    out = []
    spans = []
    for k0 in k_starts:
        k1 = k0 + ftle_len
        f1 = int(intervals[k0][0])
        f2 = int(intervals[k1 - 1][1])

        u_slice = u[f1:f2]     # (nt,NX,NY)
        v_slice = v[f1:f2]

        out.append(ftle_from_uv_series_on_grid(u_slice, v_slice, x_full, y_full, dt, direction=which))
        spans.append((f1, f2))

    return np.stack(out, axis=0), np.asarray(spans, dtype=int), x_full, y_full


def compute_scalar_series_matched(u, reg_piv, ftle):
    """
    For each info-FTLE frame, compute the mean scalar field over the matched
    fluid-frame interval used by that FTLE window.
    """
    intervals = reg_piv["intervals"]
    k_starts = _k_starts_from(reg_piv, ftle)
    ftle_len = int(ftle["ftle_len"])

    out = []
    spans = []
    for k0 in k_starts:
        k1 = k0 + ftle_len
        f1 = int(intervals[k0][0])
        f2 = int(intervals[k1 - 1][1])

        out.append(_temporal_mean_field(u[f1:f2]))
        spans.append((f1, f2))

    return np.stack(out, axis=0), np.asarray(spans, dtype=int)


def make_overlap_gif(reg_piv, ftle, results_dir, name, u, v, LX, LY, which="backward", duration=0.10):
    # info FTLE already computed
    ftle_info = np.asarray(ftle["ftle_backward"] if which.startswith("back") else ftle["ftle_forward"])

    k_starts = _k_starts_from(reg_piv, ftle)
    assert len(k_starts) == len(ftle_info), "Mismatch: k_starts vs info FTLE frames"

    # check intervals are in bounds for the u,v you passed in
    fmax = max(e for (s, e) in reg_piv["intervals"])
    assert fmax <= u.shape[0], f"Intervals require {fmax} frames but u has {u.shape[0]}"

    if v is None:
        background_series, spans = compute_scalar_series_matched(u, reg_piv, ftle)
        background_label = "mean scalar field"
        background_cmap = SCALAR_OVERLAY_CMAP
        title = "Info FTLE + matched scalar field"
    else:
        # ensure dt exists and matches what u,v represent
        assert "dt" in ftle, "ftle pickle missing dt; fluid FTLE timing may be wrong"
        background_series, spans, _, _ = compute_fluid_ftle_series_matched(
            u, v, LX, LY, reg_piv, ftle, which=which
        )
        background_label = "fluid FTLE"
        background_cmap = "Blues"
        title = "Info FTLE + fluid FTLE"
   
    # render overlap frames using plotting.py helper
    out_frames = os.path.join(results_dir, f"frames_overlap_{name}_{which}")
    save_ftle_overlap_series_plots(
        ftle_info, background_series,
        ftle["x"], ftle["y"],
        LX=LX, LY=LY,
        outdir=out_frames,
        basename=f"{name}_overlap",
        dpi=180,
        gamma=0.85,
        background_label=background_label,
        background_cmap=background_cmap,
        title=title,
    )


    out_gif = os.path.join(results_dir, f"{name}_LCS_overlap_{which}.gif")
    make_gif_from_dir(out_frames, out_gif, duration=duration)
    
    print(out_gif)
    return out_gif


if __name__ == "__main__":
    """ 
    ASSUMES pickle files for LCS and regional PIV. Also assumes that the flow feild we are generating
    is the same one used for the respective LCS and regional PIV
    """

    TOTAL_STEPS = 150      # number of frames used per flow case
    PERIOD = 100           # used by some synthetic flows   
    """
    name = "kolmogorov"
    NX, NY = 300, 300
    LX, LY = 2 * np.pi, 2 * np.pi
    DT_base = 1e-3

    u_full, v_full = generate_cfd_kolmogorov_flow(
        n_timesteps=2000,
        nx=NX, ny=NY,
        lx=LX, ly=LY,
        dt=DT_base,
        nu=2e-2,
        forcing_amp=20.0,
        kf=4,
        plot_series=False,
    )

    sample_indices = np.linspace(1500, 1900, TOTAL_STEPS, dtype=int)
    u = u_full[sample_indices]
    v = v_full[sample_indices]

    k_skip = sample_indices[1] - sample_indices[0]
    DT = k_skip * DT_base  # effective dt between stored snapshots
    """

    
    name = "moving_vortex"
    NX, NY = 300, 300
    LX, LY = 1.0, 1.0
    DT = 1.0  # or whatever makes sense for this synthetic data

    u, v = generate_moving_vortex(
        TOTAL_STEPS,
        NX, NY,
        LX, LY,
        period=PERIOD,

    )
    
    """
    name = "double_gyre"

    NX, NY = 300, 150
    LX, LY = 2.0, 1.0
    DT = 1.0

    u, v = generate_double_gyre_flow(
        TOTAL_STEPS,
        NX, NY,
        LX, LY,
        A=0.1,
        epsilon=0.5,
        period=PERIOD,
    )
    """

    
    results_dir = "ftle_series_" + name
    reg_piv, ftle = load_pickles(results_dir, name)

    make_overlap_gif(
        reg_piv=reg_piv, ftle=ftle,
        results_dir=results_dir,
        name=name,
        u=u, v=v,
        LX=LX, LY=LY,
        which="backward",   # or "forward"
        duration=0.10,
    )

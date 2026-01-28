import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from numbacs.flows import get_interp_arrays_2D, get_flow_2D
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_grid_2D

from plotting import make_gif_from_dir, load_pickles, _k_starts_from
from data_generation import *


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
    return B ** gamma


def save_ftle_overlap_series_plots(
    ftle_info_series, ftle_fluid_series,
    x_info, y_info,          # info-grid coordinates
    LX, LY,                  # full fluid domain size
    outdir, basename,
    dpi=180,
    gamma=0.85,
    alpha_max=0.90,
):
    """
    Draw overlap WITHOUT resampling:
      - Fluid FTLE: imshow on full domain extent (0..LX, 0..LY) using Blues
      - Info FTLE : imshow on its own extent (min/max of x_info/y_info) using Reds
      - Alpha is field-dependent so low values are transparent and hotspots pop.
    """
    os.makedirs(outdir, exist_ok=True)

    ftle_info_series  = np.asarray(ftle_info_series)
    ftle_fluid_series = np.asarray(ftle_fluid_series)

    if ftle_info_series.shape[0] != ftle_fluid_series.shape[0]:
        raise ValueError("Time length mismatch: info vs fluid FTLE series")

    # global normalization (stable brightness across frames)
    info_lo, info_hi   = np.percentile(ftle_info_series,  [5, 95])
    fluid_lo, fluid_hi = np.percentile(ftle_fluid_series, [5, 95])

    x_info = np.asarray(x_info)
    y_info = np.asarray(y_info)

    info_extent  = (float(x_info.min()), float(x_info.max()), float(y_info.min()), float(y_info.max()))
    fluid_extent = (0.0, float(LX), 0.0, float(LY))

    N = ftle_info_series.shape[0]
    for i in range(N):
        A = ftle_info_series[i]   # (nx_info, ny_info)
        B = ftle_fluid_series[i]  # (NX, NY)

        A01 = _norm01(A, info_lo, info_hi, gamma=gamma)
        B01 = _norm01(B, fluid_lo, fluid_hi, gamma=gamma)

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6), constrained_layout=True)
        ax.set_facecolor("black")

        # Fluid layer (full domain)
        ax.imshow(
            B01.T,
            origin="lower",
            extent=fluid_extent,
            aspect="equal",
            cmap="Blues",
            alpha=np.clip(B01.T * alpha_max, 0.0, alpha_max),   # per-pixel alpha
            interpolation="nearest",
        )

        # Info layer (its subdomain)
        ax.imshow(
            A01.T,
            origin="lower",
            extent=info_extent,
            aspect="equal",
            cmap="Reds",
            alpha=np.clip(A01.T * alpha_max, 0.0, alpha_max),   # per-pixel alpha
            interpolation="nearest",
        )

        ax.set_xlim(0, LX)
        ax.set_ylim(0, LY)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Overlap: Red=info FTLE (subdomain), Blue=fluid FTLE (full); brighter = stronger")

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


def make_overlap_gif(reg_piv, ftle, results_dir, name, u, v, LX, LY, which="backward", duration=0.10):
    # info FTLE already computed
    ftle_info = np.asarray(ftle["ftle_backward"] if which.startswith("back") else ftle["ftle_forward"])

    k_starts = _k_starts_from(reg_piv, ftle)
    assert len(k_starts) == len(ftle_info), "Mismatch: k_starts vs info FTLE frames"

    # check intervals are in bounds for the u,v you passed in
    fmax = max(e for (s, e) in reg_piv["intervals"])
    assert fmax <= u.shape[0], f"Intervals require {fmax} frames but u has {u.shape[0]}"

    # ensure dt exists and matches what u,v represent
    assert "dt" in ftle, "ftle pickle missing dt; fluid FTLE timing may be wrong"
   
    ftle_fluid, spans, _, _ = compute_fluid_ftle_series_matched(u, v, LX, LY, reg_piv, ftle, which=which)
   
    # render overlap frames using plotting.py helper
    out_frames = os.path.join(results_dir, f"frames_overlap_{name}_{which}")
    save_ftle_overlap_series_plots(
        ftle_info, ftle_fluid,
        ftle["x"], ftle["y"],
        LX=LX, LY=LY,
        outdir=out_frames,
        basename=f"{name}_overlap",
        dpi=180,
        gamma=0.85,
    )


    out_gif = os.path.join(results_dir, f"{name}_LCS_overlap_{which}.gif")
    make_gif_from_dir(out_frames, out_gif, duration=duration)
    
    print(out_gif)


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

import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from data_generation import *
from data_tranformation import *
from plotting import *
from state_concatenation import *

from regional_piv import regional_local_optimal_direction_series
from lcs import (
    compute_ftle_series_from_optimal_direction,
    save_ftle_series_plots,
)

# global settings 
WINDOW_LEN = 10        # regional PIV time_window (frames)
TOTAL_STEPS = 150      # number of frames used per flow case
PERIOD = 100           # used by some synthetic flows

FTLE_LEN = 10          # FTLE integration length in regional snapshots
STRIDE   = 1           # slide FTLE window by this many snapshots

# thread count for joblib (set via Slurm env)
N_JOBS = int(os.environ.get("PYTHON_THREADS", "1"))


# helpers
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_gif_from_dir(in_dir, out_gif, duration=0.1):
    """
    Build an animated GIF from all images in `in_dir`, saved as `out_gif`.
    """
    files = sorted(
        f for f in os.listdir(in_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not files:
        print(f"[GIF] No image files found in {in_dir}, skipping GIF.")
        return

    frames = []
    for fname in files:
        img_path = os.path.join(in_dir, fname)
        frames.append(imageio.imread(img_path))

    imageio.mimsave(out_gif, frames, duration=duration)
    print(f"[GIF] Saved GIF: {out_gif}")


def run_flow_case(
    name,
    u, v,
    LX, LY,
    DT,
    out_nx, out_ny,
    window_len=WINDOW_LEN,
    ftle_len=FTLE_LEN,
    stride=STRIDE,
    n_jobs=N_JOBS,
):
    """
    Generic pipeline for:
      - computing regional PIV sensor directions
      - computing an FTLE time series
      - saving FTLE PNGs and a GIF

    name    : short string, e.g. "kolmogorov", "moving_vortex", "double_gyre"
    u, v    : arrays (T, nx, ny)
    LX, LY  : domain size
    DT      : time between stored frames (physical units)
    out_nx, out_ny : coarse grid resolution for regional centers
    """
    print(f"\n===== Running flow case: {name} =====", flush=True)
    T, NX, NY = u.shape
    print(f"[{name}] u.shape = {u.shape}, LX={LX}, LY={LY}, DT={DT}", flush=True)
    print(f"[{name}] out_nx={out_nx}, out_ny={out_ny}, WINDOW_LEN={window_len}", flush=True)

    # --- Regional PIV / sensor-direction series ---
    res = regional_local_optimal_direction_series(
        u, v, LX, LY, DT,
        phys_window=(LX * 0.2, LY * 0.2),
        time_window=window_len,
        out_nx=out_nx, out_ny=out_ny,
        time_step=1,
        scale_mode="mean_radius",
        fixed_scale=None,
        plot_every=1,
        show=False,
        save_plots=False,
        parallel=True,
        n_jobs=n_jobs,
    )

        # --- Save FTLE plots ---
    outdir = f"ftle_series_{name}"
    basename = f"{name}_ftle"
    print(f"[{name}] Saving regional piv pickle to {outdir}/", flush=True)
    
    save_pickle(res, os.path.join(outdir, f"regional_piv_{name}.pickle"))

    # --- FTLE time series from that coarse velocity field ---
    ftle_fwd_series, ftle_bwd_series, x, y, t_centers = \
        compute_ftle_series_from_optimal_direction(
            res,
            lx=LX, ly=LY,
            time_window=window_len,
            dt=DT,
            time_step=1,
            ftle_len=ftle_len,
            stride=stride,
            parallel=True,
            n_jobs=n_jobs,
        )

    intervals = res.get("intervals", None)  # list of (s_frame, e_frame) for each V snapshot
    K = len(intervals) if intervals is not None else None

    # mirror the exact k_starts logic used inside compute_ftle_series_from_optimal_direction
    ftle_len_eff = FTLE_LEN  # or use the ftle_len variable in scope
    stride_eff = STRIDE      # or stride variable in scope

    if intervals is not None:
        k_starts = list(range(0, K - ftle_len_eff + 1, stride_eff))
        ftle_frame_spans = []
        for k0 in k_starts:
            k1 = k0 + ftle_len_eff
            start_frame = int(intervals[k0][0])
            end_frame   = int(intervals[k1 - 1][1])
            ftle_frame_spans.append((start_frame, end_frame))
        ftle_frame_spans = np.asarray(ftle_frame_spans, dtype=int)
    else:
        ftle_frame_spans = None

    ftle_payload = {
        "name": name,
        "x": np.asarray(x),
        "y": np.asarray(y),
        "t_centers": np.asarray(t_centers),
        "ftle_forward": np.asarray(ftle_fwd_series),
        "ftle_backward": np.asarray(ftle_bwd_series),
        "ftle_len": int(ftle_len_eff),
        "stride": int(stride_eff),
        "dt": float(DT),
        "dt_snap": float(DT),  # since time_step=1 in your calls; else DT*time_step
        "ftle_frame_spans": ftle_frame_spans,  # shape (N_ftle, 2) or None
    }


    save_pickle(ftle_payload, os.path.join(outdir, f"ftle_{name}.pickle"))



    save_ftle_series_plots(
        ftle_fwd_series,
        ftle_bwd_series,
        x, y,
        t_centers,
        lx=LX, ly=LY,
        outdir=outdir,
        basename=basename,
        pad_frac=0.05,
        ridge_pct=92,
        dpi=150,
        show=False,
    )

    # --- Make GIF from those plots ---
    gif_name = f"{name}_ftle.gif"
    gif_path = os.path.join(outdir, gif_name)
    make_gif_from_dir(outdir, gif_path, duration=0.1)

    print(f"===== Done flow case: {name} =====\n", flush=True)



if __name__ == "__main__":

    # 1) Moving vortex 
    NX_mv, NY_mv = 300, 300
    LX_mv, LY_mv = 1.0, 1.0
    DT_mv = 1.0  # or whatever makes sense for this synthetic data

    u_mv, v_mv = generate_moving_vortex(
        TOTAL_STEPS,
        NX_mv, NY_mv,
        LX_mv, LY_mv,
        period=PERIOD,
    )

    run_flow_case(
        name="moving_vortex",
        u=u_mv,
        v=v_mv,
        LX=LX_mv,
        LY=LY_mv,
        DT=DT_mv,
        out_nx=NX_mv // 3,
        out_ny=NY_mv // 3,
    )

    # 2) Kolmogorov flow 
    NX_k, NY_k = 300, 300
    LX_k, LY_k = 2 * np.pi, 2 * np.pi
    DT_base = 1e-3

    u_full, v_full = generate_cfd_kolmogorov_flow(
        n_timesteps=2000,
        nx=NX_k, ny=NY_k,
        lx=LX_k, ly=LY_k,
        dt=DT_base,
        nu=2e-2,
        forcing_amp=20.0,
        kf=4,
        plot_series=False,
    )

    sample_indices = np.linspace(1500, 1900, TOTAL_STEPS, dtype=int)
    u_k = u_full[sample_indices]
    v_k = v_full[sample_indices]

    k_skip = sample_indices[1] - sample_indices[0]
    DT_k = k_skip * DT_base  # effective dt between stored snapshots

    run_flow_case(
        name="kolmogorov",
        u=u_k,
        v=v_k,
        LX=LX_k,
        LY=LY_k,
        DT=DT_k,
        out_nx=NX_k // 3,
        out_ny=NY_k // 3,
    )


    # 3) Double gyre 
    NX_dg, NY_dg = 300, 150
    LX_dg, LY_dg = 2.0, 1.0
    DT_dg = 1.0

    u_dg, v_dg = generate_double_gyre_flow(
        TOTAL_STEPS,
        NX_dg, NY_dg,
        LX_dg, LY_dg,
        A=0.1,
        epsilon=0.5,
        period=PERIOD,
    )

    run_flow_case(
        name="double_gyre",
        u=u_dg,
        v=v_dg,
        LX=LX_dg,
        LY=LY_dg,
        DT=DT_dg,
        out_nx=NX_dg // 3,
        out_ny=NY_dg // 3,
    )

    print("All three flow cases completed.")
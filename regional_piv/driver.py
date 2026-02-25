import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import xarray as xr

from data_generation import *
from data_tranformation import *
from plotting import *
from state_concatenation import *

from regional_piv import regional_local_optimal_direction_series
from lcs import (
    compute_ftle_series_from_optimal_direction,
    save_ftle_series_plots,
)
from plotting import make_gif_from_dir, overlay_lcs_with_flows
from overlay_lcs import make_overlap_gif


# global settings 
WINDOW_LEN = 10        # regional PIV time_window (frames)
TOTAL_STEPS = 150      # number of frames used per flow case
PERIOD = 100           # used by some synthetic flows

FTLE_LEN = 10          # FTLE integration length in regional snapshots
STRIDE   = 1           # slide FTLE window by this many snapshots

# thread count for joblib (set via Slurm env)
N_JOBS = int(os.environ.get("PYTHON_THREADS", "1"))


# helpers
def load_oisst_wkmean_scalar_masked(
    sst_path,
    lsmask_path,
    n_steps=TOTAL_STEPS,
    t_start=0,
    coarsen_lat=0,
    coarsen_lon=0,
    lat_slice=None,
    lon_slice=None,
    mask_ocean_is_true=True,  # if mask is already "ocean=True"
    mask_threshold=0.5,       # used if coarsen mask
):
    """
    Returns:
      s: scalar SST array shaped (T, NX, NY) where X=lon, Y=lat (your convention)
         with land points = NaN
      LX, LY: domain extents in coordinate units (degrees)
      DT: timestep (1.0 week)
    """

    ds = xr.open_dataset(sst_path, engine="netcdf4")
    sst = ds["sst"].isel(time=slice(t_start, t_start + n_steps))

    # some OISST files include a singleton depth-like dim
    for dim in ["zlev", "depth", "lev"]:
        if dim in sst.dims:
            sst = sst.squeeze(dim, drop=True)

    # load land/sea mask
    ms = xr.open_dataset(lsmask_path, engine="netcdf4")
    m = ms["mask"]
    for dim in ["time", "zlev", "depth", "lev"]:
        if dim in m.dims:
            m = m.squeeze(dim, drop=True)

    # subset first (keeps alignment)
    if lat_slice is not None:
        sst = sst.sel(lat=slice(lat_slice[0], lat_slice[1]))
        m   = m.sel(lat=slice(lat_slice[0], lat_slice[1]))
    if lon_slice is not None:
        sst = sst.sel(lon=slice(lon_slice[0], lon_slice[1]))
        m   = m.sel(lon=slice(lon_slice[0], lon_slice[1]))

    # convert mask to boolean ocean mask
    ocean_mask = m.astype(bool)
    if not mask_ocean_is_true:
        ocean_mask = ~ocean_mask

    # optional coarsen (do BOTH field and mask consistently)
    if (coarsen_lat and coarsen_lat > 1) or (coarsen_lon and coarsen_lon > 1):
        sst = sst.coarsen(lat=coarsen_lat, lon=coarsen_lon, boundary="trim").mean()

        # coarsen mask: keep cell as ocean if majority ocean
        ocean_mask = (
            ocean_mask.coarsen(lat=coarsen_lat, lon=coarsen_lon, boundary="trim").mean()
            > mask_threshold
        )

    # apply mask: land -> NaN
    sst = sst.where(ocean_mask)

    # coords/extents
    lon = sst["lon"].values
    lat = sst["lat"].values
    LX = float(lon.max() - lon.min())
    LY = float(lat.max() - lat.min())
    DT = 1.0  # weekly mean

    # to numpy: (T, lat, lon) -> (T, lon, lat)
    s = sst.values.astype(np.float32)
    s = np.transpose(s, (0, 2, 1))

    return s, LX, LY, DT


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    u       : array (T, nx, ny)  (vector u-component OR scalar field if v is None)
    v       : array (T, nx, ny) or None (None => scalar mode)
    LX, LY  : domain size
    DT      : time between stored frames (physical units)
    out_nx, out_ny : coarse grid resolution for regional centers
    """
    print(f"\n===== Running flow case: {name} =====", flush=True)
    T, NX, NY = u.shape
    mode = "scalar" if v is None else "vector"
    print(f"[{name}] mode={mode}, u.shape={u.shape}, LX={LX}, LY={LY}, DT={DT}", flush=True)
    print(f"[{name}] out_nx={out_nx}, out_ny={out_ny}, WINDOW_LEN={window_len}", flush=True)

    # Regional PIV / sensor-direction series 
    reg_piv = regional_local_optimal_direction_series(
        u, v, LX, LY, DT,
        phys_window=(LX * 0.05, LY * 0.05),
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

    # Save FTLE plots 
    outdir = f"ftle_series_{name}"
    basename = f"{name}_ftle"
    print(f"[{name}] Saving regional piv pickle to {outdir}/", flush=True)
    
    save_pickle(reg_piv, os.path.join(outdir, f"regional_piv_{name}.pickle"))

    # FTLE time series from that coarse velocity field 
    ftle_fwd_series, ftle_bwd_series, x, y, t_centers = \
        compute_ftle_series_from_optimal_direction(
            reg_piv,
            lx=LX, ly=LY,
            time_window=window_len,
            dt=DT,
            time_step=1,
            ftle_len=ftle_len,
            stride=stride,
            parallel=True,
            n_jobs=n_jobs,
        )

    intervals = reg_piv.get("intervals", None)  # list of (s_frame, e_frame) for each V snapshot
    K = len(intervals) if intervals is not None else None

    # mirror the exact k_starts logic used inside compute_ftle_series_from_optimal_direction
    ftle_len_eff = int(ftle_len)
    stride_eff = int(stride)

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
        "dt_snap": float(DT),  # since time_step=1 in our calls; else DT*time_step
        "ftle_frame_spans": ftle_frame_spans,  # shape (N_ftle, 2) or None # currently unused in plotting.py
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

    # Make GIF from those plots 
    gif_name = f"{name}_ftle.gif"
    gif_path = os.path.join(outdir, gif_name)
    make_gif_from_dir(outdir, gif_path, duration=0.1)

    
    overlay_lcs_with_flows(
        reg_piv=reg_piv, 
        ftle=ftle_payload,
        results_dir=outdir,
        name=name,
        u=u, v=v,
        LX=LX, LY=LY,
        which="backward",   # "forward" or "backward"
        ridge_pct=92,
        qskip=2,           # increase if arrows too dense
        duration=0.10,
    )

    gif_path = make_overlap_gif(
        reg_piv=reg_piv, ftle=ftle_payload,
        results_dir=outdir,
        name=name,
        u=u, v=v,
        LX=LX, LY=LY,
        which="backward",   # or "forward"
        duration=0.10,
    )
    
    print(f"===== Done flow case: {name} =====\n", flush=True)



if __name__ == "__main__":

    # 1) Moving vortex 
    NX_mv, NY_mv = 900, 900
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
    NX_k, NY_k = 900, 900
    LX_k, LY_k = 2 * np.pi, 2 * np.pi
    DT_base = 1e-4

    u_full, v_full = generate_cfd_kolmogorov_flow(
        n_timesteps=20000,
        nx=NX_k, ny=NY_k,
        lx=LX_k, ly=LY_k,
        dt=DT_base,
        nu=2e-2,
        forcing_amp=20.0,
        kf=10,
        plot_series=False,
    )

    start = 1500
    end   = 1900

    # constant integer skip (closest to what linspace "meant")
    skip = max(1, int(round((end - start) / (TOTAL_STEPS - 1))))

    sample_indices = start + np.arange(TOTAL_STEPS) * skip

    # guard if you accidentally run past end (or past u_full length)
    sample_indices = sample_indices[sample_indices <= end]
    u_k = u_full[sample_indices]
    v_k = v_full[sample_indices]

    DT_base = 1e-3 # because we output something already subsampled
    DT_k = skip * DT_base  # effective dt between stored snapshots

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
    NX_dg, NY_dg = 900, 450
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

    # 4) OISST SST scalar field
    SST_PATH  = "sst.wkmean.1990-present.nc"
    MASK_PATH = "lsmask.nc"

    s_sst, LX_sst, LY_sst, DT_sst = load_oisst_wkmean_scalar_masked(
        SST_PATH, MASK_PATH,
        n_steps=TOTAL_STEPS,
        t_start=0,
        coarsen_lat=0,
        coarsen_lon=0,
        lat_slice=None,
        lon_slice=None,
        mask_ocean_is_true=True,  
    )

    T_sst, NX_sst, NY_sst = s_sst.shape

    run_flow_case(
        name="oisst_sst",
        u=s_sst,         # scalar stored in u slot
        v=None,          # scalar mode
        LX=LX_sst,
        LY=LY_sst,
        DT=DT_sst,
        out_nx=NX_sst // 3,
        out_ny=NY_sst // 3,
    )

    print("All flow cases completed.")
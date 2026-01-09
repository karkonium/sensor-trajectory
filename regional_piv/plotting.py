import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from data_generation import *

def make_gif_from_dir(in_dir, out_gif, duration=0.1):
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".png"))
    frames = [imageio.imread(os.path.join(in_dir, f)) for f in files]
    imageio.mimsave(out_gif, frames, duration=duration)
    print(f"[GIF] Saved: {out_gif}")

def load_pickles(results_dir, name):
    with open(os.path.join(results_dir, f"regional_piv_{name}.pickle"), "rb") as f:
        res = pickle.load(f)
    with open(os.path.join(results_dir, f"ftle_{name}.pickle"), "rb") as f:
        ftle = pickle.load(f)
    return res, ftle

def _k_starts_from(res, ftle):
    K = np.asarray(res["move_series"]).shape[0]
    ftle_len = int(ftle["ftle_len"])
    stride   = int(ftle["stride"])
    return list(range(0, K - ftle_len + 1, stride))

def mean_info_flow_for_idx(res, ftle, idx):
    """Mean regional-PIV/info velocity over the FTLE integration window idx."""
    move_series = np.asarray(res["move_series"])          # (K, M, 2)
    out_nx = int(res["meta"]["centers_nx"])
    out_ny = int(res["meta"]["centers_ny"])
    W = int(res["meta"]["time_window"])
    dt = float(ftle.get("dt", 1.0))
    tau = W * dt                                          # MUST match your FTLE code

    k_starts = _k_starts_from(res, ftle)
    k0 = k_starts[idx]
    k1 = k0 + int(ftle["ftle_len"])

    Vmean_M = move_series[k0:k1].mean(axis=0) / tau       # (M,2)
    Vmean = Vmean_M.reshape(out_nx, out_ny, 2)            # (out_nx,out_ny,2)
    return Vmean, (k0, k1)

def mean_fluid_flow_for_idx(u, v, LX, LY, res, ftle, idx):
    """Mean fluid velocity over ALL fluid frames used by FTLE window idx, sampled to FTLE grid."""
    intervals = res.get("intervals", None)
    if intervals is None:
        raise ValueError("res has no 'intervals'—can’t map FTLE windows to fluid frames.")

    k_starts = _k_starts_from(res, ftle)
    k0 = k_starts[idx]
    k1 = k0 + int(ftle["ftle_len"])

    s_frame = int(intervals[k0][0])
    e_frame = int(intervals[k1 - 1][1])   # end of last included regional window

    u_mean = u[s_frame:e_frame].mean(axis=0)
    v_mean = v[s_frame:e_frame].mean(axis=0)

    NX, NY = u_mean.shape
    dx = LX / (NX - 1)
    dy = LY / (NY - 1)

    x = np.asarray(ftle["x"])
    y = np.asarray(ftle["y"])

    ii = np.clip(np.rint(x / dx).astype(int), 0, NX - 1)
    jj = np.clip(np.rint(y / dy).astype(int), 0, NY - 1)

    Uc = u_mean[np.ix_(ii, jj)]
    Vc = v_mean[np.ix_(ii, jj)]
    Vmean = np.stack([Uc, Vc], axis=-1)
    return Vmean, (s_frame, e_frame)

def _normalize_for_display(V, x, y, frac=0.06):
    """Scale arrows to a fixed visible length (keeps quiver readable across frames)."""
    x = np.asarray(x); y = np.asarray(y)
    arrow_len = frac * min((x[-1] - x[0]), (y[-1] - y[0]))
    mag = np.linalg.norm(V, axis=-1, keepdims=True)
    mag = np.maximum(mag, 1e-12)
    return (V / mag) * arrow_len

def render_overlay_frames(
    ftle_field_series, x, y, LX, LY,
    V_provider,  # callable(idx)->(Vmean,(...))
    outdir, title_prefix,
    ridge_pct=92,
    qskip=1,
    cmap="gray",
    alpha=0.80,
    dpi=180,
):
    os.makedirs(outdir, exist_ok=True)
    x = np.asarray(x); y = np.asarray(y)
    Xg, Yg = np.meshgrid(x, y, indexing="ij")

    N = ftle_field_series.shape[0]
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

    for idx in range(N):
        ftle_field = ftle_field_series[idx]

        # gentle contrast so quiver pops
        vmin = np.percentile(ftle_field, 5)
        vmax = np.percentile(ftle_field, 95)

        Vmean, meta = V_provider(idx)
        Vplot = _normalize_for_display(Vmean, x, y)
        Vp = Vplot[::qskip, ::qskip, :]
        Xp = Xg[::qskip, ::qskip]
        Yp = Yg[::qskip, ::qskip]

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6), constrained_layout=True)

        ax.imshow(
            ftle_field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )

        # ridge overlay (optional but recommended)
        th = np.percentile(ftle_field, ridge_pct)
        ax.contour(Xg, Yg, ftle_field, levels=[th], colors="white", linewidths=1.2)

        # quiver overlay (make it bright)
        ax.quiver(
            Xp, Yp,
            Vp[..., 0], Vp[..., 1],
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.0028,
            color="tab:orange",
        )

        ax.set_xlim(0, LX)
        ax.set_ylim(0, LY)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_prefix}  idx={idx:04d}  meta={meta}")

        fig.savefig(os.path.join(outdir, f"frame_{idx:04d}.png"), dpi=dpi)
        plt.close(fig)

def make_two_overlay_gifs(results_dir, name, u, v, LX, LY,
                          which="forward", ridge_pct=92, qskip=2, duration=0.10):
    res, ftle = load_pickles(results_dir, name)

    x = ftle["x"]
    y = ftle["y"]

    if which.lower().startswith("back"):
        ftle_series = np.asarray(ftle["ftle_backward"])
        lcs_label = "Backward FTLE"
    else:
        ftle_series = np.asarray(ftle["ftle_forward"])
        lcs_label = "Forward FTLE"

    # --- 1) FTLE + mean info flow (regional PIV) ---
    outdir_info = os.path.join(results_dir, f"gif_frames_{name}_info")
    def info_provider(idx):
        Vmean, (k0, k1) = mean_info_flow_for_idx(res, ftle, idx)
        return Vmean, f"k[{k0},{k1})"
    render_overlay_frames(
        ftle_series, x, y, LX, LY,
        V_provider=info_provider,
        outdir=outdir_info,
        title_prefix=f"{name}: {lcs_label} + mean info-flow",
        ridge_pct=ridge_pct,
        qskip=qskip,
        cmap="gray",
        alpha=0.80,
    )
    gif_info = os.path.join(results_dir, f"{name}_ftle_plus_info.gif")
    make_gif_from_dir(outdir_info, gif_info, duration=duration)

    # --- 2) FTLE + mean fluid flow ---
    outdir_fluid = os.path.join(results_dir, f"gif_frames_{name}_fluid")
    def fluid_provider(idx):
        Vmean, (s, e) = mean_fluid_flow_for_idx(u, v, LX, LY, res, ftle, idx)
        return Vmean, f"frames[{s},{e})"
    render_overlay_frames(
        ftle_series, x, y, LX, LY,
        V_provider=fluid_provider,
        outdir=outdir_fluid,
        title_prefix=f"{name}: {lcs_label} + mean fluid-flow",
        ridge_pct=ridge_pct,
        qskip=qskip,
        cmap="gray",
        alpha=0.80,
    )
    gif_fluid = os.path.join(results_dir, f"{name}_ftle_plus_fluid.gif")
    make_gif_from_dir(outdir_fluid, gif_fluid, duration=duration)

    print("[DONE] Wrote:")
    print("  ", gif_info)
    print("  ", gif_fluid)


if __name__ == "__main__":
    name = "moving_vortex"
    results_dir = "ftle_series_moving_vortex"  # change to your folder


    # You must have the fluid u,v arrays available here:
    NX_mv, NY_mv = 300, 300
    lx, ly = 1.0, 1.0
    DT_mv = 1.0  # or whatever makes sense for this synthetic data

    WINDOW_LEN = 10        # regional PIV time_window (frames)
    TOTAL_STEPS = 150      # number of frames used per flow case
    PERIOD = 100           # used by some synthetic flows

    FTLE_LEN = 10          # FTLE integration length in regional snapshots
    STRIDE   = 1           # slide FTLE window by this many snapshots

    u, v = generate_moving_vortex(
        TOTAL_STEPS,
        NX_mv, NY_mv,
        lx, ly,
        period=PERIOD,
    )

    make_two_overlay_gifs(
        results_dir=results_dir,
        name=name,
        u=u, v=v,
        LX=lx, LY=ly,
        which="forward",   # or "backward"
        ridge_pct=92,
        qskip=2,           # increase if arrows too dense
        duration=0.10,
    )

import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import TwoSlopeNorm

from data_generation import *
from plot_style import (
    DIVERGENCE_CMAP,
    FLOW_VECTOR_COLOR,
    INFO_VECTOR_COLOR,
    RIDGE_COLOR,
    SCALAR_OVERLAY_CMAP,
    SINGLE_PANEL_FIGSIZE,
    add_frame_badge,
    presentation_plot_context,
    set_panel_title,
    style_colorbar,
    style_spatial_axis,
)

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
    # print(f"[GIF] Saved: {out_gif}")


def load_pickles(results_dir, name):
    with open(os.path.join(results_dir, f"regional_piv_{name}.pickle"), "rb") as f:
        reg_piv = pickle.load(f)
    with open(os.path.join(results_dir, f"ftle_{name}.pickle"), "rb") as f:
        ftle = pickle.load(f)
    return reg_piv, ftle


def _k_starts_from(reg_piv, ftle):
    """Starting indicies of our intervals."""
    K = np.asarray(reg_piv["move_series"]).shape[0]
    ftle_len = int(ftle["ftle_len"])
    stride   = int(ftle["stride"])
    return list(range(0, K - ftle_len + 1, stride))


def mean_info_flow_for_idx(reg_piv, ftle, idx):
    """Mean regional-PIV/info velocity over the FTLE integration window idx."""
    move_series = np.asarray(reg_piv["move_series"])          # (K, M, 2)
    out_nx = int(reg_piv["meta"]["centers_nx"])
    out_ny = int(reg_piv["meta"]["centers_ny"])
    W = int(reg_piv["meta"]["time_window"])
    dt = float(ftle.get("dt", 1.0))
    tau = W * dt                                          # MUST match our FTLE code

    # regional PIV interval used for LCS
    k_starts = _k_starts_from(reg_piv, ftle)
    k0 = k_starts[idx]
    k1 = k0 + int(ftle["ftle_len"])

    Vmean_M = move_series[k0:k1].mean(axis=0) / tau       # (M,2)
    Vmean = Vmean_M.reshape(out_nx, out_ny, 2)            # (out_nx,out_ny,2)
    return Vmean, (k0, k1)


def mean_fluid_flow_for_idx(u, v, LX, LY, reg_piv, ftle, idx):
    """
    Mean underlying field over all fluid frames used by FTLE window idx.

    Returns:
      - vector mode (v is not None): (NX, NY, 2) mean velocity
      - scalar mode (v is None):     (NX, NY) mean scalar field
    """
    intervals = reg_piv["intervals"]
    k_starts = _k_starts_from(reg_piv, ftle)
    k0 = k_starts[idx]
    k1 = k0 + int(ftle["ftle_len"])

    s_frame = int(intervals[k0][0])
    e_frame = int(intervals[k1 - 1][1])

    u_mean = np.asarray(u[s_frame:e_frame].mean(axis=0), dtype=np.float64)

    if v is None:
        return u_mean, (s_frame, e_frame)

    v_mean = np.asarray(v[s_frame:e_frame].mean(axis=0), dtype=np.float64)

    Vmean_full = np.stack([u_mean, v_mean], axis=-1)  # (NX,NY,2)
    return Vmean_full, (s_frame, e_frame)


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
    cmap="Greys",
    alpha=0.80,
    dpi=180,
):
    os.makedirs(outdir, exist_ok=True)
    x = np.asarray(x); y = np.asarray(y)
    Xg, Yg = np.meshgrid(x, y, indexing="ij")

    N = ftle_field_series.shape[0]
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

    with presentation_plot_context():
        for idx in range(N):
            ftle_field = ftle_field_series[idx]

            # Gentle contrast so the overlay remains visible without washing out the FTLE field.
            vmin = np.percentile(ftle_field, 5)
            vmax = np.percentile(ftle_field, 95)

            Vmean, meta = V_provider(idx)
            Vmean = np.asarray(Vmean)

            fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_FIGSIZE, constrained_layout=True)

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

            th = np.percentile(ftle_field, ridge_pct)
            ax.contour(Xg, Yg, ftle_field, levels=[th], colors=RIDGE_COLOR, linewidths=1.4, alpha=0.95)

            if Vmean.ndim == 3 and Vmean.shape[-1] == 2:
                if Vmean.shape[0] == len(x) and Vmean.shape[1] == len(y):
                    Xq, Yq = Xg, Yg
                    xq, yq = x, y
                else:
                    NX, NY = Vmean.shape[:2]
                    xq = np.linspace(0.0, LX, NX)
                    yq = np.linspace(0.0, LY, NY)
                    Xq, Yq = np.meshgrid(xq, yq, indexing="ij")

                Vplot = _normalize_for_display(Vmean, xq, yq)
                Vp = Vplot[::qskip, ::qskip, :]
                Xp = Xq[::qskip, ::qskip]
                Yp = Yq[::qskip, ::qskip]

                ax.quiver(
                    Xp, Yp,
                    Vp[..., 0], Vp[..., 1],
                    angles="xy",
                    scale_units="xy",
                    scale=None,
                    width=0.0032,
                    color=INFO_VECTOR_COLOR,
                    alpha=0.95,
                )
            elif Vmean.ndim == 2:
                if Vmean.shape[0] == len(x) and Vmean.shape[1] == len(y):
                    Xs, Ys = Xg, Yg
                    extent_scalar = extent
                else:
                    NX, NY = Vmean.shape
                    xs = np.linspace(0.0, LX, NX)
                    ys = np.linspace(0.0, LY, NY)
                    Xs, Ys = np.meshgrid(xs, ys, indexing="ij")
                    extent_scalar = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))

                s_lo = np.percentile(Vmean, 5)
                s_hi = np.percentile(Vmean, 95)
                if (not np.isfinite(s_lo)) or (not np.isfinite(s_hi)) or (s_hi <= s_lo):
                    s_lo = float(np.nanmin(Vmean))
                    s_hi = float(np.nanmax(Vmean) + 1e-12)

                ax.imshow(
                    Vmean.T,
                    origin="lower",
                    extent=extent_scalar,
                    aspect="equal",
                    cmap=SCALAR_OVERLAY_CMAP,
                    alpha=0.34,
                    vmin=s_lo,
                    vmax=s_hi,
                )

                if s_hi > s_lo:
                    levels = np.linspace(s_lo, s_hi, 7)
                    ax.contour(
                        Xs,
                        Ys,
                        Vmean,
                        levels=levels,
                        colors="#0F172A",
                        linewidths=0.9,
                        alpha=0.75,
                    )
            else:
                raise ValueError(
                    "V_provider must return either a vector field (NX,NY,2) or scalar field (NX,NY)."
                )

            style_spatial_axis(ax, xlim=(0.0, float(LX)), ylim=(0.0, float(LY)))
            set_panel_title(ax, title_prefix, f"Frame {idx + 1:03d} / {N:03d}")
            add_frame_badge(ax, str(meta))

            fig.savefig(os.path.join(outdir, f"frame_{idx:04d}.png"), dpi=dpi)
            plt.close(fig)


def overlay_lcs_with_flows(reg_piv, ftle, results_dir, name, u, v, LX, LY,
                          which="forward", ridge_pct=92, qskip=2, duration=0.10):

    x = ftle["x"]
    y = ftle["y"]

    if which.lower().startswith("back"):
        ftle_series = np.asarray(ftle["ftle_backward"])
        lcs_label = "Backward FTLE"
    else:
        ftle_series = np.asarray(ftle["ftle_forward"])
        lcs_label = "Forward FTLE"

    # 1) FTLE + mean info flow (regional PIV) 
    outdir_info = os.path.join(results_dir, f"gif_frames_{name}_info")
    def info_provider(idx):
        Vmean, (k0, k1) = mean_info_flow_for_idx(reg_piv, ftle, idx)
        return Vmean, f"k[{k0},{k1})"
    render_overlay_frames(
        ftle_series, x, y, LX, LY,
        V_provider=info_provider,
        outdir=outdir_info,
        title_prefix=f"{name}: {lcs_label} + mean info-flow",
        ridge_pct=ridge_pct,
        qskip=qskip,
        cmap="Greys",
        alpha=0.80,
    )
    gif_info = os.path.join(results_dir, f"{name}_ftle_plus_info.gif")
    make_gif_from_dir(outdir_info, gif_info, duration=duration)

    # 2) FTLE + mean fluid flow 
    outdir_fluid = os.path.join(results_dir, f"gif_frames_{name}_fluid")
    def fluid_provider(idx):
        Vmean, (s, e) = mean_fluid_flow_for_idx(u, v, LX, LY, reg_piv, ftle, idx)
        return Vmean, f"frames[{s},{e})"
    fluid_label = "mean fluid-flow" if v is not None else "mean scalar-field"
    render_overlay_frames(
        ftle_series, x, y, LX, LY,
        V_provider=fluid_provider,
        outdir=outdir_fluid,
        title_prefix=f"{name}: {lcs_label} + {fluid_label}",
        ridge_pct=ridge_pct,
        qskip=qskip * 3 if v is not None else qskip,
        cmap="Greys",
        alpha=0.80,
    )
    gif_fluid = os.path.join(results_dir, f"{name}_ftle_plus_fluid.gif")
    make_gif_from_dir(outdir_fluid, gif_fluid, duration=duration)

    print("[DONE] Wrote:")
    print("  ", gif_info)
    print("  ", gif_fluid)


def _divergence_2d(U, V, dx, dy, edge_order=2):
    """
    Numerical divergence div(U,V) = dU/dx + dV/dy
    Assumes: axis 0 corresponds to x, axis 1 corresponds to y
    """
    dU_dx = np.gradient(U, dx, axis=0, edge_order=edge_order)
    dV_dy = np.gradient(V, dy, axis=1, edge_order=edge_order)
    return dU_dx + dV_dy


def _info_axes_from_centers_xy(reg_piv):
    out_nx = int(reg_piv["meta"]["centers_nx"])
    out_ny = int(reg_piv["meta"]["centers_ny"])
    xy = np.asarray(reg_piv["centers_xy"], dtype=float)
    XY = xy.reshape(out_nx, out_ny, 2)
    Xc = XY[..., 0]
    Yc = XY[..., 1]
    x_info = Xc[:, 0].copy()
    y_info = Yc[0, :].copy()
    extent_info = (float(x_info[0]), float(x_info[-1]), float(y_info[0]), float(y_info[-1]))
    dx_info = float(np.mean(np.diff(x_info))) if len(x_info) > 1 else 1.0
    dy_info = float(np.mean(np.diff(y_info))) if len(y_info) > 1 else 1.0
    return x_info, y_info, extent_info, dx_info, dy_info


def divergence_info_feild(
    reg_piv, results_dir, name,
    u, v, LX, LY, dt,
    outdir=None,
    title_prefix=None,
    stride=1,
    qskip=10,
    cmap="Greys",
    # cmap_div="coolwarm",
    alpha_div=0.70,
    dpi=180,
    duration=0.10,
    div_pct=98,         # for symmetric limits on div
    global_div_limits=True,
):
    """
    GIF over REGIONAL PIV windows k:
      - Background: mean fluid speed over reg window [s:e) on full domain
      - Overlay: divergence of info velocity on info subdomain
      - Quiver: mean fluid velocity (decimated) in tab:orange (like our FTLE overlay style)

    Also prints a divergence-of-mean-flow diagnostic.
    """

    os.makedirs(results_dir, exist_ok=True)
    if outdir is None:
        outdir = os.path.join(results_dir, f"frames_div_info_{name}")
    os.makedirs(outdir, exist_ok=True)
    if title_prefix is None:
        title_prefix = f"{name}: div(info) over mean flow"

    # info velocity snapshots (K,out_nx,out_ny,2) in "velocity" units
    move_grid = np.asarray(reg_piv["move_grid"], dtype=float)
    intervals = reg_piv["intervals"]
    K = move_grid.shape[0]

    W  = int(reg_piv["meta"]["time_window"])
    tau = float(W * dt)
    V_info = move_grid / tau

    # info grid extent + spacing
    x_info, y_info, extent_info, dx_info, dy_info = _info_axes_from_centers_xy(reg_piv)

    # full fluid grid spacing
    NX, NY = u.shape[1], u.shape[2]
    dx_flow = float(LX / (NX - 1))
    dy_flow = float(LY / (NY - 1))

    # precompute global div scale to avoid flicker (like our consistent look)
    if global_div_limits:
        div_vals = []
        for k in range(0, K, stride):
            Vk = V_info[k]
            divk = _divergence_2d(Vk[..., 0], Vk[..., 1], dx_info, dy_info)
            div_vals.append(divk.ravel())
        div_vals = np.concatenate(div_vals)
        div_limit = float(np.percentile(np.abs(div_vals), div_pct))
        div_limit = max(div_limit, 1e-12)
    else:
        div_limit = None

    # diagnostics for mean-flow divergence
    flow_rms = []
    flow_max = []

    extent_full = (0.0, float(LX), 0.0, float(LY))

    frame_idx = 0
    with presentation_plot_context():
        for k in range(0, K, stride):
            s, e = intervals[k]
            s = int(s); e = int(e)

            u_mean = np.asarray(u[s:e].mean(axis=0), dtype=np.float64)
            v_mean = np.asarray(v[s:e].mean(axis=0), dtype=np.float64)
            speed  = np.sqrt(u_mean**2 + v_mean**2)

            div_flow = _divergence_2d(u_mean, v_mean, dx_flow, dy_flow)
            flow_rms.append(float(np.sqrt(np.mean(div_flow**2))))
            flow_max.append(float(np.max(np.abs(div_flow))))

            Vk = V_info[k]
            div_info = _divergence_2d(Vk[..., 0], Vk[..., 1], dx_info, dy_info)

            vmin_bg = np.percentile(speed, 5)
            vmax_bg = np.percentile(speed, 95)

            if div_limit is None:
                lim_k = float(np.percentile(np.abs(div_info), div_pct))
                lim_k = max(lim_k, 1e-12)
            else:
                lim_k = div_limit

            fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_FIGSIZE, constrained_layout=True)

            ax.imshow(
                speed.T,
                origin="lower",
                extent=extent_full,
                aspect="equal",
                cmap=cmap,
                alpha=0.34,
                vmin=vmin_bg,
                vmax=vmax_bg,
            )

            norm = TwoSlopeNorm(vmin=-lim_k, vcenter=0.0, vmax=lim_k)
            im = ax.imshow(
                div_info.T,
                origin="lower",
                extent=extent_info,
                aspect="equal",
                cmap=DIVERGENCE_CMAP,
                norm=norm,
                alpha=alpha_div,
            )

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            style_colorbar(cbar, r"$\nabla \cdot v_{\mathrm{info}}$")

            qs = max(1, int(qskip))
            xq = np.linspace(0.0, LX, NX)
            yq = np.linspace(0.0, LY, NY)
            Xq, Yq = np.meshgrid(xq, yq, indexing="ij")

            ax.quiver(
                Xq[::qs, ::qs], Yq[::qs, ::qs],
                u_mean[::qs, ::qs], v_mean[::qs, ::qs],
                angles="xy",
                scale_units="xy",
                scale=None,
                width=0.0032,
                color=FLOW_VECTOR_COLOR,
                alpha=0.90,
            )

            style_spatial_axis(ax, xlim=(0.0, float(LX)), ylim=(0.0, float(LY)))
            set_panel_title(ax, title_prefix, f"Frame {frame_idx + 1:03d}")
            add_frame_badge(ax, f"Window {k:03d}\nFrames [{s}, {e})")

            fig.savefig(os.path.join(outdir, f"frame_{frame_idx:04d}.png"), dpi=dpi)
            plt.close(fig)
            frame_idx += 1

    print(f"[div(mean flow)] mean RMS: {np.mean(flow_rms):.3e}")
    print(f"[div(mean flow)] max  RMS: {np.max(flow_rms):.3e}")
    print(f"[div(mean flow)] max |div|: {np.max(flow_max):.3e}")

    gif_path = os.path.join(results_dir, f"{name}_div_info_over_mean_flow.gif")
    make_gif_from_dir(outdir, gif_path, duration=duration)
    return gif_path


if __name__ == "__main__":
    """ 
    ASSUMES pickle files for LCS and regional PIV. Also assumes that the flow feild we are generating
    is the same one used for the respective LCS and regional PIV
    """
    TOTAL_STEPS = 150      # number of frames used per flow case
    PERIOD = 100           # used by some synthetic flows   
    
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
    # overlay_lcs_with_flows(
    #     reg_piv=reg_piv, 
    #     ftle=ftle,
    #     results_dir=results_dir,
    #     name=name,
    #     u=u, v=v,
    #     LX=LX, LY=LY,
    #     which="backward",   # "forward" or "backward"
    #     ridge_pct=92,
    #     qskip=2,           # increase if arrows too dense
    #     duration=0.10,
    # )

    divergence_info_feild(
        reg_piv, results_dir, name, u, v, LX, LY, DT
    )

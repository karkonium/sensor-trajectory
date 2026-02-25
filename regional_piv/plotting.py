import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import TwoSlopeNorm

from data_generation import *

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
    print(f"[GIF] Saved: {out_gif}")


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
        Vmean = np.asarray(Vmean)

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

        if Vmean.ndim == 3 and Vmean.shape[-1] == 2:
            # Vector field overlay.
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
                width=0.0028,
                color="tab:orange",
            )
        elif Vmean.ndim == 2:
            # Scalar field overlay: use translucent heatmap + isolines instead of quiver.
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
                cmap="viridis",
                alpha=0.35,
                vmin=s_lo,
                vmax=s_hi,
            )

            if s_hi > s_lo:
                levels = np.linspace(s_lo, s_hi, 7)
                ax.contour(Xs, Ys, Vmean, levels=levels, cmap="viridis", linewidths=0.8, alpha=0.85)
        else:
            raise ValueError(
                "V_provider must return either a vector field (NX,NY,2) or scalar field (NX,NY)."
            )

        ax.set_xlim(0, LX)
        ax.set_ylim(0, LY)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_prefix}  idx={idx:04d}  meta={meta}")

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
        cmap="gray",
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
        cmap="gray",
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
    cmap="gray",
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
        lim = float(np.percentile(np.abs(div_vals), div_pct))
        lim = max(lim, 1e-12)
    else:
        lim = None

    # diagnostics for mean-flow divergence
    flow_rms = []
    flow_max = []

    extent_full = (0.0, float(LX), 0.0, float(LY))

    frame_idx = 0
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

        # same "gentle contrast" style as our FTLE overlay:
        vmin_bg = np.percentile(speed, 5)
        vmax_bg = np.percentile(speed, 95)

        if lim is None:
            lim_k = float(np.percentile(np.abs(div_info), div_pct))
            lim_k = max(lim_k, 1e-12)
        else:
            lim_k = lim

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6), constrained_layout=True)

        lim = np.percentile(np.abs(div_info), 98)   # or a fixed lim across frames
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

        im = ax.imshow(
            div_info.T,
            origin="lower",
            extent=extent_info,
            aspect="equal",
            cmap="bwr",      # white at 0
            norm=norm,
            alpha=alpha_div,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("div(info velocity)")


        # quiver overlay (same look as ours)
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
            width=0.0028,
            color="black",
        )

        ax.set_xlim(0, LX)
        ax.set_ylim(0, LY)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_prefix}  k={k:04d}  frames[{s},{e})")

        fig.savefig(os.path.join(outdir, f"frame_{frame_idx:04d}.png"), dpi=dpi)
        plt.close(fig)
        frame_idx += 1

    print(f"[div(mean flow)] mean RMS: {np.mean(flow_rms):.3e}")
    print(f"[div(mean flow)] max  RMS: {np.max(flow_rms):.3e}")
    print(f"[div(mean flow)] max |div|: {np.max(flow_max):.3e}")

    gif_path = os.path.join(results_dir, f"{name}_div_info_over_mean_flow.gif")
    make_gif_from_dir(outdir, gif_path, duration=duration)
    return gif_path



def energy_spectrum_2d(u, v, LX, LY, nbins=60, detrend_mean=True):
    """
    Isotropic kinetic energy spectrum E(k) from one 2D snapshot on a uniform periodic grid.

    Returns:
      k_centers: (nbins,)
      E_k:       (nbins,)  ~ energy density per unit k (so slopes compare to k^-5/3, k^-3)
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    NX, NY = u.shape

    # Optional but strongly recommended: remove spatial mean (kills k~0 spike)
    if detrend_mean:
        u = u - u.mean()
        v = v - v.mean()

    # FFT
    U = np.fft.fftn(u)
    V = np.fft.fftn(v)

    # Parseval-consistent-ish scaling (keeps amplitudes sensible across NX,NY)
    # (Exact prefactor depends on FFT convention; slopes unaffected.)
    norm = (NX * NY)**2
    E2 = 0.5 * (np.abs(U)**2 + np.abs(V)**2) / norm   # energy per (kx,ky) mode

    # Wavenumbers
    _, _, K = _fft_wavenumbers_2d(NX, NY, LX, LY)
    K_flat = K.ravel()
    E_flat = E2.ravel()

    # Ignore k=0
    mask_pos = K_flat > 0
    K_flat = K_flat[mask_pos]
    E_flat = E_flat[mask_pos]

    kmin = K_flat.min()
    kmax = K_flat.max()

    # log-spaced bins
    k_edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    which = np.digitize(K_flat, k_edges) - 1

    E_shell = np.zeros(nbins, dtype=np.float64)
    counts  = np.zeros(nbins, dtype=np.int64)

    for i in range(nbins):
        m = which == i
        if np.any(m):
            E_shell[i] = E_flat[m].sum()     # total energy in that annulus
            counts[i]  = m.sum()

    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])
    dk = (k_edges[1:] - k_edges[:-1])

    # Convert “energy per shell” -> “energy density per unit k”
    # so that ∫E(k) dk ≈ total energy
    E_k = np.where(counts > 0, E_shell / dk, np.nan)

    return k_centers, E_k


def time_averaged_spectrum(u_series, v_series, LX, LY, nbins=60, t_indices=None):
    u_series = np.asarray(u_series)
    v_series = np.asarray(v_series)
    T = u_series.shape[0]
    if t_indices is None:
        t_indices = range(T)

    Ek_sum = None
    Ek_cnt = None
    k_ref = None

    for t in t_indices:
        k, Ek = energy_spectrum_2d(u_series[t], v_series[t], LX, LY, nbins=nbins)
        if Ek_sum is None:
            k_ref = k
            Ek_sum = np.zeros_like(Ek, dtype=np.float64)
            Ek_cnt = np.zeros_like(Ek, dtype=np.float64)

        m = np.isfinite(Ek)
        Ek_sum[m] += Ek[m]
        Ek_cnt[m] += 1.0

    Ek_mean = np.full_like(Ek_sum, np.nan, dtype=np.float64)
    np.divide(Ek_sum, Ek_cnt, out=Ek_mean, where=(Ek_cnt > 0))
    return k_ref, Ek_mean



def plot_fluid_vs_info_spectra(
    u, v, LX, LY,
    reg_piv, ftle,
    nbins=60,
    fluid_t_indices=None,
    info_k_indices=None,
    save_path=None,
):
    """
    Make one log-log plot comparing:
      - Fluid spectrum (full grid)
      - Info-flow spectrum (move_grid/tau on its own grid extent)

    reg_piv: dict containing move_grid, centers_xy/meta
    ftle: dict containing dt (for tau)
    """

    # Fluid spectrum

    # DEBUG
    u = np.asarray(u)
    v = np.asarray(v)
    u = u[-1:]   # keep as (T=1, NX, NY)
    v = v[-1:]
    fluid_t_indices = [0]


    k_fluid, Ek_fluid = time_averaged_spectrum(
        u, v, LX, LY, nbins=nbins, t_indices=fluid_t_indices
    )

    # Info spectrum 
    move_grid = np.asarray(reg_piv["move_grid"], dtype=np.float64)  # (K,out_nx,out_ny,2)
    K_info = [move_grid.shape[0] - 1]

    W = int(reg_piv["meta"]["time_window"])
    dt = float(ftle.get("dt", 1.0))
    tau = W * dt
    Vinfo = move_grid / tau  # velocity-like

    # info domain extents from centers_xy
    out_nx = int(reg_piv["meta"]["centers_nx"])
    out_ny = int(reg_piv["meta"]["centers_ny"])
    xy = np.asarray(reg_piv["centers_xy"], dtype=float).reshape(out_nx, out_ny, 2)
    x_info = xy[:, 0, 0]
    y_info = xy[0, :, 1]

    dx_info = float(x_info[1] - x_info[0])
    dy_info = float(y_info[1] - y_info[0])

    LX_info = dx_info * out_nx
    LY_info = dy_info * out_ny

    # choose which info snapshots to average
    if info_k_indices is None:
        info_k_indices = range(K_info)

    # average info spectra
    Ek_accum = None
    k_info_ref = None
    n_used = 0
    for k in info_k_indices:
        uk = Vinfo[k, :, :, 0]
        vk = Vinfo[k, :, :, 1]
        ki, Eki = energy_spectrum_2d(uk, vk, LX_info, LY_info, nbins=nbins)
        if Ek_accum is None:
            k_info_ref = ki
            Ek_accum = np.zeros_like(Eki, dtype=np.float64)

        mask = np.isfinite(Eki)
        Ek_accum[mask] += Eki[mask]
        n_used += 1

    Ek_info = Ek_accum / max(n_used, 1)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6), constrained_layout=True)

    ax.loglog(k_fluid, Ek_fluid, label="Fluid (full grid)")
    ax.loglog(k_info_ref, Ek_info, label="Info (move_grid/tau)")

    ax.set_xlabel("wavenumber k (rad / length)")
    ax.set_ylabel("E(k) (arb. units)")
    ax.set_title("Energy spectra: fluid vs info flow")
    ax.legend()

    # Optional reference slopes (just for visual guidance)
    # Pick an anchor point in the middle of the fluid curve for reference
    mid = np.nanargmin(np.abs(np.log10(k_fluid) - np.mean(np.log10(k_fluid))))
    k0 = k_fluid[mid]
    y0 = Ek_fluid[mid]

    # k^{-5/3}
    ax.loglog(k_fluid, y0 * (k_fluid / k0) ** (-5/3), linestyle="--", label="k^-5/3 (ref)")
    # k^{-3}
    ax.loglog(k_fluid, y0 * (k_fluid / k0) ** (-3), linestyle="--", label="k^-3 (ref)")
    ax.legend()

    
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    
    print('here')


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

import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

from data_generation import *
from plot_style import (
    CONVERGING_COLOR,
    COSINE_SIMILARITY_CMAP,
    DIVERGENCE_CMAP,
    DIVERGING_COLOR,
    FLOW_VECTOR_COLOR,
    INFO_LINE_COLOR,
    INFO_VECTOR_COLOR,
    RIDGE_COLOR,
    SCALAR_OVERLAY_CMAP,
    SINGLE_PANEL_FIGSIZE,
    WIDE_PANEL_FIGSIZE,
    add_frame_badge,
    finalize_legend,
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
    """Mean info-flow velocity over the FTLE integration window idx."""
    move_series = np.asarray(reg_piv["move_series"])          # (K, M, 2)
    out_nx = int(reg_piv["meta"]["centers_nx"])
    out_ny = int(reg_piv["meta"]["centers_ny"])
    W = int(reg_piv["meta"]["time_window"])
    dt = float(ftle.get("dt", 1.0))
    tau = W * dt                                          # MUST match our FTLE code

    # info-flow interval used for LCS
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

    u_mean = np.asarray(_temporal_mean_field(u[s_frame:e_frame]), dtype=np.float64)

    if v is None:
        return u_mean, (s_frame, e_frame)

    v_mean = np.asarray(_temporal_mean_field(v[s_frame:e_frame]), dtype=np.float64)

    Vmean_full = np.stack([u_mean, v_mean], axis=-1)  # (NX,NY,2)
    return Vmean_full, (s_frame, e_frame)


def _normalize_for_display(V, x, y, frac=0.06, domain_span=None):
    """Scale arrows to a fixed visible length (keeps quiver readable across frames)."""
    x = np.asarray(x); y = np.asarray(y)
    if domain_span is None:
        arrow_len = frac * min((x[-1] - x[0]), (y[-1] - y[0]))
    else:
        arrow_len = frac * min(float(domain_span[0]), float(domain_span[1]))
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

            Vmean, _ = V_provider(idx)
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
            set_panel_title(ax, title_prefix)
            add_frame_badge(ax, "White contour: LCS ridge")

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

    # 1) FTLE + mean info flow
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


def _finite_percentile(field, q, default=0.0):
    """Return one percentile using only finite values."""
    values = np.asarray(field, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, q))


def _stable_limit_band(limit_pairs, default=(0.0, 1.0)):
    """Collapse per-frame percentile limits into one stable display band."""
    lows = [float(lo) for lo, hi in limit_pairs if np.isfinite(lo) and np.isfinite(hi)]
    highs = [float(hi) for lo, hi in limit_pairs if np.isfinite(lo) and np.isfinite(hi)]
    if not lows or not highs:
        return default

    lo = float(np.median(lows))
    hi = float(np.median(highs))
    if hi <= lo:
        lo = float(np.min(lows))
        hi = float(np.max(highs))
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


def _sample_uniform_field_on_axes(field, x_sample, y_sample, LX, LY):
    """Bilinearly sample a uniform-grid field at a target tensor-product grid."""
    field = np.asarray(field, dtype=np.float64)
    x_sample = np.asarray(x_sample, dtype=np.float64)
    y_sample = np.asarray(y_sample, dtype=np.float64)

    NX, NY = field.shape[:2]
    if NX == 1 or NY == 1:
        out_shape = (len(x_sample), len(y_sample)) + tuple(field.shape[2:])
        return np.broadcast_to(field[0, 0], out_shape).copy()

    tx = np.clip(x_sample / max(float(LX), 1e-12) * (NX - 1), 0.0, NX - 1.0)
    ty = np.clip(y_sample / max(float(LY), 1e-12) * (NY - 1), 0.0, NY - 1.0)

    ix0 = np.floor(tx).astype(int)
    iy0 = np.floor(ty).astype(int)
    ix1 = np.clip(ix0 + 1, 0, NX - 1)
    iy1 = np.clip(iy0 + 1, 0, NY - 1)

    wx = (tx - ix0)[:, None]
    wy = (ty - iy0)[None, :]

    if field.ndim == 2:
        F00 = field[ix0[:, None], iy0[None, :]]
        F10 = field[ix1[:, None], iy0[None, :]]
        F01 = field[ix0[:, None], iy1[None, :]]
        F11 = field[ix1[:, None], iy1[None, :]]
    elif field.ndim == 3:
        F00 = field[ix0[:, None], iy0[None, :], :]
        F10 = field[ix1[:, None], iy0[None, :], :]
        F01 = field[ix0[:, None], iy1[None, :], :]
        F11 = field[ix1[:, None], iy1[None, :], :]
        wx = wx[..., None]
        wy = wy[..., None]
    else:
        raise ValueError("field must have shape (NX,NY) or (NX,NY,C)")

    out = (
        (1.0 - wx) * (1.0 - wy) * F00
        + wx * (1.0 - wy) * F10
        + (1.0 - wx) * wy * F01
        + wx * wy * F11
    )
    return out


def _add_info_domain_outline(axis, extent_info, *, edgecolor=INFO_VECTOR_COLOR, alpha=0.9):
    """Draw the info-flow subdomain outline on one axis."""
    axis.add_patch(
        Rectangle(
            (extent_info[0], extent_info[2]),
            extent_info[1] - extent_info[0],
            extent_info[3] - extent_info[2],
            fill=False,
            linewidth=1.35,
            linestyle=(0, (4, 2)),
            edgecolor=edgecolor,
            alpha=alpha,
        )
    )


def _resolve_quiver_stride(nx, ny, *, qskip=None, target_vectors=30):
    """Choose a stride that keeps quiver density near one target count per axis."""
    if qskip is not None:
        return max(1, int(qskip))

    target = max(1, int(target_vectors))
    return max(1, int(np.ceil(max(nx, ny) / target)))


def _cell_edges_from_centers(coords):
    """Convert 1D cell centers into cell-edge coordinates for pcolormesh."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 1:
        delta = 1.0
        return np.array([coords[0] - 0.5 * delta, coords[0] + 0.5 * delta], dtype=np.float64)

    mids = 0.5 * (coords[:-1] + coords[1:])
    left = coords[0] - 0.5 * (coords[1] - coords[0])
    right = coords[-1] + 0.5 * (coords[-1] - coords[-2])
    return np.concatenate([[left], mids, [right]])


def fluid_vs_regional_flow_gif(
    reg_piv,
    results_dir,
    name,
    u,
    v,
    LX,
    LY,
    dt,
    outdir=None,
    title_prefix=None,
    stride=1,
    fluid_qskip=None,
    info_qskip=None,
    target_vectors=30,
    dpi=180,
    duration=0.10,
):
    """
    Save a side-by-side GIF comparing the mean fluid flow and mean info flow.
    """
    if v is None:
        print(f"[{name}] Scalar mode: skipping fluid-vs-info-flow comparison GIF.")
        return None

    os.makedirs(results_dir, exist_ok=True)
    if outdir is None:
        outdir = os.path.join(results_dir, f"frames_flow_compare_{name}")
    os.makedirs(outdir, exist_ok=True)
    if title_prefix is None:
        title_prefix = f"{name}: fluid flow vs info flow"

    move_grid = np.asarray(reg_piv["move_grid"], dtype=np.float64)
    intervals = reg_piv["intervals"]
    K = move_grid.shape[0]
    tau = float(reg_piv["meta"]["time_window"]) * float(dt)
    V_info = move_grid / max(tau, 1e-12)

    x_info, y_info, extent_info, _, _ = _info_axes_from_centers_xy(reg_piv)
    X_info, Y_info = np.meshgrid(x_info, y_info, indexing="ij")

    NX, NY = u.shape[1], u.shape[2]
    x_full = np.linspace(0.0, LX, NX)
    y_full = np.linspace(0.0, LY, NY)
    X_full, Y_full = np.meshgrid(x_full, y_full, indexing="ij")
    extent_full = (0.0, float(LX), 0.0, float(LY))

    speed_limit_pairs = []
    for k in range(0, K, stride):
        s, e = intervals[k]
        u_mean = _temporal_mean_field(u[int(s):int(e)])
        v_mean = _temporal_mean_field(v[int(s):int(e)])
        speed = np.sqrt(u_mean**2 + v_mean**2)
        speed_limit_pairs.append(_finite_percentile_limits(speed, 5, 95))
    speed_lo, speed_hi = _stable_limit_band(speed_limit_pairs)

    with presentation_plot_context():
        frame_idx = 0
        for k in range(0, K, stride):
            s, e = intervals[k]
            s = int(s); e = int(e)

            u_mean = _temporal_mean_field(u[s:e])
            v_mean = _temporal_mean_field(v[s:e])
            V_fluid = np.stack([u_mean, v_mean], axis=-1)
            speed = np.sqrt(u_mean**2 + v_mean**2)
            V_regional = V_info[k]

            fig, axes = plt.subplots(1, 2, figsize=WIDE_PANEL_FIGSIZE, constrained_layout=True)

            for ax in axes:
                ax.imshow(
                    speed.T,
                    origin="lower",
                    extent=extent_full,
                    aspect="equal",
                    cmap="Greys",
                    alpha=0.34,
                    vmin=speed_lo,
                    vmax=speed_hi,
                )
                _add_info_domain_outline(ax, extent_info, edgecolor=INFO_LINE_COLOR)
                style_spatial_axis(ax, xlim=(0.0, float(LX)), ylim=(0.0, float(LY)))

            V_fluid_plot = _normalize_for_display(
                V_fluid,
                x_full,
                y_full,
                frac=0.055,
                domain_span=(LX, LY),
            )
            qs_fluid = _resolve_quiver_stride(
                V_fluid_plot.shape[0],
                V_fluid_plot.shape[1],
                qskip=fluid_qskip,
                target_vectors=target_vectors,
            )
            axes[0].quiver(
                X_full[::qs_fluid, ::qs_fluid],
                Y_full[::qs_fluid, ::qs_fluid],
                V_fluid_plot[::qs_fluid, ::qs_fluid, 0],
                V_fluid_plot[::qs_fluid, ::qs_fluid, 1],
                angles="xy",
                scale_units="xy",
                scale=None,
                width=0.0030,
                color=FLOW_VECTOR_COLOR,
                alpha=0.92,
            )
            set_panel_title(axes[0], "Mean fluid flow")
            add_frame_badge(axes[0], "Dashed box: info-flow domain")

            V_regional_plot = _normalize_for_display(
                V_regional,
                x_info,
                y_info,
                frac=0.055,
                domain_span=(LX, LY),
            )
            qs_info = _resolve_quiver_stride(
                V_regional_plot.shape[0],
                V_regional_plot.shape[1],
                qskip=info_qskip,
                target_vectors=target_vectors,
            )
            axes[1].quiver(
                X_info[::qs_info, ::qs_info],
                Y_info[::qs_info, ::qs_info],
                V_regional_plot[::qs_info, ::qs_info, 0],
                V_regional_plot[::qs_info, ::qs_info, 1],
                angles="xy",
                scale_units="xy",
                scale=None,
                width=0.0042,
                color=INFO_VECTOR_COLOR,
                alpha=0.95,
            )
            set_panel_title(axes[1], "Mean info flow")
            add_frame_badge(axes[1], f"Quiver density matched to fluid view")

            fig.suptitle(title_prefix, x=0.06, ha="left", color="#1F2937")
            fig.savefig(os.path.join(outdir, f"frame_{frame_idx:04d}.png"), dpi=dpi)
            plt.close(fig)
            frame_idx += 1

    gif_path = os.path.join(results_dir, f"{name}_fluid_vs_info_flow.gif")
    make_gif_from_dir(outdir, gif_path, duration=duration)
    return gif_path


def flow_cosine_similarity_gif(
    reg_piv,
    results_dir,
    name,
    u,
    v,
    LX,
    LY,
    dt,
    outdir=None,
    title_prefix=None,
    stride=1,
    dpi=180,
    duration=0.10,
):
    """
    Save a GIF of pointwise direction cosine similarity on the info-flow grid.

    At each info-flow grid point, this bilinearly samples the fluid velocity,
    normalizes both vectors to directions, and plots their cosine similarity as
    a scalar heat map on the info-flow grid.
    """
    if v is None:
        print(f"[{name}] Scalar mode: skipping flow direction cosine-similarity GIF.")
        return None

    os.makedirs(results_dir, exist_ok=True)
    if outdir is None:
        outdir = os.path.join(results_dir, f"frames_flow_cosine_{name}")
    os.makedirs(outdir, exist_ok=True)
    if title_prefix is None:
        title_prefix = f"{name}: fluid vs info-flow direction cosine similarity"

    move_grid = np.asarray(reg_piv["move_grid"], dtype=np.float64)
    intervals = reg_piv["intervals"]
    K = move_grid.shape[0]
    tau = float(reg_piv["meta"]["time_window"]) * float(dt)
    V_info = move_grid / max(tau, 1e-12)

    x_info, y_info, extent_info, _, _ = _info_axes_from_centers_xy(reg_piv)
    x_info_edges = _cell_edges_from_centers(x_info)
    y_info_edges = _cell_edges_from_centers(y_info)
    extent_full = (0.0, float(LX), 0.0, float(LY))

    speed_limit_pairs = []
    for k in range(0, K, stride):
        s, e = intervals[k]
        u_mean = _temporal_mean_field(u[int(s):int(e)])
        v_mean = _temporal_mean_field(v[int(s):int(e)])
        speed = np.sqrt(u_mean**2 + v_mean**2)
        speed_limit_pairs.append(_finite_percentile_limits(speed, 5, 95))
    speed_lo, speed_hi = _stable_limit_band(speed_limit_pairs)

    with presentation_plot_context():
        frame_idx = 0
        for k in range(0, K, stride):
            s, e = intervals[k]
            s = int(s); e = int(e)

            u_mean = _temporal_mean_field(u[s:e])
            v_mean = _temporal_mean_field(v[s:e])
            V_fluid = np.stack([u_mean, v_mean], axis=-1)
            speed = np.sqrt(u_mean**2 + v_mean**2)

            V_regional = V_info[k]
            V_fluid_on_info = _sample_uniform_field_on_axes(V_fluid, x_info, y_info, LX, LY)

            fluid_mag = np.linalg.norm(V_fluid_on_info, axis=-1)
            info_mag = np.linalg.norm(V_regional, axis=-1)
            valid = (fluid_mag > 1e-12) & (info_mag > 1e-12)

            cosine = np.full(fluid_mag.shape, np.nan, dtype=np.float64)
            if np.any(valid):
                fluid_dir = np.zeros_like(V_fluid_on_info, dtype=np.float64)
                info_dir = np.zeros_like(V_regional, dtype=np.float64)
                fluid_dir[valid] = V_fluid_on_info[valid] / fluid_mag[valid, None]
                info_dir[valid] = V_regional[valid] / info_mag[valid, None]
                cosine[valid] = np.clip(np.sum(fluid_dir[valid] * info_dir[valid], axis=-1), -1.0, 1.0)

            fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_FIGSIZE, constrained_layout=True)
            ax.imshow(
                speed.T,
                origin="lower",
                extent=extent_full,
                aspect="equal",
                cmap="Greys",
                alpha=0.34,
                vmin=speed_lo,
                vmax=speed_hi,
            )

            norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            im = ax.pcolormesh(
                x_info_edges,
                y_info_edges,
                np.ma.masked_invalid(cosine).T,
                cmap=COSINE_SIMILARITY_CMAP,
                norm=norm,
                shading="flat",
                alpha=0.92,
            )

            _add_info_domain_outline(ax, extent_info, edgecolor=INFO_LINE_COLOR)
            style_spatial_axis(ax, xlim=(0.0, float(LX)), ylim=(0.0, float(LY)))
            set_panel_title(ax, title_prefix)
            add_frame_badge(ax, "Heat map evaluated on the info-flow grid")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            style_colorbar(cbar, r"$\cos(\theta)$")

            fig.savefig(os.path.join(outdir, f"frame_{frame_idx:04d}.png"), dpi=dpi)
            plt.close(fig)
            frame_idx += 1

    gif_path = os.path.join(results_dir, f"{name}_flow_cosine_similarity.gif")
    make_gif_from_dir(outdir, gif_path, duration=duration)
    return gif_path


def dual_lcs_overlay_gif(
    reg_piv,
    ftle,
    results_dir,
    name,
    u,
    v,
    LX,
    LY,
    title_prefix=None,
    ridge_pct=92,
    qskip=10,
    outdir=None,
    dpi=180,
    duration=0.10,
):
    """
    Save a GIF with the underlying flow plus both attracting and repelling LCS overlays.
    Attracting structures use the same blue family as converging divergence regions and
    repelling structures use the same red family as diverging divergence regions.
    """
    os.makedirs(results_dir, exist_ok=True)
    if outdir is None:
        outdir = os.path.join(results_dir, f"frames_dual_lcs_{name}")
    os.makedirs(outdir, exist_ok=True)
    if title_prefix is None:
        if v is None:
            title_prefix = f"{name}: attracting and repelling LCS over mean scalar field"
        else:
            title_prefix = f"{name}: attracting and repelling LCS over mean flow"

    ftle_forward = np.asarray(ftle["ftle_forward"], dtype=np.float64)
    ftle_backward = np.asarray(ftle["ftle_backward"], dtype=np.float64)
    x = np.asarray(ftle["x"], dtype=np.float64)
    y = np.asarray(ftle["y"], dtype=np.float64)
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    N = ftle_forward.shape[0]
    extent_info = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    extent_full = (0.0, float(LX), 0.0, float(LY))

    attract_threshold = _finite_percentile(ftle_backward, ridge_pct, default=0.0)
    repel_threshold = _finite_percentile(ftle_forward, ridge_pct, default=0.0)

    background_limit_pairs = []
    for idx in range(N):
        background, _ = mean_fluid_flow_for_idx(u, v, LX, LY, reg_piv, ftle, idx)
        if background.ndim == 3:
            background_field = np.linalg.norm(background, axis=-1)
        else:
            background_field = background
        background_limit_pairs.append(_finite_percentile_limits(background_field, 5, 95))
    background_lo, background_hi = _stable_limit_band(background_limit_pairs)

    if v is not None:
        NX, NY = u.shape[1], u.shape[2]
        x_full = np.linspace(0.0, LX, NX)
        y_full = np.linspace(0.0, LY, NY)
        X_full, Y_full = np.meshgrid(x_full, y_full, indexing="ij")

    legend_handles = [
        Patch(facecolor=CONVERGING_COLOR, edgecolor=CONVERGING_COLOR, alpha=0.28, label="Attracting LCS"),
        Patch(facecolor=DIVERGING_COLOR, edgecolor=DIVERGING_COLOR, alpha=0.28, label="Repelling LCS"),
    ]

    with presentation_plot_context():
        for idx in range(N):
            background, _ = mean_fluid_flow_for_idx(u, v, LX, LY, reg_piv, ftle, idx)
            if background.ndim == 3:
                background_field = np.linalg.norm(background, axis=-1)
            else:
                background_field = background

            fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_FIGSIZE, constrained_layout=True)
            ax.imshow(
                background_field.T,
                origin="lower",
                extent=extent_full,
                aspect="equal",
                cmap="Greys" if background.ndim == 3 else SCALAR_OVERLAY_CMAP,
                alpha=0.36 if background.ndim == 3 else 0.44,
                vmin=background_lo,
                vmax=background_hi,
            )

            if background.ndim == 3:
                V_plot = _normalize_for_display(
                    background,
                    x_full,
                    y_full,
                    frac=0.055,
                    domain_span=(LX, LY),
                )
                qs = max(1, int(qskip))
                ax.quiver(
                    X_full[::qs, ::qs],
                    Y_full[::qs, ::qs],
                    V_plot[::qs, ::qs, 0],
                    V_plot[::qs, ::qs, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=None,
                    width=0.0030,
                    color=FLOW_VECTOR_COLOR,
                    alpha=0.88,
                )

            ftle_b = ftle_backward[idx]
            ftle_f = ftle_forward[idx]

            if np.any(np.isfinite(ftle_b)) and np.nanmax(ftle_b) > attract_threshold:
                ax.contourf(
                    Xg,
                    Yg,
                    ftle_b,
                    levels=[attract_threshold, float(np.nanmax(ftle_b)) + 1e-12],
                    colors=[CONVERGING_COLOR],
                    alpha=0.28,
                )
                ax.contour(
                    Xg,
                    Yg,
                    ftle_b,
                    levels=[attract_threshold],
                    colors=[CONVERGING_COLOR],
                    linewidths=1.35,
                    alpha=0.95,
                )

            if np.any(np.isfinite(ftle_f)) and np.nanmax(ftle_f) > repel_threshold:
                ax.contourf(
                    Xg,
                    Yg,
                    ftle_f,
                    levels=[repel_threshold, float(np.nanmax(ftle_f)) + 1e-12],
                    colors=[DIVERGING_COLOR],
                    alpha=0.26,
                )
                ax.contour(
                    Xg,
                    Yg,
                    ftle_f,
                    levels=[repel_threshold],
                    colors=[DIVERGING_COLOR],
                    linewidths=1.35,
                    alpha=0.95,
                )

            style_spatial_axis(ax, xlim=(0.0, float(LX)), ylim=(0.0, float(LY)))
            set_panel_title(ax, title_prefix)
            finalize_legend(ax, handles=legend_handles, loc="lower right")

            fig.savefig(os.path.join(outdir, f"frame_{idx:04d}.png"), dpi=dpi)
            plt.close(fig)

    gif_suffix = "mean_scalar" if v is None else "mean_flow"
    gif_path = os.path.join(results_dir, f"{name}_dual_lcs_over_{gif_suffix}.gif")
    make_gif_from_dir(outdir, gif_path, duration=duration)
    return gif_path


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
      - Vector mode: mean fluid speed over reg window [s:e) + mean-flow quiver
      - Scalar mode: mean scalar field over reg window [s:e)
      - Overlay: divergence of info velocity on info subdomain

    Also prints a divergence-of-mean-flow diagnostic in vector mode.
    """

    os.makedirs(results_dir, exist_ok=True)
    if outdir is None:
        outdir = os.path.join(results_dir, f"frames_div_info_{name}")
    os.makedirs(outdir, exist_ok=True)
    if title_prefix is None:
        if v is None:
            title_prefix = f"{name}: div(info) over mean scalar field"
        else:
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

    # full-domain grid geometry
    NX, NY = u.shape[1], u.shape[2]
    xq = np.linspace(0.0, LX, NX)
    yq = np.linspace(0.0, LY, NY)
    Xq, Yq = np.meshgrid(xq, yq, indexing="ij")
    if v is not None:
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

            if v is None:
                background = _temporal_mean_field(u[s:e])
                background_cmap = SCALAR_OVERLAY_CMAP if cmap == "Greys" else cmap
                background_label = "Mean scalar field"
            else:
                u_mean = _temporal_mean_field(u[s:e])
                v_mean = _temporal_mean_field(v[s:e])
                background = np.sqrt(u_mean**2 + v_mean**2)
                background_cmap = cmap
                background_label = "Mean fluid speed"

                div_flow = _divergence_2d(u_mean, v_mean, dx_flow, dy_flow)
                flow_rms.append(float(np.sqrt(np.mean(div_flow**2))))
                flow_max.append(float(np.max(np.abs(div_flow))))

            Vk = V_info[k]
            div_info = _divergence_2d(Vk[..., 0], Vk[..., 1], dx_info, dy_info)

            vmin_bg, vmax_bg = _finite_percentile_limits(background, 5, 95)

            if div_limit is None:
                lim_k = float(np.percentile(np.abs(div_info), div_pct))
                lim_k = max(lim_k, 1e-12)
            else:
                lim_k = div_limit

            fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_FIGSIZE, constrained_layout=True)

            ax.imshow(
                background.T,
                origin="lower",
                extent=extent_full,
                aspect="equal",
                cmap=background_cmap,
                alpha=0.42 if v is None else 0.34,
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

            if v is not None:
                qs = max(1, int(qskip))
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
            set_panel_title(ax, title_prefix)
            add_frame_badge(ax, f"Background: {background_label}")

            fig.savefig(os.path.join(outdir, f"frame_{frame_idx:04d}.png"), dpi=dpi)
            plt.close(fig)
            frame_idx += 1

    if v is not None:
        print(f"[div(mean flow)] mean RMS: {np.mean(flow_rms):.3e}")
        print(f"[div(mean flow)] max  RMS: {np.max(flow_rms):.3e}")
        print(f"[div(mean flow)] max |div|: {np.max(flow_max):.3e}")
    else:
        print("[div(mean scalar)] scalar mode: skipped mean-flow divergence diagnostics.")

    gif_suffix = "mean_scalar" if v is None else "mean_flow"
    gif_path = os.path.join(results_dir, f"{name}_div_info_over_{gif_suffix}.gif")
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

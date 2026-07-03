"""Plotting helpers for the state-weighted POD-QR experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common.plot_style import apply_axis_style, pretty_flow_name
from experiments.state_weighted_pod_qr.pipeline import (
    METHOD_ORDER,
    STANDARD_METHOD,
    WEIGHTED_PSI_METHOD,
    WEIGHTED_PSIW_METHOD,
)


METRIC_COLUMN = "L2_h"
METRIC_LABEL = r"Relative $L_h^2$ Error"
CONDITION_LABEL = r"$\kappa(C_i\,\Phi_{\mathrm{recon}})$"

METHOD_STYLES = {
    STANDARD_METHOD: {
        "color": "#264653",
        "marker": "o",
        "label": "Standard POD-QR / recon Psi",
    },
    WEIGHTED_PSI_METHOD: {
        "color": "#E76F51",
        "marker": "s",
        "label": "Instant weighted / recon Psi",
    },
    WEIGHTED_PSIW_METHOD: {
        "color": "#2A9D8F",
        "marker": "^",
        "label": "Instant weighted / recon PsiW",
    },
}

PNG_RC_PARAMS = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
}


def _method_style(method_name):
    """Return plotting style for a method name."""
    return METHOD_STYLES.get(
        str(method_name),
        {
            "color": "#6B7280",
            "marker": "o",
            "label": str(method_name),
        },
    )


def _ordered_methods(values):
    """Return methods in preferred order followed by any extras."""
    available = list(pd.Series(values).dropna().unique())
    ordered = [method for method in METHOD_ORDER if method in set(available)]
    ordered.extend(sorted(method for method in available if method not in set(ordered)))
    return ordered


def _stable_y_limits(values, log_scale):
    """Compute stable y limits shared by all progressive frames."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return (1e-12, 1.0) if log_scale else (0.0, 1.0)

    if log_scale:
        positive = values[values > 0.0]
        ymin = float(positive.min()) * 0.8
        ymax = float(positive.max()) * 1.25
        if ymin == ymax:
            ymin *= 0.5
            ymax *= 2.0
        return max(ymin, 1e-16), ymax

    ymin = min(0.0, float(values.min()) * 0.95)
    ymax = float(values.max()) * 1.10
    if ymin == ymax:
        ymax = ymin + 1.0
    return ymin, ymax


def _should_use_log_scale(values):
    """Use a log error axis when all finite error values are positive."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return bool(not values.empty and (values > 0.0).all())


def _plot_error_over_time(axis, flow_raw_df, current_t=None):
    """Plot relative L2_h over time for one flow."""
    for method_name in _ordered_methods(flow_raw_df["method"]):
        method_df = flow_raw_df[flow_raw_df["method"] == method_name].sort_values("t")
        style = _method_style(method_name)
        axis.plot(
            method_df["t"],
            method_df[METRIC_COLUMN],
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=1.8,
            markersize=3.8,
            label=style["label"],
        )

    if current_t is not None:
        axis.axvline(int(current_t), color="#667085", linestyle=":", linewidth=1.1)

    if _should_use_log_scale(flow_raw_df[METRIC_COLUMN]):
        axis.set_yscale("log")
    axis.set_xlabel("Time Index")
    axis.set_ylabel(METRIC_LABEL)
    axis.set_title("Reconstruction Error")
    apply_axis_style(axis, x_grid=True, y_grid=True)
    axis.legend(loc="best", frameon=True)


def _plot_flow_with_sensors(axis, u, v, domain, flow_sensor_df, current_t, quiver_step=8):
    """Plot one velocity-magnitude snapshot with standard and weighted sensors."""
    t_idx = int(current_t)
    speed = np.hypot(u[t_idx], v[t_idx])
    image = axis.imshow(
        speed.T,
        origin="lower",
        extent=[0.0, domain.lx, 0.0, domain.ly],
        aspect="auto",
        cmap="viridis",
    )

    step = max(1, int(quiver_step))
    x_grid = np.linspace(0.0, domain.lx, u.shape[1])
    y_grid = np.linspace(0.0, domain.ly, u.shape[2])
    x_mesh, y_mesh = np.meshgrid(x_grid[::step], y_grid[::step], indexing="ij")
    quiver_scale = max(float(np.nanmax(speed)), 1e-12) * 20.0
    axis.quiver(
        x_mesh,
        y_mesh,
        u[t_idx, ::step, ::step],
        v[t_idx, ::step, ::step],
        color="white",
        alpha=0.42,
        linewidth=0.25,
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
    )

    current_sensor_df = flow_sensor_df[flow_sensor_df["t"] == t_idx]
    for method_name in _ordered_methods(current_sensor_df["method"]):
        method_sensor_df = current_sensor_df[current_sensor_df["method"] == method_name]
        style = _method_style(method_name)
        axis.scatter(
            method_sensor_df["x"],
            method_sensor_df["y"],
            s=42,
            marker=style["marker"],
            facecolors="none",
            edgecolors=style["color"],
            linewidths=1.5,
            label=style["label"],
        )

    axis.set_xlim(0.0, domain.lx)
    axis.set_ylim(0.0, domain.ly)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"Flow and Sensors, t={t_idx}")
    axis.legend(loc="upper right", frameon=True)
    return image


def _plot_condition_over_time(axis, flow_condition_df, current_t=None):
    """Plot cond(C Psi) over time for one flow."""
    for method_name in _ordered_methods(flow_condition_df["method"]):
        method_df = flow_condition_df[flow_condition_df["method"] == method_name].sort_values("t")
        style = _method_style(method_name)
        condition_values = pd.to_numeric(method_df["condition_number"], errors="coerce")
        finite_values = condition_values.replace([np.inf, -np.inf], np.nan)
        axis.plot(
            method_df["t"],
            finite_values,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=1.8,
            markersize=3.8,
            label=style["label"],
        )

    if current_t is not None:
        axis.axvline(int(current_t), color="#667085", linestyle=":", linewidth=1.1)

    finite_positive = flow_condition_df["condition_number"].replace([np.inf, -np.inf], np.nan)
    finite_positive = finite_positive[finite_positive > 0.0].dropna()
    if not finite_positive.empty:
        axis.set_yscale("log")
    axis.set_xlabel("Time Index")
    axis.set_ylabel(CONDITION_LABEL)
    axis.set_title("Sensing Matrix Condition")
    apply_axis_style(axis, x_grid=True, y_grid=True)
    axis.legend(loc="best", frameon=True)


def save_flow_diagnostic_plot(
    flow_name,
    u,
    v,
    domain,
    raw_df,
    condition_df,
    sensor_df,
    out_path,
    current_t=None,
    quiver_step=8,
):
    """Save a three-panel diagnostic plot for one flow."""
    flow_raw_df = raw_df[raw_df["flow"] == flow_name].copy()
    flow_condition_df = condition_df[condition_df["flow"] == flow_name].copy()
    flow_sensor_df = sensor_df[sensor_df["flow"] == flow_name].copy()
    if flow_raw_df.empty:
        raise ValueError(f"No raw records found for flow {flow_name!r}")

    if current_t is None:
        unique_t = np.sort(flow_raw_df["t"].unique())
        current_t = int(unique_t[len(unique_t) // 2])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PNG_RC_PARAMS):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16.4, 5.2),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.16, 1.0, 1.16]},
        )
        fig.suptitle(f"{pretty_flow_name(flow_name)}: State-Weighted POD-QR", fontsize=13)
        _plot_error_over_time(axes[0], flow_raw_df, current_t=current_t)
        image = _plot_flow_with_sensors(
            axes[1],
            u,
            v,
            domain,
            flow_sensor_df,
            current_t=current_t,
            quiver_step=quiver_step,
        )
        _plot_condition_over_time(axes[2], flow_condition_df, current_t=current_t)
        fig.colorbar(image, ax=axes[1], shrink=0.78, label="Speed")
        fig.savefig(out_path)
        plt.close(fig)

    return out_path


def save_flow_diagnostic_frames(
    flow_name,
    u,
    v,
    domain,
    raw_df,
    condition_df,
    sensor_df,
    frames_dir,
    frame_t_indices=None,
    quiver_step=8,
):
    """Save per-time three-panel diagnostic frames for GIF creation."""
    flow_raw_df = raw_df[raw_df["flow"] == flow_name].copy()
    if frame_t_indices is None:
        frame_t_indices = np.sort(flow_raw_df["t"].unique())

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for frame_idx, t_idx in enumerate(frame_t_indices):
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        save_flow_diagnostic_plot(
            flow_name=flow_name,
            u=u,
            v=v,
            domain=domain,
            raw_df=raw_df,
            condition_df=condition_df,
            sensor_df=sensor_df,
            out_path=frame_path,
            current_t=int(t_idx),
            quiver_step=quiver_step,
        )
        frame_paths.append(frame_path)

    return frame_paths


def make_gif_from_frames(frame_paths, gif_path, duration=0.12):
    """Create a GIF from diagnostic PNG frames if imageio is available."""
    frame_paths = [Path(frame_path) for frame_path in frame_paths]
    if not frame_paths:
        return None

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio is not available; skipping diagnostic GIF creation")
        return None

    with imageio.get_writer(gif_path, mode="I", duration=float(duration)) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    return gif_path


def save_progressive_error_plots(raw_df, flow_name, output_dir, make_gif=False, gif_duration=0.10):
    """Save cumulative error plots t_0 through t_i for one flow.

    Progressive plots show how the method comparison evolves over time rather
    than only showing the final full trajectory.
    """
    flow_df = raw_df[raw_df["flow"] == flow_name].copy()
    if flow_df.empty:
        raise ValueError(f"No raw records found for flow {flow_name!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_values = np.sort(flow_df["t"].unique())
    log_scale = _should_use_log_scale(flow_df[METRIC_COLUMN])
    y_limits = _stable_y_limits(flow_df[METRIC_COLUMN], log_scale=log_scale)

    frame_paths = []
    with plt.rc_context(PNG_RC_PARAMS):
        for frame_idx in range(1, len(t_values)):
            current_t = int(t_values[frame_idx])
            cumulative_df = flow_df[flow_df["t"] <= current_t]
            fig, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
            _plot_error_over_time(axis, cumulative_df, current_t=None)
            axis.set_ylim(*y_limits)
            axis.set_title(f"{pretty_flow_name(flow_name)}: Errors Through t={current_t}")
            frame_path = output_dir / f"progressive_error_{frame_idx:04d}.png"
            fig.savefig(frame_path)
            plt.close(fig)
            frame_paths.append(frame_path)

        fig, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
        _plot_error_over_time(axis, flow_df, current_t=None)
        axis.set_ylim(*y_limits)
        axis.set_title(f"{pretty_flow_name(flow_name)}: Full Error Trajectory")
        full_plot_path = output_dir / "error_over_time_full.png"
        fig.savefig(full_plot_path)
        plt.close(fig)

    gif_path = None
    if make_gif and frame_paths:
        gif_path = make_gif_from_frames(frame_paths, output_dir / "progressive_error.gif", duration=gif_duration)

    return {
        "frames": frame_paths,
        "full_plot": full_plot_path,
        "gif": gif_path,
    }


def save_l2h_boxplot(raw_df, flow_name, output_path):
    """Save a box plot of relative L2_h error by method."""
    flow_df = raw_df[raw_df["flow"] == flow_name].copy()
    methods = _ordered_methods(flow_df["method"])
    series = [flow_df.loc[flow_df["method"] == method, METRIC_COLUMN].dropna().to_numpy() for method in methods]
    labels = [_method_style(method)["label"] for method in methods]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PNG_RC_PARAMS):
        fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
        boxplot = axis.boxplot(
            series,
            labels=labels,
            vert=False,
            patch_artist=True,
            showfliers=True,
            whis=[5, 95],
            medianprops={"color": "#111827", "linewidth": 1.5},
            whiskerprops={"color": "#475467", "linewidth": 1.0},
            capprops={"color": "#475467", "linewidth": 1.0},
        )
        for patch, method in zip(boxplot["boxes"], methods):
            patch.set_facecolor(_method_style(method)["color"])
            patch.set_alpha(0.62)
            patch.set_edgecolor("#344054")
            patch.set_linewidth(0.9)

        if _should_use_log_scale(flow_df[METRIC_COLUMN]):
            axis.set_xscale("log")
        axis.set_xlabel(METRIC_LABEL)
        axis.set_ylabel("Method")
        axis.set_title(f"{pretty_flow_name(flow_name)}: Error Distribution")
        apply_axis_style(axis, x_grid=True, y_grid=False)
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def save_condition_over_time_plot(raw_df, flow_name, output_path):
    """Save condition-number-over-time plot for sampled reconstruction bases."""
    flow_df = raw_df[raw_df["flow"] == flow_name].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PNG_RC_PARAMS):
        fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
        for method in _ordered_methods(flow_df["method"]):
            method_df = flow_df[flow_df["method"] == method].sort_values("t")
            style = _method_style(method)
            cond_values = pd.to_numeric(method_df["cond_reconstruction_basis"], errors="coerce")
            cond_values = cond_values.replace([np.inf, -np.inf], np.nan)
            axis.plot(
                method_df["t"],
                cond_values,
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="white",
                markeredgewidth=0.85,
                linewidth=1.9,
                markersize=3.8,
                label=style["label"],
            )

        finite_positive = flow_df["cond_reconstruction_basis"].replace([np.inf, -np.inf], np.nan)
        finite_positive = finite_positive[finite_positive > 0.0].dropna()
        if not finite_positive.empty:
            axis.set_yscale("log")
        axis.set_xlabel("Time Index")
        axis.set_ylabel(CONDITION_LABEL)
        axis.set_title(f"{pretty_flow_name(flow_name)}: Reconstruction Basis Conditioning")
        apply_axis_style(axis, x_grid=True, y_grid=True)
        axis.legend(loc="best", frameon=True)
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def save_sensor_overlap_plot(overlap_df, flow_name, output_path):
    """Save overlap_i = |S_i intersect S_{i-1}| / p for weighted targets."""
    flow_df = overlap_df[overlap_df["flow"] == flow_name].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PNG_RC_PARAMS):
        fig, axis = plt.subplots(figsize=(8.8, 4.4), constrained_layout=True)
        if not flow_df.empty:
            axis.plot(
                flow_df["t"],
                flow_df["overlap"],
                color="#E76F51",
                marker="s",
                markerfacecolor="white",
                markeredgewidth=0.85,
                linewidth=1.9,
                markersize=3.8,
                label="Instant weighted POD-QR",
            )
            axis.legend(loc="best", frameon=True)

        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Time Index")
        axis.set_ylabel(r"$|S_i \cap S_{i-1}| / p$")
        axis.set_title(f"{pretty_flow_name(flow_name)}: Weighted Sensor-Set Overlap")
        apply_axis_style(axis, x_grid=True, y_grid=True)
        fig.savefig(output_path)
        plt.close(fig)
    return output_path

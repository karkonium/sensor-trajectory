"""Shared presentation-style Matplotlib helpers for regional PIV visuals."""

from contextlib import contextmanager

import matplotlib.pyplot as plt


TEXT_COLOR = "#1F2937"
SPINE_COLOR = "#667085"
GRID_MAJOR_COLOR = "#D0D7E2"
GRID_MINOR_COLOR = "#E5EAF1"
PANEL_FACE_COLOR = "#FBFCFE"
LEGEND_EDGE_COLOR = "#D0D7E2"

FTLE_CMAP = "magma"
DIVERGENCE_CMAP = "RdBu_r"
SCALAR_OVERLAY_CMAP = "viridis"
COSINE_SIMILARITY_CMAP = "RdYlBu"

_DIVERGENCE_MAP = plt.get_cmap(DIVERGENCE_CMAP)
CONVERGING_COLOR = _DIVERGENCE_MAP(0.14)
DIVERGING_COLOR = _DIVERGENCE_MAP(0.86)

FLOW_VECTOR_COLOR = "#111827"
INFO_VECTOR_COLOR = "#B54708"
FLUID_LINE_COLOR = "#264653"
INFO_LINE_COLOR = "#B56576"
REFERENCE_LINE_COLOR = "#98A2B3"
RIDGE_COLOR = "#F8FAFC"

SINGLE_PANEL_FIGSIZE = (8.8, 6.2)
WIDE_PANEL_FIGSIZE = (13.4, 5.9)

PRESENTATION_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stixsans",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.titleweight": "semibold",
    "axes.labelsize": 16,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "axes.facecolor": PANEL_FACE_COLOR,
    "axes.edgecolor": SPINE_COLOR,
    "axes.linewidth": 1.0,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "legend.frameon": True,
    "legend.framealpha": 0.96,
    "legend.edgecolor": LEGEND_EDGE_COLOR,
    "legend.fancybox": False,
    "legend.fontsize": 12.5,
    "legend.title_fontsize": 13,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "grid.color": GRID_MAJOR_COLOR,
    "grid.linewidth": 0.85,
    "grid.alpha": 0.95,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def presentation_plot_context():
    """Temporarily apply the regional-PIV presentation plotting style."""
    with plt.rc_context(PRESENTATION_RC_PARAMS):
        yield


def apply_axis_style(axis, *, x_grid=False, y_grid=False):
    """Apply a clean academic presentation style to one axis."""
    axis.set_facecolor(PANEL_FACE_COLOR)
    axis.tick_params(width=1.0, colors=TEXT_COLOR)
    axis.minorticks_on()

    axis.grid(False)
    if x_grid:
        axis.xaxis.grid(True, which="major", color=GRID_MAJOR_COLOR, linewidth=0.85)
        axis.xaxis.grid(True, which="minor", color=GRID_MINOR_COLOR, linewidth=0.55, alpha=0.85)
    if y_grid:
        axis.yaxis.grid(True, which="major", color=GRID_MAJOR_COLOR, linewidth=0.85)
        axis.yaxis.grid(True, which="minor", color=GRID_MINOR_COLOR, linewidth=0.55, alpha=0.85)

    for spine_name in ("left", "bottom"):
        if spine_name in axis.spines:
            axis.spines[spine_name].set_color(SPINE_COLOR)
            axis.spines[spine_name].set_linewidth(1.0)


def style_spatial_axis(axis, *, xlim, ylim, xlabel="x", ylabel="y"):
    """Apply consistent styling for spatial-domain image and quiver plots."""
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    apply_axis_style(axis, x_grid=False, y_grid=False)


def set_panel_title(axis, title, subtitle=None):
    """Set a left-aligned title with an optional subtitle line."""
    if subtitle:
        axis.set_title(f"{title}\n{subtitle}", loc="left")
    else:
        axis.set_title(title, loc="left")


def add_frame_badge(axis, text, *, loc="upper left"):
    """Add a compact annotation badge inside an axis."""
    x = 0.02 if loc.endswith("left") else 0.98
    y = 0.98 if loc.startswith("upper") else 0.02
    ha = "left" if loc.endswith("left") else "right"
    va = "top" if loc.startswith("upper") else "bottom"
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha=ha,
        va=va,
        color=TEXT_COLOR,
        fontsize=11.5,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": LEGEND_EDGE_COLOR,
            "linewidth": 0.8,
            "alpha": 0.94,
        },
    )


def style_colorbar(colorbar, label=None):
    """Apply consistent styling to a colorbar."""
    if label is not None:
        colorbar.set_label(label, color=TEXT_COLOR)
    colorbar.ax.tick_params(colors=TEXT_COLOR, width=0.9, length=4)
    colorbar.outline.set_edgecolor(SPINE_COLOR)
    colorbar.outline.set_linewidth(0.8)


def finalize_legend(axis, **legend_kwargs):
    """Create a legend with the shared frame styling."""
    legend = axis.legend(**legend_kwargs)
    if legend is None:
        return None

    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE_COLOR)
    frame.set_linewidth(0.8)
    return legend

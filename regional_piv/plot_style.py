"""Shared presentation-style Matplotlib helpers for info-flow visuals."""

from contextlib import contextmanager

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


TEXT_COLOR = "#1F2937"
SPINE_COLOR = "#667085"
GRID_MAJOR_COLOR = "#D0D7E2"
GRID_MINOR_COLOR = "#E5EAF1"
PANEL_FACE_COLOR = "#FBFCFE"
LEGEND_EDGE_COLOR = "#D0D7E2"
TEXT_COLOR_DARK = "#F8FAFC"
SPINE_COLOR_DARK = "#E2E8F0"
GRID_MAJOR_COLOR_DARK = "#334155"
GRID_MINOR_COLOR_DARK = "#1E293B"
PANEL_FACE_COLOR_DARK = "#000000"
FIGURE_FACE_COLOR_DARK = "#000000"
LEGEND_EDGE_COLOR_DARK = "#475569"

FTLE_CMAP = "magma"
DIVERGENCE_CMAP = "RdBu_r"
SCALAR_OVERLAY_CMAP = "cividis"
COSINE_SIMILARITY_CMAP = DIVERGENCE_CMAP

_DIVERGENCE_MAP = plt.get_cmap(DIVERGENCE_CMAP)
CONVERGING_COLOR = _DIVERGENCE_MAP(0.14)
DIVERGING_COLOR = _DIVERGENCE_MAP(0.86)

FLOW_VECTOR_COLOR = "#F8FAFC"
INFO_VECTOR_COLOR = "#2DD4BF"
FLUID_LINE_COLOR = "#60A5FA"
INFO_LINE_COLOR = "#2DD4BF"
REFERENCE_LINE_COLOR = "#98A2B3"
RIDGE_COLOR = "#F8FAFC"

SINGLE_PANEL_FIGSIZE = (8.8, 6.2)
WIDE_PANEL_FIGSIZE = (13.4, 5.9)

PRESENTATION_RC_PARAMS = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
    "font.family": "lmodern",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "axes.facecolor": PANEL_FACE_COLOR,
    "axes.edgecolor": SPINE_COLOR,
    "axes.linewidth": 0.9,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 8.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": LEGEND_EDGE_COLOR,
    "legend.fancybox": False,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "grid.color": GRID_MAJOR_COLOR,
    "grid.linewidth": 0.75,
    "grid.alpha": 0.9,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PRESENTATION_DARK_RC_PARAMS = PRESENTATION_RC_PARAMS.copy()
PRESENTATION_DARK_RC_PARAMS.update(
    {
        "axes.labelcolor": TEXT_COLOR_DARK,
        "axes.titlecolor": TEXT_COLOR_DARK,
        "axes.facecolor": PANEL_FACE_COLOR_DARK,
        "axes.edgecolor": SPINE_COLOR_DARK,
        "figure.facecolor": FIGURE_FACE_COLOR_DARK,
        "savefig.facecolor": FIGURE_FACE_COLOR_DARK,
        "legend.edgecolor": LEGEND_EDGE_COLOR_DARK,
        "xtick.color": TEXT_COLOR_DARK,
        "ytick.color": TEXT_COLOR_DARK,
        "grid.color": GRID_MAJOR_COLOR_DARK,
    }
)


@contextmanager
def presentation_plot_context():
    """Temporarily apply the shared presentation plotting style."""
    with plt.rc_context(PRESENTATION_DARK_RC_PARAMS):
        yield


def apply_axis_style(axis, *, x_grid=False, y_grid=False):
    """Apply a clean academic presentation style to one axis."""
    axis.set_facecolor(PANEL_FACE_COLOR_DARK)
    axis.tick_params(width=1.0, colors=TEXT_COLOR_DARK)
    axis.minorticks_on()

    axis.grid(False)
    if x_grid:
        axis.xaxis.grid(True, which="major", color=GRID_MAJOR_COLOR_DARK, linewidth=0.85)
        axis.xaxis.grid(True, which="minor", color=GRID_MINOR_COLOR_DARK, linewidth=0.55, alpha=0.85)
    if y_grid:
        axis.yaxis.grid(True, which="major", color=GRID_MAJOR_COLOR_DARK, linewidth=0.85)
        axis.yaxis.grid(True, which="minor", color=GRID_MINOR_COLOR_DARK, linewidth=0.55, alpha=0.85)

    for spine_name in ("left", "bottom"):
        if spine_name in axis.spines:
            axis.spines[spine_name].set_color(SPINE_COLOR_DARK)
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
    """Set a centered title with an optional subtitle line."""
    if subtitle:
        axis.set_title(f"{title}\n{subtitle}", loc="center")
    else:
        axis.set_title(title, loc="center")


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
        color=TEXT_COLOR_DARK,
        fontsize=11.5,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#000000",
            "edgecolor": LEGEND_EDGE_COLOR_DARK,
            "linewidth": 0.8,
            "alpha": 0.94,
        },
    )


def style_colorbar(colorbar, label=None):
    """Apply consistent styling to a colorbar."""
    if label is not None:
        colorbar.set_label(label, color=TEXT_COLOR_DARK)
    colorbar.ax.tick_params(colors=TEXT_COLOR_DARK, width=0.9, length=4)
    colorbar.ax.set_facecolor(PANEL_FACE_COLOR_DARK)
    colorbar.outline.set_edgecolor(SPINE_COLOR_DARK)
    colorbar.outline.set_linewidth(0.8)


def add_spatial_colorbar(fig, axis, mappable, label=None, *, size="4.5%", pad=0.10):
    """Append a colorbar whose height matches the spatial plot axis."""
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size=size, pad=pad)
    colorbar = fig.colorbar(mappable, cax=cax)
    style_colorbar(colorbar, label)
    return colorbar


def finalize_legend(axis, **legend_kwargs):
    """Create a legend with the shared frame styling."""
    legend = axis.legend(**legend_kwargs)
    if legend is None:
        return None

    frame = legend.get_frame()
    frame.set_facecolor("#000000")
    frame.set_edgecolor(LEGEND_EDGE_COLOR_DARK)
    frame.set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR_DARK)
    title = legend.get_title()
    if title is not None:
        title.set_color(TEXT_COLOR_DARK)
    return legend

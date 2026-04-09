"""Shared publication-style plotting helpers for experiment figures."""

from contextlib import contextmanager

import matplotlib.pyplot as plt


TEXT_COLOR = "#1F2937"
SPINE_COLOR = "#667085"
GRID_MAJOR_COLOR = "#D0D7E2"
GRID_MINOR_COLOR = "#E5EAF1"
PANEL_FACE_COLOR = "#FBFCFE"
LEGEND_EDGE_COLOR = "#D0D7E2"

METHOD_COLORS = {
    "Static QR": "#264653",
    "Teleport QR": "#E76F51",
    "QR teleport": "#E76F51",
    "Lagrangian": "#2A9D8F",
    "Moving QR": "#4C78A8",
    "Moving POD-QR": "#4C78A8",
    "Fixed": "#8D99AE",
    "Eulerian": "#8D99AE",
}

DISPLAY_LABELS = {
    "Fixed": "Eulerian",
}

BASIS_COLORS = {
    "Global POD": "#355070",
    "Window POD": "#B56576",
}

BASIS_LINESTYLES = {
    "Global POD": "-",
    "Window POD": (0, (5, 1.8)),
}

BASIS_MARKERS = {
    "Global POD": "o",
    "Window POD": "s",
}

BASIS_HATCHES = {
    "Global POD": "",
    "Window POD": "////",
}

PAPER_RC_PARAMS = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
    "font.family": "lmodern",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
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
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
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


@contextmanager
def paper_plot_context():
    """Temporarily apply a paper-style Matplotlib configuration."""
    with plt.rc_context(PAPER_RC_PARAMS):
        yield


def pretty_flow_name(flow_name):
    """Format internal flow identifiers into title-style labels."""
    return str(flow_name).replace("_", " ").title()


def color_for_method(method_name):
    """Return a consistent color for one placement/method label."""
    return METHOD_COLORS.get(str(method_name), "#6B7280")


def display_label(label):
    """Return the publication-facing display label for a method/placement."""
    return DISPLAY_LABELS.get(str(label), str(label))


def color_for_basis(basis_name):
    """Return a consistent color for one basis label."""
    return BASIS_COLORS.get(str(basis_name), "#475467")


def linestyle_for_basis(basis_name):
    """Return the preferred line style for one basis label."""
    return BASIS_LINESTYLES.get(str(basis_name), "-")


def marker_for_basis(basis_name):
    """Return the preferred marker shape for one basis label."""
    return BASIS_MARKERS.get(str(basis_name), "o")


def hatch_for_basis(basis_name):
    """Return the preferred hatch pattern for one basis label."""
    return BASIS_HATCHES.get(str(basis_name), "")


def apply_axis_style(axis, *, x_grid=False, y_grid=True):
    """Apply polished academic-style axis cosmetics."""
    axis.set_facecolor(PANEL_FACE_COLOR)
    axis.tick_params(width=0.9, colors=TEXT_COLOR)
    axis.minorticks_on()

    axis.grid(False)
    if x_grid:
        axis.xaxis.grid(True, which="major", color=GRID_MAJOR_COLOR, linewidth=0.75)
        axis.xaxis.grid(True, which="minor", color=GRID_MINOR_COLOR, linewidth=0.5, alpha=0.8)
    if y_grid:
        axis.yaxis.grid(True, which="major", color=GRID_MAJOR_COLOR, linewidth=0.75)
        axis.yaxis.grid(True, which="minor", color=GRID_MINOR_COLOR, linewidth=0.5, alpha=0.8)

    for spine_name in ("left", "bottom"):
        if spine_name in axis.spines:
            axis.spines[spine_name].set_color(SPINE_COLOR)
            axis.spines[spine_name].set_linewidth(0.9)


def finalize_legend(axis, **legend_kwargs):
    """Create a polished legend and normalize its frame styling."""
    legend = axis.legend(**legend_kwargs)
    if legend is None:
        return None

    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE_COLOR)
    frame.set_linewidth(0.8)
    return legend

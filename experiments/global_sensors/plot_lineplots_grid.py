"""Plot a 2x2 grid of global-sensor L2_h line plots from saved CSV results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import pandas as pd

from experiments.common.plot_style import (
    apply_axis_style,
    color_for_method,
    display_label,
    marker_for_basis,
    paper_plot_context,
    pretty_flow_name,
)


METRIC_COLUMN = "L2_h"
METRIC_LABEL = r"Relative $L_h^2$ Error"

DEFAULT_CSV_PATH = REPO_ROOT / "experiments/global_sensors/artifacts/results/aggregated_mean_l2h.csv"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "experiments/global_sensors/artifacts/plots/all_flows_l2h_vs_sensors_2x2.png"
)

FLOW_ORDER = ["double_gyre", "moving_vortex", "kolmogorov", "cylinder_wake"]
COMBO_ORDER = [
    "Moving QR | Window POD",
    "Static QR | Window POD",
    "Teleport QR | Window POD",
    "Lagrangian | Window POD",
]
FIGSIZE = (17.0, 11.0)
FONT_SCALE = 2.8125
SCALED_TEXT_RC_PARAMS = {
    "font.size": 11 * FONT_SCALE,
    "axes.titlesize": 11 * FONT_SCALE,
    "axes.labelsize": 11 * FONT_SCALE,
    "legend.fontsize": 9 * FONT_SCALE,
    "legend.title_fontsize": 9 * FONT_SCALE,
    "xtick.labelsize": 9 * FONT_SCALE,
    "ytick.labelsize": 9 * FONT_SCALE,
}
PNG_TEXT_RC_PARAMS = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
}
CONSTRAINED_LAYOUT_PADS = {
    "w_pad": 0.07,
    "h_pad": 0.09,
    "wspace": 0.13,
    "hspace": 0.18,
}


def _ordered_available(values, preferred_order):
    """Return preferred values that exist, followed by any remaining values."""
    available = list(pd.Series(values).dropna().unique())
    available_set = set(available)
    ordered_values = [value for value in preferred_order if value in available_set]
    ordered_values.extend(sorted(value for value in available if value not in set(ordered_values)))
    return ordered_values


def _split_combo_label(combo_label):
    """Split a combined series label into method and basis names."""
    combo_label = str(combo_label)
    if " | " not in combo_label:
        return combo_label, None
    return combo_label.split(" | ", 1)


def _display_combo_label(combo_label, method_counts):
    """Format a combined method/basis label for the side legend."""
    method_name, basis_name = _split_combo_label(combo_label)
    if basis_name is None or method_counts.get(method_name, 0) <= 1:
        return display_label(method_name)
    return f"{display_label(method_name)} ({basis_name})"


def _read_aggregated_l2h(csv_path):
    """Read global-sensor CSV records and aggregate to one value per line-plot point."""
    results_df = pd.read_csv(csv_path)
    required_columns = {"flow", "num_sensors", "basis", "method", METRIC_COLUMN}
    missing_columns = required_columns.difference(results_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

    results_df = results_df.copy()
    results_df["num_sensors"] = pd.to_numeric(results_df["num_sensors"], errors="coerce")
    results_df[METRIC_COLUMN] = pd.to_numeric(results_df[METRIC_COLUMN], errors="coerce")
    results_df = results_df.dropna(subset=["flow", "num_sensors", "basis", "method", METRIC_COLUMN])
    results_df = results_df[results_df[METRIC_COLUMN] > 0.0]

    aggregated_df = results_df.groupby(
        ["flow", "num_sensors", "basis", "method"],
        as_index=False,
    )[METRIC_COLUMN].mean()
    aggregated_df["num_sensors"] = aggregated_df["num_sensors"].astype(int)

    if aggregated_df.empty:
        raise ValueError(f"{csv_path} does not contain positive {METRIC_COLUMN} values to plot")
    return aggregated_df


def _build_flow_pivot(aggregated_df, flow_name, combo_order):
    """Build a num-sensors indexed pivot table for one flow."""
    flow_df = aggregated_df[aggregated_df["flow"] == flow_name].copy()
    flow_df["combo"] = flow_df["method"] + " | " + flow_df["basis"]
    pivot_df = (
        flow_df.pivot_table(
            index="num_sensors",
            columns="combo",
            values=METRIC_COLUMN,
            aggfunc="mean",
        )
        .sort_index()
        .dropna(axis=1, how="all")
    )

    if combo_order is not None:
        ordered_columns = [combo_label for combo_label in combo_order if combo_label in pivot_df.columns]
        pivot_df = pivot_df[ordered_columns]
    return pivot_df


def save_lineplot_grid(csv_path, output_path, flow_order=None, combo_order=None):
    """Read global-sensor records and save a 2x2 line-plot figure by flow."""
    flow_order = FLOW_ORDER if flow_order is None else flow_order

    aggregated_df = _read_aggregated_l2h(csv_path)
    flow_order_use = _ordered_available(aggregated_df["flow"], flow_order)
    if len(flow_order_use) > 4:
        omitted_flows = ", ".join(flow_order_use[4:])
        print(f"Only the first four flows fit in a 2x2 grid; omitting: {omitted_flows}")
        flow_order_use = flow_order_use[:4]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    legend_combos = []
    with paper_plot_context(), plt.rc_context(PNG_TEXT_RC_PARAMS), plt.rc_context(SCALED_TEXT_RC_PARAMS):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
        fig.set_constrained_layout_pads(**CONSTRAINED_LAYOUT_PADS)
        axes_flat = axes.ravel()

        for axis, flow_name in zip(axes_flat, flow_order_use):
            pivot_df = _build_flow_pivot(aggregated_df, flow_name, combo_order)

            if pivot_df.empty:
                axis.set_axis_off()
                continue

            for combo_label in pivot_df.columns:
                method_name, basis_name = _split_combo_label(combo_label)
                values = pivot_df[combo_label].dropna()
                if values.empty:
                    continue

                axis.plot(
                    values.index,
                    values.to_numpy(),
                    color=color_for_method(method_name),
                    linestyle="-",
                    marker=marker_for_basis(basis_name),
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    linewidth=2.1,
                    markersize=5.4,
                )
                legend_combos.append(combo_label)

            axis.set_yscale("log")
            axis.xaxis.set_major_locator(FixedLocator(list(pivot_df.index)))
            axis.set_title(pretty_flow_name(flow_name))
            axis.set_xlabel("Number of Sensors")
            axis.set_ylabel(METRIC_LABEL)
            axis.margins(x=0.04)
            apply_axis_style(axis, x_grid=True, y_grid=True)

        for axis in axes_flat[len(flow_order_use) :]:
            axis.set_axis_off()

        ordered_legend_combos = []
        if combo_order is not None:
            ordered_legend_combos.extend(combo for combo in combo_order if combo in set(legend_combos))
        ordered_legend_combos.extend(
            combo for combo in sorted(set(legend_combos)) if combo not in set(ordered_legend_combos)
        )

        method_counts = {}
        for combo_label in ordered_legend_combos:
            method_name, _basis_name = _split_combo_label(combo_label)
            method_counts[method_name] = method_counts.get(method_name, 0) + 1

        legend_handles = []
        for combo_label in ordered_legend_combos:
            method_name, basis_name = _split_combo_label(combo_label)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color_for_method(method_name),
                    linestyle="-",
                    marker=marker_for_basis(basis_name),
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    linewidth=2.1,
                    markersize=5.4,
                    label=_display_combo_label(combo_label, method_counts),
                )
            )

        legend = fig.legend(
            handles=legend_handles,
            title="Method",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=1,
            borderaxespad=0.0,
        )
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("#D0D7E2")
            frame.set_linewidth(0.8)

        fig.savefig(output_path)
        plt.close(fig)

    return output_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a 2x2 relative L2_h line-plot grid from global-sensor records.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to aggregated_mean_l2h.csv or raw_window_records.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output figure path.",
    )
    parser.add_argument(
        "--flow-order",
        nargs="+",
        default=FLOW_ORDER,
        help="Flow order for the 2x2 subplots.",
    )
    parser.add_argument(
        "--combo-order",
        nargs="+",
        default=COMBO_ORDER,
        help='Series order, as quoted "method | basis" labels.',
    )
    parser.add_argument(
        "--all-combos",
        action="store_true",
        help="Plot every available method/basis combination instead of the default Moving QR Window POD series.",
    )
    return parser.parse_args()


def main():
    """Run the standalone plotter."""
    args = parse_args()
    combo_order = None if args.all_combos else args.combo_order
    output_path = save_lineplot_grid(
        csv_path=args.csv,
        output_path=args.out,
        flow_order=args.flow_order,
        combo_order=combo_order,
    )
    print(f"Saved 2x2 line-plot grid to {output_path}")


if __name__ == "__main__":
    main()

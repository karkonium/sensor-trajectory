"""Plot a 2x2 grid of random-trial L2_h boxplots from saved raw records."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import pandas as pd

from experiments.common.plot_style import (
    apply_axis_style,
    color_for_method,
    display_label,
    paper_plot_context,
    pretty_flow_name,
)


METRIC_COLUMN = "L2_h"
METRIC_LABEL = r"Relative $L_h^2$ Error"

DEFAULT_CSV_PATH = REPO_ROOT / "experiments/random_trials/artifacts/results/raw_window_records.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "experiments/random_trials/artifacts/plots/all_flows_l2h_boxplots_2x2.png"

FLOW_ORDER = ["double_gyre", "moving_vortex", "kolmogorov", "cylinder_wake"]
PLACEMENT_ORDER = ["Fixed", "Lagrangian", "Moving POD-QR", "QR teleport"]
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
CONSTRAINED_LAYOUT_PADS = {
    "w_pad": 0.07,
    "h_pad": 0.09,
    "wspace": 0.12,
    "hspace": 0.17,
}
BOX_WIDTH = 0.62
BOX_LOG_PADDING = 0.015
MIN_BOX_LOG_SPAN = 0.045
MIN_BOX_LOG_SPAN_FRACTION = 0.02
NARROW_AXIS_LOG_SPAN = 2.5
NARROW_AXIS_TICK_MULTIPLIERS = (1, 2, 3, 5)


def _ordered_available(values, preferred_order):
    """Return preferred values that exist, followed by any remaining values."""
    available = list(pd.Series(values).dropna().unique())
    available_set = set(available)
    ordered_values = [value for value in preferred_order if value in available_set]
    ordered_values.extend(sorted(value for value in available if value not in set(ordered_values)))
    return ordered_values


def _build_boxplot_data(flow_df, placement_order):
    """Collect trial-mean boxplot arrays and labels for one flow."""
    placement_order_use = _ordered_available(flow_df["placement"], placement_order)
    series_data = []
    placement_names = []
    placement_labels = []

    for placement_name in placement_order_use:
        values = pd.to_numeric(
            flow_df.loc[flow_df["placement"] == placement_name, METRIC_COLUMN],
            errors="coerce",
        ).dropna()
        positive_values = values[values > 0.0]

        if positive_values.empty:
            continue

        series_data.append(positive_values.to_numpy())
        placement_names.append(placement_name)
        placement_labels.append(display_label(placement_name))

    return series_data, placement_names, placement_labels


def _pad_box_patch(box_patch, values, min_log_span):
    """Add a small visual x-padding to a horizontal log-scale box patch."""
    values = pd.Series(values).dropna()
    values = values[values > 0.0]
    if values.empty:
        return

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    if q1 <= 0.0 or q3 <= 0.0:
        return

    log_q1 = math.log10(q1)
    log_q3 = math.log10(q3)
    center = 0.5 * (log_q1 + log_q3)
    padded_low = log_q1 - BOX_LOG_PADDING
    padded_high = log_q3 + BOX_LOG_PADDING
    if padded_high - padded_low < min_log_span:
        padded_low = center - 0.5 * min_log_span
        padded_high = center + 0.5 * min_log_span

    vertices = box_patch.get_path().vertices
    if len(vertices) < 5:
        return

    vertices[:2, 0] = 10.0**padded_low
    vertices[2:4, 0] = 10.0**padded_high
    vertices[4:, 0] = 10.0**padded_low


def _format_log_tick(value, _position):
    """Format one positive log tick using compact powers of ten."""
    if value <= 0.0:
        return ""

    exponent = math.floor(math.log10(value))
    coefficient = value / (10.0**exponent)
    rounded_coefficient = round(coefficient)
    if math.isclose(coefficient, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        return rf"$10^{{{exponent}}}$"
    if math.isclose(coefficient, rounded_coefficient, rel_tol=1e-6, abs_tol=1e-8):
        return rf"${rounded_coefficient}\times 10^{{{exponent}}}$"
    return rf"${value:.1e}$"


def _set_labeled_log_ticks(axis, flow_name):
    """Set explicit x-axis log ticks so each visible tick has a label."""
    x_min, x_max = axis.get_xlim()
    if x_min <= 0.0 or x_max <= 0.0:
        return

    log_min = math.floor(math.log10(x_min))
    log_max = math.ceil(math.log10(x_max))
    log_span = math.log10(x_max) - math.log10(x_min)

    multipliers = NARROW_AXIS_TICK_MULTIPLIERS if log_span <= NARROW_AXIS_LOG_SPAN else (1,)
    exponent_step = 2 if flow_name == "double_gyre" and log_span > NARROW_AXIS_LOG_SPAN else 1
    tick_values = []
    for exponent in range(log_min, log_max + 1, exponent_step):
        for multiplier in multipliers:
            tick_value = multiplier * (10.0**exponent)
            if x_min <= tick_value <= x_max:
                tick_values.append(tick_value)

    axis.xaxis.set_major_locator(FixedLocator(tick_values))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    axis.xaxis.set_minor_locator(NullLocator())


def save_boxplot_grid(csv_path, output_path, flow_order=None, placement_order=None):
    """Read raw random-trial records and save the requested 2x2 boxplot figure."""
    flow_order = FLOW_ORDER if flow_order is None else flow_order
    placement_order = PLACEMENT_ORDER if placement_order is None else placement_order

    raw_df = pd.read_csv(csv_path)
    required_columns = {"flow", "placement", "trial", METRIC_COLUMN}
    missing_columns = required_columns.difference(raw_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

    group_columns = ["flow", "placement", "trial"]
    if "num_sensors" in raw_df.columns:
        group_columns.insert(2, "num_sensors")
    raw_df = raw_df.groupby(group_columns, as_index=False)[METRIC_COLUMN].mean()

    flow_order_use = _ordered_available(raw_df["flow"], flow_order)
    if len(flow_order_use) > 4:
        omitted_flows = ", ".join(flow_order_use[4:])
        print(f"Only the first four flows fit in a 2x2 grid; omitting: {omitted_flows}")
        flow_order_use = flow_order_use[:4]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    legend_names = []
    with paper_plot_context(), plt.rc_context(SCALED_TEXT_RC_PARAMS):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
        fig.set_constrained_layout_pads(**CONSTRAINED_LAYOUT_PADS)
        axes_flat = axes.ravel()

        for axis, flow_name in zip(axes_flat, flow_order_use):
            flow_df = raw_df[raw_df["flow"] == flow_name]
            series_data, placement_names, placement_labels = _build_boxplot_data(
                flow_df,
                placement_order,
            )

            if not series_data:
                axis.set_axis_off()
                continue

            flow_values = pd.concat([pd.Series(values) for values in series_data])
            flow_values = flow_values[flow_values > 0.0]
            flow_log_span = math.log10(flow_values.max()) - math.log10(flow_values.min())
            min_box_log_span = max(MIN_BOX_LOG_SPAN, MIN_BOX_LOG_SPAN_FRACTION * flow_log_span)

            boxplot = axis.boxplot(
                series_data,
                labels=placement_labels,
                vert=False,
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=True,
                whis=[5, 95],
                medianprops={"color": "#111827", "linewidth": 1.5},
                whiskerprops={"color": "#475467", "linewidth": 1.0},
                capprops={"color": "#475467", "linewidth": 1.0},
                flierprops={
                    "marker": "o",
                    "markersize": 3.8,
                    "markerfacecolor": "#98A2B3",
                    "markeredgecolor": "#98A2B3",
                    "alpha": 0.55,
                },
            )

            for patch, placement_name, values in zip(boxplot["boxes"], placement_names, series_data):
                _pad_box_patch(patch, values, min_box_log_span)
                patch.set_facecolor(color_for_method(placement_name))
                patch.set_alpha(0.62)
                patch.set_edgecolor("#344054")
                patch.set_linewidth(0.9)

            axis.set_xscale("log")
            axis.set_title(pretty_flow_name(flow_name))
            axis.set_xlabel(METRIC_LABEL)
            apply_axis_style(axis, x_grid=True, y_grid=False)
            _set_labeled_log_ticks(axis, flow_name)
            axis.tick_params(axis="y", labelleft=False)
            legend_names.extend(placement_names)

        for axis in axes_flat[len(flow_order_use) :]:
            axis.set_axis_off()

        ordered_legend_names = [
            placement_name for placement_name in placement_order if placement_name in set(legend_names)
        ]
        ordered_legend_names.extend(
            sorted(
                placement_name
                for placement_name in set(legend_names)
                if placement_name not in set(ordered_legend_names)
            )
        )
        legend_handles = [
            Patch(
                facecolor=color_for_method(placement_name),
                edgecolor="#344054",
                alpha=0.62,
                label=display_label(placement_name),
            )
            for placement_name in ordered_legend_names
        ]
        legend = fig.legend(
            handles=legend_handles,
            title="Path Planning Method",
            loc="outside upper right",
            ncol=2,
            borderaxespad=0.25,
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
        description="Create a 2x2 path-planning-method boxplot grid from trial-mean random-trial records.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to raw_window_records.csv.",
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
        "--placement-order",
        nargs="+",
        default=PLACEMENT_ORDER,
        help="Path planning method order within each boxplot.",
    )
    return parser.parse_args()


def main():
    """Run the standalone plotter."""
    args = parse_args()
    output_path = save_boxplot_grid(
        csv_path=args.csv,
        output_path=args.out,
        flow_order=args.flow_order,
        placement_order=args.placement_order,
    )
    print(f"Saved 2x2 boxplot grid to {output_path}")


if __name__ == "__main__":
    main()

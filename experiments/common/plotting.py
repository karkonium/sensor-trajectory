"""Shared plotting helpers for experiment modules."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .plot_style import (
    apply_axis_style,
    color_for_basis,
    color_for_method,
    display_label,
    finalize_legend,
    hatch_for_basis,
    linestyle_for_basis,
    marker_for_basis,
    paper_plot_context,
    pretty_flow_name,
)


METRIC_COLUMN = "L2_h"
METRIC_LABEL = r"Relative $L_h^2$ Error"


def _split_combo_label(combo_label):
    """Split a combined series label into method and basis names."""
    combo_label = str(combo_label)
    if " | " not in combo_label:
        return combo_label, None
    return combo_label.split(" | ", 1)


def _display_combo_label(combo_label):
    """Format a combined method/basis label for plot display."""
    method_name, basis_name = _split_combo_label(combo_label)
    if basis_name is None:
        return display_label(method_name)
    return f"{display_label(method_name)} | {basis_name}"


def plot_grouped_l2h_summary(results, label_key=None):
    """Print and plot mean relative L2_h error grouped by label and basis.

    Args:
        results: DataFrame with L2_h records.
        label_key: Optional grouping key. Auto-detected when omitted.

    Returns:
        Grouped mean series.
    """
    if label_key is None:
        if "placement" in results.columns:
            label_key = "placement"
        elif "method" in results.columns:
            label_key = "method"
        else:
            raise ValueError("results must include 'placement' or 'method' column")

    if "basis" in results.columns:
        grouped_means = results.groupby([label_key, "basis"])[METRIC_COLUMN].mean().sort_values()
        print(grouped_means)

        plot_df = grouped_means.unstack("basis")
        plot_df.index = [display_label(index_label) for index_label in plot_df.index]
        with paper_plot_context():
            fig, axis = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
            plot_df.plot(
                kind="barh",
                logx=True,
                ax=axis,
                color=[color_for_basis(column) for column in plot_df.columns],
                width=0.74,
                edgecolor="#344054",
                linewidth=0.6,
            )

            for container, basis_name in zip(axis.containers, plot_df.columns):
                hatch = hatch_for_basis(basis_name)
                for patch in container.patches:
                    patch.set_hatch(hatch)
                    patch.set_alpha(0.92)

            axis.set_xlabel(f"Mean {METRIC_LABEL}")
            axis.set_ylabel(label_key.replace("_", " ").title())
            axis.set_title(f"Mean {METRIC_LABEL} by {label_key.replace('_', ' ').title()} and Basis")
            apply_axis_style(axis, x_grid=True, y_grid=False)
            finalize_legend(axis, title="Basis", loc="lower right")
            plt.show()
        return grouped_means

    grouped_means = results.groupby([label_key])[METRIC_COLUMN].mean().sort_values()
    grouped_means.index = [display_label(index_label) for index_label in grouped_means.index]
    print(grouped_means)

    with paper_plot_context():
        fig, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
        grouped_means.plot(
            kind="barh",
            logx=True,
            ax=axis,
            color=[color_for_method(label) for label in grouped_means.index],
            width=0.72,
            edgecolor="#344054",
            linewidth=0.6,
        )
        axis.set_xlabel(f"Mean {METRIC_LABEL}")
        axis.set_ylabel(label_key.replace("_", " ").title())
        axis.set_title(f"Mean {METRIC_LABEL} by {label_key.replace('_', ' ').title()}")
        apply_axis_style(axis, x_grid=True, y_grid=False)
        plt.show()
    return grouped_means


def plot_l2h_timeseries(results, label_key=None):
    """Plot relative L2_h error curves per method/placement and basis.

    Args:
        results: DataFrame with at least window and L2_h columns.
        label_key: Optional label column. Auto-detected when omitted.

    Returns:
        Pivoted DataFrame used for plotting.
    """
    if label_key is None:
        if "placement" in results.columns:
            label_key = "placement"
        elif "method" in results.columns:
            label_key = "method"
        else:
            raise ValueError("results must include 'placement' or 'method' column")

    results_df = results.copy()
    if "basis" in results_df.columns:
        results_df["combo"] = results_df[label_key] + " | " + results_df["basis"]
    else:
        results_df["combo"] = results_df[label_key]

    pivot_df = results_df.pivot(index="window", columns="combo", values=METRIC_COLUMN)

    with paper_plot_context():
        fig, axis = plt.subplots(figsize=(10.4, 5.4), constrained_layout=True)
        for combo_label in pivot_df.columns:
            method_name, basis_name = _split_combo_label(combo_label)
            display_combo_label = _display_combo_label(combo_label)
            axis.plot(
                pivot_df.index,
                pivot_df[combo_label],
                label=display_combo_label,
                color=color_for_method(method_name),
                linestyle=linestyle_for_basis(basis_name),
                marker=marker_for_basis(basis_name),
                markerfacecolor="white",
                markeredgewidth=1.1,
                linewidth=2.1,
                markersize=5.2,
            )

        axis.set_yscale("log")
        axis.set_xlabel("Window")
        axis.set_ylabel(METRIC_LABEL)
        axis.set_title(f"{METRIC_LABEL} Over Window")
        apply_axis_style(axis, x_grid=True, y_grid=True)
        legend_cols = 1 if len(pivot_df.columns) <= 4 else 2
        finalize_legend(axis, loc="upper right", ncol=legend_cols)
        plt.show()
    return pivot_df


def save_mean_l2h_vs_sensor_count(aggregated_df, output_dir):
    """Save per-flow line plots of mean relative L2_h error vs sensor count.

    Args:
        aggregated_df: DataFrame containing flow, num_sensors, method, basis, L2_h.
        output_dir: Directory path for plot files.

    Returns:
        None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for flow_name in sorted(aggregated_df["flow"].unique()):
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
        )

        with paper_plot_context():
            fig, axis = plt.subplots(figsize=(10.4, 5.4), constrained_layout=True)
            for combo_label in pivot_df.columns:
                method_name, basis_name = _split_combo_label(combo_label)
                display_combo_label = _display_combo_label(combo_label)
                axis.plot(
                    pivot_df.index,
                    pivot_df[combo_label],
                    label=display_combo_label,
                    color=color_for_method(method_name),
                    linestyle=linestyle_for_basis(basis_name),
                    marker=marker_for_basis(basis_name),
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    linewidth=2.1,
                    markersize=5.4,
                )

            axis.set_yscale("log")
            axis.set_xlabel("Number of Sensors")
            axis.set_ylabel(f"Mean {METRIC_LABEL}")
            axis.set_title(f"{pretty_flow_name(flow_name)}: Mean {METRIC_LABEL} vs Sensor Count")
            apply_axis_style(axis, x_grid=True, y_grid=True)
            finalize_legend(axis, loc="upper right", ncol=2)
            fig.savefig(output_dir / f"{flow_name}_mean_l2h_vs_sensors.png")
            plt.close(fig)


def save_grouped_barh_by_flow(aggregated_df, output_dir, method_order=None, basis_order=None):
    """Save per-flow grouped horizontal bar charts by method and basis.

    Args:
        aggregated_df: DataFrame containing flow, method, basis, L2_h.
        output_dir: Directory path for plot files.
        method_order: Optional ordered list for method axis.
        basis_order: Optional ordered list for basis columns.

    Returns:
        None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for flow_name in sorted(aggregated_df["flow"].unique()):
        flow_df = aggregated_df[aggregated_df["flow"] == flow_name].copy()

        pivot_df = flow_df.pivot_table(
            index="method",
            columns="basis",
            values=METRIC_COLUMN,
            aggfunc="mean",
        )
        pivot_df.index = [display_label(index_label) for index_label in pivot_df.index]

        if method_order is not None:
            pivot_df = pivot_df.reindex([display_label(method_name) for method_name in method_order])

        if basis_order is not None:
            ordered_basis_cols = [basis_name for basis_name in basis_order if basis_name in pivot_df.columns]
            if ordered_basis_cols:
                pivot_df = pivot_df[ordered_basis_cols]

        with paper_plot_context():
            fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
            pivot_df.plot(
                kind="barh",
                logx=True,
                ax=axis,
                color=[color_for_basis(column) for column in pivot_df.columns],
                width=0.74,
                edgecolor="#344054",
                linewidth=0.6,
            )

            for container, basis_name in zip(axis.containers, pivot_df.columns):
                hatch = hatch_for_basis(basis_name)
                for patch in container.patches:
                    patch.set_hatch(hatch)
                    patch.set_alpha(0.92)

            axis.set_xlabel(f"Mean {METRIC_LABEL}")
            axis.set_ylabel("Method")
            axis.set_title(f"{pretty_flow_name(flow_name)}: Mean {METRIC_LABEL}")
            apply_axis_style(axis, x_grid=True, y_grid=False)
            finalize_legend(axis, title="Basis", loc="lower right")
            fig.savefig(output_dir / f"{flow_name}_mean_l2h_barh.png")
            plt.close(fig)


def save_boxplots_per_flow(raw_df, output_dir, placement_order=None):
    """Save one horizontal relative L2_h error boxplot per flow.

    Args:
        raw_df: DataFrame with flow, placement, L2_h columns.
        output_dir: Directory path for plot files.
        placement_order: Optional ordered list for placement axis.

    Returns:
        None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for flow_name in sorted(raw_df["flow"].unique()):
        flow_df = raw_df[raw_df["flow"] == flow_name].copy()

        if placement_order is not None:
            flow_df["placement"] = pd.Categorical(
                flow_df["placement"],
                categories=placement_order,
                ordered=True,
            )
            flow_df = flow_df.sort_values("placement")

        placement_order_use = list(flow_df["placement"].dropna().unique())
        series_data = []
        placement_labels = []
        for placement_name in placement_order_use:
            values = flow_df.loc[flow_df["placement"] == placement_name, METRIC_COLUMN].dropna()
            if not values.empty:
                placement_labels.append(display_label(placement_name))
                series_data.append(values.to_numpy())

        with paper_plot_context():
            fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
            boxplot = axis.boxplot(
                series_data,
                labels=placement_labels,
                vert=False,
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

            for patch, placement_name in zip(boxplot["boxes"], placement_labels):
                patch.set_facecolor(color_for_method(placement_name))
                patch.set_alpha(0.62)
                patch.set_edgecolor("#344054")
                patch.set_linewidth(0.9)

            axis.set_xscale("log")
            axis.set_xlabel(METRIC_LABEL)
            axis.set_ylabel("Placement")
            axis.set_title(f"{pretty_flow_name(flow_name)}: {METRIC_LABEL} Distribution")
            apply_axis_style(axis, x_grid=True, y_grid=False)
            fig.savefig(output_dir / f"{flow_name}_l2h_boxplot.png")
            plt.close(fig)


def plot_grouped_rmse_summary(results, label_key=None):
    """Backward-compatible wrapper for plot_grouped_l2h_summary."""
    return plot_grouped_l2h_summary(results, label_key=label_key)


def plot_rmse_timeseries(results, label_key=None):
    """Backward-compatible wrapper for plot_l2h_timeseries."""
    return plot_l2h_timeseries(results, label_key=label_key)


def save_mean_rmse_vs_sensor_count(aggregated_df, output_dir):
    """Backward-compatible wrapper for save_mean_l2h_vs_sensor_count."""
    return save_mean_l2h_vs_sensor_count(aggregated_df, output_dir)

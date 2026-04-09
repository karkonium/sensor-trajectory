"""Shared plotting helpers for experiment modules."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMN = "L2_h"
METRIC_LABEL = "Relative L2_h Error"


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

        plt.figure(figsize=(10, 5))
        grouped_means.unstack("basis").plot(kind="barh", logx=True)
        plt.xlabel(f"mean {METRIC_LABEL}")
        plt.tight_layout()
        plt.show()
        return grouped_means

    grouped_means = results.groupby([label_key])[METRIC_COLUMN].mean().sort_values()
    print(grouped_means)
    plt.figure(figsize=(8, 4))
    grouped_means.plot(kind="barh", logx=True)
    plt.xlabel(f"mean {METRIC_LABEL}")
    plt.tight_layout()
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

    plt.figure(figsize=(10, 5))
    axis = plt.gca()
    pivot_df.plot(marker="o", linewidth=1.5, ax=axis)

    axis.set_yscale("log")
    axis.set_xlabel("window")
    axis.set_ylabel(f"{METRIC_LABEL} (log scale)")
    axis.set_title(f"{METRIC_LABEL} Over Window (log scale)")
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))

    plt.tight_layout()
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

        fig, axis = plt.subplots(figsize=(10, 5))
        pivot_df.plot(marker="o", linewidth=1.5, ax=axis)
        axis.set_yscale("log")
        axis.set_xlabel("number of sensors")
        axis.set_ylabel(f"mean {METRIC_LABEL} (log scale)")
        axis.set_title(f"Mean {METRIC_LABEL} vs Sensors: {flow_name}")
        axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
        fig.tight_layout()
        fig.savefig(output_dir / f"{flow_name}_mean_l2h_vs_sensors.png", dpi=150)
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

        if method_order is not None:
            pivot_df = pivot_df.reindex(method_order)

        if basis_order is not None:
            ordered_basis_cols = [basis_name for basis_name in basis_order if basis_name in pivot_df.columns]
            if ordered_basis_cols:
                pivot_df = pivot_df[ordered_basis_cols]

        fig, axis = plt.subplots(figsize=(9, 4.5))
        pivot_df.plot(kind="barh", logx=True, ax=axis)
        axis.set_xlabel(f"mean {METRIC_LABEL}")
        axis.set_ylabel("method")
        axis.set_title(f"Mean {METRIC_LABEL} by Method/Basis: {flow_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{flow_name}_mean_l2h_barh.png", dpi=150)
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

        fig, axis = plt.subplots(figsize=(9, 4.5))
        flow_df.boxplot(
            column=METRIC_COLUMN,
            by="placement",
            vert=False,
            grid=False,
            showfliers=True,
            whis=[5, 95],
            ax=axis,
        )

        fig.suptitle("")
        axis.set_title(f"{METRIC_LABEL} Distribution Across Windows and Trials: {flow_name}")
        axis.set_xlabel(METRIC_LABEL)
        axis.set_ylabel("placement")
        axis.set_xscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / f"{flow_name}_l2h_boxplot.png", dpi=150)
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

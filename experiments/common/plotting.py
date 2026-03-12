"""Shared plotting helpers for experiment variants."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_grouped_rmse_summary(results, label_key=None):
    """Print and plot mean RMSE grouped by label and basis when available.

    Args:
        results: DataFrame with RMSE records.
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
        grouped_means = results.groupby([label_key, "basis"])["RMSE"].mean().sort_values()
        print(grouped_means)

        plt.figure(figsize=(10, 5))
        grouped_means.unstack("basis").plot(kind="barh", logx=True)
        plt.xlabel("mean RMSE")
        plt.tight_layout()
        plt.show()
        return grouped_means

    grouped_means = results.groupby([label_key])["RMSE"].mean().sort_values()
    print(grouped_means)
    plt.figure(figsize=(8, 4))
    grouped_means.plot(kind="barh", logx=True)
    plt.xlabel("mean RMSE")
    plt.tight_layout()
    plt.show()
    return grouped_means


def plot_rmse_timeseries(results, label_key=None):
    """Plot RMSE curves per method/placement and basis on log y-axis.

    Args:
        results: DataFrame with at least window and RMSE columns.
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

    pivot_df = results_df.pivot(index="window", columns="combo", values="RMSE")

    plt.figure(figsize=(10, 5))
    axis = plt.gca()
    pivot_df.plot(marker="o", linewidth=1.5, ax=axis)

    axis.set_yscale("log")
    axis.set_xlabel("window")
    axis.set_ylabel("RMSE (log scale)")
    axis.set_title("RMSE over window (log scale)")
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))

    plt.tight_layout()
    plt.show()
    return pivot_df


def save_mean_rmse_vs_sensor_count(aggregated_df, output_dir):
    """Save per-flow line plots of mean RMSE vs sensor count.

    Args:
        aggregated_df: DataFrame containing flow, num_sensors, method, basis, RMSE.
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
                values="RMSE",
                aggfunc="mean",
            )
            .sort_index()
        )

        fig, axis = plt.subplots(figsize=(10, 5))
        pivot_df.plot(marker="o", linewidth=1.5, ax=axis)
        axis.set_yscale("log")
        axis.set_xlabel("number of sensors")
        axis.set_ylabel("mean RMSE (log scale)")
        axis.set_title(f"Mean RMSE vs sensors: {flow_name}")
        axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
        fig.tight_layout()
        fig.savefig(output_dir / f"{flow_name}_mean_rmse_vs_sensors.png", dpi=150)
        plt.close(fig)


def save_grouped_barh_by_flow(aggregated_df, output_dir, method_order=None, basis_order=None):
    """Save per-flow grouped horizontal bar charts by method and basis.

    Args:
        aggregated_df: DataFrame containing flow, method, basis, RMSE.
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
            values="RMSE",
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
        axis.set_xlabel("mean RMSE")
        axis.set_ylabel("method")
        axis.set_title(f"Mean RMSE by method/basis: {flow_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{flow_name}_mean_rmse_barh.png", dpi=150)
        plt.close(fig)


def save_boxplots_per_flow(raw_df, output_dir, placement_order=None):
    """Save one horizontal RMSE boxplot per flow.

    Args:
        raw_df: DataFrame with flow, placement, RMSE columns.
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
            column="RMSE",
            by="placement",
            vert=False,
            grid=False,
            showfliers=True,
            whis=[5, 95],
            ax=axis,
        )

        fig.suptitle("")
        axis.set_title(f"RMSE distribution across windows and trials: {flow_name}")
        axis.set_xlabel("RMSE")
        axis.set_ylabel("placement")
        axis.set_xscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / f"{flow_name}_rmse_boxplot.png", dpi=150)
        plt.close(fig)

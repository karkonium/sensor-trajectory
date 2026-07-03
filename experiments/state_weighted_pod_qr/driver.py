"""Driver for the state-weighted POD-QR sensor-selection experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.config import ExperimentConfig
from experiments.common.flow_cases import generate_standard_flow_cases
from experiments.common.paths import build_artifact_paths, ensure_artifact_dirs
from experiments.state_weighted_pod_qr.pipeline import run_state_weighted_pod_qr
from experiments.state_weighted_pod_qr.plotting import (
    make_gif_from_frames,
    save_condition_over_time_plot,
    save_flow_diagnostic_frames,
    save_flow_diagnostic_plot,
    save_l2h_boxplot,
    save_progressive_error_plots,
    save_sensor_overlap_plot,
)


TOTAL_STEPS = 200
PERIOD = 80
FLOW_NAMES = ["double_gyre", "moving_vortex", "kolmogorov", "cylinder_wake"]

NUM_SENSORS = 10
MAX_BASIS_DIM = 10
SEED = 90
WEIGHT_EPS = 1e-8

EVAL_START = 0
EVAL_END = None
EVAL_STRIDE = 1

SHOW_PROGRESS = True
SAVE_DIAGNOSTIC_FRAMES = True
MAKE_DIAGNOSTIC_GIFS = True
PLOT_FRAME_STRIDE = 5
GIF_DURATION = 0.12
SAVE_PROGRESSIVE_ERROR_FRAMES = True
MAKE_PROGRESSIVE_ERROR_GIFS = True
PROGRESSIVE_GIF_DURATION = 0.10

RAW_CSV_NAME = "raw_l2h_records.csv"
SUMMARY_CSV_NAME = "summary_l2h.csv"
CONDITION_CSV_NAME = "condition_numbers.csv"
SENSOR_CSV_NAME = "sensor_records.csv"
OVERLAP_CSV_NAME = "weighted_sensor_overlap.csv"


def _hyperparams_dict():
    """Return the state-weighted experiment hyperparameters."""
    return {
        "total_steps": TOTAL_STEPS,
        "period": PERIOD,
        "flow_names": FLOW_NAMES,
        "num_sensors": NUM_SENSORS,
        "max_basis_dim": MAX_BASIS_DIM,
        "seed": SEED,
        "weight_eps": WEIGHT_EPS,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "eval_stride": EVAL_STRIDE,
        "save_diagnostic_frames": SAVE_DIAGNOSTIC_FRAMES,
        "make_diagnostic_gifs": MAKE_DIAGNOSTIC_GIFS,
        "plot_frame_stride": PLOT_FRAME_STRIDE,
        "save_progressive_error_frames": SAVE_PROGRESSIVE_ERROR_FRAMES,
        "make_progressive_error_gifs": MAKE_PROGRESSIVE_ERROR_GIFS,
    }


def _run_single_flow(flow_case):
    """Run the state-weighted POD-QR comparison for one flow payload."""
    experiment_config = ExperimentConfig(
        domain=flow_case.domain_config,
        num_sensors=NUM_SENSORS,
        max_basis_dim=MAX_BASIS_DIM,
        seed=SEED,
    )

    print(
        f"\n=== State-weighted POD-QR flow: {flow_case.flow_name} "
        f"| shape={flow_case.u.shape} "
        f"| dt={flow_case.dt_actual} "
        f"| periodic={flow_case.is_periodic} ==="
    )

    return run_state_weighted_pod_qr(
        flow_case.u,
        flow_case.v,
        num_sensors=NUM_SENSORS,
        max_basis_dim=MAX_BASIS_DIM,
        seed=SEED,
        eps=WEIGHT_EPS,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        eval_stride=EVAL_STRIDE,
        config=experiment_config,
        show_progress=SHOW_PROGRESS,
        flow=flow_case.flow_name,
    )


def _summarize_l2h(raw_df):
    """Compute flow/method summary statistics for relative L2_h records."""
    group_cols = ["flow", "method"]
    for optional_col in ("selection_basis", "reconstruction_basis"):
        if optional_col in raw_df.columns:
            group_cols.append(optional_col)

    summary_df = raw_df.groupby(group_cols, as_index=False).agg(
        mean=("L2_h", "mean"),
        median=("L2_h", "median"),
        std=("L2_h", lambda values: float(values.std(ddof=0))),
        q25=("L2_h", lambda values: float(values.quantile(0.25))),
        q75=("L2_h", lambda values: float(values.quantile(0.75))),
        num_sensors=("num_sensors", "first"),
        max_basis_dim=("max_basis_dim", "first"),
    )
    ordered_cols = [
        "flow",
        "method",
        "selection_basis",
        "reconstruction_basis",
        "num_sensors",
        "max_basis_dim",
        "mean",
        "median",
        "std",
        "q25",
        "q75",
    ]
    return summary_df[[col for col in ordered_cols if col in summary_df.columns]]


def _save_flow_plots(flow_case, artifact_paths, raw_df, condition_df, sensor_df, overlap_df):
    """Save summary, progressive, condition, box, and overlap plots for one flow."""
    flow_name = flow_case.flow_name
    flow_raw_df = raw_df[raw_df["flow"] == flow_name]
    eval_t = np.sort(flow_raw_df["t"].unique())
    summary_t = int(eval_t[len(eval_t) // 2])

    plot_path = artifact_paths.plots_dir / f"{flow_name}_state_weighted_pod_qr.png"
    save_flow_diagnostic_plot(
        flow_name=flow_name,
        u=flow_case.u,
        v=flow_case.v,
        domain=flow_case.domain_config,
        raw_df=raw_df,
        condition_df=condition_df,
        sensor_df=sensor_df,
        out_path=plot_path,
        current_t=summary_t,
        quiver_step=flow_case.domain_config.nx // 40 or 1,
    )

    if SAVE_DIAGNOSTIC_FRAMES and artifact_paths.frames_dir is not None:
        frame_t_indices = eval_t[:: max(1, int(PLOT_FRAME_STRIDE))]
        frames_dir = artifact_paths.frames_dir / flow_name
        frame_paths = save_flow_diagnostic_frames(
            flow_name=flow_name,
            u=flow_case.u,
            v=flow_case.v,
            domain=flow_case.domain_config,
            raw_df=raw_df,
            condition_df=condition_df,
            sensor_df=sensor_df,
            frames_dir=frames_dir,
            frame_t_indices=frame_t_indices,
            quiver_step=flow_case.domain_config.nx // 40 or 1,
        )

        if MAKE_DIAGNOSTIC_GIFS:
            gif_path = artifact_paths.frames_dir / f"{flow_name}_state_weighted_pod_qr.gif"
            make_gif_from_frames(frame_paths, gif_path, duration=GIF_DURATION)

    if SAVE_PROGRESSIVE_ERROR_FRAMES:
        progressive_dir = artifact_paths.plots_dir / "progressive_errors" / flow_name
        save_progressive_error_plots(
            raw_df=raw_df,
            flow_name=flow_name,
            output_dir=progressive_dir,
            make_gif=MAKE_PROGRESSIVE_ERROR_GIFS,
            gif_duration=PROGRESSIVE_GIF_DURATION,
        )

    flow_plot_dir = artifact_paths.plots_dir / flow_name
    save_l2h_boxplot(
        raw_df=raw_df,
        flow_name=flow_name,
        output_path=flow_plot_dir / "l2h_error_boxplot.png",
    )
    save_condition_over_time_plot(
        raw_df=raw_df,
        flow_name=flow_name,
        output_path=flow_plot_dir / "condition_number_over_time.png",
    )
    save_sensor_overlap_plot(
        overlap_df=overlap_df,
        flow_name=flow_name,
        output_path=flow_plot_dir / "weighted_sensor_overlap.png",
    )


def main():
    """Run the state-weighted POD-QR experiment across standard flow cases."""
    print("\n=== State-Weighted POD-QR Hyperparameters ===")
    print(_hyperparams_dict())

    artifact_paths = build_artifact_paths("state_weighted_pod_qr", include_frames=True)
    ensure_artifact_dirs(artifact_paths)

    flow_cases = generate_standard_flow_cases(
        total_steps=TOTAL_STEPS,
        period=PERIOD,
        flow_names=FLOW_NAMES,
    )

    raw_parts = []
    condition_parts = []
    sensor_parts = []
    overlap_parts = []
    for flow_case in flow_cases:
        result = _run_single_flow(flow_case)
        raw_parts.append(result.raw_records)
        condition_parts.append(result.condition_records)
        sensor_parts.append(result.sensor_records)
        overlap_parts.append(result.overlap_records)

    raw_df = pd.concat(raw_parts, ignore_index=True)
    condition_df = pd.concat(condition_parts, ignore_index=True)
    sensor_df = pd.concat(sensor_parts, ignore_index=True)
    overlap_df = pd.concat(overlap_parts, ignore_index=True)
    summary_df = _summarize_l2h(raw_df)

    raw_csv_path = Path(artifact_paths.results_dir) / RAW_CSV_NAME
    summary_csv_path = Path(artifact_paths.results_dir) / SUMMARY_CSV_NAME
    condition_csv_path = Path(artifact_paths.results_dir) / CONDITION_CSV_NAME
    sensor_csv_path = Path(artifact_paths.results_dir) / SENSOR_CSV_NAME
    overlap_csv_path = Path(artifact_paths.results_dir) / OVERLAP_CSV_NAME

    raw_df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    condition_df.to_csv(condition_csv_path, index=False)
    sensor_df.to_csv(sensor_csv_path, index=False)
    overlap_df.to_csv(overlap_csv_path, index=False)

    print(f"Saved raw L2_h records to {raw_csv_path}")
    print(f"Saved summary records to {summary_csv_path}")
    print(f"Saved condition-number records to {condition_csv_path}")
    print(f"Saved sensor records to {sensor_csv_path}")
    print(f"Saved weighted sensor-overlap records to {overlap_csv_path}")

    for flow_case in flow_cases:
        _save_flow_plots(flow_case, artifact_paths, raw_df, condition_df, sensor_df, overlap_df)

    print(f"Saved diagnostic plots to {artifact_paths.plots_dir}")
    if artifact_paths.frames_dir is not None:
        print(f"Saved diagnostic frames/GIFs to {artifact_paths.frames_dir}")


if __name__ == "__main__":
    main()

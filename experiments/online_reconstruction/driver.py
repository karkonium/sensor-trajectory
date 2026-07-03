"""Driver for the online reconstructed-window Moving POD-QR experiment."""

from pathlib import Path

import pandas as pd

from experiments.common.config import ExperimentConfig
from experiments.common.flow_cases import generate_standard_flow_cases
from experiments.common.paths import build_artifact_paths, ensure_artifact_dirs
from experiments.online_reconstruction.pipeline import METHOD_ORDER, run_online_reconstructed_window
from experiments.online_reconstruction.plotting import save_flow_artifacts


TOTAL_STEPS = 500
PERIOD = 80
NUM_SENSORS = 10
GLOBAL_BASIS_DIM = 15
WINDOW_LEN = 25
WINDOW_BASIS_DIM = 8
TRAIN_FRACTION = 0.5
MIN_DIST_PCT = 0.05
FLOW_NAMES = ['kolmogorov']# ["double_gyre", "moving_vortex", "kolmogorov", "cylinder_wake"]

SEED = 90
QUIVER_STEP = 4
SHOW_PROGRESS = True
PLOT_FRAME_STRIDE = 1
MAKE_GIFS = True
GIF_DURATION = 0.10

RUN_LABEL = "online_reconstructed_window"
RAW_CSV_NAME = "raw_online_reconstructed_window_records.csv"
SUMMARY_CSV_NAME = "summary_online_reconstructed_window.csv"
DIAGNOSTIC_CSV_NAME = "diagnostics_online_reconstructed_window.csv"


def _hyperparams_dict():
    """Return the online reconstructed-window experiment hyperparameters."""
    return {
        "total_steps": TOTAL_STEPS,
        "period": PERIOD,
        "flow_names": FLOW_NAMES,
        "num_sensors": NUM_SENSORS,
        "global_basis_dim": GLOBAL_BASIS_DIM,
        "window_len": WINDOW_LEN,
        "window_basis_dim": WINDOW_BASIS_DIM,
        "train_fraction": TRAIN_FRACTION,
        "min_dist_pct": MIN_DIST_PCT,
        "seed": SEED,
        "quiver_step": QUIVER_STEP,
        "plot_frame_stride": PLOT_FRAME_STRIDE,
        "make_gifs": MAKE_GIFS,
    }


def _run_single_flow(flow_case, artifact_paths):
    """Run the online reconstructed-window experiment for one flow payload."""
    experiment_config = ExperimentConfig(
        domain=flow_case.domain_config,
        num_sensors=NUM_SENSORS,
        max_basis_dim=max(GLOBAL_BASIS_DIM, WINDOW_BASIS_DIM),
        seed=SEED,
        quiver_step=QUIVER_STEP,
    )

    print(
        f"\n=== Online reconstructed-window flow: {flow_case.flow_name} "
        f"| shape={flow_case.u.shape} "
        f"| dt={flow_case.dt_actual} "
        f"| periodic={flow_case.is_periodic} ==="
    )

    result = run_online_reconstructed_window(
        flow_case.u,
        flow_case.v,
        train_fraction=TRAIN_FRACTION,
        window_len=WINDOW_LEN,
        min_dist_pct=MIN_DIST_PCT,
        dt=flow_case.dt_actual,
        periodic=flow_case.is_periodic,
        config=experiment_config,
        global_basis_dim=GLOBAL_BASIS_DIM,
        window_basis_dim=WINDOW_BASIS_DIM,
        flow=flow_case.flow_name,
        show_progress=SHOW_PROGRESS,
    )
    save_flow_artifacts(
        flow_case,
        result,
        artifact_paths,
        run_label=RUN_LABEL,
        quiver_step=QUIVER_STEP,
        frame_stride=PLOT_FRAME_STRIDE,
        make_gifs=MAKE_GIFS,
        gif_duration=GIF_DURATION,
        show_progress=SHOW_PROGRESS,
    )
    return result.raw_records, result.diagnostic_records


def _summarize_final_cumulative(raw_df):
    """Return final running mean error per flow and method."""
    sorted_df = raw_df.sort_values(["flow", "method", "t"])
    summary_df = sorted_df.groupby(["flow", "method"], as_index=False).tail(1)
    summary_df = summary_df.rename(columns={"cumulative_L2_h": "final_cumulative_L2_h"})
    summary_df = summary_df[
        [
            "flow",
            "method",
            "final_cumulative_L2_h",
            "num_sensors",
            "global_basis_dim",
            "window_len",
            "window_basis_dim",
            "train_fraction",
        ]
    ]
    method_order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    summary_df["method_order"] = summary_df["method"].map(method_order)
    summary_df = summary_df.sort_values(["flow", "method_order"]).drop(columns="method_order")
    return summary_df.reset_index(drop=True)


def main():
    """Run online reconstructed-window Moving POD-QR across standard flows."""
    print("\n=== Online Reconstructed-Window Hyperparameters ===")
    print(_hyperparams_dict())

    artifact_paths = build_artifact_paths("online_reconstruction", include_frames=True)
    ensure_artifact_dirs(artifact_paths)

    raw_parts = []
    diagnostic_parts = []
    for flow_name in FLOW_NAMES:
        flow_case = generate_standard_flow_cases(
            total_steps=TOTAL_STEPS,
            period=PERIOD,
            flow_names=[flow_name],
        )[0]
        raw_records, diagnostic_records = _run_single_flow(flow_case, artifact_paths)
        raw_parts.append(raw_records)
        diagnostic_parts.append(diagnostic_records)

    raw_df = pd.concat(raw_parts, ignore_index=True)
    diagnostic_df = pd.concat(diagnostic_parts, ignore_index=True)
    summary_df = _summarize_final_cumulative(raw_df)

    raw_csv_path = Path(artifact_paths.results_dir) / RAW_CSV_NAME
    summary_csv_path = Path(artifact_paths.results_dir) / SUMMARY_CSV_NAME
    diagnostic_csv_path = Path(artifact_paths.results_dir) / DIAGNOSTIC_CSV_NAME
    raw_df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    diagnostic_df.to_csv(diagnostic_csv_path, index=False)

    print(f"\nSaved raw records to {raw_csv_path}")
    print(f"Saved final cumulative summary to {summary_csv_path}")
    print(f"Saved online target diagnostics to {diagnostic_csv_path}")
    print("\nFinal cumulative relative L2_h error:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

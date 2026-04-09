"""Driver for random-initialization Window-POD trials across standard flows."""

from pathlib import Path

import pandas as pd

from experiments.common.config import ExperimentConfig
from experiments.common.flow_cases import generate_standard_flow_cases
from experiments.common.paths import build_artifact_paths, ensure_artifact_dirs
from experiments.common.plotting import save_boxplots_per_flow
from experiments.random_trials.pipeline import run_random_trials_window_pod


TOTAL_STEPS = 160
PERIOD = 80
FLOW_NAMES = ['kolmogorov']# ["double_gyre", "moving_vortex", "kolmogorov", "cylinder_wake"]

NUM_SENSORS = 10
N_TRIALS = 50
SEED = 42
MAX_BASIS_DIM = 10

WINDOW_LEN = 13
STEP_SIZE = 1
MIN_DIST_PCT = 0.05
SHOW_PROGRESS = True

RAW_CSV_NAME = "raw_window_records.csv"
AGGREGATED_CSV_NAME = "aggregated_mean_l2h.csv"

PLACEMENT_ORDER = ["Fixed", "Lagrangian", "Moving POD-QR", "QR teleport"]


def _run_single_flow(flow_case):
    """Run random trials for one flow payload.

    Args:
        flow_case: FlowCasePayload from generate_standard_flow_cases.

    Returns:
        DataFrame with per-window, per-trial relative L2_h records.
    """
    experiment_config = ExperimentConfig(
        domain=flow_case.domain_config,
        num_sensors=NUM_SENSORS,
        max_basis_dim=MAX_BASIS_DIM,
        seed=SEED,
    )

    return run_random_trials_window_pod(
        flow_case.u,
        flow_case.v,
        window_len=WINDOW_LEN,
        step_size=STEP_SIZE,
        min_dist_pct=MIN_DIST_PCT,
        n_trials=N_TRIALS,
        seed=SEED,
        dt=flow_case.dt_actual,
        periodic=flow_case.is_periodic,
        config=experiment_config,
        show_progress=SHOW_PROGRESS,
        flow=flow_case.flow_name,
    )


def main():
    """Run random-trials experiment across all standard flow cases.

    Args:
        None.

    Returns:
        None.
    """
    artifact_paths = build_artifact_paths("random_trials", include_frames=False)
    ensure_artifact_dirs(artifact_paths)

    flow_cases = generate_standard_flow_cases(
        total_steps=TOTAL_STEPS,
        period=PERIOD,
        flow_names=FLOW_NAMES,
    )

    all_records = []
    for flow_case in flow_cases:
        print(
            f"\n=== Random-trials flow: {flow_case.flow_name} "
            f"| shape={flow_case.u.shape} "
            f"| dt={flow_case.dt_actual} "
            f"| periodic={flow_case.is_periodic} ==="
        )
        flow_records = _run_single_flow(flow_case)
        all_records.append(flow_records)

    raw_df = pd.concat(all_records, ignore_index=True)
    aggregated_df = raw_df.groupby(["flow", "num_sensors", "placement"], as_index=False)["L2_h"].mean()

    raw_csv_path = Path(artifact_paths.results_dir) / RAW_CSV_NAME
    aggregated_csv_path = Path(artifact_paths.results_dir) / AGGREGATED_CSV_NAME

    raw_df.to_csv(raw_csv_path, index=False)
    aggregated_df.to_csv(aggregated_csv_path, index=False)

    print(f"Saved raw records to {raw_csv_path}")
    print(f"Saved aggregated records to {aggregated_csv_path}")

    save_boxplots_per_flow(raw_df, artifact_paths.plots_dir, placement_order=PLACEMENT_ORDER)
    print(f"Saved boxplots to {artifact_paths.plots_dir}")


if __name__ == "__main__":
    main()

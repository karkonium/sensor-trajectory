"""Driver for the Global-POD versus Window-POD comparison experiment."""

from pathlib import Path

import pandas as pd

from experiments.common.config import ExperimentConfig
from experiments.common.flow_cases import generate_standard_flow_cases
from experiments.common.paths import build_artifact_paths, ensure_artifact_dirs
from experiments.common.plotting import save_grouped_barh_by_flow, save_mean_rmse_vs_sensor_count
from experiments.global_sensors.pipeline import run_pod_basis_comparison


TOTAL_STEPS = 160
PERIOD = 80
FLOW_NAMES = ["double_gyre", "moving_vortex", "kolmogorov", "cylinder_wake"]

SENSOR_COUNTS = [1, 2, 4, 8, 16]
MAX_BASIS_DIM = 10
SEED = 90

WINDOW_LEN = 13
STEP_SIZE = 1
MIN_DIST_PCT = 0.05
SHOW_PROGRESS = True

RAW_CSV_NAME = "raw_window_records.csv"
AGGREGATED_CSV_NAME = "aggregated_mean_rmse.csv"

METHOD_ORDER = ["Static QR", "Teleport QR", "Lagrangian", "Moving QR"]
BASIS_ORDER = ["Global POD", "Window POD"]


def _run_single_sensor_count(flow_case, num_sensors):
    """Run one flow with one sensor count.

    Args:
        flow_case: FlowCasePayload from generate_standard_flow_cases.
        num_sensors: Number of sensors for this run.

    Returns:
        DataFrame with per-window comparison records.
    """
    experiment_config = ExperimentConfig(
        domain=flow_case.domain_config,
        num_sensors=num_sensors,
        max_basis_dim=MAX_BASIS_DIM,
        seed=SEED,
    )

    return run_pod_basis_comparison(
        flow_case.u,
        flow_case.v,
        window_len=WINDOW_LEN,
        step_size=STEP_SIZE,
        min_dist_pct=MIN_DIST_PCT,
        dt=flow_case.dt_actual,
        periodic=flow_case.is_periodic,
        config=experiment_config,
        show_progress=SHOW_PROGRESS,
        flow=flow_case.flow_name,
    )


def main():
    """Run Global-vs-Window POD comparison across all standard flows and sensor counts.

    Args:
        None.

    Returns:
        None.
    """
    artifact_paths = build_artifact_paths("global_sensors", include_frames=False)
    ensure_artifact_dirs(artifact_paths)

    flow_cases = generate_standard_flow_cases(
        total_steps=TOTAL_STEPS,
        period=PERIOD,
        flow_names=FLOW_NAMES,
    )

    all_records = []
    for flow_case in flow_cases:
        print(
            f"\n=== Global-sensors flow: {flow_case.flow_name} "
            f"| shape={flow_case.u.shape} "
            f"| dt={flow_case.dt_actual} "
            f"| periodic={flow_case.is_periodic} ==="
        )

        for num_sensors in SENSOR_COUNTS:
            print(f"[{flow_case.flow_name}] num_sensors={num_sensors}")
            sensor_records = _run_single_sensor_count(flow_case, num_sensors)
            all_records.append(sensor_records)

    raw_df = pd.concat(all_records, ignore_index=True)
    aggregated_df = raw_df.groupby(["flow", "num_sensors", "basis", "method"], as_index=False)["RMSE"].mean()

    raw_csv_path = Path(artifact_paths.results_dir) / RAW_CSV_NAME
    aggregated_csv_path = Path(artifact_paths.results_dir) / AGGREGATED_CSV_NAME

    raw_df.to_csv(raw_csv_path, index=False)
    aggregated_df.to_csv(aggregated_csv_path, index=False)

    print(f"Saved raw records to {raw_csv_path}")
    print(f"Saved aggregated records to {aggregated_csv_path}")

    save_mean_rmse_vs_sensor_count(aggregated_df, artifact_paths.plots_dir)
    save_grouped_barh_by_flow(
        aggregated_df,
        artifact_paths.plots_dir,
        method_order=METHOD_ORDER,
        basis_order=BASIS_ORDER,
    )
    print(f"Saved summary plots to {artifact_paths.plots_dir}")


if __name__ == "__main__":
    main()

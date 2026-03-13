"""Driver for the sliding moving-sensor experiment."""

from pathlib import Path

import pandas as pd

from experiments.common.config import ExperimentConfig
from experiments.common.flow_cases import generate_standard_flow_cases
from experiments.common.paths import build_artifact_paths, ensure_artifact_dirs
from experiments.sliding.pipeline import run_experiment_sliding


TOTAL_STEPS = 160
PERIOD = 80
FLOW_NAMES = ["double_gyre", "moving_vortex", "kolmogorov", "cylinder_wake"]

NUM_SENSORS = 10
MAX_BASIS_DIM = 10
SEED = 90

WINDOW_LEN = 13
STEP_SIZE = 1
MIN_DIST_PCT = 0.05

SHOW_PROGRESS = True
PLOT_WINDOWS = True
SAVE_WINDOW_FRAMES = True
MAKE_GIF = True
GIF_DURATION = 0.10

SAVE_RAW_CSV = True
SAVE_AGGREGATED_CSV = True
RAW_CSV_NAME = "raw_window_records.csv"
AGGREGATED_CSV_NAME = "aggregated_mean_rmse.csv"


def _run_single_flow(flow_case, artifact_paths):
    """Run sliding experiment for one flow payload.

    Args:
        flow_case: FlowCasePayload from generate_standard_flow_cases.
        artifact_paths: ArtifactPaths for sliding outputs.

    Returns:
        DataFrame with per-window RMSE records for one flow.
    """
    experiment_config = ExperimentConfig(
        domain=flow_case.domain_config,
        num_sensors=NUM_SENSORS,
        max_basis_dim=MAX_BASIS_DIM,
        seed=SEED,
    )

    flow_frames_dir = None
    flow_gif_path = None
    if artifact_paths.frames_dir is not None:
        flow_frames_dir = artifact_paths.frames_dir / flow_case.flow_name
        flow_gif_path = artifact_paths.frames_dir / f"{flow_case.flow_name}.gif"

    print(
        f"\n=== Sliding flow: {flow_case.flow_name} "
        f"| shape={flow_case.u.shape} "
        f"| dt={flow_case.dt_actual} "
        f"| periodic={flow_case.is_periodic} ==="
    )

    result_df = run_experiment_sliding(
        flow_case.u,
        flow_case.v,
        window_len=WINDOW_LEN,
        step_size=STEP_SIZE,
        min_dist_pct=MIN_DIST_PCT,
        dt=flow_case.dt_actual,
        periodic=flow_case.is_periodic,
        config=experiment_config,
        plot_windows=PLOT_WINDOWS,
        save_window_frames=SAVE_WINDOW_FRAMES,
        frames_dir=str(flow_frames_dir) if flow_frames_dir is not None else None,
        make_gif=MAKE_GIF,
        gif_path=str(flow_gif_path) if flow_gif_path is not None else None,
        gif_duration=GIF_DURATION,
        run_name=flow_case.flow_name,
        show_progress=SHOW_PROGRESS,
    )

    result_df = result_df.copy()
    result_df["flow"] = flow_case.flow_name
    return result_df


def main():
    """Run sliding experiment over all standard flow cases and persist artifacts.

    Args:
        None.

    Returns:
        None.
    """
    artifact_paths = build_artifact_paths("sliding", include_frames=True)
    ensure_artifact_dirs(artifact_paths)

    flow_cases = generate_standard_flow_cases(
        total_steps=TOTAL_STEPS,
        period=PERIOD,
        flow_names=FLOW_NAMES,
    )

    all_records = []
    for flow_case in flow_cases:
        flow_records = _run_single_flow(flow_case, artifact_paths)
        all_records.append(flow_records)

    combined_df = pd.concat(all_records, ignore_index=True)

    if SAVE_RAW_CSV:
        raw_csv_path = Path(artifact_paths.results_dir) / RAW_CSV_NAME
        combined_df.to_csv(raw_csv_path, index=False)
        print(f"Saved raw records to {raw_csv_path}")

    if SAVE_AGGREGATED_CSV:
        aggregated_df = (
            combined_df.groupby(["flow", "placement", "basis"], as_index=False)["RMSE"].mean()
        )
        aggregated_csv_path = Path(artifact_paths.results_dir) / AGGREGATED_CSV_NAME
        aggregated_df.to_csv(aggregated_csv_path, index=False)
        print(f"Saved aggregated records to {aggregated_csv_path}")


if __name__ == "__main__":
    main()

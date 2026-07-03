"""Driver for the sliding moving-sensor experiment."""

from pathlib import Path

import pandas as pd

from experiments.common.config import ExperimentConfig
from experiments.common.flow_cases import generate_standard_flow_cases
from experiments.common.paths import build_artifact_paths, ensure_artifact_dirs
from experiments.sliding.pipeline import run_experiment_sliding


FLOW_NAMES = ["cylinder_wake", "kolmogorov", "moving_vortex", "double_gyre"]
DEFAULT_FLOW_CONFIG = {
    "total_steps": 200,
    "period": 80,
    "num_sensors": 4,
    "max_basis_dim": 4,
    "seed": 910,
    "window_len": 13,
    "step_size": 1,
    "min_dist_pct": 0.05,
    "quiver_step": 4,
    "show_progress": True,
    "plot_windows": True,
    "plot_trajectories": True,
    "save_window_frames": True,
    "make_flow_gif": True,
    "make_sensor_motion_gif": True,
    "gif_duration": 0.10,
    "sensor_tail_length": 100,
    "trajectory_snapshot_indices": None,
}
FLOW_CONFIGS = {
    "double_gyre": {
        "total_steps": 200,
        "trajectory_snapshot_indices": [0, 100, 199],
    },
    "kolmogorov": {
        "trajectory_snapshot_indices": [0, 100, 199],
    },
    "cylinder_wake": {
        "seed": 300,
        "trajectory_snapshot_indices": [0, 100, 199],
    },
    "moving_vortex": {
        "total_steps": 100,
        "period": 200,
        "trajectory_snapshot_indices": [0, 50, 99],
    },
}

SAVE_RAW_CSV = True
SAVE_AGGREGATED_CSV = True
RAW_CSV_NAME = "raw_window_records.csv"
AGGREGATED_CSV_NAME = "aggregated_mean_l2h.csv"


def _hyperparams_dict():
    """Return the sliding experiment hyperparameters."""
    return {
        "flow_names": FLOW_NAMES,
        "default_flow_config": DEFAULT_FLOW_CONFIG,
        "flow_configs": {
            flow_name: _resolved_flow_config(flow_name)
            for flow_name in FLOW_NAMES
        },
    }


def _resolved_flow_config(flow_name):
    """Return the effective sliding config for one flow."""
    flow_config = DEFAULT_FLOW_CONFIG.copy()
    flow_config.update(FLOW_CONFIGS.get(flow_name, {}))
    return flow_config


def _run_single_flow(flow_case, artifact_paths, flow_config):
    """Run sliding experiment for one flow payload.

    Args:
        flow_case: FlowCasePayload from generate_standard_flow_cases.
        artifact_paths: ArtifactPaths for sliding outputs.

    Returns:
        DataFrame with per-window L2_h records for one flow.
    """
    experiment_config = ExperimentConfig(
        domain=flow_case.domain_config,
        num_sensors=flow_config["num_sensors"],
        max_basis_dim=flow_config["max_basis_dim"],
        seed=flow_config["seed"],
        quiver_step=flow_config["quiver_step"],
    )

    flow_frames_dir = None
    flow_gif_path = None
    sensor_motion_gif_path = None
    flow_trajectory_plot_path = artifact_paths.plots_dir / f"{flow_case.flow_name}_sensor_trajectories.png"
    if artifact_paths.frames_dir is not None:
        flow_frames_dir = artifact_paths.frames_dir / flow_case.flow_name
        flow_gif_path = artifact_paths.frames_dir / f"{flow_case.flow_name}_flow.gif"
        sensor_motion_gif_path = artifact_paths.frames_dir / f"{flow_case.flow_name}_sensor_motion.gif"

    print(
        f"\n=== Sliding flow: {flow_case.flow_name} "
        f"| shape={flow_case.u.shape} "
        f"| dt={flow_case.dt_actual} "
        f"| periodic={flow_case.is_periodic} "
        f"| config={flow_config} ==="
    )

    result_df = run_experiment_sliding(
        flow_case.u,
        flow_case.v,
        window_len=flow_config["window_len"],
        step_size=flow_config["step_size"],
        min_dist_pct=flow_config["min_dist_pct"],
        dt=flow_case.dt_actual,
        periodic=flow_case.is_periodic,
        config=experiment_config,
        plot_windows=flow_config["plot_windows"],
        save_window_frames=flow_config["save_window_frames"],
        frames_dir=str(flow_frames_dir) if flow_frames_dir is not None else None,
        make_flow_gif=flow_config["make_flow_gif"],
        flow_gif_path=str(flow_gif_path) if flow_gif_path is not None else None,
        make_sensor_motion_gif=flow_config["make_sensor_motion_gif"],
        sensor_motion_gif_path=(
            str(sensor_motion_gif_path) if sensor_motion_gif_path is not None else None
        ),
        sensor_tail_length=flow_config["sensor_tail_length"],
        plot_trajectories=flow_config["plot_trajectories"],
        trajectory_plot_path=str(flow_trajectory_plot_path),
        trajectory_snapshot_indices=flow_config["trajectory_snapshot_indices"],
        gif_duration=flow_config["gif_duration"],
        run_name=flow_case.flow_name,
        show_progress=flow_config["show_progress"],
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
    print("\n=== Sliding Hyperparameters ===")
    print(_hyperparams_dict())

    artifact_paths = build_artifact_paths("sliding", include_frames=True)
    ensure_artifact_dirs(artifact_paths)

    all_records = []
    for flow_name in FLOW_NAMES:
        flow_config = _resolved_flow_config(flow_name)
        flow_case = generate_standard_flow_cases(
            total_steps=flow_config["total_steps"],
            period=flow_config["period"],
            flow_names=[flow_name],
        )[0]
        flow_records = _run_single_flow(flow_case, artifact_paths, flow_config)
        all_records.append(flow_records)

    combined_df = pd.concat(all_records, ignore_index=True)

    if SAVE_RAW_CSV:
        raw_csv_path = Path(artifact_paths.results_dir) / RAW_CSV_NAME
        combined_df.to_csv(raw_csv_path, index=False)

    if SAVE_AGGREGATED_CSV:
        aggregated_df = combined_df.groupby(["flow", "placement", "basis"], as_index=False).agg(
            L2_h=("L2_h", "mean"),
            L2_h_variance=("L2_h", lambda s: float(s.var(ddof=0))),
        )
        aggregated_csv_path = Path(artifact_paths.results_dir) / AGGREGATED_CSV_NAME
        aggregated_df.to_csv(aggregated_csv_path, index=False)
        print("\nAggregated relative L2_h error summary:")
        print(aggregated_df.to_string(index=False))


if __name__ == "__main__":
    main()

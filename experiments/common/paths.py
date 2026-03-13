"""Path helpers for experiment artifact outputs."""

from pathlib import Path


class ArtifactPaths:
    """Container for experiment artifact directories."""

    def __init__(self, experiment_name, experiment_dir, artifacts_dir, results_dir, plots_dir, frames_dir):
        """
        Args:
            experiment_name: Experiment name.
            experiment_dir: Root directory of the experiment.
            artifacts_dir: Root artifact directory for the experiment.
            results_dir: Result file directory.
            plots_dir: Plot file directory.
            frames_dir: Optional frame directory.

        Returns:
            None.
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.artifacts_dir = artifacts_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.frames_dir = frames_dir


def build_artifact_paths(experiment_name, include_frames=False):
    """Build standardized artifact paths for one experiment.

    Args:
        experiment_name: Experiment folder name under `experiments/`.
        include_frames: Whether a frames directory should be created.

    Returns:
        ArtifactPaths with experiment-local artifact directories.
    """
    experiment_dir = Path("experiments") / str(experiment_name)
    artifacts_dir = experiment_dir / "artifacts"
    results_dir = artifacts_dir / "results"
    plots_dir = artifacts_dir / "plots"
    frames_dir = artifacts_dir / "frames" if include_frames else None

    return ArtifactPaths(
        experiment_name=experiment_name,
        experiment_dir=experiment_dir,
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        plots_dir=plots_dir,
        frames_dir=frames_dir,
    )


def ensure_artifact_dirs(artifact_paths):
    """Create artifact directories on disk for an experiment.

    Args:
        artifact_paths: ArtifactPaths instance.

    Returns:
        None.
    """
    artifact_paths.results_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths.plots_dir.mkdir(parents=True, exist_ok=True)
    if artifact_paths.frames_dir is not None:
        artifact_paths.frames_dir.mkdir(parents=True, exist_ok=True)

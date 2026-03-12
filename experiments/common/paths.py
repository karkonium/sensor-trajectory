"""Path helpers for experiment artifact outputs."""

from pathlib import Path


class VariantArtifactPaths:
    """Container for variant artifact directories."""

    def __init__(self, variant_name, variant_dir, artifacts_dir, results_dir, plots_dir, frames_dir):
        """
        Args:
            variant_name: Experiment variant name.
            variant_dir: Root directory of the variant.
            artifacts_dir: Root artifact directory for the variant.
            results_dir: Result file directory.
            plots_dir: Plot file directory.
            frames_dir: Optional frame directory.

        Returns:
            None.
        """
        self.variant_name = variant_name
        self.variant_dir = variant_dir
        self.artifacts_dir = artifacts_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.frames_dir = frames_dir


def build_variant_artifact_paths(variant_name, include_frames=False):
    """Build standardized artifact paths for one experiment variant.

    Args:
        variant_name: Variant folder name under experiments/.
        include_frames: Whether a frames directory should be created.

    Returns:
        VariantArtifactPaths with variant-local artifact directories.
    """
    variant_dir = Path("experiments") / str(variant_name)
    artifacts_dir = variant_dir / "artifacts"
    results_dir = artifacts_dir / "results"
    plots_dir = artifacts_dir / "plots"
    frames_dir = artifacts_dir / "frames" if include_frames else None

    return VariantArtifactPaths(
        variant_name=variant_name,
        variant_dir=variant_dir,
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        plots_dir=plots_dir,
        frames_dir=frames_dir,
    )


def ensure_artifact_dirs(variant_paths):
    """Create artifact directories on disk for a variant.

    Args:
        variant_paths: VariantArtifactPaths instance.

    Returns:
        None.
    """
    variant_paths.results_dir.mkdir(parents=True, exist_ok=True)
    variant_paths.plots_dir.mkdir(parents=True, exist_ok=True)
    if variant_paths.frames_dir is not None:
        variant_paths.frames_dir.mkdir(parents=True, exist_ok=True)

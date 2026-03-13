"""Shared configuration objects for experiment modules."""


class DomainConfig:
    """Grid/domain metadata for velocity fields with shape (T, nx, ny)."""

    def __init__(self, nx, ny, lx=1.0, ly=1.0):
        """
        Args:
            nx: Number of grid points along x.
            ny: Number of grid points along y.
            lx: Physical domain length in x.
            ly: Physical domain length in y.

        Returns:
            None.
        """
        if nx <= 1 or ny <= 1:
            raise ValueError("nx and ny must both be > 1")
        if lx <= 0.0 or ly <= 0.0:
            raise ValueError("lx and ly must both be > 0")

        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)


class ExperimentConfig:
    """Configuration for sensor-placement experiments."""

    def __init__(self, domain, num_sensors=10, max_basis_dim=10, seed=90, quiver_step=4):
        """
        Args:
            domain: DomainConfig describing grid/domain metadata.
            num_sensors: Number of sensors to place.
            max_basis_dim: Maximum POD basis dimension.
            seed: Random seed for reproducibility.
            quiver_step: Subsampling step for quiver plotting.

        Returns:
            None.
        """
        if not isinstance(domain, DomainConfig):
            raise TypeError("domain must be a DomainConfig")
        if num_sensors <= 0:
            raise ValueError("num_sensors must be > 0")
        if max_basis_dim <= 0:
            raise ValueError("max_basis_dim must be > 0")
        if quiver_step <= 0:
            raise ValueError("quiver_step must be > 0")

        self.domain = domain
        self.num_sensors = int(num_sensors)
        self.max_basis_dim = int(max_basis_dim)
        self.seed = int(seed)
        self.quiver_step = int(quiver_step)


def config_from_arrays(
    u_shape,
    lx=1.0,
    ly=1.0,
    num_sensors=10,
    max_basis_dim=10,
    seed=90,
):
    """Create ExperimentConfig from an input array shape and defaults.

    Args:
        u_shape: Tuple shaped as (T, nx, ny).
        lx: Physical domain length in x.
        ly: Physical domain length in y.
        num_sensors: Number of sensors to place.
        max_basis_dim: Maximum POD basis dimension.
        seed: Random seed.

    Returns:
        ExperimentConfig instance.
    """
    if len(u_shape) != 3:
        raise ValueError("Expected u shape (T, nx, ny)")

    _, nx, ny = u_shape
    return ExperimentConfig(
        domain=DomainConfig(nx=nx, ny=ny, lx=lx, ly=ly),
        num_sensors=num_sensors,
        max_basis_dim=max_basis_dim,
        seed=seed,
    )

import numpy as np
import matplotlib.pyplot as plt


def generate_chladni_square_phi(
    n_timesteps,
    nx,
    ny,
    n,
    m,
    L=1.0,
    phase_speed=0.0,
    plot_series=False,
    nodal_threshold=0.05,
    nodal_alpha=0.9,
):
    """
    Generate square-plate Chladni field using:
        Phi(x,y) = cos(n*pi*x/L)cos(m*pi*y/L) - cos(m*pi*x/L)cos(n*pi*y/L)
        [https://www.dynamicmath.xyz/chladni-patterns/]

    Parameters
    ----------
    n_timesteps : int
        Number of frames.
    nx, ny : int
        Grid resolution.
    n, m : int
        Mode numbers.
    L : float
        Side length; x,y in [0,L].
    phase_speed : float
        If 0, all frames are identical (static Phi).
        If nonzero, adds a time phase to make a changing series:
            Phi_t = cos(n*pi*x/L + w*t)*cos(m*pi*y/L) - cos(m*pi*x/L + w*t)*cos(n*pi*y/L)
        This is *just for generating a time series* (not physical damping etc).
    plot_series : bool
        Plot the time series.
    nodal_threshold : float
        Where abs(Phi) < threshold is considered "nodal line" for overlay.
    nodal_alpha : float
        Alpha for the nodal overlay.

    Returns
    -------
    phi_field : ndarray, shape (n_timesteps, nx, ny)
        Raw Phi values for each timestep.
    """
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    phi_field = np.zeros((n_timesteps, nx, ny), dtype=float)

    for t in range(n_timesteps):
        w = phase_speed * t
        phi = (
            np.cos(n * np.pi * X / L + w) * np.cos(m * np.pi * Y / L)
            - np.cos(m * np.pi * X / L + w) * np.cos(n * np.pi * Y / L)
        )
        phi_field[t] = phi

    if plot_series:
        fig, axes = plt.subplots(1, n_timesteps, figsize=(4 * n_timesteps, 4), squeeze=False)

        for t in range(n_timesteps):
            ax = axes[0, t]

            # Plot Phi
            im = ax.imshow(
                phi_field[t].T,
                origin="lower",
                extent=[0, L, 0, L],
                aspect="equal",
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Overlay nodal region: abs(Phi) < threshold
            nodal = (np.abs(phi_field[t]) < nodal_threshold).T  # transpose to match imshow
            # show mask as an overlay (binary)
            ax.imshow(
                nodal,
                origin="lower",
                extent=[0, L, 0, L],
                aspect="equal",
                alpha=nodal_alpha,
            )

            ax.set_title(f"Chladni square Phi (n={n}, m={m}, t={t})")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        plt.show()

    return phi_field


if __name__ == "__main__":
    phi = generate_chladni_square_phi(
        n_timesteps=5,
        nx=250,
        ny=250,
        n=2,
        m=5,
        L=1.0,
        phase_speed=0.35, # set 0.0 for static frames
        plot_series=True,
        nodal_threshold=0.05, # threshold for plotting
        nodal_alpha=0.95,
    )

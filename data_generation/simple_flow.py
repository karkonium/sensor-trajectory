import numpy as np


def generate_simple_flow(n_timesteps, nx, ny):
    # create a moving Gaussian "blob" that travels across the domain.
    data = np.zeros((n_timesteps, nx, ny))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    Xgrid, Ygrid = np.meshgrid(x, y, indexing='ij')

    for t in range(n_timesteps):
        # The center of the blob moves linearly through the domain.
        cx = 0.8 - 0.6 * (t / n_timesteps)
        cy = 0.2 + 0.6 * (t / n_timesteps)
        data[t] = np.exp(-((Xgrid - cx)**2 + (Ygrid - cy)**2) / 0.01)

    return data
import numpy as np


def combine_fields(u_field, v_field, horizontal_concat=True):
    """
    Combine two vector fields (u and v) into a single augmented snapshot.
    
    Parameters:
      u_field : np.ndarray, shape (n_timesteps, nx, ny)
      v_field : np.ndarray, shape (n_timesteps, nx, ny)
      horizontal_concat : bool, optional
             True will concatenate side by side (axis=2),
             resulting in shape (n_timesteps, nx, 2*ny).
             False will concatenate top-to-bottom (axis=1),
             resulting in shape (n_timesteps, 2*nx, ny).
    """
    if horizontal_concat:
        combined_field = np.concatenate((u_field, v_field), axis=2)
    else:
        combined_field = np.concatenate((u_field, v_field), axis=1)
    
    return combined_field


if __name__ == "__main__":
    from data_generation import generate_double_gyre_flow
    import matplotlib.pyplot as plt

    n_timesteps = 4
    nx, ny= 50, 25
    u_field, v_field = generate_double_gyre_flow(
        n_timesteps=n_timesteps, nx=nx, ny=ny
    )

    combined_field_h = combine_fields(u_field, v_field)
    combined_field_v = combine_fields(u_field, v_field, horizontal_concat=False)


    fig, axes = plt.subplots(4, n_timesteps, figsize=(4 * n_timesteps, 4))
    for t in range(0, n_timesteps, 1):
        for i, data in enumerate([u_field, v_field, combined_field_h, combined_field_v]):
            ax = axes[i, t]
            # Plot magnitute
            _, t_nx, t_ny = data.shape
            im = ax.imshow(data[t].T, origin='lower', extent=[0, t_nx, 0, t_ny], cmap='viridis')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
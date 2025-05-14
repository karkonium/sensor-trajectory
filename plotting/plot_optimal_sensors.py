import matplotlib.pyplot as plt
import numpy as np


def plot_optimal_sensors(nx, ny,
                         data_interval, sensor_coords, 
                         ax, plot_title, plot='imshow'):
        
    if plot == 'imshow':
        # For visualization, use mean
        avg_field = data_interval.mean(axis=0)
        im = ax.imshow(avg_field.T, origin='lower', extent=[0, nx, 0, ny], cmap='viridis')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    elif plot == 'quiver':
        # expect a tuple/list (u, v) with shape (T, nx, ny) or (nx, ny)
        u_field, v_field = data_interval

        # if time-dependent, collapse along time axis
        if u_field.ndim == 3:
            u_field = u_field.mean(axis=0)
            v_field = v_field.mean(axis=0)

        # build grid in the SAME coordinates as imshow / sensors
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

        ax.quiver(X, Y, u_field, v_field,
                  color='black',
                  scale_units='xy', scale=None,
                  width=0.005, pivot='mid')

    sensor_x = sensor_coords[:, 0] 
    sensor_y = sensor_coords[:, 1] 

    ax.scatter(sensor_x, sensor_y, 
                color='red', marker='o', s=50)
    ax.set_title(plot_title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
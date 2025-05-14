import matplotlib.pyplot as plt
import numpy as np
from .plot_optimal_sensors import plot_optimal_sensors


def plot_all_intervals(
    data,
    nx, ny,
    intervals,
    sensor_coords_list
):
    """
    Plot one subplot per interval.
    """
    plot = "imshow"
    if type(data) is tuple:
        plot = 'quiver'
    elif np.iscomplexobj(data):
        data = (np.real(data), np.imag(data))
        # Force mode to imshow for now.
        plot = 'quiver'

    n = len(intervals)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for ax, coords, (s, e) in zip(axes, sensor_coords_list, intervals):
        data_interval = (data[0][s:e], data[1][s:e]) if plot == "quiver" else data[s:e]
        plot_optimal_sensors(
            nx, ny,
            data_interval, coords,
            ax,
            plot_title=f"{s}→{e}",
            plot=plot
        )
    plt.tight_layout()
    plt.show()
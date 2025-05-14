import numpy as np

def rotate_data_90(data):
    """
    Rotate each snapshot by 90 degrees counterclockwise.
    
    Parameters
    ----------
    data : np.ndarray of shape (n_timesteps, nx, ny)
    """
    # We'll collect each rotated snapshot, then stack them back into an array.
    data_rotated_list = []
    for t in range(data.shape[0]):
        # Rotate data[t] (shape (nx, ny)) by 90 deg counterclockwise
        # np.rot90() outputs shape (ny, nx)
        slice_rot = np.rot90(data[t])
        data_rotated_list.append(slice_rot)
    
    # Stack them along the time dimension again
    data_rotated = np.stack(data_rotated_list, axis=0)
    return data_rotated

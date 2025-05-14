def reflect_data_y(data):
    """
    Reflect each snapshot about the horizontal midline in y (about y = L_y/2).
    
    Parameters
    ----------
    data : np.ndarray of shape (n_timesteps, nx, ny)
    """
    data_reflected = data[..., ::-1]
    return data_reflected
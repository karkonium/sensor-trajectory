import numpy as np

def split_state(X_aug, nx_c, ny_c, horizontal_concat=True):
    """
    Split augmented snapshots back into u and v arrays.
      nx_c, ny_c  = shape of combined snapshot (not original!)
    """
    if horizontal_concat:
        # X_aug columns = nx_c * ny_c = nx * (2ny)
        u_flat, v_flat = np.split(X_aug, 2, axis=1)
        nx_u, ny_u = nx_c, ny_c          # original dims
    else:
        # vertically stacked → first nx rows are u, next nx rows are v
        nx_orig = nx_c // 2              # recover original nx
        half    = nx_orig * ny_c
        u_flat  = X_aug[:, :half]
        v_flat  = X_aug[:, half:]
        nx_u, ny_u = nx_orig, ny_c

    u = u_flat.reshape(-1, nx_u, ny_u)
    v = v_flat.reshape(-1, nx_u, ny_u)
    return u, v

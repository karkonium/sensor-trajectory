import numpy as np


def extract_flow_subregion(u, v, x_range, y_range, coords="index", lx=None, ly=None, return_metadata=True):
    """Crop a full (u, v) flow timeseries to one spatial subregion.

    Parameters
    ----------
    u, v : np.ndarray
        Flow timeseries with shape (T, nx, ny).
    x_range, y_range : tuple
        Subregion bounds.
        If coords="index", use end-exclusive index bounds: (start, end).
        If coords="physical", use physical-coordinate bounds in the original domain.
    coords : {"index", "physical"}
        Whether x_range/y_range are index bounds or physical-coordinate bounds.
    lx, ly : float, optional
        Original domain lengths. Required when coords="physical".
    return_metadata : bool
        Whether to also return a small metadata dict for the cropped region.

    Returns
    -------
    tuple
        By default returns (u_sub, v_sub, metadata). If return_metadata=False,
        returns (u_sub, v_sub).
    """
    u = np.asarray(u)
    v = np.asarray(v)

    if u.shape != v.shape:
        raise ValueError("u and v must have identical shape")
    if u.ndim != 3:
        raise ValueError("u and v must have shape (T, nx, ny)")
    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError("x_range and y_range must each have exactly 2 values")
    if coords not in {"index", "physical"}:
        raise ValueError("coords must be either 'index' or 'physical'")

    _, nx, ny = u.shape

    if coords == "index":
        x_start, x_end = int(x_range[0]), int(x_range[1])
        y_start, y_end = int(y_range[0]), int(y_range[1])
    else:
        if lx is None or ly is None:
            raise ValueError("lx and ly must be provided when coords='physical'")

        x_grid = np.linspace(0.0, float(lx), nx)
        y_grid = np.linspace(0.0, float(ly), ny)

        x_start = int(np.searchsorted(x_grid, float(x_range[0]), side="left"))
        x_end = int(np.searchsorted(x_grid, float(x_range[1]), side="right"))
        y_start = int(np.searchsorted(y_grid, float(y_range[0]), side="left"))
        y_end = int(np.searchsorted(y_grid, float(y_range[1]), side="right"))

    if x_start < 0 or x_end > nx or x_start >= x_end:
        raise ValueError(f"Invalid x_range={x_range} for spatial size nx={nx}")
    if y_start < 0 or y_end > ny or y_start >= y_end:
        raise ValueError(f"Invalid y_range={y_range} for spatial size ny={ny}")

    u_sub = u[:, x_start:x_end, y_start:y_end]
    v_sub = v[:, x_start:x_end, y_start:y_end]

    if not return_metadata:
        return u_sub, v_sub

    metadata = {
        "coords": coords,
        "x_slice": (x_start, x_end),
        "y_slice": (y_start, y_end),
        "u_shape_full": tuple(int(dim) for dim in u.shape),
        "u_shape_sub": tuple(int(dim) for dim in u_sub.shape),
        "v_shape_full": tuple(int(dim) for dim in v.shape),
        "v_shape_sub": tuple(int(dim) for dim in v_sub.shape),
    }

    if lx is not None:
        metadata["lx_full"] = float(lx)
        metadata["lx_sub"] = float(lx) * (x_end - x_start) / max(nx - 1, 1)
    if ly is not None:
        metadata["ly_full"] = float(ly)
        metadata["ly_sub"] = float(ly) * (y_end - y_start) / max(ny - 1, 1)

    return u_sub, v_sub, metadata


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pathlib import Path

    from data_generation.kolmogorov_flow import generate_cfd_kolmogorov_flow

    nx = 900
    ny = 900
    lx = 2 * np.pi
    ly = 2 * np.pi

    u, v = generate_cfd_kolmogorov_flow(
        n_timesteps=20000,
        nx=nx,
        ny=ny,
        dt=1e-4,
        nu=2e-2,
        forcing_amp=20.0,
        kf=10,
        plot_series=False,
        plot_every=200,
    )
    print("Snapshots:", u.shape, "  v-rms:", np.sqrt((v**2).mean()))

    # Example subregion: centered box covering the middle third of the domain.
    x_range = (300, 600)
    y_range = (300, 600)
    u_sub, v_sub, meta = extract_flow_subregion(u, v, x_range=x_range, y_range=y_range)
    print("Subregion metadata:", meta)

    output_path = Path(__file__).resolve().with_name("kolmogorov_subregion_comparison.png")

    snapshot_indices = np.linspace(0, u.shape[0] - 1, 10, dtype=int)
    full_speed = np.sqrt(u**2 + v**2)
    sub_speed = np.sqrt(u_sub**2 + v_sub**2)

    x_full = np.linspace(0.0, lx, nx)
    y_full = np.linspace(0.0, ly, ny)
    x_sub = x_full[x_range[0] : x_range[1]]
    y_sub = y_full[y_range[0] : y_range[1]]

    fig, axes = plt.subplots(2, len(snapshot_indices), figsize=(3.5 * len(snapshot_indices), 7), constrained_layout=True)

    for col_idx, snapshot_idx in enumerate(snapshot_indices):
        full_axis = axes[0, col_idx]
        sub_axis = axes[1, col_idx]

        full_im = full_axis.imshow(
            full_speed[snapshot_idx].T,
            origin="lower",
            extent=[0.0, lx, 0.0, ly],
            cmap="viridis",
            aspect="equal",
        )
        full_axis.add_patch(
            Rectangle(
                (x_full[x_range[0]], y_full[y_range[0]]),
                x_full[x_range[1] - 1] - x_full[x_range[0]],
                y_full[y_range[1] - 1] - y_full[y_range[0]],
                fill=False,
                edgecolor="white",
                linewidth=1.5,
            )
        )
        full_axis.set_title(f"full t={snapshot_idx}")
        full_axis.set_xlabel("x")
        if col_idx == 0:
            full_axis.set_ylabel("y")

        sub_im = sub_axis.imshow(
            sub_speed[snapshot_idx].T,
            origin="lower",
            extent=[x_sub[0], x_sub[-1], y_sub[0], y_sub[-1]],
            cmap="viridis",
            aspect="equal",
        )
        sub_axis.set_title(f"sub t={snapshot_idx}")
        sub_axis.set_xlabel("x")
        if col_idx == 0:
            sub_axis.set_ylabel("y")

    fig.colorbar(full_im, ax=axes[0, :], fraction=0.015, pad=0.02, label="speed")
    fig.colorbar(sub_im, ax=axes[1, :], fraction=0.015, pad=0.02, label="speed")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")

import numpy as np
import xarray as xr
from pathlib import Path


def generate_cylinder_wake_from_netcdf(
    dataset_path=None,
    total_steps=None,
    start_idx=0,
    end_idx=None,
    u_name="u",
    v_name="v",
    time_dim="tdim",
    x_dim="xdim",
    y_dim="ydim",
    engine="netcdf4",
):
    """Load cylinder-wake velocity fields from NetCDF and optionally resample.

    Args:
        dataset_path: Path to the NetCDF dataset. If None, uses
            data_generation/cylinder2d.nc.
        total_steps: Optional number of snapshots to return after resampling.
        start_idx: First raw time index to include.
        end_idx: Last raw time index to include, inclusive.
        u_name: Name of the u-velocity variable in the dataset.
        v_name: Name of the v-velocity variable in the dataset.
        time_dim: Name of the time dimension.
        x_dim: Name of the x-grid dimension.
        y_dim: Name of the y-grid dimension.
        engine: xarray NetCDF backend engine.

    Returns:
        Tuple (u, v, meta), where:
            u: Array shaped (T, nx, ny).
            v: Array shaped (T, nx, ny).
            meta: Dictionary with nx, ny, lx, ly, dt, and selected indices.
    """

    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parent / "cylinder2d.nc"

    ds = xr.open_dataset(dataset_path, engine=engine)
    try:
        missing_vars = [name for name in (u_name, v_name) if name not in ds.variables]
        if missing_vars:
            raise ValueError(f"Missing velocity variables in dataset: {missing_vars}")

        missing_dims = [name for name in (time_dim, x_dim, y_dim) if name not in ds.coords]
        if missing_dims:
            raise ValueError(f"Missing coordinate variables in dataset: {missing_dims}")

        # Keep state layout consistent with pipeline expectations: (time, x, y).
        u = ds[u_name].transpose(time_dim, x_dim, y_dim).values
        v = ds[v_name].transpose(time_dim, x_dim, y_dim).values

        if u.shape != v.shape:
            raise ValueError("u and v must have the same shape in the NetCDF dataset")
        if u.ndim != 3:
            raise ValueError("Expected velocity arrays with shape (time, x, y)")

        n_time, nx, ny = u.shape
        if n_time < 2:
            raise ValueError("Dataset must contain at least 2 timesteps")

        start = int(start_idx)
        stop = n_time - 1 if end_idx is None else int(end_idx)
        if start < 0 or stop < start or stop >= n_time:
            raise ValueError(
                f"Invalid index window start={start}, end={stop} for n_time={n_time}"
            )

        raw_idx = np.arange(start, stop + 1, dtype=int)
        u_sel = u[raw_idx]
        v_sel = v[raw_idx]

        if total_steps is not None:
            steps = int(total_steps)
            if steps <= 1:
                raise ValueError("total_steps must be > 1 when provided")
            # Uniformly sample across the selected raw segment.
            local_idx = np.linspace(0, u_sel.shape[0] - 1, steps, dtype=int)
            selected_idx = raw_idx[local_idx]
            u_out = u_sel[local_idx]
            v_out = v_sel[local_idx]
        else:
            selected_idx = raw_idx
            u_out = u_sel
            v_out = v_sel

        x_vals = ds[x_dim].values
        y_vals = ds[y_dim].values
        t_vals = ds[time_dim].values

        lx = float(np.max(x_vals) - np.min(x_vals))
        ly = float(np.max(y_vals) - np.min(y_vals))

        dt_raw = t_vals[1] - t_vals[0]
        if np.issubdtype(type(dt_raw), np.timedelta64):
            dt = float(dt_raw / np.timedelta64(1, "s"))
        else:
            dt = float(dt_raw)

        meta = {
            "nx": int(nx),
            "ny": int(ny),
            "lx": lx,
            "ly": ly,
            "dt": dt,
            "n_time_available": int(n_time),
            "selected_indices": selected_idx,
        }

        return u_out, v_out, meta
    finally:
        ds.close()

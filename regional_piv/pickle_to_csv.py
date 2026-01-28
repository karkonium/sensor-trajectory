import os
import pickle
import numpy as np

def regional_piv_pickle_to_table(
    pickle_path,
    out_path,
    *,
    dt,
    as_velocity,
    add_speed,
):
    """
    Convert regional_piv_*.pickle (dict from regional_local_optimal_direction_series)
    into a flat table suitable for MATLAB.

    Output columns (CSV/XLSX):
      k, s_frame, e_frame, t_center, tile_id, x, y, u, v, (speed)

    Parameters
    ----------
    pickle_path : str
        Path to regional_piv_<name>.pickle
    out_path : str
        Output path ending in .csv or .xlsx
    dt : float
        Physical time per frame (used to compute t_center, and tau if as_velocity=True).
    as_velocity : bool
        If False: u,v are "move vectors" (displacement-like pointing vectors).
        If True : u,v are converted to a velocity by dividing by tau = time_window * dt.
                (Matches your FTLE conversion choice.)
    add_speed : bool
        Add speed = sqrt(u^2+v^2)

    Notes
    -----
    - move_series shape: (K, M, 2)
    - centers_xy shape: (M, 2)
    - intervals: list[(s,e)] of length K, in frame indices
    - meta contains time_window (W frames) and time_step
    """
    with open(pickle_path, "rb") as f:
        reg = pickle.load(f)

    move_series = np.asarray(reg["move_series"], dtype=float)   # (K,M,2)
    centers_xy  = np.asarray(reg["centers_xy"], dtype=float)    # (M,2)
    intervals   = reg.get("intervals", None)
    meta        = reg.get("meta", {})

    K, M, _ = move_series.shape

    # Convert to velocity if requested
    if as_velocity:
        W = int(meta.get("time_window", 1))
        tau = float(W) * float(dt)
        if tau <= 0:
            raise ValueError("tau <= 0; check dt and meta['time_window']")
        uv = move_series / tau
    else:
        uv = move_series

    # Build rows
    # Columns: k, s_frame, e_frame, t_center, tile_id, x, y, u, v, speed?
    rows = []
    for k in range(K):
        if intervals is not None:
            s_frame, e_frame = intervals[k]
            t_center = 0.5 * (s_frame + e_frame) * float(dt)
        else:
            # fallback if intervals missing
            s_frame, e_frame = np.nan, np.nan
            t_center = np.nan

        u_k = uv[k, :, 0]
        v_k = uv[k, :, 1]

        if add_speed:
            sp = np.sqrt(u_k**2 + v_k**2)
        else:
            sp = None

        for m in range(M):
            x, y = centers_xy[m, 0], centers_xy[m, 1]
            if add_speed:
                rows.append([k, s_frame, e_frame, t_center, m, x, y, u_k[m], v_k[m], sp[m]])
            else:
                rows.append([k, s_frame, e_frame, t_center, m, x, y, u_k[m], v_k[m]])

    # Write output
    header = ["k", "s_frame", "e_frame", "t_center", "tile_id", "x", "y", "u", "v"]
    if add_speed:
        header.append("speed")

    out_ext = os.path.splitext(out_path)[1].lower()
    if out_ext == ".csv":
        import csv
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"[OK] Wrote CSV: {out_path}")

    elif out_ext in (".xlsx", ".xls"):
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "regional_piv"

        ws.append(header)
        for r in rows:
            ws.append(r)

        wb.save(out_path)
        print(f"[OK] Wrote Excel: {out_path}")

    else:
        raise ValueError("out_path must end with .csv or .xlsx")


if __name__ == "__main__":
    # Example usage:
    # python pickle_to_regional_piv_table.py
    name = "double_gyre"
    results_dir = f"ftle_series_{name}"
    pickle_path = os.path.join(results_dir, f"regional_piv_{name}.pickle")

    # Use the same dt you used when generating that flow case
    dt = 1.0  # set appropriately (e.g., DT_k for kolmogorov)

    out_csv = os.path.join(results_dir, f"regional_piv_{name}.csv")

    regional_piv_pickle_to_table(pickle_path, out_csv, dt=dt, as_velocity=False, add_speed=True)
# run_regional_pod_ftle.py

import os
import numpy as np
import matplotlib.pyplot as plt

from data_generation import *
from data_tranformation import *
from plotting import *
from state_concatenation import *

from regional_piv import regional_local_optimal_direction_series
from lcs import *


WINDOW_LEN = 20
MAX_BASIS_DIM = 11  # you still use this for global POD in plotting
TOTAL_STEPS = WINDOW_LEN + 30
PERIOD = TOTAL_STEPS * 10

NX, NY = 300, 300
LX, LY = 1.0, 1.0
U_max = 1.02
DT = 0.05 / U_max

# generate data
u, v = generate_moving_vortex(TOTAL_STEPS, NX, NY, LX, LY, period=PERIOD)

# how many threads to use for joblib
N_JOBS = int(os.environ.get("PYTHON_THREADS", "1"))

res = regional_local_optimal_direction_series(
    u, v, LX, LY, DT,
    phys_window=(LX * 0.2, LY * 0.2),
    time_window=WINDOW_LEN,
    out_nx=(NX // 5), out_ny=(NY // 5),
    time_step=1,
    scale_mode="mean_radius",
    fixed_scale=None,
    plot_every=1,
    show=False,           # plotting off for HPC run
    save_plots=True,
    parallel=True,        # <--- enable parallel windows
    n_jobs=N_JOBS,
)

# Compute FTLE (using τ=W*dt by default)
ftle_fwd, ftle_bwd, x, y, V_grid = compute_ftle_from_optimal_direction(
    res, u, v, LX, LY,
    time_window=WINDOW_LEN, dt=DT, time_step=1,
    tau=None,
)

# Basic plots (optional; on HPC you might save instead of show)
plot_ftle(ftle_fwd, ftle_bwd, LX, LY, pad_frac=0.05, ridge_pct=92)
quiver_row(res, 1, N=5)

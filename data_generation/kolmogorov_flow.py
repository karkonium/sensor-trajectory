import json, hashlib, tempfile, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from fluidsim.solvers.ns2d.solver import Simul as SimulBase
from fluidsim.base.forcing.kolmogorov import extend_simul_class, KolmogorovFlow

Simul = extend_simul_class(SimulBase, KolmogorovFlow)

CURRENT_DIR = Path(__file__).resolve().parent
CACHE_DIR = CURRENT_DIR / ".flow_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _param_hash(dic: dict) -> str:
    return hashlib.md5(json.dumps(dic, sort_keys=True).encode()).hexdigest()


def generate_cfd_kolmogorov_flow(n_timesteps: int,
                                 nx: int,
                                 ny: int,
                                 lx: float = 2*np.pi,
                                 ly: float = 2*np.pi,
                                 dt: float = 1e-3,
                                 nu: float = 1e-3,
                                 forcing_amp: float = 0.1,
                                 kf: int = 4,
                                 use_cache: bool = True,
                                 cache_dir: Path = CACHE_DIR,
                                 plot_series: bool = False,
                                 plot_every: int = 1):
    """
    Returns
    -------
    u_field, v_field : ndarray
        Shapes (n_timesteps, nx, ny)
    """
    # cache lookup
    key = _param_hash(dict(n_timesteps=n_timesteps, nx=nx, ny=ny, lx=lx, ly=ly,
                           dt=dt, nu=nu, forcing_amp=forcing_amp, kf=kf))
    print(cache_dir)
    cache_file = cache_dir / f"kolmo_{key}.npz"
    if use_cache and cache_file.exists():
        data = np.load(cache_file)
        u_field, v_field = data["u_field"], data["v_field"]
    else:
        # FluidSim parameter 
        params = Simul.create_default_params()
        params.oper.nx, params.oper.ny = nx, ny
        params.oper.Lx, params.oper.Ly = lx, ly
        params.oper.type_fft = "fft2d.with_pyfftw"

        params.nu_2 = nu

        # Fixed time step; we step manually
        params.time_stepping.USE_CFL = False
        params.time_stepping.deltat0 = dt
        params.time_stepping.t_end = n_timesteps * dt

        # Kolmogorov forcing
        params.forcing.enable = True
        params.forcing.type = "kolmogorov_flow"
        params.forcing.kolmo.ik = kf
        params.forcing.kolmo.amplitude = forcing_amp

        # Noise initial condition 
        params.init_fields.type = "noise"
        params.init_fields.noise.length = ly / kf 

        # Silence outputs and save nothing on disk
        params.output.sub_directory = tempfile.mkdtemp()
        params.output.HAS_TO_SAVE = False
        params.output.periods_print.print_stdout = n_timesteps + 1

        # run solver 
        sim = Simul(params)
        sim.state.statephys_from_statespect()  # create physical arrays 

        u_field = np.empty((n_timesteps, nx, ny), dtype=np.float32)
        v_field = np.empty_like(u_field)

        for it in range(n_timesteps):
            # store current physical fields
            u_field[it] = sim.state.get_var("ux").copy()
            v_field[it] = sim.state.get_var("uy").copy()

            # advance one RK4 step, except after the last snapshot
            if it < n_timesteps - 1:
                sim.time_stepping.one_time_step()
                sim.state.statephys_from_statespect()

        if use_cache:
            np.savez_compressed(cache_file, u_field=u_field, v_field=v_field)

    if plot_series:
        n_show = n_timesteps // plot_every
        idx = np.linspace(0, n_timesteps-1, n_show, dtype=int)
        x = np.linspace(0, lx, nx, endpoint=False)
        y = np.linspace(0, ly, ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        stride = max(1, nx // 32)

        fig, axs = plt.subplots(1, n_show, figsize=(4*n_show, 4))
        for ax, it_snap in zip(axs, idx):
            speed = np.sqrt(u_field[it_snap]**2 + v_field[it_snap]**2)
            im = ax.imshow(speed.T, origin="lower",
                           extent=[0, lx, 0, ly], cmap="viridis", aspect='equal')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                      u_field[it_snap, ::stride, ::stride],
                      v_field[it_snap, ::stride, ::stride],
                      color='black',scale_units='xy', scale=None,
                      width=0.005, pivot='mid')
            ax.set_title(f"t = {it_snap*dt:.3f}")
            ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.tight_layout(); plt.show()

    return u_field, v_field


if __name__ == "__main__":
    u, v = generate_cfd_kolmogorov_flow(
        n_timesteps=2000,
        nx=128, ny=128,      
        dt=1e-3,              
        nu=2e-2, 
        forcing_amp=20.0, 
        kf=4,
        plot_series=True,
        plot_every=200)
    print("Snapshots:", u.shape, "  v-rms:", np.sqrt((v**2).mean()))

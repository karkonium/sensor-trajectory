import json, hashlib, tempfile, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from fluidsim.solvers.ns2d.solver import Simul as SimulBase
from fluidsim.base.forcing.taylor_green import extend_simul_class, TaylorGreen

Simul = extend_simul_class(SimulBase, TaylorGreen)

CURRENT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = CURRENT_DIR / ".flow_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _param_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def generate_cfd_taylor_green_flow(n_timesteps: int,
                                   nx: int, ny: int,
                                   lx: float = 2*np.pi,
                                   ly: float = 2*np.pi,
                                   dt: float = 1e-3,
                                   Re: float = 2000,
                                   forcing_amp: float = 1.0,   # F0
                                   k: int = 1,                 # wavenumber
                                   use_cache: bool = True,
                                   cache_dir: Path = CACHE_DIR,
                                   plot_series: bool = False,
                                   plot_every: int = 1):
    """
    Forced 2-D Taylor–Green flow.

    Parameters
    ----------
    n_timesteps : int
    nx, ny      : grid resolution
    lx, ly      : domain (default 2π × 2π)
    dt          : time step
    Re          : Reynolds number  (sets ν = U*L/Re, with U≈√F0 )
    forcing_amp : amplitude F0 of the TG body-force
    k           : integer wavenumber (cells per 2π)
    """

    key = _param_hash(dict(n_timesteps=n_timesteps, nx=nx, ny=ny, lx=lx, ly=ly,
                           dt=dt, Re=Re, forcing_amp=forcing_amp, k=k))
    cache_file = cache_dir / f"tg2d_forced_{key}.npz"
    if use_cache and cache_file.exists():
        data = np.load(cache_file)
        return data["u_field"], data["v_field"]

    # FluidSim parameters 
    params = Simul.create_default_params()
    params.oper.nx, params.oper.ny = nx, ny
    params.oper.Lx, params.oper.Ly = lx, ly
    params.oper.type_fft = "fft2d.with_pyfftw"

    # viscosity from Re  (use length L = lx/2π, velocity U ≈ √F0)
    Uchar = np.sqrt(forcing_amp)
    params.nu_2 = Uchar * (lx/(2*np.pi)) / Re

    params.time_stepping.USE_CFL = False
    params.time_stepping.deltat0 = dt
    params.time_stepping.t_end   = n_timesteps * dt   # guard only

    # TG forcing configuration
    params.forcing.enable = True
    params.forcing.type   = "taylor_green"
    params.forcing.taylor_green.amplitude = forcing_amp
    params.forcing.taylor_green.kx = k
    params.forcing.taylor_green.ky = k

    # small random noise to seed asymmetry
    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = 1e-3
    params.init_fields.noise.length   = ly / k

    # silence disk I/O and stdout
    params.output.sub_directory = tempfile.mkdtemp()
    params.output.HAS_TO_SAVE   = False
    params.output.periods_print.print_stdout = 0

    # construct solver 
    sim = Simul(params)
    sim.state.statephys_from_statespect()         # phys arrays exist
    sim.output.one_time_step = lambda : None      # mute output plugin

    u_field = np.empty((n_timesteps, nx, ny), dtype=np.float32)
    v_field = np.empty_like(u_field)

    # manual RK4 stepping 
    for it in range(n_timesteps):
        u_field[it] = sim.state.get_var("ux")
        v_field[it] = sim.state.get_var("uy")

        if it < n_timesteps - 1:
            sim.time_stepping.one_time_step()
            sim.state.statephys_from_statespect()

    if use_cache:
        np.savez_compressed(cache_file, u_field=u_field, v_field=v_field)

    # plotting 
    if plot_series:
        frames = np.arange(0, n_timesteps, plot_every)[:4]
        speed  = np.sqrt(u_field**2 + v_field**2)
        x = np.linspace(0, lx, nx, endpoint=False)
        y = np.linspace(0, ly, ny, endpoint=False)
        Xg, Yg = np.meshgrid(x, y, indexing="ij")
        stride = max(1, nx//32)

        fig, axs = plt.subplots(1, len(frames), figsize=(4*len(frames),4))
        for ax, t in zip(axs, frames):
            im = ax.imshow(speed[t].T, origin="lower",
                           extent=[0,lx,0,ly], cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.quiver(Xg[::stride,::stride], Yg[::stride,::stride],
                      u_field[t,::stride,::stride],
                      v_field[t,::stride,::stride],
                      color="black", scale_units="xy",
                      width=0.004, pivot="mid")
            ax.set_title(f"t={t*dt:.2f}")
        plt.tight_layout(); plt.show()

    return u_field, v_field


if __name__ == "__main__":
    u, v = generate_cfd_taylor_green_flow(
        n_timesteps=1500,
        nx=128, ny=128,
        dt=2e-3,
        Re=3000,
        forcing_amp=1.0,
        k=1,
        plot_series=True,
        plot_every=500,
        use_cache=False
    )
    print("Returned arrays:", u.shape, v.shape)

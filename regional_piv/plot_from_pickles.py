import os
import sys


REGIONAL_PIV_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REGIONAL_PIV_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
RESULTS_ROOT = os.environ.get(
    "REGIONAL_PIV_RESULTS_ROOT",
    os.path.join(REGIONAL_PIV_DIR, "results"),
)

CASES = [
    ("double_gyre", os.path.join(RESULTS_ROOT, "ftle_series_double_gyre")),
    ("kolmogorov", os.path.join(RESULTS_ROOT, "ftle_series_kolmogorov")),
    ("moving_vortex", os.path.join(RESULTS_ROOT, "ftle_series_moving_vortex")),
]

WHICH = "backward"
RIDGE_PCT = 92
QSKIP = 2
DURATION = 0.10
DPI = 150
TOTAL_STEPS = 150
PERIOD = 100


def _load_flow_case(name):
    import numpy as np
    from data_generation import (
        generate_cfd_kolmogorov_flow,
        generate_double_gyre_flow,
        generate_moving_vortex,
    )

    if name == "moving_vortex":
        nx, ny = 300, 300
        lx, ly = 1.0, 1.0
        dt = 1.0
        u, v = generate_moving_vortex(TOTAL_STEPS, nx, ny, lx, ly, period=PERIOD)
        return u, v, lx, ly, dt

    if name == "kolmogorov":
        nx, ny = 300, 300
        lx, ly = 2 * np.pi, 2 * np.pi
        dt_base = 1e-3
        u_full, v_full = generate_cfd_kolmogorov_flow(
            n_timesteps=2000,
            nx=nx,
            ny=ny,
            lx=lx,
            ly=ly,
            dt=dt_base,
            nu=2e-2,
            forcing_amp=20.0,
            kf=4,
            plot_series=False,
        )

        start = 1500
        end = 1900
        skip = max(1, int(round((end - start) / (TOTAL_STEPS - 1))))
        sample_indices = start + np.arange(TOTAL_STEPS) * skip
        sample_indices = sample_indices[sample_indices <= end]
        u = u_full[sample_indices]
        v = v_full[sample_indices]
        dt = skip * dt_base
        return u, v, lx, ly, dt

    if name == "double_gyre":
        nx, ny = 300, 150
        lx, ly = 2.0, 1.0
        dt = 1.0
        u, v = generate_double_gyre_flow(
            TOTAL_STEPS,
            nx,
            ny,
            lx,
            ly,
            A=0.1,
            epsilon=0.5,
            period=PERIOD,
        )
        return u, v, lx, ly, dt

    raise ValueError(f"Unsupported flow case for redraw: {name!r}")


def redraw_case(
    results_dir,
    name,
):
    import numpy as np

    from lcs import save_ftle_series_plots
    from plotting import (
        divergence_info_feild,
        dual_lcs_overlay_gif,
        fluid_vs_regional_flow_gif,
        flow_cosine_similarity_gif,
        info_lcs_overlay_gif_from_pickles,
        load_pickles,
        make_gif_from_dir,
    )

    print(f"[{name}] Loading pickles from {results_dir}", flush=True)
    missing = [
        path for path in (
            os.path.join(results_dir, f"regional_piv_{name}.pickle"),
            os.path.join(results_dir, f"ftle_{name}.pickle"),
        )
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required pickle file(s):\n"
            + "\n".join(f"  {path}" for path in missing)
            + f"\nResolved RESULTS_ROOT={RESULTS_ROOT}"
            + "\nSet REGIONAL_PIV_RESULTS_ROOT to override this location."
        )

    reg_piv, ftle = load_pickles(results_dir, name)
    print(f"[{name}] Reconstructing underlying flow field...", flush=True)
    u, v, lx, ly, dt = _load_flow_case(name)

    written = []

    basename = f"{name}_ftle"
    print(f"[{name}] Redrawing FTLE PNG frames...", flush=True)
    save_ftle_series_plots(
        np.asarray(ftle["ftle_forward"]),
        np.asarray(ftle["ftle_backward"]),
        np.asarray(ftle["x"]),
        np.asarray(ftle["y"]),
        np.asarray(ftle["t_centers"]),
        lx=lx,
        ly=ly,
        outdir=results_dir,
        basename=basename,
        pad_frac=0.05,
        ridge_pct=RIDGE_PCT,
        dpi=DPI,
        show=False,
    )

    gif_path = os.path.join(results_dir, f"{name}_ftle.gif")
    make_gif_from_dir(results_dir, gif_path, duration=DURATION)
    written.append(gif_path)

    print(f"[{name}] Redrawing FTLE + info-flow overlay GIF...", flush=True)
    written.append(
        info_lcs_overlay_gif_from_pickles(
            reg_piv=reg_piv,
            ftle=ftle,
            results_dir=results_dir,
            name=name,
            which=WHICH,
            ridge_pct=RIDGE_PCT,
            qskip=QSKIP,
            duration=DURATION,
        )
    )

    if v is not None:
        print(f"[{name}] Redrawing fluid-vs-InfoFlo GIF...", flush=True)
        written.append(
            fluid_vs_regional_flow_gif(
                reg_piv=reg_piv,
                results_dir=results_dir,
                name=name,
                u=u,
                v=v,
                LX=lx,
                LY=ly,
                dt=dt,
                duration=DURATION,
            )
        )

        print(f"[{name}] Redrawing flow direction cosine-similarity GIF...", flush=True)
        written.append(
            flow_cosine_similarity_gif(
                reg_piv=reg_piv,
                results_dir=results_dir,
                name=name,
                u=u,
                v=v,
                LX=lx,
                LY=ly,
                dt=dt,
                duration=DURATION,
            )
        )

    print(f"[{name}] Redrawing info-flow divergence GIF...", flush=True)
    written.append(
        divergence_info_feild(
            reg_piv=reg_piv,
            results_dir=results_dir,
            name=name,
            u=u,
            v=v,
            LX=lx,
            LY=ly,
            dt=dt,
            ftle=ftle,
            duration=DURATION,
        )
    )

    print(f"[{name}] Redrawing dual-LCS mean-flow GIF...", flush=True)
    written.append(
        dual_lcs_overlay_gif(
            reg_piv=reg_piv,
            ftle=ftle,
            results_dir=results_dir,
            name=name,
            u=u,
            v=v,
            LX=lx,
            LY=ly,
            ridge_pct=RIDGE_PCT,
            duration=DURATION,
        )
    )

    print(f"[{name}] Done.", flush=True)
    return written


def main():
    print(f"Redrawing plots for {len(CASES)} hard-coded case(s).", flush=True)
    for name, results_dir in CASES:
        redraw_case(results_dir, name)


if __name__ == "__main__":
    main()

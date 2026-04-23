"""
2D isotropic kinetic energy spectrum utilities (Kolmogorov / 2D turbulence).

What this script does:
- Computes an isotropic kinetic energy spectrum E(k) from 2D velocity snapshots (u,v)
  by FFT -> per-mode energy -> radial binning in |k|.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from data_generation import generate_cfd_kolmogorov_flow
from plotting import load_pickles
from plot_style import (
    FLUID_LINE_COLOR,
    INFO_LINE_COLOR,
    REFERENCE_LINE_COLOR,
    SINGLE_PANEL_FIGSIZE,
    apply_axis_style,
    finalize_legend,
    presentation_plot_context,
    set_panel_title,
)



def _fft_wavenumbers_2d(NX: int, NY: int, LX: float, LY: float):
    """
    Return physical wavenumbers (rad/length) consistent with np.fft.fftn on a periodic domain.

    Assumes uniform grid:
        x_j = j * LX / NX, j=0..NX-1
        y_j = j * LY / NY, j=0..NY-1

    Returns:
        KX, KY : (NX, NY) arrays of wavenumber components
        K      : (NX, NY) array of wavenumber magnitudes sqrt(KX^2 + KY^2)
    """
    dx = LX / NX
    dy = LY / NY

    kx = 2.0 * np.pi * np.fft.fftfreq(NX, d=dx)  # rad/length
    ky = 2.0 * np.pi * np.fft.fftfreq(NY, d=dy)

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    return KX, KY, K


def _info_grid_from_reg_piv(reg_piv):
    """
    Infer coarse info-grid axes and domain lengths from regional PIV centers.

    centers_xy are tile centers, so domain length is approximated as:
        L ~ (max - min) + dx
    """
    out_nx = int(reg_piv["meta"]["centers_nx"])
    out_ny = int(reg_piv["meta"]["centers_ny"])
    xy = np.asarray(reg_piv["centers_xy"], dtype=float)
    XY = xy.reshape(out_nx, out_ny, 2)
    Xc = XY[..., 0]
    Yc = XY[..., 1]

    x_info = Xc[:, 0].copy()
    y_info = Yc[0, :].copy()

    dx_info = float(np.mean(np.diff(x_info))) if len(x_info) > 1 else 1.0
    dy_info = float(np.mean(np.diff(y_info))) if len(y_info) > 1 else 1.0

    LX_info = float(x_info[-1] - x_info[0] + dx_info)
    LY_info = float(y_info[-1] - y_info[0] + dy_info)
    return (out_nx, out_ny), (LX_info, LY_info), (x_info, y_info), (dx_info, dy_info)


def parseval_energy_check(u: np.ndarray, v: np.ndarray):
    """
    Quick sanity check: mean kinetic energy in physical space vs spectral space.

    With NumPy's FFT convention:
        u(x) <-> U(k) where inverse FFT includes 1/(NX*NY),
    then:
        mean(u^2) == sum(|U|^2) / (NX*NY)^2   (up to floating error)

    Returns:
        E_phys_mean : 0.5 * mean(u^2 + v^2)
        E_spec_mean : 0.5 * (sum(|U|^2)+sum(|V|^2)) / (NX*NY)^2
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    NX, NY = u.shape

    U = np.fft.fftn(u)
    V = np.fft.fftn(v)

    E_phys_mean = 0.5 * np.mean(u**2 + v**2)
    E_spec_mean = 0.5 * (np.sum(np.abs(U)**2) + np.sum(np.abs(V)**2)) / (NX * NY) ** 2
    return E_phys_mean, E_spec_mean



class IsotropicSpectrum2D:
    """
    Precomputes the radial binning for a given grid (NX,NY) and domain (LX,LY),
    so you can compute many spectra quickly and time-average them.

    Usage:
        spec = IsotropicSpectrum2D(NX, NY, LX, LY, nbins=200)
        k, Ek = spec.spectrum(u_snap, v_snap)
        k, Ek_mean = spec.time_average(u_series, v_series, t_indices=range(100,200))
    """

    def __init__(
        self,
        NX: int,
        NY: int,
        LX: float,
        LY: float,
        nbins: int = 200,
        kmax: float | None = None,
        bin_width: float | None = None,
        exclude_k0: bool = True,
    ):
        """
        Args:
            NX, NY     : grid size
            LX, LY     : domain size
            nbins      : number of radial bins (used if bin_width is None)
            kmax       : max |k| included (defaults to max representable |k|)
            bin_width  : if provided, use uniform linear bins of this width in |k|
                         (rad/length). If None, bins are linear from 0..kmax with nbins.
            exclude_k0 : drop the k=0 mode from the spectrum
        """
        self.NX = int(NX)
        self.NY = int(NY)
        self.LX = float(LX)
        self.LY = float(LY)

        # Build |k| array once
        _, _, K = _fft_wavenumbers_2d(self.NX, self.NY, self.LX, self.LY)
        K_flat = K.ravel()

        # Optional: exclude k=0 (mean / DC mode)
        if exclude_k0:
            mask = K_flat > 0
        else:
            mask = np.ones_like(K_flat, dtype=bool)

        self._mask_flat = mask
        K_flat = K_flat[mask]

        # Choose kmax
        if kmax is None:
            kmax = float(K_flat.max())
        self.kmax = float(kmax)

        # Choose bins
        if bin_width is None:
            # Linear bins from 0..kmax with nbins bins.
            # (Linear bins tend to be easier to interpret than log-bins for spectra.)
            self.nbins = int(nbins)
            self.k_edges = np.linspace(0.0, self.kmax, self.nbins + 1)
        else:
            # Uniform bins of a given width
            bin_width = float(bin_width)
            self.k_edges = np.arange(0.0, self.kmax + bin_width, bin_width)
            self.nbins = len(self.k_edges) - 1

        self.dk = np.diff(self.k_edges)  # (nbins,)
        self.k_centers = 0.5 * (self.k_edges[:-1] + self.k_edges[1:])  # (nbins,)

        # Precompute bin index per Fourier gridpoint (flattened)
        which = np.digitize(K_flat, self.k_edges) - 1  # in [0, nbins-1] ideally
        valid = (which >= 0) & (which < self.nbins)

        # Store only valid indices; keeps bincount clean
        self._K_valid = K_flat[valid]
        self._bin_idx = which[valid].astype(np.int64)
        self._valid_selector_in_masked = valid  # aligns with masked flattening
        self._window_cache = {}

    def _get_window(self, kind: str | None):
        if kind is None or str(kind).lower() in ("none", "off", "false"):
            return None
        kind_key = str(kind).lower()
        if kind_key in self._window_cache:
            return self._window_cache[kind_key]

        if kind_key in ("hann", "hanning"):
            wx = np.hanning(self.NX)
            wy = np.hanning(self.NY)
            W = np.outer(wx, wy)
        else:
            raise ValueError(f"Unsupported window kind: {kind!r} (use 'hann' or None).")

        # Normalize so mean energy is preserved on average
        w2_mean = float(np.mean(W**2))
        if w2_mean > 0:
            W = W / np.sqrt(w2_mean)

        self._window_cache[kind_key] = W
        return W

    def spectrum(
        self,
        u: np.ndarray,
        v: np.ndarray,
        detrend_mean: bool = True,
        window: str | None = None,
    ):
        """
        Compute isotropic kinetic energy spectrum E(k) for a single snapshot.

        Steps:
        1) optionally remove spatial mean from u and v (kills k=0 spike)
        2) FFT to U,V
        3) per-mode energy: 0.5(|U|^2+|V|^2)/(NX*NY)^2  (Parseval-consistent for mean energy)
        4) sum energy in rings (bins) of constant |k|
        5) divide by dk to get a density per unit k: integral E(k) dk ~ total mean energy

        Returns:
            k_centers : (nbins,) radial k values (rad/length)
            E_k       : (nbins,) energy density per unit k
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if u.shape != (self.NX, self.NY) or v.shape != (self.NX, self.NY):
            raise ValueError(f"Expected u,v shape ({self.NX},{self.NY}), got {u.shape} and {v.shape}")

        # Remove mean flow (especially important in 2D if there is a strong box-scale mode)
        if detrend_mean:
            u = u - u.mean()
            v = v - v.mean()

        # Optional taper for non-periodic fields
        W = self._get_window(window)
        if W is not None:
            u = u * W
            v = v * W

        U = np.fft.fftn(u)
        V = np.fft.fftn(v)

        # Energy per Fourier mode (mean-energy consistent with numpy FFT)
        E2 = 0.5 * (np.abs(U) ** 2 + np.abs(V) ** 2) / (self.NX * self.NY) ** 2

        # Flatten and apply the same k-mask we used when building bins
        E_flat_masked = E2.ravel()[self._mask_flat]

        # Apply "valid bins" selector (so bin_idx aligns with weights)
        E_weights = E_flat_masked[self._valid_selector_in_masked]

        # Sum energy into radial bins
        E_shell = np.bincount(self._bin_idx, weights=E_weights, minlength=self.nbins).astype(np.float64)

        # Convert shell energy -> density per unit k
        # (This makes ∫ E(k) dk ≈ total mean kinetic energy)
        E_k = np.divide(E_shell, self.dk, out=np.full_like(E_shell, np.nan), where=(self.dk > 0))

        return self.k_centers, E_k

    def time_average(
        self,
        u_series: np.ndarray,
        v_series: np.ndarray,
        t_indices=None,
        detrend_mean: bool = True,
        window: str | None = None,
    ):
        """
        Time-average E(k) over selected snapshots.

        Args:
            u_series, v_series : arrays shaped (T, NX, NY)
            t_indices          : iterable of time indices (default: all)
            detrend_mean       : remove mean from each snapshot before FFT

        Returns:
            k_centers : (nbins,)
            E_k_mean  : (nbins,)
        """
        u_series = np.asarray(u_series, dtype=np.float64)
        v_series = np.asarray(v_series, dtype=np.float64)

        if u_series.ndim != 3 or v_series.ndim != 3:
            raise ValueError("u_series and v_series must have shape (T, NX, NY).")

        if u_series.shape != v_series.shape:
            raise ValueError("u_series and v_series must have the same shape.")

        T, NX, NY = u_series.shape
        if (NX, NY) != (self.NX, self.NY):
            raise ValueError(f"Expected series shape (T,{self.NX},{self.NY}), got (T,{NX},{NY}).")

        if t_indices is None:
            t_indices = range(T)

        Ek_sum = np.zeros(self.nbins, dtype=np.float64)
        Ek_cnt = np.zeros(self.nbins, dtype=np.float64)

        for t in t_indices:
            k, Ek = self.spectrum(
                u_series[t],
                v_series[t],
                detrend_mean=detrend_mean,
                window=window,
            )
            m = np.isfinite(Ek)
            Ek_sum[m] += Ek[m]
            Ek_cnt[m] += 1.0

        Ek_mean = np.divide(Ek_sum, Ek_cnt, out=np.full_like(Ek_sum, np.nan), where=(Ek_cnt > 0))
        return self.k_centers, Ek_mean


def plot_fluid_spectrum(
    u_series: np.ndarray,
    v_series: np.ndarray,
    LX: float,
    LY: float,
    kf: float | None = None,
    nbins: int = 200,
    t_indices=None,
    save_path: str | None = None,
    title: str = "2D kinetic energy spectrum",
):
    """
    Compute a time-averaged E(k) and plot it on a log-log axis.

    Args:
        u_series, v_series : arrays shaped (T, NX, NY)
        LX, LY             : domain sizes
        kf                 : forcing wavenumber magnitude (rad/length). If our forcing is
                             e.g. sin(kf*y) in a 2π box with integer kf, then kf is that integer.
        nbins              : number of radial bins
        t_indices          : which times to average (e.g. range(20,120))
        save_path          : if provided, saves figure to this file
        title              : plot title
    """
    u_series = np.asarray(u_series)
    v_series = np.asarray(v_series)
    T, NX, NY = u_series.shape

    # Precompute binner once
    spec = IsotropicSpectrum2D(NX, NY, LX, LY, nbins=nbins)

    # Time-averaged spectrum
    k, Ek = spec.time_average(u_series, v_series, t_indices=t_indices, detrend_mean=True)

    # Basic cleanup for plotting
    m = np.isfinite(Ek) & (Ek > 0) & (k > 0)
    k_plot = k[m]
    Ek_plot = Ek[m]

    with presentation_plot_context():
        fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE, constrained_layout=True)
        ax.loglog(k_plot, Ek_plot, color=FLUID_LINE_COLOR, label="Fluid (time-avg)")

        ax.set_xlabel("wavenumber $k$ (rad / length)")
        ax.set_ylabel("$E(k)$ (energy density per unit $k$)")
        set_panel_title(ax, title)
        apply_axis_style(ax, x_grid=True, y_grid=True)

        if kf is not None and kf > 0:
            ax.axvline(kf, linestyle=":", linewidth=1.35, color=REFERENCE_LINE_COLOR, label=f"$k_f={kf}$")

            def _anchor_at(k_target):
                idx = np.argmin(np.abs(np.log(k_plot) - np.log(k_target)))
                return k_plot[idx], Ek_plot[idx]

            if np.any(k_plot < kf):
                k0, y0 = _anchor_at(max(k_plot.min(), 0.7 * kf))
                ax.loglog(
                    k_plot,
                    y0 * (k_plot / k0) ** (-5 / 3),
                    linestyle="--",
                    linewidth=1.2,
                    color=REFERENCE_LINE_COLOR,
                    label=r"$k^{-5/3}$ (ref, inverse side)",
                )

            if np.any(k_plot > kf):
                k0, y0 = _anchor_at(min(k_plot.max(), 2.0 * kf))
                ax.loglog(
                    k_plot,
                    y0 * (k_plot / k0) ** (-3),
                    linestyle="--",
                    linewidth=1.2,
                    color="#667085",
                    label=r"$k^{-3}$ (ref, forward side)",
                )

        finalize_legend(ax, loc="best")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=180)
            plt.close(fig)
        else:
            plt.show()


def plot_fluid_vs_info_spectrum(
    u_series: np.ndarray,
    v_series: np.ndarray,
    LX: float,
    LY: float,
    reg_piv: dict,
    dt: float = 1.0,
    kf: float | None = None,
    nbins: int = 200,
    fluid_t_indices=None,
    info_k_indices=None,
    info_window: str | None = "hann",
    save_path: str | None = None,
    title: str = "Energy spectra: fluid vs info-flow",
):
    """
    Compare time-averaged spectra for:
      - Fluid velocity (full grid, periodic)
      - Info-flow velocity (regional PIV grid, non-periodic -> windowed)
    """
    u_series = np.asarray(u_series)
    v_series = np.asarray(v_series)

    # Fluid spectrum
    T, NX, NY = u_series.shape
    spec_fluid = IsotropicSpectrum2D(NX, NY, LX, LY, nbins=nbins)
    k_fluid, Ek_fluid = spec_fluid.time_average(
        u_series, v_series, t_indices=fluid_t_indices, detrend_mean=True, window=None
    )

    # Info-flow spectrum
    move_grid = np.asarray(reg_piv["move_grid"], dtype=float)  # (K,out_nx,out_ny,2)
    W = int(reg_piv["meta"]["time_window"])
    tau = float(W * dt)
    V_info = move_grid / max(tau, 1e-12)

    (out_nx, out_ny), (LX_info, LY_info), _, _ = _info_grid_from_reg_piv(reg_piv)
    spec_info = IsotropicSpectrum2D(out_nx, out_ny, LX_info, LY_info, nbins=nbins)

    u_info = V_info[..., 0]
    v_info = V_info[..., 1]
    k_info, Ek_info = spec_info.time_average(
        u_info, v_info, t_indices=info_k_indices, detrend_mean=True, window=info_window
    )

    m_fluid = np.isfinite(Ek_fluid) & (Ek_fluid > 0) & (k_fluid > 0)
    m_info = np.isfinite(Ek_info) & (Ek_info > 0) & (k_info > 0)

    k_plot = k_fluid[m_fluid]
    Ek_plot = Ek_fluid[m_fluid]

    with presentation_plot_context():
        fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE, constrained_layout=True)
        ax.loglog(k_plot, Ek_plot, color=FLUID_LINE_COLOR, label="Fluid (time-avg)")
        ax.loglog(
            k_info[m_info],
            Ek_info[m_info],
            color=INFO_LINE_COLOR,
            linestyle=(0, (5, 2)),
            label=f"Info-flow (windowed: {info_window})",
        )

        ax.set_xlabel("wavenumber $k$ (rad / length)")
        ax.set_ylabel("$E(k)$ (energy density per unit $k$)")
        set_panel_title(ax, title)
        apply_axis_style(ax, x_grid=True, y_grid=True)

        if kf is not None and kf > 0:
            ax.axvline(kf, linestyle=":", linewidth=1.35, color=REFERENCE_LINE_COLOR, label=f"$k_f={kf}$")

            def _anchor_at(k_target):
                idx = np.argmin(np.abs(np.log(k_plot) - np.log(k_target)))
                return k_plot[idx], Ek_plot[idx]

            if np.any(k_plot < kf):
                k0, y0 = _anchor_at(max(k_plot.min(), 0.7 * kf))
                ax.loglog(
                    k_plot,
                    y0 * (k_plot / k0) ** (-5 / 3),
                    linestyle="--",
                    linewidth=1.2,
                    color=REFERENCE_LINE_COLOR,
                    label=r"$k^{-5/3}$ (ref, inverse side)",
                )

            if np.any(k_plot > kf):
                k0, y0 = _anchor_at(min(k_plot.max(), 2.0 * kf))
                ax.loglog(
                    k_plot,
                    y0 * (k_plot / k0) ** (-3),
                    linestyle="--",
                    linewidth=1.2,
                    color="#667085",
                    label=r"$k^{-3}$ (ref, forward side)",
                )

        finalize_legend(ax, loc="best")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=180)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    """
    Example: generate 2D Kolmogorov flow and plot time-averaged spectrum.

    IMPORTANT FIX vs our original snippet:
    - Removed the DEBUG override that forced T=1 (that prevented time-averaging).
    """

    name = "kolmogorov"

    # Simulation params (ours)
    NX, NY = 900, 900
    LX, LY = 2 * np.pi, 2 * np.pi
    DT_base = 1e-4

    # Kolmogorov forcing scale (in a 2π box, integer kf corresponds to physical k magnitude)
    kf = 10

    # Generate velocity snapshots (our generator must return arrays shaped (T, NX, NY))
    u_full, v_full = generate_cfd_kolmogorov_flow(
        n_timesteps=20000,
        nx=NX, ny=NY,
        lx=LX, ly=LY,
        dt=DT_base,
        nu=2e-2,
        forcing_amp=20.0,
        kf=kf,
        plot_series=False,
    )

    # Subsample to reduce storage/plot cost (ours)
    TOTAL_STEPS = 150
    sample_indices = np.linspace(1500, 1900, TOTAL_STEPS, dtype=int)

    u = u_full[sample_indices]
    v = v_full[sample_indices]
    DT_base = 1e-3 # because we output something already subsampled
    k_skip = sample_indices[1] - sample_indices[0]

    DT = k_skip * DT_base

    # OPTIONAL: check Parseval once on a representative snapshot
    E_phys, E_spec = parseval_energy_check(u[0] - u[0].mean(), v[0] - v[0].mean())
    print(f"Parseval check (mean energy): physical={E_phys:.6e}, spectral={E_spec:.6e}, ratio={E_spec/E_phys:.6f}")

    # Choose averaging window 
    fluid_t_indices = range(20, 120)

    results_dir = f"ftle_series_{name}"
    save_path = os.path.join(results_dir, "spectra_fluid_only.png")

    plot_fluid_spectrum(
        u_series=u,
        v_series=v,
        LX=LX, LY=LY,
        kf=kf,
        nbins=220,                 # more bins = smoother 
        t_indices=fluid_t_indices, # time-average window
        save_path=save_path,
        title="2D Kolmogorov flow: time-averaged kinetic energy spectrum",
    )

    print(f"Saved: {save_path}")

    # Info-flow spectrum (from regional PIV pickles)
    results_dir = f"ftle_series_{name}"
    reg_piv, ftle = load_pickles(results_dir, name)

    save_path_info = os.path.join(results_dir, "spectra_fluid_vs_info.png")
    plot_fluid_vs_info_spectrum(
        u_series=u,
        v_series=v,
        LX=LX, LY=LY,
        reg_piv=reg_piv,
        dt=DT,
        kf=kf,
        nbins=180,
        fluid_t_indices=fluid_t_indices,
        info_k_indices=None,      # all info snapshots
        info_window="hann",       # taper for non-periodic info flow
        save_path=save_path_info,
        title="Kolmogorov: fluid vs info-flow spectra",
    )
    print(f"Saved: {save_path_info}")

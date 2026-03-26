"""Standardized flow generation and sampling cases shared by experiments."""

from dataclasses import dataclass

import numpy as np

from data_generation.cylinder_wake_flow import generate_cylinder_wake_from_netcdf
from data_generation.double_gyre import generate_double_gyre_flow
from data_generation.kolmogorov_flow import generate_cfd_kolmogorov_flow
from data_generation.moving_vortex import generate_moving_vortex

from .config import DomainConfig


@dataclass
class FlowCasePayload:
    """Container for one standardized flow experiment input."""

    flow_name: str
    u: np.ndarray
    v: np.ndarray
    domain_config: DomainConfig
    dt_actual: float
    is_periodic: bool


def _sample_kolmogorov_segment(u, v, total_steps, start_idx, end_idx, original_dt):
    """Subsample a requested index range from Kolmogorov snapshots.

    Args:
        u: u snapshots shaped (T_available, nx, ny).
        v: v snapshots shaped (T_available, nx, ny).
        total_steps: Number of output snapshots to select.
        start_idx: Start index of the segment to sample.
        end_idx: End index of the segment to sample, inclusive.
        original_dt: Time spacing between consecutive stored Kolmogorov snapshots.

    Returns:
        Tuple (u_sel, v_sel, idx_sel, dt_advect) with selected snapshots, used indices,
        and the effective time spacing after subsampling.
    """
    if u.shape != v.shape:
        raise ValueError("u and v must have identical shape")
    if total_steps <= 1:
        raise ValueError("total_steps must be > 1")
    if original_dt <= 0.0:
        raise ValueError("original_dt must be > 0")

    n_available = int(u.shape[0])
    if n_available < 2:
        raise ValueError("Kolmogorov output must contain at least 2 snapshots")

    segment_start = max(0, int(start_idx))
    segment_end = min(int(end_idx), n_available - 1)
    if segment_end < segment_start:
        raise ValueError(
            f"Invalid segment [{start_idx}, {end_idx}] for available snapshots {n_available}"
        )

    stride = (segment_end - segment_start) // (total_steps - 1)
    idx_sel = segment_start + stride * np.arange(total_steps)    
    k_skip = int(idx_sel[1] - idx_sel[0])
    if k_skip <= 0 or k_skip != stride:
        raise ValueError(
            f"Kolmogorov subsampling produced invalid spacing {k_skip} for total_steps={total_steps}"
        )

    dt_advect = float(k_skip * original_dt)
    return u[idx_sel], v[idx_sel], idx_sel, dt_advect


def _generate_double_gyre_case(total_steps, period):
    """Generate the canonical double-gyre test case.

    Args:
        total_steps: Number of snapshots.
        period: Flow period.

    Returns:
        FlowCasePayload for double gyre.
    """
    nx, ny = 600, 300
    lx, ly = 2.0, 1.0
    u, v = generate_double_gyre_flow(
        total_steps,
        nx,
        ny,
        lx,
        ly,
        A=0.1,
        epsilon=0.5,
        period=period,
    )
    return FlowCasePayload(
        flow_name="double_gyre",
        u=u,
        v=v,
        domain_config=DomainConfig(nx=nx, ny=ny, lx=lx, ly=ly),
        dt_actual=1.0,
        is_periodic=False,
    )


def _generate_moving_vortex_case(total_steps, period):
    """Generate the canonical moving-vortex test case.

    Args:
        total_steps: Number of snapshots.
        period: Flow period.

    Returns:
        FlowCasePayload for moving vortex.
    """
    nx, ny = 600, 600
    lx, ly = 1.0, 1.0
    u, v = generate_moving_vortex(total_steps, nx, ny, lx, ly, period=period)
    return FlowCasePayload(
        flow_name="moving_vortex",
        u=u,
        v=v,
        domain_config=DomainConfig(nx=nx, ny=ny, lx=lx, ly=ly),
        dt_actual=1.0,
        is_periodic=False,
    )


def _generate_kolmogorov_case(total_steps):
    """Generate the canonical Kolmogorov-flow test case.

    Args:
        total_steps: Number of snapshots used by experiment runs.

    Returns:
        FlowCasePayload for kolmogorov flow.
    """
    nx, ny = 300, 300
    solver_dt = 1e-3
    stored_snapshot_dt = 1e-3

    u_raw, v_raw = generate_cfd_kolmogorov_flow(
        n_timesteps=2000,
        nx=nx,
        ny=ny,
        dt=solver_dt,
        nu=2e-2,
        forcing_amp=20.0,
        kf=4,
        plot_series=False,
    )

    u, v, selected_idx, dt_advect = _sample_kolmogorov_segment(
        u_raw,
        v_raw,
        total_steps=total_steps,
        start_idx=1100,
        end_idx=1999,
        original_dt=stored_snapshot_dt,
    )

    print(
        f"Kolmogorov selection: available={u_raw.shape[0]} snapshots, "
        f"selected idx range=[{int(selected_idx[0])}, {int(selected_idx[-1])}], "
        f"k_skip={int(selected_idx[1] - selected_idx[0])}, dt_advect={dt_advect}"
    )

    return FlowCasePayload(
        flow_name="kolmogorov",
        u=u,
        v=v,
        domain_config=DomainConfig(nx=nx, ny=ny, lx=2 * np.pi, ly=2 * np.pi),
        dt_actual=dt_advect,
        is_periodic=True,
    )


def _generate_cylinder_wake_case():
    """Load and sample the canonical precomputed cylinder-wake test case.

    Args:
        None.

    Returns:
        FlowCasePayload for cylinder wake.
    """
    u_raw, v_raw, meta = generate_cylinder_wake_from_netcdf()

    cylinder_steps = 400
    sample_indices = np.linspace(
        int(u_raw.shape[0] // 2),
        int(u_raw.shape[0] - 1),
        cylinder_steps,
        dtype=int,
    )
    u = u_raw[sample_indices]
    v = v_raw[sample_indices]

    k_skip = int(sample_indices[1] - sample_indices[0])
    dt_advect = float(k_skip * meta["dt"])

    print(
        "Cylinder wake selection: "
        f"available={meta['n_time_available']} snapshots, "
        f"selected={u.shape[0]}, "
        f"idx range=[{int(sample_indices[0])}, {int(sample_indices[-1])}], "
        f"k_skip={k_skip}, dt_advect={dt_advect}"
    )

    return FlowCasePayload(
        flow_name="cylinder_wake",
        u=u,
        v=v,
        domain_config=DomainConfig(
            nx=meta["nx"],
            ny=meta["ny"],
            lx=meta["lx"],
            ly=meta["ly"],
        ),
        dt_actual=dt_advect,
        is_periodic=False,
    )


def generate_standard_flow_cases(total_steps=160, period=80, flow_names=None):
    """Generate standardized flow payloads for selected flow names.

    Args:
        total_steps: Number of output snapshots for generated flows.
        period: Shared period argument for supported generators.
        flow_names: Optional ordered list of flow names to include.

    Returns:
        List of FlowCasePayload objects in requested order.
    """
    default_order = ["double_gyre", "moving_vortex", "kolmogorov", "cylinder_wake"]
    selected_flow_names = list(flow_names) if flow_names is not None else default_order

    payloads = []
    for flow_name in selected_flow_names:
        if flow_name == "double_gyre":
            payloads.append(_generate_double_gyre_case(total_steps, period))
            continue
        if flow_name == "moving_vortex":
            payloads.append(_generate_moving_vortex_case(total_steps, period))
            continue
        if flow_name == "kolmogorov":
            payloads.append(_generate_kolmogorov_case(total_steps))
            continue
        if flow_name == "cylinder_wake":
            payloads.append(_generate_cylinder_wake_case())
            continue

        raise ValueError(f"Unsupported flow case name: {flow_name!r}")

    return payloads

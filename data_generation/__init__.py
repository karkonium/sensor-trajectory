from .double_gyre import generate_double_gyre_flow
from .moving_vortex import generate_moving_vortex
from .simple_flow import generate_simple_flow
from .kolmogorov_flow import generate_cfd_kolmogorov_flow

__all__ = [
    "generate_double_gyre_flow",
    "generate_moving_vortex",
    "generate_simple_flow",
    "generate_cfd_kolmogorov_flow"
]
import numpy as np

def to_complex_cartesian(u, v):
    return u + 1j * v

def to_complex_polar(u, v):
    r = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)
    return r * np.exp(1j * theta)
"""Shared sliding-window utilities."""


def get_sliding_intervals(n_timesteps, window, step):
    """Build fixed-length sliding intervals.

    Args:
        n_timesteps: Total number of time steps.
        window: Window length.
        step: Shift between consecutive windows.

    Returns:
        List of (start, end) index tuples.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")

    intervals = []
    end = window
    while end <= n_timesteps:
        start = end - window
        intervals.append((start, end))
        end += step

    return intervals

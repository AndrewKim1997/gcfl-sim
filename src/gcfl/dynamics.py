"""
Deterministic state-update maps and utilities.

These helpers are optional and can be swapped for your paper-specific dynamics.
They are written to be numerically stable and testable.
"""
from __future__ import annotations
from typing import Callable, Iterable, Tuple
import numpy as np

MapFn = Callable[[np.ndarray], np.ndarray]

def linear_damped_towards_scalar(alpha: float, step: float = 0.05) -> MapFn:
    """
    Return a map f(u) = (1 - step*alpha) u + step*alpha * m,
    where m must be supplied by partially binding this function or
    by closing over m in a lambda. This factory only sets the damping.
    """
    damp = max(0.0, 1.0 - step * float(alpha))
    def _map(u: np.ndarray, m: float = 0.0) -> np.ndarray:
        return damp * u + (1.0 - damp) * float(m)
    return _map  # usage: f = linear_damped_towards_scalar(alpha); f(u, m=m_val)

def logistic_clip(a: float) -> MapFn:
    """
    Classic logistic-like map with gentle clipping to avoid overflow:
        f(u) = u + a * u * (1 - u), evaluated elementwise and clipped to [-1e6, 1e6].
    """
    a = float(a)
    def _map(u: np.ndarray) -> np.ndarray:
        v = u + a * u * (1.0 - u)
        return np.clip(v, -1e6, 1e6)
    return _map

def iterate_map(u0: np.ndarray, f: Callable[..., np.ndarray], T: int, **kwargs) -> np.ndarray:
    """
    Iterate u_{t+1} = f(u_t, **kwargs) for T steps; returns the final state.
    """
    u = np.asarray(u0, dtype=float)
    for _ in range(int(T)):
        u = f(u, **kwargs)
    return u

def trajectory(u0: np.ndarray, f: Callable[..., np.ndarray], T: int, **kwargs) -> np.ndarray:
    """
    Return the full trajectory matrix of shape (T+1, *u.shape).
    """
    u = np.asarray(u0, dtype=float)
    traj = np.empty((int(T) + 1, *u.shape), dtype=float)
    traj[0] = u
    for t in range(1, int(T) + 1):
        u = f(u, **kwargs)
        traj[t] = u
    return traj

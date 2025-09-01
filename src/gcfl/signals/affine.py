from __future__ import annotations
from typing import Any, Tuple
import numpy as np
from ..registry import register_signal

"""
Affine signal model with optional heteroskedastic noise and clipping.

s = a * u + b + ε,   ε ~ N(0, σ^2) elementwise

Parameters (kwargs accepted through the registry/engine):
    a: float                  # linear scale (default 1.0)
    b: float                  # bias shift (default 0.0)
    noise_sigma: float | array-like
                              # std-dev; scalar or array broadcastable to u.shape (default 0.5)
    sigma: alias for noise_sigma
    clip: tuple[float | None, float | None] | None
                              # optional (lo, hi) clipping; use None for open bound

Returns:
    np.ndarray of same shape as u
"""


def _as_sigma(sigma: float | np.ndarray | None, u_shape: tuple[int, ...]) -> np.ndarray:
    if sigma is None:
        sigma = 0.5
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 0:
        arr = np.broadcast_to(arr, u_shape)
    return arr


@register_signal("affine")
def model(
    u: np.ndarray,
    rng: np.random.Generator,
    *,
    a: float = 1.0,
    b: float = 0.0,
    noise_sigma: float | np.ndarray = 0.5,
    sigma: float | np.ndarray | None = None,  # alias
    clip: Tuple[float | None, float | None] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Generate monitoring signals from latent utilities u via an affine map with Gaussian noise.
    """
    u = np.asarray(u, dtype=float)
    sig = _as_sigma(sigma if sigma is not None else noise_sigma, u.shape)
    noise = rng.normal(loc=0.0, scale=sig, size=u.shape)
    s = float(a) * u + float(b) + noise
    if clip is not None:
        lo, hi = clip
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        s = np.clip(s, lo, hi)
    return s

"""
Optional Numba-accelerated kernels with safe fallbacks.

Public API (stable):
- trimmed_mean(values, trim_ratio=0.10, nan_policy="omit")
- sorted_weighted(values, weights=None, nan_policy="omit")

Behavior matches Python aggregators (gcfl.aggregates.*) as closely as possible.
If Numba is unavailable, pure NumPy fallbacks are used.
"""
from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np

try:
    import numba as nb  # type: ignore
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

# --------- helpers ---------

def _clean(values: np.ndarray, nan_policy: str = "omit") -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    if nan_policy == "omit":
        v = v[np.isfinite(v)]
    return v

def _np_trimmed_mean(values: np.ndarray, trim_ratio: float, nan_policy: str = "omit") -> float:
    v = _clean(values, nan_policy=nan_policy)
    if v.size == 0:
        return float("nan")
    v = np.sort(v)
    n = v.size
    r = float(trim_ratio)
    if not np.isfinite(r):
        r = 0.0
    r = min(max(r, 0.0), 0.5)
    k = int(round(r * n))
    if 2 * k >= n:
        return float(v.mean())
    return float(v[k : n - k].mean())

def _resample_weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    if not weights:
        return np.ones(n, dtype=float) / max(n, 1)
    w = np.asarray(list(weights), dtype=float).ravel()
    w = np.clip(w, 0, np.inf)
    if w.size == n:
        out = w
    else:
        x_src = np.linspace(0.0, 1.0, num=w.size)
        x_tgt = np.linspace(0.0, 1.0, num=n)
        out = np.interp(x_tgt, x_src, w)
    s = out.sum()
    return (out / s) if s > 0 else np.ones(n, dtype=float) / max(n, 1)

def _np_sorted_weighted(values: np.ndarray, weights: Sequence[float] | None, nan_policy: str = "omit") -> float:
    v = _clean(values, nan_policy=nan_policy)
    if v.size == 0:
        return float("nan")
    v = np.sort(v)
    w = _resample_weights(weights, n=v.size)
    return float(np.dot(v, w))

# --------- numba kernels ---------

if HAS_NUMBA:
    @nb.njit(cache=True, fastmath=True)
    def _trimmed_mean_sorted_numba(v_sorted: np.ndarray, k: int) -> float:
        n = v_sorted.shape[0]
        if n == 0:
            return np.nan
        if 2 * k >= n:
            return float(v_sorted.mean())
        s = 0.0
        cnt = 0
        for i in range(k, n - k):
            s += v_sorted[i]
            cnt += 1
        return s / max(1, cnt)

    @nb.njit(cache=True, fastmath=True)
    def _trimmed_mean_numba(v: np.ndarray, trim_ratio: float) -> float:
        n = v.shape[0]
        if n == 0:
            return np.nan
        v = np.sort(v.copy())
        r = trim_ratio
        if not np.isfinite(r) or r < 0.0:
            r = 0.0
        if r > 0.5:
            r = 0.5
        k = int(round(r * n))
        return _trimmed_mean_sorted_numba(v, k)

    @nb.njit(cache=True, fastmath=True)
    def _sorted_weighted_same_len_numba(v: np.ndarray, w: np.ndarray) -> float:
        # assumes v not sorted; sorts ascending, normalizes w (same length), nonneg clip
        n = v.shape[0]
        if n == 0:
            return np.nan
        v = np.sort(v.copy())
        w = w.copy()
        for i in range(n):
            if w[i] < 0.0 or not np.isfinite(w[i]):
                w[i] = 0.0
        s = 0.0
        for i in range(n):
            s += w[i]
        if s <= 0.0:
            return float(v.mean())
        for i in range(n):
            w[i] /= s
        dot = 0.0
        for i in range(n):
            dot += v[i] * w[i]
        return dot

def trimmed_mean(values: Iterable[float], trim_ratio: float = 0.10, nan_policy: str = "omit") -> float:
    v = _clean(np.asarray(values), nan_policy=nan_policy)
    if not HAS_NUMBA:
        return _np_trimmed_mean(v, trim_ratio, nan_policy=nan_policy)
    return float(_trimmed_mean_numba(v, float(trim_ratio)))

def sorted_weighted(values: Iterable[float], weights: Sequence[float] | None = None, nan_policy: str = "omit") -> float:
    v = _clean(np.asarray(values), nan_policy=nan_policy)
    if v.size == 0:
        return float("nan")
    if not HAS_NUMBA or weights is None:
        return _np_sorted_weighted(v, weights, nan_policy=nan_policy)
    w = np.asarray(list(weights), dtype=float).ravel()
    if w.size != v.size:
        # Numba path expects same length; fall back to NumPy (with interpolation)
        return _np_sorted_weighted(v, weights, nan_policy=nan_policy)
    return float(_sorted_weighted_same_len_numba(v, w))

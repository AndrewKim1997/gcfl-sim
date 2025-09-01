from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
from ..registry import register_aggregator


def _clean(values: np.ndarray, nan_policy: str = "omit") -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    if nan_policy == "omit":
        v = v[np.isfinite(v)]
    return v


def _resample_weights(weights: Sequence[float], n: int) -> np.ndarray:
    """Resample an arbitrary-length weight vector to length n via linear interpolation."""
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        return np.ones(n, dtype=float) / max(n, 1)
    # Negative weights are clipped to zero before normalization
    w = np.clip(w, 0.0, np.inf)
    if w.size == n:
        out = w
    else:
        x_src = np.linspace(0.0, 1.0, num=w.size)
        x_tgt = np.linspace(0.0, 1.0, num=n)
        out = np.interp(x_tgt, x_src, w)
    s = out.sum()
    return (out / s) if s > 0 else np.ones(n, dtype=float) / max(n, 1)


@register_aggregator("sorted_weighted")
def aggregate(
    values: Iterable[float],
    *,
    weights: Sequence[float] | None = None,
    nan_policy: str = "omit",
    **_: dict,
) -> float:
    """
    Sorted-weighted mean:
      1) sort values ascending,
      2) resample/normalize the given weights to match length n,
      3) return dot(sorted(values), weights).

    Args:
      weights: arbitrary-length nonnegative weights; will be interpolated to n and normalized.
      nan_policy: "omit" to ignore NaN/inf.
    """
    v = _clean(np.asarray(values), nan_policy=nan_policy)
    if v.size == 0:
        return float("nan")
    v = np.sort(v)
    w = _resample_weights(weights if weights is not None else (), n=v.size)
    return float(np.dot(v, w))

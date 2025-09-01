from __future__ import annotations
from typing import Iterable
import numpy as np
from ..registry import register_aggregator

def _clean(values: np.ndarray, nan_policy: str = "omit") -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    if nan_policy == "omit":
        v = v[np.isfinite(v)]
    return v

@register_aggregator("trimmed")
def aggregate(
    values: Iterable[float],
    *,
    trim_ratio: float = 0.10,
    nan_policy: str = "omit",
    **_: dict,
) -> float:
    """
    Symmetric trimmed mean:
      - Sort ascending, drop k = round(trim_ratio * n) from each side, then average.
      - If 2k >= n, fall back to plain mean (or NaN if empty after cleaning).
    Args:
      trim_ratio: in [0, 0.5]. Values outside are clipped into this range.
      nan_policy: "omit" to ignore NaN/inf.
    """
    v = _clean(np.asarray(values), nan_policy=nan_policy)
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

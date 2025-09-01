from __future__ import annotations
from typing import Iterable
import numpy as np
from ..registry import register_aggregator

def _clean(values: np.ndarray, nan_policy: str = "omit") -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    if nan_policy == "omit":
        v = v[np.isfinite(v)]
    return v

@register_aggregator("median")
def aggregate(values: Iterable[float], *, nan_policy: str = "omit", **_: dict) -> float:
    """
    Median with optional NaN/inf omission.
    """
    v = _clean(np.asarray(values), nan_policy=nan_policy)
    if v.size == 0:
        return float("nan")
    return float(np.median(v))

from __future__ import annotations
from typing import Iterable
import numpy as np
from ..registry import register_aggregator


def _clean(values: np.ndarray, nan_policy: str = "omit") -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    if nan_policy == "omit":
        v = v[np.isfinite(v)]
    return v


@register_aggregator("mean")
def aggregate(values: Iterable[float], *, nan_policy: str = "omit", **_: dict) -> float:
    """
    Arithmetic mean with optional NaN/inf omission.
    Args:
      values: 1-D numeric array.
      nan_policy: "omit" to ignore NaN/inf, anything else to propagate.
    """
    v = _clean(np.asarray(values), nan_policy=nan_policy)
    if v.size == 0:
        return float("nan")
    return float(v.mean())

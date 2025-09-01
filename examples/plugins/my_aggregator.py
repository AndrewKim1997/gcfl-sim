"""
Example custom aggregator plugin for gcfl-sim.

Implements a 'topk_mean' aggregator:
  - sorts values descending and averages the top k fraction.
  - handles NaN/inf robustly (omit policy).
  - registers itself under the name "topk_mean".

Usage:
  1) Ensure this module is importable (e.g., PYTHONPATH=examples or move into your package).
  2) Import it once before running:
       import examples.plugins.my_aggregator  # noqa: F401
     (or `python -c "import examples.plugins.my_aggregator; import gcfl.run as r; r.main([...])"`)
  3) In config/CLI, set: aggregator.kind=topk_mean and optionally aggregator.k_frac=0.2
"""
from __future__ import annotations
from typing import Iterable
import numpy as np
from gcfl.registry import register_aggregator

def _clean(values: Iterable[float], nan_policy: str = "omit") -> np.ndarray:
    v = np.asarray(list(values), dtype=float).ravel()
    if nan_policy == "omit":
        v = v[np.isfinite(v)]
    return v

@register_aggregator("topk_mean")
def aggregate(
    values: Iterable[float],
    *,
    k_frac: float = 0.2,
    nan_policy: str = "omit",
    **kwargs,
) -> float:
    """
    Average the top k fraction (descending) of values.

    Args:
      k_frac: fraction in (0,1]; e.g., 0.2 means top 20%.
      nan_policy: "omit" to ignore NaN/inf.
    """
    v = _clean(values, nan_policy=nan_policy)
    if v.size == 0:
        return float("nan")
    k_frac = float(k_frac)
    if not np.isfinite(k_frac) or k_frac <= 0.0:
        k_frac = 0.2
    if k_frac > 1.0:
        k_frac = 1.0
    k = int(max(1, round(k_frac * v.size)))
    # sort descending and take top-k
    idx = np.argpartition(v, -k)[-k:]
    return float(v[idx].mean())

"""
Metric helpers: summaries with confidence intervals, frontiers on sign maps, etc.
"""
from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import numpy as np
import pandas as pd

def summarize_mean_ci(df: pd.DataFrame, value: str, by: Sequence[str]) -> pd.DataFrame:
    """
    Group by `by` and compute mean, std, count, sem, and Wald 95% CI for `value`.
    """
    g = df.groupby(list(by))[value].agg(["mean", "std", "count"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    g["ci95"] = 1.96 * g["sem"]
    return g

def first_zero_crossing_frontier(
    alphas: np.ndarray, phis: np.ndarray, deltaU: np.ndarray
) -> pd.DataFrame:
    """
    Given grids alpha (na,), phi (nf,), and a matrix deltaU[nf, na],
    compute φ*(α) as the smallest phi where ΔU crosses from ≤0 to >0 (row-wise search).

    Returns a DataFrame with columns: alpha, phi_star (or NaN if no crossing).
    """
    alphas = np.asarray(alphas).ravel()
    phis = np.asarray(phis).ravel()
    M = np.asarray(deltaU)
    if M.shape != (len(phis), len(alphas)):
        raise ValueError("deltaU must have shape (len(phis), len(alphas))")

    phi_star = np.full(len(alphas), np.nan, dtype=float)
    signM = np.sign(M)

    for j, a in enumerate(alphas):
        col = signM[:, j]
        # find first index where sign becomes >0 (assuming rows ascend in phi)
        idx = np.argmax(col > 0)
        if (col > 0).any():
            # linear interpolation with previous row if possible
            i1 = int(idx)
            i0 = max(0, i1 - 1)
            y0, y1 = M[i0, j], M[i1, j]
            x0, x1 = phis[i0], phis[i1]
            if y1 == y0:
                phi_star[j] = float(x1)
            else:
                t = (0.0 - y0) / (y1 - y0)
                phi_star[j] = float(x0 + t * (x1 - x0))

    return pd.DataFrame({"alpha": alphas, "phi_star": phi_star})

def sign_map(M: np.ndarray) -> np.ndarray:
    """
    Return sign(ΔU) map as -1, 0, +1. Useful for quick plotting or counting regions.
    """
    return np.sign(np.asarray(M))

from __future__ import annotations
"""
u_orth_penalty: penalize the component of the monitoring signal s that is orthogonal to u.

Intuition (vector-in-R^N view):
- Center both u and s (remove means) and decompose s into components parallel/orthogonal to u.
- Penalize the orthogonal magnitude; report ΔU as the negative penalty.
- Optionally apply an "information mixing" with parameter π: s_eff = (1-π)*s + π*η*u.
- Provide a benign zone: if orth magnitude ≤ benign_threshold and m is non-negative, neutralize ΔU.

Returned metrics (keys match the engine/repro scripts):
    M       := m (the aggregator output)
    PoG     := max(0, m)           # proxy for "gain"
    PoC     := max(0, -m)          # proxy for "cost"
    DeltaU  := - φ * ||s_orth||_avg  (possibly neutralized in benign region)

Parameters (from cfg["mechanism"]):
    alpha: float           # not used directly here, but recorded/available for dynamics elsewhere
    pi: float              # information mixing ratio in [0,1]
    eta: float             # mixing scale for u (default 1.0)
    phi: float             # penalty strength
    benign_threshold: float
    neutralize_when_deltaU_ge_0: bool
"""
from typing import Dict, MutableMapping, Any
import numpy as np
from ..registry import register_mechanism

def _center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x - x.mean()

def _orth_component(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return the component of s that is orthogonal to u (both treated as centered)."""
    su = _center(s)
    uu = _center(u)
    denom = float(np.dot(uu, uu))
    if denom <= 0.0:
        # u has no direction (constant); treat entire signal as orthogonal
        return su
    proj = (np.dot(su, uu) / denom) * uu
    return su - proj

def _orth_magnitude_avg(s: np.ndarray, u: np.ndarray) -> float:
    """Average absolute orthogonal magnitude (L1/N), robust to zeros/NaNs."""
    orth = _orth_component(s, u)
    if orth.size == 0:
        return 0.0
    finite = np.isfinite(orth)
    if not finite.any():
        return 0.0
    return float(np.mean(np.abs(orth[finite])))

@register_mechanism("u_orth_penalty")
def mechanism(
    state: MutableMapping[str, Any],
    u: np.ndarray,
    s: np.ndarray,
    m: float,
    rng: np.random.Generator,
    *,
    alpha: float = 0.5,
    pi: float = 0.2,
    eta: float = 1.0,
    phi: float = 1.0,
    benign_threshold: float = 0.10,
    neutralize_when_deltaU_ge_0: bool = True,
    **kwargs: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute monitoring-based penalty on the u-orthogonal component of signals.
    """
    # Information design (optional): mix s with u before assessing orthogonality.
    pi = float(np.clip(pi, 0.0, 1.0))
    s_eff = (1.0 - pi) * np.asarray(s, dtype=float) + pi * float(eta) * np.asarray(u, dtype=float)

    # Orthogonal magnitude (average L1 per client)
    orth_mag = _orth_magnitude_avg(s_eff, u)

    # Base ΔU: negative penalty
    delta_u = -float(phi) * orth_mag

    # Benign neutralization: if m non-negative and orthogonality small, neutralize ΔU to 0
    if neutralize_when_deltaU_ge_0 and (m >= 0.0) and (orth_mag <= float(benign_threshold)):
        delta_u = 0.0

    metrics = {
        "M": float(m),
        "PoG": float(max(0.0, m)),
        "PoC": float(max(0.0, -m)),
        "DeltaU": float(delta_u),
    }
    # No state mutation by default, but you can store round-level stats if desired:
    # state["last_orth_mag"] = orth_mag
    return metrics

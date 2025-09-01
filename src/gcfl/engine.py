from __future__ import annotations
from typing import Any, Dict, MutableMapping, Callable
import numpy as np
import pandas as pd

from .rng import RngBundle, make_seedseq
from .types import LogRow

# ========= helpers =========


def _clean(values: np.ndarray, nan_policy: str = "omit") -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    if nan_policy == "omit":
        v = v[np.isfinite(v)]
    return v


# ========= built-in aggregators (engine-level) =========


def agg_mean(values, *, nan_policy: str = "omit", **_: Any) -> float:
    v = _clean(np.asarray(values), nan_policy=nan_policy)
    return float("nan") if v.size == 0 else float(v.mean())


def agg_median(values, *, nan_policy: str = "omit", **_: Any) -> float:
    v = _clean(np.asarray(values), nan_policy=nan_policy)
    return float("nan") if v.size == 0 else float(np.median(v))


def agg_trimmed(values, *, trim_ratio: float = 0.10, nan_policy: str = "omit", **_: Any) -> float:
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


def _resample_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=float) / max(n, 1)
    w = np.asarray(weights, dtype=float).ravel()
    w = np.clip(w, 0.0, np.inf)
    if w.size == n:
        out = w
    else:
        x_src = np.linspace(0.0, 1.0, num=w.size)
        x_tgt = np.linspace(0.0, 1.0, num=n)
        out = np.interp(x_tgt, x_src, w)
    s = out.sum()
    return (out / s) if s > 0 else np.ones(n, dtype=float) / max(n, 1)


def agg_sorted_weighted(values, *, weights=None, nan_policy: str = "omit", **_: Any) -> float:
    v = _clean(np.asarray(values), nan_policy=nan_policy)
    if v.size == 0:
        return float("nan")
    v = np.sort(v)
    w = _resample_weights(weights, v.size)
    return float(np.dot(v, w))


AGGREGATORS: Dict[str, Callable[..., float]] = {
    "mean": agg_mean,
    "median": agg_median,
    "trimmed": agg_trimmed,
    "sorted_weighted": agg_sorted_weighted,
}

# ========= built-in mechanism (engine-level) =========


def _center(x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float).ravel()
    return xx - xx.mean()


def _orth_component(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    su = _center(s)
    uu = _center(u)
    denom = float(np.dot(uu, uu))
    if denom <= 0.0:
        return su
    proj = (np.dot(su, uu) / denom) * uu
    return su - proj


def mech_u_orth_penalty(
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
    **kwargs: Any,
) -> Dict[str, float]:
    pi = float(np.clip(pi, 0.0, 1.0))
    s_eff = (1.0 - pi) * np.asarray(s, dtype=float) + pi * float(eta) * np.asarray(u, dtype=float)

    orth = _orth_component(s_eff, u)
    finite = np.isfinite(orth)
    orth_mag = float(np.mean(np.abs(orth[finite]))) if finite.any() else 0.0

    delta_u = -float(phi) * orth_mag
    if neutralize_when_deltaU_ge_0 and (m >= 0.0) and (orth_mag <= float(benign_threshold)):
        delta_u = 0.0

    return {
        "M": float(m),
        "PoG": float(max(0.0, m)),
        "PoC": float(max(0.0, -m)),
        "DeltaU": float(delta_u),
    }


MECHANISMS: Dict[str, Callable[..., Dict[str, float]]] = {
    "u_orth_penalty": mech_u_orth_penalty,
}

# ========= signals registry seed (engine exposes names; impl은 패키지에서 로드) =========
# 신호는 빌트인 이름만 노출하고, 실제 구현은 gcfl.signals.* 에서 등록/해결합니다.
SIGNALS: Dict[str, Callable[..., np.ndarray]] = {
    "affine": None,  # registry에서 실제 함수를 resolve
}

# ========= reference loop used by tests (unchanged API) =========


def run_experiment(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Minimal reference loop used by tests. Deterministic given meta.seed_root.
    """
    R = int(cfg["engine"]["repeats"])
    T = int(cfg["engine"]["rounds"])
    N = int(cfg["engine"]["clients"])

    # resolve functions: prefer engine's built-ins for aggregators/mechanisms
    agg = AGGREGATORS[cfg["aggregator"]["kind"]]
    # signals/mechanisms can come from registry; tests seed registry via seed_with(...)
    from .registry import get_signal, get_mechanism, seed_with

    seed_with(AGGREGATORS, SIGNALS, MECHANISMS)
    sig = get_signal(cfg["signals"]["model"])
    mech = get_mechanism(cfg["mechanism"]["policy"])

    ss = make_seedseq(cfg["meta"].get("seed_root"))
    rngs = RngBundle(ss)

    rows: list[LogRow] = []
    for r in range(R):
        g_rep = rngs.for_repeat(r)
        u = g_rep.normal(loc=0.0, scale=1.0, size=(N,))
        for t in range(T):
            g_round = rngs.for_round(r, t)
            s = sig(u, g_round, **cfg["signals"])
            m = float(agg(s, **cfg["aggregator"]))
            metrics = mech({}, u, s, m, g_round, **cfg["mechanism"])

            rows.append(
                {
                    "repeat": r,
                    "round": t,
                    "N": N,
                    "aggregator": cfg["aggregator"]["kind"],
                    "mechanism": cfg["mechanism"]["policy"],
                    "alpha": float(cfg["mechanism"].get("alpha", 0.0)),
                    "pi": float(cfg["mechanism"].get("pi", 0.0)),
                    "M": float(metrics["M"]),
                    "PoG": float(metrics["PoG"]),
                    "PoC": float(metrics["PoC"]),
                    "DeltaU": float(metrics["DeltaU"]),
                }
            )

            # toy dynamics (same as reference backend)
            alpha = float(cfg["mechanism"].get("alpha", 0.0))
            u = (1.0 - 0.05 * alpha) * u + 0.05 * alpha * m

    return pd.DataFrame.from_records(rows)

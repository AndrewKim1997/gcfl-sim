from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .types import EngineConfig, LogRow, Aggregator, SignalModel, Mechanism
from .rng import make_seedseq, RngBundle
from .params import load_config

# ---------------- Built-in minimal plugins ----------------

def agg_mean(values: np.ndarray, **_: dict) -> float:
    return float(np.mean(values))

def agg_median(values: np.ndarray, **_: dict) -> float:
    return float(np.median(values))

def agg_trimmed(values: np.ndarray, trim_ratio: float = 0.10, **_: dict) -> float:
    v = np.sort(values)
    n = len(v)
    k = int(max(0, min(n // 2, round(trim_ratio * n))))
    if 2 * k >= n:
        return float(v.mean())
    return float(v[k : n - k].mean())

def agg_sorted_weighted(values: np.ndarray, weights: List[float] | Tuple[float, ...] = (), **_: dict) -> float:
    v = np.sort(values)
    if not weights:
        return float(v.mean())
    w = np.asarray(weights, dtype=float)
    if len(w) != len(v):
        # interpolate weights to length n
        w = np.interp(np.linspace(0, 1, len(v)), np.linspace(0, 1, len(weights)), w)
    w = w / w.sum()
    return float(np.dot(v, w))

AGGREGATORS: Dict[str, Aggregator] = {
    "mean": agg_mean,
    "median": agg_median,
    "trimmed": agg_trimmed,
    "sorted_weighted": agg_sorted_weighted,
}

def signal_affine(u: np.ndarray, rng: np.random.Generator, a: float = 1.0, b: float = 0.0, noise_sigma: float = 0.5, **kwargs) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=noise_sigma, size=u.shape)
    return a * u + b + noise

SIGNALS: Dict[str, SignalModel] = {
    "affine": signal_affine,
}

def mech_u_orth_penalty(state, u: np.ndarray, s: np.ndarray, m: float, rng: np.random.Generator, alpha: float = 0.5, pi: float = 0.2, phi: float = 1.0, **kwargs):
    """
    Toy mechanism:
    - Monitoring signal is the aggregator output m.
    - 'Penalty' proportional to average deviation |s - u|.
    - Report a few generic metrics often seen in the paper nomenclature:
        M     := m
        PoG   := max(0, m)           (placeholder "gain"-like proxy)
        PoC   := max(0, -m)          (placeholder "cost"-like proxy)
        DeltaU:= -phi * mean(|s - u|)
    You should replace this with your actual mechanism when ready.
    """
    deviation = np.abs(s - u).mean()
    delta_u = -phi * deviation
    metrics = {
        "M": float(m),
        "PoG": float(max(0.0, m)),
        "PoC": float(max(0.0, -m)),
        "DeltaU": float(delta_u),
    }
    # This toy mechanism does not mutate state; in real models you might update it here.
    return metrics

MECHANISMS: Dict[str, Mechanism] = {
    "u_orth_penalty": mech_u_orth_penalty,
}

# ---------------- Engine ----------------

def _select_plugins(cfg: EngineConfig) -> tuple[Aggregator, SignalModel, Mechanism]:
    agg_kind = cfg["aggregator"]["kind"]
    if agg_kind not in AGGREGATORS:
        raise KeyError(f"Unknown aggregator: {agg_kind}")
    sig_model = cfg["signals"]["model"]
    if sig_model not in SIGNALS:
        raise KeyError(f"Unknown signal model: {sig_model}")
    mech_policy = cfg["mechanism"]["policy"]
    if mech_policy not in MECHANISMS:
        raise KeyError(f"Unknown mechanism policy: {mech_policy}")
    return AGGREGATORS[agg_kind], SIGNALS[sig_model], MECHANISMS[mech_policy]

def _write(df: pd.DataFrame, out: str, fmt: str) -> None:
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
            df.to_parquet(out, index=False)
        except Exception:
            # graceful fallback to CSV if parquet writer not available
            csv_out = out.rsplit(".", 1)[0] + ".csv"
            df.to_csv(csv_out, index=False)
    else:
        df.to_csv(out, index=False)

def run_experiment(config: EngineConfig | dict, seed: int | None = None, out: str | None = None) -> pd.DataFrame:
    """
    Run a single experiment defined by `config`.
    Returns a tidy DataFrame of per-round metrics; optionally writes to `out` (csv/parquet).
    """
    cfg = load_config(config) if isinstance(config, dict) else load_config(config)  # both paths supported
    agg_fn, sig_fn, mech_fn = _select_plugins(cfg)

    N = int(cfg["engine"]["clients"])
    T = int(cfg["engine"]["rounds"])
    R = int(cfg["engine"]["repeats"])
    agg_kind = str(cfg["aggregator"]["kind"])
    mech_kind = str(cfg["mechanism"]["policy"])
    alpha = float(cfg["mechanism"]["alpha"])
    pi = float(cfg["mechanism"]["pi"])

    ss = make_seedseq(cfg["meta"].get("seed_root", seed))
    rngs = RngBundle(ss)

    rows: List[LogRow] = []
    for r in range(R):
        # Persistent latent u for this repeat
        g_rep = rngs.for_repeat(r)
        u = g_rep.normal(loc=0.0, scale=1.0, size=(N,))
        state: Dict[str, object] = {"repeat": r}

        for t in range(T):
            g_round = rngs.for_round(r, t)
            s = sig_fn(u, g_round, **cfg["signals"])  # signals from u
            m = agg_fn(s, **cfg["aggregator"])        # aggregated monitoring signal
            metrics = mech_fn(state, u, s, m, g_round, **cfg["mechanism"])

            row: LogRow = {
                "repeat": r,
                "round": t,
                "N": N,
                "aggregator": agg_kind,
                "mechanism": mech_kind,
                "alpha": alpha,
                "pi": pi,
                "M": float(metrics.get("M", m)),
                "PoG": float(metrics.get("PoG", 0.0)),
                "PoC": float(metrics.get("PoC", 0.0)),
                "DeltaU": float(metrics.get("DeltaU", 0.0)),
                "signal_mean": float(s.mean()),
                "signal_std": float(s.std(ddof=0)),
                "u_mean": float(u.mean()),
                "u_std": float(u.std(ddof=0)),
            }
            rows.append(row)

            # --- optional dynamics (toy): mild drift towards M with damping alpha ---
            # Comment out or replace with your model-specific update.
            u = (1.0 - 0.05 * alpha) * u + 0.05 * alpha * m

    df = pd.DataFrame.from_records(rows)
    if out:
        _write(df, out, cfg["logging"]["out_format"])
    return df

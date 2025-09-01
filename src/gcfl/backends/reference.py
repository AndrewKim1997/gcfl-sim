"""
Reference backend — single-threaded, fully deterministic.
"""

from __future__ import annotations
from typing import Dict, List
import pandas as pd

from ..rng import RngBundle
from ..types import LogRow


def _run_one_repeat(
    r: int,
    cfg: dict,
    agg_fn,
    sig_fn,
    mech_fn,
    rngs: RngBundle,
) -> List[LogRow]:
    N = int(cfg["engine"]["clients"])
    T = int(cfg["engine"]["rounds"])
    alpha = float(cfg["mechanism"]["alpha"])
    agg_kind = str(cfg["aggregator"]["kind"])
    mech_kind = str(cfg["mechanism"]["policy"])
    pi = float(cfg["mechanism"]["pi"])

    g_rep = rngs.for_repeat(r)
    u = g_rep.normal(loc=0.0, scale=1.0, size=(N,))
    state: Dict[str, object] = {"repeat": r}

    rows: List[LogRow] = []
    for t in range(T):
        g_round = rngs.for_round(r, t)
        s = sig_fn(u, g_round, **cfg["signals"])
        m = float(agg_fn(s, **cfg["aggregator"]))
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

        # toy dynamics: drift towards scalar m with damping ~ alpha
        u = (1.0 - 0.05 * alpha) * u + 0.05 * alpha * m

    return rows


def run(cfg: dict, agg_fn, sig_fn, mech_fn, seedseq) -> pd.DataFrame:
    """
    Run the experiment deterministically on a single process.
    """
    R = int(cfg["engine"]["repeats"])
    rngs = RngBundle(seedseq)
    all_rows: List[LogRow] = []
    for r in range(R):
        all_rows.extend(_run_one_repeat(r, cfg, agg_fn, sig_fn, mech_fn, rngs))
    return pd.DataFrame.from_records(all_rows)

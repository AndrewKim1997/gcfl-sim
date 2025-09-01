#!/usr/bin/env python
"""
Micro-benchmarks for gcfl-sim backends.

Examples:
  python scripts/benchmark.py --backends reference scale --clients 200 1000 --rounds 60 --repeats 5 20
"""
from __future__ import annotations
import argparse, time
from typing import Any, Dict, List

import pandas as pd

from gcfl import params, rng as _rng
from gcfl.registry import seed_with, get_aggregator, get_signal, get_mechanism
from gcfl import engine as _engine
from gcfl.backends import get_backend
from gcfl.utils.logging import get_logger
from gcfl.utils.profiling import Timer

def run_once(cfg: Dict[str, Any], backend_name: str) -> tuple[pd.DataFrame, float]:
    backend_run = get_backend(backend_name)
    agg_fn = get_aggregator(cfg["aggregator"]["kind"])
    sig_fn = get_signal(cfg["signals"]["model"])
    mech_fn = get_mechanism(cfg["mechanism"]["policy"])
    ss = _rng.make_seedseq(cfg["meta"]["seed_root"])
    t0 = time.perf_counter()
    df = backend_run(cfg, agg_fn, sig_fn, mech_fn, ss)
    dt = time.perf_counter() - t0
    return df, dt

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/base.yaml")
    ap.add_argument("--backends", nargs="+", default=["reference", "scale"])
    ap.add_argument("--clients", nargs="+", type=int, default=[200])
    ap.add_argument("--rounds", nargs="+", type=int, default=[60])
    ap.add_argument("--repeats", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--workers", type=int, default=0, help="scale backend workers")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    log = get_logger("gcfl.bench", level="DEBUG" if args.verbose else "INFO")
    seed_with(_engine.AGGREGATORS, _engine.SIGNALS, _engine.MECHANISMS)

    base = params.load_config(args.config)
    results: List[dict] = []

    for N in args.clients:
        for T in args.rounds:
            for R in args.repeats:
                cfg = params.load_config({
                    **base,
                    "engine": {"clients": N, "rounds": T, "repeats": R},
                    "execution": {**base["execution"], "parallel_workers": args.workers},
                })
                rows_expected = N * T * R  # per-round, per-repeat rows have N only in metadata

                for backend in args.backends:
                    with Timer(f"{backend} N={N} T={T} R={R}", logger=log) as tm:
                        df, dt = run_once(cfg, backend)
                    rows = len(df)
                    results.append({
                        "backend": backend,
                        "clients": N,
                        "rounds": T,
                        "repeats": R,
                        "rows": rows,
                        "sec": round(dt, 4),
                        "rows_per_sec": round(rows / max(dt, 1e-9), 1),
                    })

    table = pd.DataFrame(results).sort_values(["clients", "rounds", "repeats", "backend"])
    print("\n=== Benchmark results ===")
    try:
        print(table.to_markdown(index=False))
    except Exception:
        print(table)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

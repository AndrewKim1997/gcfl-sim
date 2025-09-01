#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict

import pandas as pd

from . import params, rng as _rng
from .io import write_table, write_provenance
from .utils.logging import get_logger, log_provenance
from .utils.profiling import Timer
from .registry import seed_with, get_aggregator, get_signal, get_mechanism
from . import aggregates as _agg_pkg  # noqa: F401  # trigger decorators
from . import signals as _sig_pkg     # noqa: F401
from . import mechanisms as _mech_pkg # noqa: F401
from . import engine as _engine       # seed from engine builtins too
from .backends import get_backend, list_backends


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides to a loaded config (in-place, but also return)."""
    c = cfg
    # Execution / logging
    if args.backend:          c["execution"]["backend"] = args.backend
    if args.workers is not None: c["execution"]["parallel_workers"] = int(args.workers)
    if args.out_format:       c["logging"]["out_format"] = args.out_format

    # Engine
    if args.clients is not None:  c["engine"]["clients"] = int(args.clients)
    if args.rounds is not None:   c["engine"]["rounds"] = int(args.rounds)
    if args.repeats is not None:  c["engine"]["repeats"] = int(args.repeats)

    # Signals
    if args.signal_model:     c["signals"]["model"] = args.signal_model
    if args.a is not None:    c["signals"]["a"] = float(args.a)
    if args.b is not None:    c["signals"]["b"] = float(args.b)
    if args.noise_sigma is not None: c["signals"]["noise_sigma"] = float(args.noise_sigma)

    # Aggregator
    if args.aggregator:       c["aggregator"]["kind"] = args.aggregator
    if args.trim_ratio is not None: c["aggregator"]["trim_ratio"] = float(args.trim_ratio)

    # Mechanism
    if args.policy:           c["mechanism"]["policy"] = args.policy
    if args.alpha is not None:c["mechanism"]["alpha"] = float(args.alpha)
    if args.pi is not None:   c["mechanism"]["pi"] = float(args.pi)
    if args.phi is not None:  c["mechanism"]["phi"] = float(args.phi)

    # Seed
    if args.seed is not None: c["meta"]["seed_root"] = int(args.seed)

    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gcfl-run",
        description="Run a single GCFL-sim experiment and write a table (Parquet/CSV).",
    )
    ap.add_argument("-c", "--config", required=True, help="Path to base YAML config")
    ap.add_argument("-o", "--out", default="results/logs/run.parquet", help="Output table path (.parquet or .csv)")
    ap.add_argument("--out-format", dest="out_format", choices=["parquet", "csv"], help="Override output format")

    # Execution / engine
    ap.add_argument("--backend", choices=list_backends(), help="Execution backend")
    ap.add_argument("--workers", type=int, help="Parallel workers (scale backend)")
    ap.add_argument("--clients", type=int)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--repeats", type=int)

    # Signals
    ap.add_argument("--signal-model", dest="signal_model", help="Signal model name (e.g., affine)")
    ap.add_argument("--a", type=float, help="Signal scale (affine)")
    ap.add_argument("--b", type=float, help="Signal bias (affine)")
    ap.add_argument("--noise-sigma", type=float, help="Signal noise sigma (affine)")

    # Aggregator & mechanism
    ap.add_argument("--aggregator", help="Aggregator name (mean|median|trimmed|sorted_weighted or plugin)")
    ap.add_argument("--trim-ratio", type=float, help="Trimmed ratio (for trimmed)")
    ap.add_argument("--policy", help="Mechanism policy name (e.g., u_orth_penalty)")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--pi", type=float)
    ap.add_argument("--phi", type=float)

    ap.add_argument("--seed", type=int, help="Seed root (deterministic)")

    ap.add_argument("--list", action="store_true", help="List available backends/aggregators/signals/mechanisms and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    log = get_logger("gcfl.run", level="DEBUG" if args.verbose else "INFO")

    # Seed plugin registry with built-ins (engine) and decorator-registered packages
    seed_with(_engine.AGGREGATORS, _engine.SIGNALS, _engine.MECHANISMS)

    if args.list:
        from .registry import list_aggregators, list_signals, list_mechanisms
        payload = {
            "backends": list_backends(),
            "aggregators": list_aggregators(),
            "signals": list_signals(),
            "mechanisms": list_mechanisms(),
        }
        log.info(json.dumps(payload, indent=2))
        return 0

    # Load config and apply overrides
    cfg = params.load_config(args.config)
    cfg = _apply_overrides(cfg, args)

    # Resolve plugins
    agg_fn = get_aggregator(cfg["aggregator"]["kind"])
    sig_fn = get_signal(cfg["signals"]["model"])
    mech_fn = get_mechanism(cfg["mechanism"]["policy"])

    # Pick backend and seed
    backend_run = get_backend(cfg["execution"]["backend"])
    ss = _rng.make_seedseq(cfg["meta"].get("seed_root"))

    # Run
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with Timer("experiment", logger=log):
        df: pd.DataFrame = backend_run(cfg, agg_fn, sig_fn, mech_fn, ss)

    # Write outputs + provenance
    written = write_table(df, args.out, cfg["logging"]["out_format"])
    meta = {
        "config": cfg,
        "rows": int(len(df)),
    }
    write_provenance(os.path.splitext(written)[0], meta)
    log_provenance(log, extra={"rows": int(len(df)), "out": written})
    log.info(f"[done] wrote {written} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

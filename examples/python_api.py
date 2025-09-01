#!/usr/bin/env python
"""
Minimal, end-to-end usage of gcfl-sim via the Python API.

What it shows:
  1) Load/compose a config (dict or YAML)
  2) Resolve plugins (aggregator/signal/mechanism) via the registry
  3) Pick a backend and run deterministically
  4) Write a table (Parquet if available, else CSV) + quick peek
"""
from __future__ import annotations
import os
import pandas as pd

from gcfl import params, rng as _rng
from gcfl.backends import get_backend
from gcfl.registry import seed_with, get_aggregator, get_signal, get_mechanism

# Import built-ins so their @register_* decorators run:
from gcfl import engine as _engine               # seeds registry dicts
from gcfl import aggregates as _agg_pkg  # noqa: F401
from gcfl import signals as _sig_pkg     # noqa: F401
from gcfl import mechanisms as _mech_pkg # noqa: F401
from gcfl.io import write_table, write_provenance


def main() -> int:
    # 1) Compose a config in-memory (you could also: params.load_config("configs/base.yaml"))
    cfg = params.load_config({
        "meta": {"experiment": "examples_python_api", "seed_root": 20250901},
        "execution": {"backend": "reference", "parallel_workers": 0},
        "engine": {"clients": 200, "rounds": 60, "repeats": 3},
        "signals": {"model": "affine", "a": 1.0, "b": 0.0, "noise_sigma": 0.5},
        "aggregator": {"kind": "mean"},              # try: median | trimmed | sorted_weighted
        "mechanism": {"policy": "u_orth_penalty", "alpha": 1.2, "pi": 0.2, "phi": 1.0},
        "logging": {"out_format": "parquet", "float_precision": 6},
    })

    # 2) Seed the registry with built-ins exposed from engine, then resolve plugins
    seed_with(_engine.AGGREGATORS, _engine.SIGNALS, _engine.MECHANISMS)
    agg_fn = get_aggregator(cfg["aggregator"]["kind"])
    sig_fn = get_signal(cfg["signals"]["model"])
    mech_fn = get_mechanism(cfg["mechanism"]["policy"])

    # 3) Select backend & seed
    backend_run = get_backend(cfg["execution"]["backend"])
    ss = _rng.make_seedseq(cfg["meta"]["seed_root"])

    # 4) Run + write output
    os.makedirs("results/logs", exist_ok=True)
    df: pd.DataFrame = backend_run(cfg, agg_fn, sig_fn, mech_fn, ss)
    out = write_table(df, "results/logs/examples_python_api.parquet", cfg["logging"]["out_format"])
    write_provenance(os.path.splitext(out)[0], {"config": cfg, "rows": int(len(df))})

    # Quick peek
    print(f"[ok] wrote {out} with {len(df)} rows")
    print(df.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

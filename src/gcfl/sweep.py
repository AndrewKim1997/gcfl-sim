#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

from . import params, rng as _rng
from .io import write_table, write_provenance
from .utils.logging import get_logger, log_provenance
from .utils.profiling import Timer
from .registry import seed_with, get_aggregator, get_signal, get_mechanism
from . import aggregates as _agg_pkg  # noqa: F401
from . import signals as _sig_pkg  # noqa: F401
from . import mechanisms as _mech_pkg  # noqa: F401
from . import engine as _engine
from .backends import get_backend

# ---- helpers ----


def _set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _expand_spec(spec: Any) -> List[Any]:
    if spec is None:
        return [None]
    if isinstance(spec, (list, tuple)):
        return list(spec)
    if isinstance(spec, dict) and {"start", "stop", "num"} <= set(spec):
        start = float(spec["start"])
        stop = float(spec["stop"])
        num = int(spec["num"])
        return list(np.linspace(start, stop, num))
    return [spec]


def _load_sweep_config(path: str) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    raw = yaml.safe_load(open(path, "r", encoding="utf-8").read()) or {}
    base_path = raw.get("base")
    base_cfg = params.load_config(base_path) if base_path else params.load_config({})
    # merge any inline overrides at top level (optional)
    for k, v in raw.items():
        if k not in {"base", "sweep", "output"}:
            _set_nested(base_cfg, k, v)
    grid_raw = (raw.get("sweep") or {}).get("grid") or {}
    grid: Dict[str, List[Any]] = {k: _expand_spec(v) for k, v in grid_raw.items()}
    return base_cfg, grid


def _product_dict(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


# ---- main ----


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gcfl-sweep",
        description="Run a parameter sweep (grid) and write a single combined table.",
    )
    ap.add_argument(
        "-c", "--config", required=True, help="Sweep YAML (with 'sweep.grid') or base config YAML"
    )
    ap.add_argument(
        "-o", "--out", default="results/logs/sweep.parquet", help="Output combined table path"
    )
    ap.add_argument("--out-format", choices=["parquet", "csv"], help="Override output format")

    # If --config points to a base YAML, you may provide an inline grid via --grid KEY=SPEC
    # SPEC can be a comma list (e.g., 0.1,0.2,0.5) or a linspace dict JSON: {"start":0,"stop":1,"num":11}
    ap.add_argument(
        "--grid",
        action="append",
        metavar="KEY=SPEC",
        help='Inline grid, e.g. --grid "mechanism.alpha=0.5,1.0,2.0"',
    )
    ap.add_argument(
        "--experiments-workers", type=int, default=0, help="Parallel experiments (0/1 sequential)"
    )
    ap.add_argument("--seed", type=int, help="Seed root (deterministic)")

    # Execution override commonly useful for all experiments
    ap.add_argument("--backend", help="Backend name (reference|scale|...)")
    ap.add_argument("--workers", type=int, help="Parallel workers for scale backend")

    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    log = get_logger("gcfl.sweep", level="DEBUG" if args.verbose else "INFO")
    seed_with(_engine.AGGREGATORS, _engine.SIGNALS, _engine.MECHANISMS)

    # Load sweep/base config + grid
    if args.grid:
        base_cfg = params.load_config(args.config)
        grid: Dict[str, List[Any]] = {}
        for item in args.grid:
            if "=" not in item:
                raise SystemExit(f"--grid must be KEY=SPEC, got: {item}")
            key, spec = item.split("=", 1)
            spec = spec.strip()
            if spec.startswith("{"):  # JSON dict for linspace
                vals = _expand_spec(json.loads(spec))
            else:
                vals = [float(x) if x.replace(".", "", 1).isdigit() else x for x in spec.split(",")]
            grid[key.strip()] = vals
    else:
        base_cfg, grid = _load_sweep_config(args.config)

    # Apply global overrides
    if args.backend:
        base_cfg["execution"]["backend"] = args.backend
    if args.workers is not None:
        base_cfg["execution"]["parallel_workers"] = int(args.workers)
    if args.out_format:
        base_cfg["logging"]["out_format"] = args.out_format
    if args.seed is not None:
        base_cfg["meta"]["seed_root"] = int(args.seed)

    # Resolve plugins once
    agg_fn = get_aggregator(base_cfg["aggregator"]["kind"])
    sig_fn = get_signal(base_cfg["signals"]["model"])
    mech_fn = get_mechanism(base_cfg["mechanism"]["policy"])
    backend_run = get_backend(base_cfg["execution"]["backend"])

    # Deterministic seed root; substream per experiment index
    seed_root = base_cfg["meta"].get("seed_root")
    root_ss = _rng.make_seedseq(seed_root)

    # Prepare jobs
    combos = list(_product_dict(grid)) if grid else [dict()]
    log.info(f"[sweep] total experiments: {len(combos)}")

    def _run_one(idx_val: Tuple[int, Dict[str, Any]]) -> pd.DataFrame:
        idx, vals = idx_val
        cfg = deepcopy(base_cfg)
        for k, v in vals.items():
            _set_nested(cfg, k, v)
        # derive a child seed per experiment
        ss = _rng.substream(root_ss, 0x5A11EE, idx)  # type: ignore[attr-defined]
        df = backend_run(cfg, agg_fn, sig_fn, mech_fn, ss)
        # annotate sweep params onto the rows
        for k, v in vals.items():
            df[k] = v
        return df

    # Run
    with Timer("sweep", logger=log):
        if args.experiments_workers and args.experiments_workers > 1:
            import concurrent.futures as _cf

            rows: List[pd.DataFrame] = []
            with _cf.ThreadPoolExecutor(max_workers=args.experiments_workers) as ex:
                for df in ex.map(_run_one, list(enumerate(combos))):
                    rows.append(df)
            big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        else:
            parts = [_run_one((i, vals)) for i, vals in enumerate(combos)]
            big = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Write outputs + provenance
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = write_table(big, args.out, base_cfg["logging"]["out_format"])
    meta = {
        "base_config": base_cfg,
        "grid": grid,
        "rows": int(len(big)),
        "experiments": len(combos),
    }
    write_provenance(os.path.splitext(written)[0], meta)
    log_provenance(log, extra={"rows": int(len(big)), "experiments": len(combos), "out": written})
    log.info(f"[done] wrote {written} ({len(big)} rows across {len(combos)} experiment(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

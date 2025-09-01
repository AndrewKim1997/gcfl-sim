"""
Scale backend — parallelize repeats using threads; deterministic per-repeat streams.

Notes:
- Keeps numerical behavior identical to reference for a fixed RNG stream per repeat.
- Parallelism is per-repeat; inner round loop is still sequential (stateful).
"""
from __future__ import annotations
from typing import List
import concurrent.futures as _cf
import pandas as pd

from ..rng import RngBundle
from ..types import LogRow
from .reference import _run_one_repeat  # reuse core logic for exact agreement

def run(cfg: dict, agg_fn, sig_fn, mech_fn, seedseq) -> pd.DataFrame:
    R = int(cfg["engine"]["repeats"])
    workers = int(cfg["execution"].get("parallel_workers", 0)) or None  # None => default
    rngs = RngBundle(seedseq)

    # We derive child SeedSequences deterministically per repeat to keep results stable.
    rep_seeds = [rngs.gen(0xA11CE, r).bit_generator._seed_seq for r in range(R)]  # type: ignore[attr-defined]

    rows: List[LogRow] = []
    if (workers or 0) > 1:
        with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for r in range(R):
                local_rngs = RngBundle(rep_seeds[r])  # isolate substreams
                futs.append(ex.submit(_run_one_repeat, r, cfg, agg_fn, sig_fn, mech_fn, local_rngs))
            for f in _cf.as_completed(futs):
                rows.extend(f.result())
    else:
        # fallback to sequential if workers<=1
        local_rngs = RngBundle(seedseq)
        for r in range(R):
            rows.extend(_run_one_repeat(r, cfg, agg_fn, sig_fn, mech_fn, local_rngs))

    df = pd.DataFrame.from_records(rows)
    # Stable ordering: (repeat, round)
    df.sort_values(["repeat", "round"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

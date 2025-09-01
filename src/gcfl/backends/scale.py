from __future__ import annotations
from typing import List
import concurrent.futures as _cf
import pandas as pd

from ..rng import RngBundle
from ..types import LogRow
from .reference import _run_one_repeat  # exact same inner loop

def run(cfg: dict, agg_fn, sig_fn, mech_fn, seedseq) -> pd.DataFrame:
    R = int(cfg["engine"]["repeats"])
    workers = int(cfg["execution"].get("parallel_workers", 0)) or 0
    rngs = RngBundle(seedseq)  # stateless → safe to share

    rows: List[LogRow] = []
    if workers > 1:
        with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one_repeat, r, cfg, agg_fn, sig_fn, mech_fn, rngs) for r in range(R)]
            for f in _cf.as_completed(futs):
                rows.extend(f.result())
    else:
        for r in range(R):
            rows.extend(_run_one_repeat(r, cfg, agg_fn, sig_fn, mech_fn, rngs))

    df = pd.DataFrame.from_records(rows)
    df.sort_values(["repeat", "round"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

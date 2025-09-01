"""
Dask backend — distributed repeats via Dask.

Requires: `pip install dask distributed`. If Dask is not available, importing this module will fail.
"""

from __future__ import annotations
from typing import List
import pandas as pd

try:  # import check at module load
    from dask import delayed, compute  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Dask backend requires `dask` and `distributed` to be installed.") from e

from ..rng import RngBundle, substream
from ..types import LogRow
from .reference import _run_one_repeat  # reuse core logic


def _repeat_task(r: int, cfg: dict, agg_fn, sig_fn, mech_fn, seedseq) -> List[LogRow]:
    rngs = RngBundle(seedseq)
    return _run_one_repeat(r, cfg, agg_fn, sig_fn, mech_fn, rngs)


def run(cfg: dict, agg_fn, sig_fn, mech_fn, seedseq) -> pd.DataFrame:
    R = int(cfg["engine"]["repeats"])
    tasks = [
        delayed(_repeat_task)(r, cfg, agg_fn, sig_fn, mech_fn, substream(seedseq, 0xA11CE, r))
        for r in range(R)
    ]
    results = compute(*tasks, scheduler="threads")  # or "processes" / distributed client
    rows: List[LogRow] = [row for part in results for row in part]
    df = pd.DataFrame.from_records(rows)
    df.sort_values(["repeat", "round"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

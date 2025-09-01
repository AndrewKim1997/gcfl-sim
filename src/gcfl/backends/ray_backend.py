"""
Ray backend — distributed repeats via Ray tasks.

Requires: `pip install ray`. If Ray is not available, importing this module will fail.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

try:  # import check at module load
    import ray  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Ray backend requires `ray` to be installed.") from e

from ..rng import RngBundle, substream, make_seedseq
from ..types import LogRow
from .reference import _run_one_repeat  # reuse core logic

@ray.remote
def _repeat_task(r: int, cfg: dict, agg_bytes: bytes, sig_bytes: bytes, mech_bytes: bytes, seed_state: bytes) -> List[LogRow]:
    import pickle
    agg_fn = pickle.loads(agg_bytes)
    sig_fn = pickle.loads(sig_bytes)
    mech_fn = pickle.loads(mech_bytes)
    seedseq = pickle.loads(seed_state)
    rngs = RngBundle(seedseq)
    return _run_one_repeat(r, cfg, agg_fn, sig_fn, mech_fn, rngs)

def run(cfg: dict, agg_fn, sig_fn, mech_fn, seedseq) -> pd.DataFrame:
    """
    Launch one Ray task per repeat. This preserves determinism by spawning
    a child SeedSequence per repeat and shipping it with the task.
    """
    import pickle

    R = int(cfg["engine"]["repeats"])
    # Ensure Ray is initialized (no-op if already)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Serialize callables & seeds for Ray
    agg_b = pickle.dumps(agg_fn)
    sig_b = pickle.dumps(sig_fn)
    mech_b = pickle.dumps(mech_fn)

    roots = [substream(seedseq, 0xA11CE, r) for r in range(R)]
    tasks = [
        _repeat_task.remote(r, cfg, agg_b, sig_b, mech_b, pickle.dumps(roots[r]))
        for r in range(R)
    ]
    results = ray.get(tasks)
    rows: List[LogRow] = [row for part in results for row in part]
    df = pd.DataFrame.from_records(rows)
    df.sort_values(["repeat", "round"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

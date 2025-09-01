"""
gcfl-sim: Lightweight simulator for gaming & cooperation in federated learning.

Public surface (stable):
- gcfl.engine.run_experiment(config: dict, seed: int | None, out: str | None) -> pd.DataFrame
- gcfl.params.load_config(path_or_dict) -> dict
- gcfl.rng.make_seedseq(seed_root), gcfl.rng.substream(...)
"""
from __future__ import annotations

from . import engine, params, rng, types  # re-export modules
from .utils_version import __version__ if False else None  # placeholder if you add versioning

__all__ = ["engine", "params", "rng", "types"]
__version__ = "0.1.0"

"""
gcfl-sim: Lightweight simulator for gaming & cooperation in federated learning.

Public surface (stable):
- gcfl.engine.run_experiment(config: dict, seed: int | None, out: str | None) -> pd.DataFrame
- gcfl.params.load_config(path_or_dict) -> dict
- gcfl.rng.make_seedseq(seed_root), gcfl.rng.substream(...)
"""

from __future__ import annotations

from . import engine, params, rng, types  # re-export modules
from .utils.version import package_version

__all__ = ["engine", "params", "rng", "types"]
__version__ = package_version()

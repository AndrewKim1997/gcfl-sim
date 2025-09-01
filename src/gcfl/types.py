from __future__ import annotations
from typing import Protocol, Mapping, MutableMapping, TypedDict, Any
import numpy as np

# ---------- Plugin protocols (interfaces) ----------

class Aggregator(Protocol):
    def __call__(self, values: np.ndarray, **kwargs) -> float: ...

class SignalModel(Protocol):
    def __call__(self, u: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray: ...

class Mechanism(Protocol):
    def __call__(
        self,
        state: MutableMapping[str, Any],
        u: np.ndarray,
        s: np.ndarray,
        m: float,
        rng: np.random.Generator,
        **kwargs,
    ) -> Mapping[str, float]:
        """
        Apply monitoring/enforcement to the current round.
        Returns a dict of round-level metrics (e.g., {'M': m, 'PoG': ..., 'PoC': ..., 'DeltaU': ...}).
        """

# ---------- Config & logging types ----------

class EngineConfig(TypedDict, total=False):
    meta: dict
    execution: dict
    engine: dict
    signals: dict
    aggregator: dict
    mechanism: dict
    logging: dict

class LogRow(TypedDict, total=False):
    repeat: int
    round: int
    N: int
    aggregator: str
    mechanism: str
    alpha: float
    pi: float
    M: float
    signal_mean: float
    signal_std: float
    u_mean: float
    u_std: float
    PoG: float
    PoC: float
    DeltaU: float

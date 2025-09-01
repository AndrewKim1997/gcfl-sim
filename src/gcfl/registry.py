"""
Lightweight plugin registry with decorator-based registration.

Usage:
    from gcfl.registry import register_aggregator, get_aggregator
    @register_aggregator("my_mean")
    def my_mean(values: np.ndarray, **kw) -> float: ...

    fn = get_aggregator("my_mean")
"""

from __future__ import annotations
from typing import Callable, Dict

# Internal registries
_AGG: Dict[str, Callable] = {}
_SIG: Dict[str, Callable] = {}
_MECH: Dict[str, Callable] = {}

# ----- decorators -----


def register_aggregator(name: str):
    def _decorator(fn: Callable):
        key = str(name).strip()
        if not key:
            raise ValueError("Aggregator name must be non-empty")
        _AGG[key] = fn
        return fn

    return _decorator


def register_signal(name: str):
    def _decorator(fn: Callable):
        key = str(name).strip()
        if not key:
            raise ValueError("Signal name must be non-empty")
        _SIG[key] = fn
        return fn

    return _decorator


def register_mechanism(name: str):
    def _decorator(fn: Callable):
        key = str(name).strip()
        if not key:
            raise ValueError("Mechanism name must be non-empty")
        _MECH[key] = fn
        return fn

    return _decorator


# ----- getters & listings -----


def get_aggregator(name: str) -> Callable:
    try:
        return _AGG[name]
    except KeyError as e:
        raise KeyError(f"Unknown aggregator: {name}. Available: {sorted(_AGG)}") from e


def get_signal(name: str) -> Callable:
    try:
        return _SIG[name]
    except KeyError as e:
        raise KeyError(f"Unknown signal: {name}. Available: {sorted(_SIG)}") from e


def get_mechanism(name: str) -> Callable:
    try:
        return _MECH[name]
    except KeyError as e:
        raise KeyError(f"Unknown mechanism: {name}. Available: {sorted(_MECH)}") from e


def list_aggregators() -> list[str]:
    return sorted(_AGG)


def list_signals() -> list[str]:
    return sorted(_SIG)


def list_mechanisms() -> list[str]:
    return sorted(_MECH)


# ----- optional: seed with builtins (call this from engine/__init__ if desired) -----


def seed_with(
    builtin_agg: Dict[str, Callable] | None = None,
    builtin_sig: Dict[str, Callable] | None = None,
    builtin_mech: Dict[str, Callable] | None = None,
) -> None:
    """
    Seed the registry with built-in plugin dictionaries.
    Safe to call multiple times; later calls overwrite on key collision.
    """
    if builtin_agg:
        _AGG.update(builtin_agg)
    if builtin_sig:
        _SIG.update(builtin_sig)
    if builtin_mech:
        _MECH.update(builtin_mech)

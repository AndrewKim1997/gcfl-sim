from __future__ import annotations

from typing import Callable, Dict, Optional

# Internal registries
_AGG: Dict[str, Callable[..., float]] = {}
_SIG: Dict[str, Callable[..., object]] = {}
_MECH: Dict[str, Callable[..., dict]] = {}


# ----- registration decorators -----
def register_aggregator(name: str):
    def _wrap(fn: Callable[..., float]):
        _AGG[str(name)] = fn
        return fn

    return _wrap


def register_signal(name: str):
    def _wrap(fn: Callable[..., object]):
        _SIG[str(name)] = fn
        return fn

    return _wrap


def register_mechanism(name: str):
    def _wrap(fn: Callable[..., dict]):
        _MECH[str(name)] = fn
        return fn

    return _wrap


# ----- helpers -----
def seed_with(
    aggs: Optional[Dict[str, Callable[..., float]]] = None,
    sigs: Optional[Dict[str, Callable[..., object]]] = None,
    mechs: Optional[Dict[str, Callable[..., dict]]] = None,
) -> None:
    """
    Prime registries with provided dictionaries, but:
      - do not override existing entries,
      - ignore entries where the value is None (placeholder names).
    """
    if aggs:
        for k, v in aggs.items():
            if v is None:
                continue
            _AGG.setdefault(k, v)
    if sigs:
        for k, v in sigs.items():
            if v is None:
                continue
            _SIG.setdefault(k, v)
    if mechs:
        for k, v in mechs.items():
            if v is None:
                continue
            _MECH.setdefault(k, v)


def _lazy_import_signals() -> None:
    # Import inside function to avoid circulars at module import time
    try:
        import gcfl.signals as _  # noqa: F401
    except Exception:
        pass  # keep silent; caller will error if unresolved


def _lazy_import_aggregates() -> None:
    try:
        import gcfl.aggregates as _  # noqa: F401
    except Exception:
        pass


def _lazy_import_mechanisms() -> None:
    try:
        import gcfl.mechanisms as _  # noqa: F401
    except Exception:
        pass


# ----- getters (with lazy import fallback) -----
def get_aggregator(name: str) -> Callable[..., float]:
    fn = _AGG.get(name)
    if fn is None:
        _lazy_import_aggregates()
        fn = _AGG.get(name)
    if fn is None:
        raise KeyError(f"unknown aggregator: {name!r}")
    return fn


def get_signal(name: str):
    fn = _SIG.get(name)
    if fn is None:
        _lazy_import_signals()
        fn = _SIG.get(name)
    if fn is None:
        raise KeyError(f"unknown signal: {name!r}")
    return fn


def get_mechanism(name: str) -> Callable[..., dict]:
    fn = _MECH.get(name)
    if fn is None:
        _lazy_import_mechanisms()
        fn = _MECH.get(name)
    if fn is None:
        raise KeyError(f"unknown mechanism: {name!r}")
    return fn


# ----- listings -----
def list_aggregators() -> list[str]:
    return sorted(_AGG.keys())


def list_signals() -> list[str]:
    return sorted(_SIG.keys())


def list_mechanisms() -> list[str]:
    return sorted(_MECH.keys())

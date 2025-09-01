"""
Execution backends for gcfl-sim.

Each backend exposes:
    run(cfg, agg_fn, sig_fn, mech_fn, seedseq) -> pd.DataFrame

Use `get_backend(name)` to resolve by string.
"""
from __future__ import annotations
from typing import Callable, Dict

from . import reference as _reference
from . import scale as _scale

_BACKENDS: Dict[str, Callable] = {
    "reference": _reference.run,
    "scale": _scale.run,
}

# Optional backends — register if importable
try:
    from . import ray_backend as _ray
    _BACKENDS["ray"] = _ray.run
except Exception:  # pragma: no cover
    pass

try:
    from . import dask_backend as _dask
    _BACKENDS["dask"] = _dask.run
except Exception:  # pragma: no cover
    pass


def get_backend(name: str) -> Callable:
    try:
        return _BACKENDS[name]
    except KeyError as e:
        raise KeyError(f"Unknown backend: {name}. Available: {sorted(_BACKENDS)}") from e


def list_backends() -> list[str]:
    return sorted(_BACKENDS)

from __future__ import annotations
from typing import Any, Mapping
from copy import deepcopy
import pathlib as _p
import yaml

_DEFAULTS: dict[str, Any] = {
    "meta": {"seed_root": 20250901, "experiment": "gcfl_sim"},
    "execution": {"backend": "reference", "parallel_workers": 0, "log_every": 1},
    "engine": {"clients": 200, "rounds": 50, "repeats": 5},
    "signals": {"model": "affine", "a": 1.0, "b": 0.0, "noise_sigma": 0.5},
    "aggregator": {"kind": "mean", "trim_ratio": 0.10},
    "mechanism": {"policy": "u_orth_penalty", "alpha": 0.5, "pi": 0.20, "phi": 1.0},
    "logging": {"out_format": "parquet", "float_precision": 6},
}


def _merge(a: dict, b: Mapping[str, Any]) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(out[k], v)  # type: ignore
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path_or_obj: str | _p.Path | Mapping[str, Any]) -> dict:
    """
    Load YAML config (or use given mapping), merge with defaults, and normalize types.
    """
    if isinstance(path_or_obj, Mapping):
        raw = dict(path_or_obj)
    else:
        p = _p.Path(path_or_obj)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = _merge(_DEFAULTS, raw)

    # Basic normalization & validation
    eng = cfg["engine"]
    for key in ("clients", "rounds", "repeats"):
        if not isinstance(eng[key], int) or eng[key] < 1:
            raise ValueError(f"engine.{key} must be a positive integer")
    sig = cfg["signals"]
    if sig["model"] != "affine":
        raise ValueError("signals.model currently supports only 'affine' in the base package")
    for key in ("a", "b", "noise_sigma"):
        sig[key] = float(sig[key])

    agg = cfg["aggregator"]
    if agg["kind"] not in {"mean", "median", "trimmed", "sorted_weighted"}:
        raise ValueError("aggregator.kind must be one of: mean|median|trimmed|sorted_weighted")
    agg["trim_ratio"] = float(agg.get("trim_ratio", 0.10))
    if "weights" in agg and agg["kind"] == "sorted_weighted":
        ws = [float(x) for x in agg["weights"]]
        s = sum(ws)
        if s <= 0:
            raise ValueError("aggregator.sorted_weighted.weights must sum to > 0")
        agg["weights"] = [w / s for w in ws]

    mech = cfg["mechanism"]
    mech["alpha"] = float(mech.get("alpha", 0.5))
    mech["pi"] = float(mech.get("pi", 0.2))
    mech["phi"] = float(mech.get("phi", 1.0))
    if mech.get("policy", "u_orth_penalty") != "u_orth_penalty":
        raise ValueError("mechanism.policy currently supports only 'u_orth_penalty'")

    # Logging
    fmt = cfg["logging"].get("out_format", "parquet").lower()
    if fmt not in {"parquet", "csv"}:
        raise ValueError("logging.out_format must be 'parquet' or 'csv'")
    cfg["logging"]["out_format"] = fmt

    return cfg

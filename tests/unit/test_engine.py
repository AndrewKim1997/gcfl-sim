from __future__ import annotations
import pandas as pd
from gcfl import engine
from gcfl.registry import seed_with
from gcfl import engine as _engine  # for builtins
from gcfl import aggregates as _agg_pkg  # noqa: F401
from gcfl import signals as _sig_pkg     # noqa: F401
from gcfl import mechanisms as _mech_pkg # noqa: F401

def _cfg():
    return {
        "meta": {"seed_root": 123},
        "execution": {"backend": "reference", "parallel_workers": 0},
        "engine": {"clients": 30, "rounds": 7, "repeats": 2},
        "signals": {"model": "affine", "a": 1.0, "b": 0.0, "noise_sigma": 0.3},
        "aggregator": {"kind": "mean"},
        "mechanism": {"policy": "u_orth_penalty", "alpha": 1.2, "pi": 0.2, "phi": 1.0},
        "logging": {"out_format": "csv"},
    }

def test_run_experiment_shapes_and_columns():
    # engine.run_experiment uses its own built-ins; still seed the registry for completeness
    seed_with(_engine.AGGREGATORS, _engine.SIGNALS, _engine.MECHANISMS)
    df: pd.DataFrame = engine.run_experiment(_cfg())
    assert len(df) == _cfg()["engine"]["rounds"] * _cfg()["engine"]["repeats"]
    for col in ["repeat", "round", "N", "aggregator", "mechanism", "M", "PoG", "PoC", "DeltaU"]:
        assert col in df.columns

def test_run_experiment_deterministic_same_seed():
    df1 = engine.run_experiment(_cfg())
    df2 = engine.run_experiment(_cfg())
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))

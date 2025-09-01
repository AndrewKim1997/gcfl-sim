from __future__ import annotations
import pandas as pd
from gcfl.registry import seed_with, get_aggregator, get_signal, get_mechanism
from gcfl import engine as _engine
from gcfl.backends import reference, scale

def _cfg():
    return {
        "meta": {"seed_root": 20250901},
        "execution": {"backend": "reference", "parallel_workers": 0},  # set workers=0 to avoid thread scheduling nondeterminism
        "engine": {"clients": 40, "rounds": 10, "repeats": 3},
        "signals": {"model": "affine", "a": 1.0, "b": 0.0, "noise_sigma": 0.5},
        "aggregator": {"kind": "trimmed", "trim_ratio": 0.1},
        "mechanism": {"policy": "u_orth_penalty", "alpha": 0.8, "pi": 0.2, "phi": 1.0},
        "logging": {"out_format": "csv"},
    }

def test_reference_equals_scale_sequential():
    # Seed registry and resolve plugins
    seed_with(_engine.AGGREGATORS, _engine.SIGNALS, _engine.MECHANISMS)
    cfg = _cfg()
    agg = get_aggregator(cfg["aggregator"]["kind"])
    sig = get_signal(cfg["signals"]["model"])
    mech = get_mechanism(cfg["mechanism"]["policy"])

    ss = _engine.make_seedseq(cfg["meta"]["seed_root"])  # reuse engine utility
    df_ref = reference.run(cfg, agg, sig, mech, ss)
    # scale with workers<=1 falls back to sequential but same code path; should be identical
    df_scale = scale.run(cfg, agg, sig, mech, ss)

    pd.testing.assert_frame_equal(
        df_ref.sort_values(["repeat", "round"]).reset_index(drop=True),
        df_scale.sort_values(["repeat", "round"]).reset_index(drop=True),
        check_exact=True,
    )

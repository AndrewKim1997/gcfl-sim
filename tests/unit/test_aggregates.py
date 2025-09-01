from __future__ import annotations
import numpy as np
from gcfl.registry import seed_with, get_aggregator
from gcfl import engine as _engine
from gcfl import aggregates as _agg_pkg  # noqa: F401  # trigger decorators

def _v():
    # includes NaN/inf; 'omit' policy should ignore them
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.inf, -np.inf])

def test_mean_and_median():
    seed_with(_engine.AGGREGATORS, None, None)
    mean = get_aggregator("mean")
    median = get_aggregator("median")
    assert mean(_v(), nan_policy="omit") == 3.0
    assert median(_v(), nan_policy="omit") == 3.0

def test_trimmed_basic():
    seed_with(_engine.AGGREGATORS, None, None)
    trimmed = get_aggregator("trimmed")
    out = trimmed(_v(), trim_ratio=0.2, nan_policy="omit")
    assert np.isclose(out, 3.0)  # [2,3,4] mean

def test_sorted_weighted_interpolation():
    seed_with(_engine.AGGREGATORS, None, None)
    sw = get_aggregator("sorted_weighted")
    # Weights favor the middle via interpolation (length 3 → resampled to 5)
    out = sw(_v(), weights=[0.0, 1.0, 0.0], nan_policy="omit")
    assert 2.5 < out < 3.5

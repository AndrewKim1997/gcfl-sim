from __future__ import annotations
import numpy as np
from gcfl.registry import seed_with, get_aggregator, get_signal, get_mechanism
from gcfl import engine as _engine
from gcfl import aggregates as _agg_pkg  # noqa: F401
from gcfl import signals as _sig_pkg  # noqa: F401
from gcfl import mechanisms as _mech_pkg  # noqa: F401


def test_aggregator_outputs_within_range():
    seed_with(_engine.AGGREGATORS, None, None)
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    vmin, vmax = vals.min(), vals.max()
    for name in ["mean", "median", "trimmed", "sorted_weighted"]:
        agg = get_aggregator(name)
        out = agg(vals, trim_ratio=0.2, weights=[0, 1, 0]) if name != "median" else agg(vals)
        assert vmin - 1e-12 <= out <= vmax + 1e-12


def test_signal_affine_zero_noise_is_exact():
    seed_with(None, _engine.SIGNALS, None)
    sig = get_signal("affine")
    rng = np.random.default_rng(0)
    u = rng.normal(size=50)
    s = sig(u, rng, a=2.0, b=-1.0, noise_sigma=0.0)
    assert np.allclose(s, 2.0 * u - 1.0)


def test_mechanism_pog_poc_exclusive():
    seed_with(None, None, _engine.MECHANISMS)
    mech = get_mechanism("u_orth_penalty")
    rng = np.random.default_rng(0)
    u = rng.normal(size=16)
    s = u.copy()
    for m in [-1.0, 0.0, 0.5]:
        metrics = mech({}, u, s, m, rng, alpha=0.5, pi=0.0, phi=1.0)
        pog, poc = metrics["PoG"], metrics["PoC"]
        assert not (pog > 0 and poc > 0)
        if m > 0:
            assert pog > 0 and poc == 0
        elif m < 0:
            assert poc > 0 and pog == 0
        else:
            assert pog == 0 and poc == 0

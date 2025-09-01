from __future__ import annotations
import numpy as np
from gcfl.registry import seed_with, get_mechanism
from gcfl import engine as _engine
from gcfl import mechanisms as _mech_pkg  # noqa: F401  # trigger decorators

def test_u_orth_penalty_parallel_signal_zero_penalty():
    seed_with(None, None, _engine.MECHANISMS)
    mech = get_mechanism("u_orth_penalty")
    rng = np.random.default_rng(0)
    u = rng.normal(size=64)
    s = 1.7 * u  # perfectly parallel
    m = 0.8      # positive monitoring → PoG > 0, PoC = 0
    metrics = mech({}, u, s, m, rng, alpha=0.5, pi=0.0, phi=1.0, benign_threshold=1.0)
    assert metrics["PoG"] > 0 and metrics["PoC"] == 0
    assert abs(metrics["DeltaU"]) < 1e-12  # no orthogonal component

def test_u_orth_penalty_strictly_orthogonal_negative_delta():
    seed_with(None, None, _engine.MECHANISMS)
    mech = get_mechanism("u_orth_penalty")
    rng = np.random.default_rng(1)
    u = rng.normal(size=128)
    u -= u.mean()
    r = rng.normal(size=128)
    r -= r.mean()
    # make s orthogonal to u: remove projection
    proj = (np.dot(r, u) / np.dot(u, u)) * u
    s = r - proj
    m = -0.5
    metrics = mech({}, u, s, m, rng, alpha=0.5, pi=0.0, phi=1.0,
                   benign_threshold=0.0, neutralize_when_deltaU_ge_0=False)
    assert metrics["PoC"] > 0 and metrics["PoG"] == 0
    assert metrics["DeltaU"] < 0

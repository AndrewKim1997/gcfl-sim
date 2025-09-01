from __future__ import annotations
import numpy as np
from gcfl.rng import make_seedseq, RngBundle, substream

def test_substream_determinism_and_distinctness():
    root1 = make_seedseq(123)
    root2 = make_seedseq(123)
    # identical roots → identical first draws
    g1 = np.random.default_rng(substream(root1, 1, 2, 3))
    g2 = np.random.default_rng(substream(root2, 1, 2, 3))
    assert np.allclose(g1.random(4), g2.random(4))

    # different keys → different streams
    g3 = np.random.default_rng(substream(root1, 1, 2, 4))
    assert not np.allclose(g1.random(4), g3.random(4))

def test_rngbundle_streams_are_stable():
    ss = make_seedseq(42)
    rngs = RngBundle(ss)
    a = rngs.for_repeat(0).random(3)
    b = rngs.for_repeat(0).random(3)  # same repeat → independent gens but same sequence start
    # New generator replays the same stream start
    assert np.allclose(a, b)

    # round/client substreams differ
    r0 = rngs.for_round(0, 0).integers(0, 1000)
    r1 = rngs.for_round(0, 1).integers(0, 1000)
    assert r0 != r1

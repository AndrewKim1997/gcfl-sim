from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _root_int(seed_or_ss: int | None | np.random.SeedSequence) -> int:
    if isinstance(seed_or_ss, np.random.SeedSequence):
        return int(seed_or_ss.generate_state(1, dtype=np.uint32)[0])
    if seed_or_ss is None:
        return 0
    return int(seed_or_ss) & 0xFFFFFFFF


def make_seedseq(seed_root: int | None = None) -> np.random.SeedSequence:
    """Create a SeedSequence from a plain integer (deterministic)."""
    return np.random.SeedSequence(_root_int(seed_root))


def substream(
    seed_or_ss: int | np.random.SeedSequence | None, *keys: int
) -> np.random.SeedSequence:
    """
    Deterministically derive a child SeedSequence from (root, keys...),
    without consuming state (no .spawn()).
    """
    root = _root_int(seed_or_ss)
    entropy = [root] + [int(k) & 0xFFFFFFFF for k in keys]
    return np.random.SeedSequence(entropy)


# ---- bundle ----


@dataclass(frozen=True)
class RngBundle:
    """
    Stateless RNG factory.
    Every call returns a brand-new Generator at the *start* of its stream
    for the given (tag, keys). Repeated calls with the same arguments
    produce identical sequences.
    """

    _root: int

    def __init__(self, seed_or_ss: int | np.random.SeedSequence | None):
        object.__setattr__(self, "_root", _root_int(seed_or_ss))

    def gen(self, tag: int, *keys: int) -> np.random.Generator:
        ss = np.random.SeedSequence(
            [self._root, int(tag) & 0xFFFFFFFF, *[int(k) & 0xFFFFFFFF for k in keys]]
        )
        return np.random.default_rng(ss)

    # convenience substreams
    def for_repeat(self, r: int) -> np.random.Generator:
        return self.gen(0xA11CE, r)

    def for_round(self, r: int, t: int) -> np.random.Generator:
        return self.gen(0x0D0C, r, t)

    def for_client(self, r: int, t: int, i: int) -> np.random.Generator:
        return self.gen(0xC1E17, r, t, i)

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np

# Deterministic substreams using SeedSequence spawning

def make_seedseq(seed_root: int | None) -> np.random.SeedSequence:
    if seed_root is None:
        # fixed default for reproducibility unless user provides a seed
        seed_root = 20250901
    return np.random.SeedSequence(seed_root)

def substream(ss: np.random.SeedSequence, *keys: int) -> np.random.SeedSequence:
    """
    Spawn a child SeedSequence deterministically from a tuple of integer keys.
    Example: substream(root, run_id, repeat, round, client_id)
    """
    return ss.spawn(1 + sum(1 for _ in keys))[-1] if keys else ss.spawn(1)[0]

@dataclass(frozen=True)
class RngBundle:
    """Convenience wrapper to derive repeat/round/client generators."""
    root: np.random.SeedSequence

    def gen(self, *keys: int) -> np.random.Generator:
        return np.random.default_rng(substream(self.root, *keys))

    def for_repeat(self, r: int) -> np.random.Generator:
        return self.gen(0xA11CE, r)

    def for_round(self, r: int, t: int) -> np.random.Generator:
        return self.gen(0xA11CE, r, t)

    def for_client(self, r: int, t: int, i: int) -> np.random.Generator:
        return self.gen(0xA11CE, r, t, i)

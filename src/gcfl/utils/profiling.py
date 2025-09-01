"""
Lightweight profiling helpers (time, CPU profile, and peak memory).
No external dependencies; safe to import anywhere.
"""

from __future__ import annotations
from typing import Any, Callable, Tuple
import os
import time
import tracemalloc
import cProfile
import pstats
import io as _io


class Timer:
    """Context manager for wall-clock timing with perf_counter()."""

    def __init__(self, label: str | None = None, logger=None):
        self.label = label
        self.logger = logger
        self.start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        end = time.perf_counter()
        self.elapsed = end - (self.start or end)
        if self.logger and self.label:
            self.logger.info(f"[timer] {self.label}: {self.elapsed:.6f}s")
        return False


def profile_callable(
    fn: Callable[..., Any],
    *args,
    sort: str = "cumtime",
    save_to: str | None = None,
    **kwargs,
) -> Tuple[Any, str]:
    """
    Run cProfile on a single callable invocation.
    Returns (result, stats_text). If save_to is given, write pstats text there.
    """
    pr = cProfile.Profile()
    result = pr.runcall(fn, *args, **kwargs)
    s = _io.StringIO()
    pstats.Stats(pr).strip_dirs().sort_stats(sort).print_stats(30, stream=s)  # top 30 lines
    stats_text = s.getvalue()
    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            f.write(stats_text)
    return result, stats_text


def trace_peak_memory(fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, int]:
    """
    Execute `fn(*args, **kwargs)` while measuring peak allocated bytes with tracemalloc.
    Returns (result, peak_bytes).
    """
    tracemalloc.start()
    try:
        result = fn(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        return result, int(peak)
    finally:
        tracemalloc.stop()


def profile_if_env(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator: enable cProfile when GCFL_PROFILE=1 is set in the environment.
    Writes a short report to stdout (and returns the wrapped result).
    """

    def wrapper(*args, **kwargs):
        if os.getenv("GCFL_PROFILE", "0") != "1":
            return fn(*args, **kwargs)
        pr = cProfile.Profile()
        res = pr.runcall(fn, *args, **kwargs)
        pstats.Stats(pr).strip_dirs().sort_stats("cumtime").print_stats(30)
        return res

    return wrapper

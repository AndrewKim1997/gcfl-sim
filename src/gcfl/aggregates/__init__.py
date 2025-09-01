"""
Aggregator plugins for gcfl-sim.

Importing this package registers built-in aggregators into the global registry:
- mean
- median
- trimmed
- sorted_weighted
"""
from __future__ import annotations

# Import submodules to trigger @register_aggregator decorators
from . import mean as _mean  # noqa: F401
from . import median as _median  # noqa: F401
from . import trimmed as _trimmed  # noqa: F401
from . import sorted_weighted as _sorted_weighted  # noqa: F401

__all__ = ["_mean", "_median", "_trimmed", "_sorted_weighted"]

"""
Signal/noise models for gcfl-sim.

Importing this package registers built-in signal models into the global registry:
- affine
"""

from __future__ import annotations

# Import submodules to trigger @register_signal decorators
from . import affine as _affine  # noqa: F401

__all__ = ["_affine"]

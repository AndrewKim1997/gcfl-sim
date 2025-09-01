"""
Mechanism plugins for gcfl-sim.

Importing this package registers built-in mechanisms into the global registry:
- u_orth_penalty
"""

from __future__ import annotations

# Import submodules to trigger @register_mechanism decorators
from . import u_orth_penalty as _u_orth_penalty  # noqa: F401

__all__ = ["_u_orth_penalty"]

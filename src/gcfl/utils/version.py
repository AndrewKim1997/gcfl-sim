"""
Version helpers. Try to read the installed package version; fall back to a constant.
Optionally include a short git description for local worktrees.
"""
from __future__ import annotations
from typing import Any, Dict
import os
import subprocess

try:  # Python 3.8+
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
except Exception:  # pragma: no cover
    _pkg_version = None  # type: ignore
    class PackageNotFoundError(Exception):  # type: ignore
        pass

__version__ = "0.1.0"  # fallback when metadata is unavailable

def package_version(dist_name: str = "gcfl-sim") -> str:
    """Return the installed distribution version if available; else fallback."""
    if _pkg_version is None:
        return __version__
    try:
        return _pkg_version(dist_name)
    except PackageNotFoundError:
        return __version__

def _git_info() -> Dict[str, Any]:
    """Best-effort git summary for local checkouts; safe if git is missing."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "short": short, "branch": branch, "dirty": dirty}
    except Exception:
        return {}

def full_version(dist_name: str = "gcfl-sim", with_git: bool = True) -> Dict[str, Any]:
    """
    Structured version info for logging/metadata sidecars.
    """
    info: Dict[str, Any] = {"version": package_version(dist_name)}
    if with_git:
        gi = _git_info()
        if gi:
            info["git"] = gi
    return info

"""
I/O utilities: safe writing to CSV/Parquet and lightweight provenance capture.
"""

from __future__ import annotations
from typing import Any, Dict
import json
import os
import platform
import socket
import subprocess
import shutil
import pandas as pd


def write_table(df: pd.DataFrame, out_path: str, fmt: str = "parquet") -> str:
    """
    Write `df` to `out_path` (with `fmt` = parquet|csv). Returns the written path.
    Falls back to CSV if Parquet writers are unavailable.
    """
    fmt = (fmt or "parquet").lower()
    base, ext = os.path.splitext(out_path)
    if fmt == "parquet":
        try:
            import pyarrow  # noqa

            path = base + ".parquet" if ext.lower() not in {".parq", ".parquet"} else out_path
            df.to_parquet(path, index=False)
            return path
        except Exception:
            # graceful fallback
            path = base + ".csv"
            df.to_csv(path, index=False)
            return path
    else:
        path = base + ".csv" if ext.lower() != ".csv" else out_path
        df.to_csv(path, index=False)
        return path


def _git_info() -> Dict[str, Any]:
    if not shutil.which("git"):
        return {"git": False}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        dirty = bool(status.strip())
        return {"git": True, "commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"git": False}


def provenance(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Collect a compact, privacy-respecting provenance dictionary.
    """
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": {
            "pandas": getattr(__import__("pandas"), "__version__", "unknown"),
            "numpy": getattr(__import__("numpy"), "__version__", "unknown"),
            "pyyaml": getattr(__import__("yaml"), "__version__", "unknown"),
            "pyarrow": getattr(__import__("pyarrow"), "__version__", "missing"),
        },
        "git": _git_info(),
    }
    if extra:
        info.update(extra)
    return info


def write_provenance(path_no_ext: str, meta: Dict[str, Any]) -> str:
    """
    Write provenance JSON sidecar next to a table. Returns the JSON path.
    """
    jpath = path_no_ext + ".provenance.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return jpath

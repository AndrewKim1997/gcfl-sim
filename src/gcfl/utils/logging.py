"""
Tiny logging setup helpers.
We keep Python's stdlib 'logging' but provide sane defaults + JSON helper.
"""

from __future__ import annotations
from typing import Any, Mapping
import json
import logging as _L
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "gcfl", level: int | str = "INFO") -> _L.Logger:
    """
    Return a configured logger with a single StreamHandler to stderr.
    Reuses existing handlers if already configured.
    """
    logger = _L.getLogger(name)
    if not logger.handlers:
        handler = _L.StreamHandler(stream=sys.stderr)
        fmt = _L.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    set_verbosity(level, logger)
    return logger


def set_verbosity(level: int | str, logger: _L.Logger | None = None) -> None:
    """
    Set the verbosity for a given logger (or the root 'gcfl' logger if None).
    Accepts numeric levels or strings like 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    """
    lv = _L.getLevelName(level) if isinstance(level, str) else int(level)
    (logger or _L.getLogger("gcfl")).setLevel(lv)


def log_json(
    logger: _L.Logger, payload: Mapping[str, Any], level: str = "INFO", prefix: str | None = None
) -> None:
    """
    Emit a compact JSON log line. Example:
        log_json(log, {"event":"run_end","rows":1024})
    """
    msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if prefix:
        msg = f"{prefix} {msg}"
    logger.log(_L.getLevelName(level) if isinstance(level, str) else int(level), msg)


def log_provenance(logger: _L.Logger, *, extra: Mapping[str, Any] | None = None) -> None:
    """
    Capture and log a compact provenance record (python/platform/git, selected packages).
    """
    try:
        from ..io import provenance as _prov

        payload = _prov(dict(extra) if extra else None)
    except Exception:  # fallback: still log what we have
        payload = dict(extra) if extra else {}
    log_json(logger, {"event": "provenance", **payload})

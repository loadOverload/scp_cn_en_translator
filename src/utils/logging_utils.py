"""Logging setup: console + optional file, consistently formatted."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None, name: str = "scp") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        logger.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        logger.propagate = False
        _CONFIGURED = True
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve()
            for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
            logger.addHandler(fh)
    return logger


def get_logger(name: str = "scp") -> logging.Logger:
    return logging.getLogger(name if name.startswith("scp") else f"scp.{name}")

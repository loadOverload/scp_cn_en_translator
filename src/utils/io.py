"""Small IO helpers shared by every script."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Atomically write JSONL (never corrupts an existing file on crash)."""
    p = ensure_parent(path)
    n = 0
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return n


def write_json(path: str | Path, obj: Any) -> None:
    """Atomic JSON write; unknown objects (e.g. torch.dtype) become strings."""
    p = ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False, default=str)
            fh.write("\n")
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def write_text(path: str | Path, text: str) -> None:
    p = ensure_parent(path)
    p.write_text(text, encoding="utf-8", newline="\n")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def sha1_short(text: str, n: int = 12) -> str:
    return sha1(text)[:n]

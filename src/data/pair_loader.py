"""Load raw English/Chinese SCP page pairs into a uniform record.

Supported layouts (``data.source`` in the config):

``dir``    two trees of raw files, ``data/raw/en/**`` and ``data/raw/zh/**``.
           Page ids are derived from filenames (``SCP-173.txt`` == ``scp-173.zh.txt``).
``jsonl``  one JSON object per line
``json``   a list, or ``{"pairs": [...]}``, or ``{id: {...}}``
``csv``    CSV/TSV with a header row
``sqlite`` a table with id/source/target columns

Nothing is ever written back to the raw inputs.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.io import read_text

# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass
class Pair:
    id: str
    source: str
    target: str
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def source_len(self) -> int:
        return len(self.source)

    @property
    def target_len(self) -> int:
        return len(self.target)

    @property
    def length_ratio(self) -> float:
        return self.target_len / max(self.source_len, 1)


# ---------------------------------------------------------------------------
# id normalisation
# ---------------------------------------------------------------------------

_FULLWIDTH = str.maketrans(
    {
        "－": "-",
        "—": "-",
        "–": "-",
        "：": ":",
        "＿": "_",
        "　": " ",
        "\u3000": " ",
    }
)

_ID_NOISE_RE = re.compile(r"[\s_]+")


def strip_extension(name: str, extensions: Sequence[str]) -> str:
    lowered = name.lower()
    for ext in sorted((e for e in extensions if e), key=len, reverse=True):
        if lowered.endswith(ext.lower()):
            return name[: -len(ext)]
    return name


def derive_id(path: Path, cfg: Mapping[str, Any]) -> str:
    name = strip_extension(path.name, cfg.get("extensions") or [".txt"])
    suffixes = cfg.get("id_strip_suffixes") or []
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if suffix and name.lower().endswith(str(suffix).lower()):
                name = name[: -len(suffix)]
                changed = True
    return name.strip()


def normalize_id(value: str) -> str:
    """Case/space/underscore-insensitive key used to pair the two languages."""
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).translate(_FULLWIDTH)
    value = _ID_NOISE_RE.sub("-", value.strip())
    value = re.sub(r"-{2,}", "-", value)
    return value.upper()


# ---------------------------------------------------------------------------
# Field detection for tabular sources
# ---------------------------------------------------------------------------


def _norm_key(key: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


def _pick(record: Mapping[str, Any], candidates: Sequence[str]) -> Optional[str]:
    lookup = {_norm_key(k): k for k in record.keys()}
    for cand in candidates:
        key = lookup.get(_norm_key(cand))
        if key is not None and record[key] is not None:
            value = record[key]
            if isinstance(value, str):
                return value
            return str(value)
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def scan_dir(directory: str | Path, cfg: Mapping[str, Any]) -> Dict[str, Path]:
    root = Path(directory)
    out: Dict[str, Path] = {}
    if not root.exists():
        return out
    pattern = "**/*" if cfg.get("recursive", True) else "*"
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.name.startswith("."):
            continue
        out.setdefault(normalize_id(derive_id(path, cfg)), path)
    return out


def load_from_dirs(cfg: Mapping[str, Any], logger=None) -> List[Pair]:
    en_dir = cfg.get("raw_en_dir")
    zh_dir = cfg.get("raw_zh_dir")
    en_map = scan_dir(en_dir, cfg) if en_dir else {}
    zh_map = scan_dir(zh_dir, cfg) if zh_dir else {}
    if logger:
        logger.info("scanned raw dirs: %d english files, %d chinese files", len(en_map), len(zh_map))
    keys = sorted(set(en_map) | set(zh_map))
    pairs: List[Pair] = []
    unmatched_en, unmatched_zh = [], []
    for key in keys:
        en_path, zh_path = en_map.get(key), zh_map.get(key)
        if en_path is None:
            unmatched_zh.append(str(zh_path))
            continue
        if zh_path is None:
            unmatched_en.append(str(en_path))
            continue
        pairs.append(
            Pair(
                id=derive_id(en_path, cfg) or key,
                source=read_text(en_path),
                target=read_text(zh_path),
                source_path=str(en_path),
                target_path=str(zh_path),
            )
        )
    if logger and (unmatched_en or unmatched_zh):
        logger.warning(
            "unpaired files: %d english-only, %d chinese-only (e.g. %s%s)",
            len(unmatched_en),
            len(unmatched_zh),
            (unmatched_en or unmatched_zh)[:2],
            " ..." if len(unmatched_en) + len(unmatched_zh) > 2 else "",
        )
    return pairs


def _records_from_json(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        for key in ("pairs", "data", "items", "records", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
        # {id: {source, target}}
        out = []
        for key, value in data.items():
            if isinstance(value, Mapping):
                out.append({"id": key, **value})
        if out:
            return out
    raise ValueError(f"unsupported JSON structure in {path}")


def _records_from_csv(path: Path) -> List[Mapping[str, Any]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def _records_from_sqlite(path: Path, table: Optional[str] = None) -> List[Mapping[str, Any]]:
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if not table:
            cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
            tables = [r[0] for r in cur.fetchall()]
            best, best_score = None, -1
            for name in tables:
                cur.execute(f'PRAGMA table_info("{name}")')
                cols = {_norm_key(c[1]) for c in cur.fetchall()}
                score = len(cols & {"source", "en", "english", "source_text"} ) + len(cols & {"target", "zh", "chinese", "target_text"})
                if score > best_score:
                    best, best_score = name, score
            if best is None:
                raise ValueError(f"no table found in {path}")
            table = best
        cur.execute(f'SELECT * FROM "{table}"')
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def load_from_file(cfg: Mapping[str, Any], logger=None) -> List[Pair]:
    path = Path(str(cfg.get("pairs_file") or ""))
    if not path.exists():
        raise FileNotFoundError(f"data.pairs_file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    elif suffix == ".json":
        records = _records_from_json(path)
    elif suffix in {".csv", ".tsv"}:
        records = _records_from_csv(path)
    elif suffix in {".sqlite", ".db", ".sqlite3"}:
        records = _records_from_sqlite(path, cfg.get("sqlite_table"))
    else:
        raise ValueError(f"unsupported pairs file type: {path.suffix}")

    id_fields = cfg.get("id_fields") or ["id"]
    src_fields = cfg.get("source_fields") or ["source"]
    tgt_fields = cfg.get("target_fields") or ["target"]

    pairs: List[Pair] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        source = _pick(record, src_fields)
        target = _pick(record, tgt_fields)
        pid = _pick(record, id_fields) or f"sample-{index}"
        pairs.append(Pair(id=str(pid), source=source or "", target=target or ""))
    if logger:
        logger.info("loaded %d raw pairs from %s", len(pairs), path)
    return pairs


def merged_data_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """``data:`` section, with the directory keys inherited from ``paths:``.

    The raw directories are configured under ``paths`` (so other scripts can
    find them too) while the loading behaviour lives under ``data``.
    """
    data_cfg: Dict[str, Any] = dict(cfg.get("data") or {})
    paths = cfg.get("paths") or {}
    for key in ("raw_dir", "raw_en_dir", "raw_zh_dir"):
        if not data_cfg.get(key) and paths.get(key):
            data_cfg[key] = paths[key]
    return data_cfg


def load_pairs(cfg: Mapping[str, Any], logger=None) -> List[Pair]:
    """Dispatch on ``data.source`` (``auto`` sniffs the available inputs)."""
    data_cfg = merged_data_cfg(cfg)
    source = str(data_cfg.get("source", "auto")).lower()

    if source == "auto":
        en_dir, zh_dir = data_cfg.get("raw_en_dir"), data_cfg.get("raw_zh_dir")
        has_dirs = bool(en_dir and Path(en_dir).exists() and any(Path(en_dir).glob("**/*"))) or bool(
            zh_dir and Path(zh_dir).exists() and any(Path(zh_dir).glob("**/*"))
        )
        pairs_file = data_cfg.get("pairs_file")
        if has_dirs:
            source = "dir"
        elif pairs_file and Path(str(pairs_file)).exists():
            source = Path(str(pairs_file)).suffix.lstrip(".").lower()
            source = {"ndjson": "jsonl", "db": "sqlite", "sqlite3": "sqlite"}.get(source, source)
        else:
            raise FileNotFoundError(
                "no raw data found: put files in data/raw/en + data/raw/zh, "
                "or set data.pairs_file to a jsonl/json/csv/sqlite file"
            )

    if source == "dir":
        return load_from_dirs(data_cfg, logger)
    return load_from_file(data_cfg, logger)

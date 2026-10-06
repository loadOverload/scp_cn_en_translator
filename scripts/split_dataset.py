#!/usr/bin/env python
"""Split a built dataset into train/validation **by document**.

    ./py scripts/split_dataset.py --val-ratio 0.02

Every file in ``data/conversion`` that is not already a validation file is split.
The split key is the document id, hashed deterministically, so the short set and
the long set hold out *the same documents* -- a paragraph pair from a document
never appears in training while a window from that document sits in validation.
Paragraph-level splitting would leak almost completely here, because a window is
literally made of the pairs the short set contains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DIR = "data/conversion"


def document_of(row: dict) -> str:
    """Document id, from whichever field the row carries."""
    metadata = row.get("metadata") or {}
    if metadata.get("scp_id"):
        return str(metadata["scp_id"])
    row_id = str(row.get("id", ""))
    # ids look like scp-003_p0012 / scp-003_w000
    for separator in ("_p", "_w"):
        if separator in row_id:
            return row_id.split(separator)[0]
    return row_id


def is_validation(document: str, ratio: float, salt: str = "scp-split-v1") -> bool:
    """Deterministic, stable across files and runs."""
    digest = hashlib.sha1(f"{salt}:{document}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < ratio


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Split a dataset by document")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-split files that are already split")
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    files = sorted(p for p in directory.glob(args.pattern)
                   if ".val." not in p.name and ".tmp" not in p.name)
    if not files:
        raise SystemExit(f"no files in {directory}")

    for path in files:
        stem = path.name[: -len(".jsonl")]
        if stem.endswith(".train") and not args.overwrite:
            # already a train file from a previous split: skip unless asked
            if (directory / f"{stem[:-len('.train')]}.val.jsonl").exists():
                print(f"  {path.name}: already split, skipping")
                continue
        base = stem[: -len(".train")] if stem.endswith(".train") else stem
        train_path = directory / f"{base}.train.jsonl"
        val_path = directory / f"{base}.val.jsonl"
        train_tmp = directory / f"{base}.train.jsonl.tmp"
        val_tmp = directory / f"{base}.val.jsonl.tmp"

        counts = {"train": 0, "val": 0}
        documents = {"train": set(), "val": set()}
        with path.open("r", encoding="utf-8") as source, \
                train_tmp.open("w", encoding="utf-8") as train_sink, \
                val_tmp.open("w", encoding="utf-8") as val_sink:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                document = document_of(row)
                which = "val" if is_validation(document, args.val_ratio) else "train"
                (val_sink if which == "val" else train_sink).write(line)
                counts[which] += 1
                documents[which].add(document)

        train_tmp.replace(train_path)
        val_tmp.replace(val_path)
        if path != train_path and path.exists():
            path.unlink()
        print(f"  {base:<26} train {counts['train']:>8,} ({len(documents['train']):>5,} 篇)   "
              f"val {counts['val']:>7,} ({len(documents['val']):>4,} 篇)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

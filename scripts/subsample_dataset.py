#!/usr/bin/env python
"""Subsample a dataset, spread evenly over documents.

    ./py scripts/subsample_dataset.py --input data/conversion/short.train.chat.jsonl \
        --output data/conversion/short.train.chat.jsonl --target 30000

A plain random 30,000 out of 265,923 would be dominated by the largest documents:
the biggest pages contribute hundreds of paragraph pairs each, so a uniform draw
mostly re-samples the same few dozen pages and the model sees far less variety
than the count suggests.

Instead this takes samples **round-robin across documents**: one per document
first, then a second each, and so on, until the target is reached. Every document
gets its first sample before any document gets its second, so the result covers
as many distinct pages as the budget allows.

The file is rewritten in place by default (via a temporary file), so the training
config keeps pointing at the same path.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def document_of(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("scp_id"):
        return str(metadata["scp_id"])
    row_id = str(row.get("id", ""))
    for separator in ("_p", "_w"):
        if separator in row_id:
            return row_id.split(separator)[0]
    return row_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Document-stratified subsampling")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="", help="default: rewrite the input in place")
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    source = Path(args.input)
    output = Path(args.output) if args.output else source
    rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    print(f"  input   {source.name}: {len(rows):,} 条")
    if args.target >= len(rows):
        print(f"  target {args.target:,} >= 输入，无需抽样")
        return 0

    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_doc[document_of(row)].append(row)

    rng = random.Random(args.seed)
    order = sorted(by_doc)                       # deterministic document order
    rng.shuffle(order)
    for doc in order:
        rng.shuffle(by_doc[doc])

    chosen: List[Dict[str, Any]] = []
    taken: Counter = Counter()
    depth = 0
    # round-robin: give every document its n-th sample before any gets its (n+1)-th
    while len(chosen) < args.target:
        added = 0
        for doc in order:
            if len(chosen) >= args.target:
                break
            if taken[doc] > depth:
                continue
            bucket = by_doc[doc]
            if taken[doc] < len(bucket):
                chosen.append(bucket[taken[doc]])
                taken[doc] += 1
                added += 1
        if added == 0:
            break
        depth += 1

    rng.shuffle(chosen)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in chosen:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)

    per_doc = Counter(document_of(r) for r in chosen)
    def user_text(row):
        """chat rows carry messages; raw rows carry source."""
        if row.get("source") is not None:
            return str(row["source"])
        for message in (row.get("messages") or []):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    sources = [len(user_text(r)) for r in chosen]
    docs = len(per_doc)
    print(f"  output  {output.name}: {len(chosen):,} 条  覆盖 {docs:,} 篇文档 "
          f"(原 {len(by_doc):,} 篇，覆盖率 {100*docs/len(by_doc):.1f}%)")
    print(f"  每篇样本数  min {min(per_doc.values())}  p50 {sorted(per_doc.values())[len(per_doc)//2]}  "
          f"max {max(per_doc.values())}")
    print(f"  源字符      p50 {sorted(sources)[len(sources)//2]}  max {max(sources):,}")
    print(f"  → 约 {len(chosen):,} 条 × 0.29 s/条 ≈ {len(chosen)*0.29/3600:.1f} 小时（1 轮）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

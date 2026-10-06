"""Split cleaned page pairs into train/val/test.

Rules enforced here:

* the split unit is a whole SCP page -- a page is never cut into pieces
* variant pages (``SCP-173`` / ``SCP-173-D`` ...) are grouped by
  ``split.variant_markers`` so two versions never straddle two splits
* identical content cannot appear in more than one split
  (``split.dedupe_by_content_across_splits``)
* the split is deterministic for a given seed
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..utils.io import sha1
from .clean import CleanRecord


def group_key(page_id: str, markers: Sequence[str]) -> str:
    """Map a page id to the variant-group it belongs to."""
    key = page_id.strip()
    norm = key.upper()
    for marker in sorted((m for m in markers if m), key=len, reverse=True):
        suffix = str(marker).upper()
        if norm.endswith(suffix) and len(norm) > len(suffix):
            return key[: -len(str(marker))].rstrip("-_ ")
    return key


def split_records(
    records: Sequence[CleanRecord],
    split_cfg: Mapping[str, Any],
    seed: int = 42,
    logger=None,
) -> Tuple[Dict[str, List[CleanRecord]], Dict[str, Any]]:
    train_ratio = float(split_cfg.get("train", 0.8))
    val_ratio = float(split_cfg.get("val", 0.1))
    test_ratio = float(split_cfg.get("test", 0.1))
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split ratios must be positive")
    train_ratio, val_ratio, test_ratio = (r / total_ratio for r in (train_ratio, val_ratio, test_ratio))

    markers = split_cfg.get("variant_markers") or []
    dedupe = bool(split_cfg.get("dedupe_by_content_across_splits", True))
    shuffle = bool(split_cfg.get("shuffle", True))

    groups: Dict[str, List[CleanRecord]] = defaultdict(list)
    for record in records:
        groups[group_key(record.id, markers)].append(record)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(keys)

    n_groups = len(keys)
    n_train = int(round(n_groups * train_ratio))
    n_val = int(round(n_groups * val_ratio))
    n_train = min(n_train, n_groups)
    n_val = min(n_val, max(n_groups - n_train, 0))
    # guarantee at least one page in val/test when the corpus allows it
    if n_groups >= 3:
        n_train = min(n_train, n_groups - 2)
        n_val = max(1, min(n_val, n_groups - n_train - 1))

    buckets: Dict[str, List[CleanRecord]] = {"train": [], "val": [], "test": []}
    for i, key in enumerate(keys):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        buckets[split].extend(groups[key])

    leakage_removed = {"val_from_train": 0, "test_from_train": 0, "test_from_val": 0}
    if dedupe:
        train_hashes = {r.source_sha1 for r in buckets["train"]} | {r.target_sha1 for r in buckets["train"]}
        kept_val: List[CleanRecord] = []
        for record in buckets["val"]:
            if record.source_sha1 in train_hashes or record.target_sha1 in train_hashes:
                leakage_removed["val_from_train"] += 1
                continue
            kept_val.append(record)
        buckets["val"] = kept_val

        used_hashes = set(train_hashes)
        for record in buckets["val"]:
            used_hashes.add(record.source_sha1)
            used_hashes.add(record.target_sha1)
        kept_test: List[CleanRecord] = []
        for record in buckets["test"]:
            if record.source_sha1 in used_hashes or record.target_sha1 in used_hashes:
                # decide which side it leaked from, for the report
                if record.source_sha1 in {r.source_sha1 for r in buckets["val"]}:
                    leakage_removed["test_from_val"] += 1
                else:
                    leakage_removed["test_from_train"] += 1
                continue
            kept_test.append(record)
        buckets["test"] = kept_test

    stats: Dict[str, Any] = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "groups": n_groups,
        "group_examples": {k: [r.id for r in groups[k]][:3] for k in keys[:5]},
        "splits": {
            name: {
                "pages": len(records_),
                "groups": len({group_key(r.id, markers) for r in records_}),
                "source_chars": sum(r.source_len for r in records_),
                "target_chars": sum(r.target_len for r in records_),
                "flagged": sum(1 for r in records_ if r.flags),
                "ids_head": [r.id for r in records_[:10]],
            }
            for name, records_ in buckets.items()
        },
        "leakage_removed": leakage_removed,
        "total_pages": sum(len(v) for v in buckets.values()),
    }
    if logger:
        for name in ("train", "val", "test"):
            logger.info(
                "split %-5s pages=%-6d groups=%-6d source_chars=%d",
                name,
                stats["splits"][name]["pages"],
                stats["splits"][name]["groups"],
                stats["splits"][name]["source_chars"],
            )
        if any(leakage_removed.values()):
            logger.info("removed cross-split content leakage: %s", leakage_removed)
    return buckets, stats

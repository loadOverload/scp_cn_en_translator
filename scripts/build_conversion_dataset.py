#!/usr/bin/env python
"""Build the two conversion training sets from ``data/raw/crawl.db``.

The task is framed as a conversion between two *text types* -- English SCP
Wikidot source and Chinese SCP Wikidot source -- rather than as translation.

Two data sets are produced:

**short** -- every page is split into paragraphs at blank lines, the two sides
are aligned with the bge-m3 monotonic aligner, and each matched pair becomes one
1:1 training sample. This teaches the paragraph-level mapping.

**long** -- consecutive aligned regions are packed into windows bounded by
``--max-tokens`` (16,384 by default, the context the model will be trained at).
A window is built greedily from the page's first paragraph onwards: keep adding
the next region while the whole sample (prompt + English + Chinese) still fits,
then start a new window at the next paragraph. This teaches the conversion with
real document context, which is where cross-paragraph references matter.

Region construction keeps **every** paragraph: an unmatched English paragraph is
attached to the region of the pair that follows it, so nothing is dropped and the
two sides still cover the same part of the document. (Paragraphs the Chinese
translation legitimately omits -- ``[[module Rate]]``, the license footer --
therefore stay on the English side, which is exactly the behaviour to learn.)

    ./py scripts/build_conversion_dataset.py --limit 30                 # smoke
    ./py scripts/build_conversion_dataset.py                            # full corpus
    ./py scripts/build_conversion_dataset.py --only long --max-tokens 16384
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys

import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.align.embedding_cache import DEFAULT_CACHE_DIR, embed_documents  # noqa: E402
from src.align.paragraph_align import (                       # noqa: E402
    DEFAULT_BATCH_SIZE, DEFAULT_GAP_PENALTY, DEFAULT_MAX_LENGTH, DEFAULT_MODEL,
    DEFAULT_POSITION_WEIGHT,
    ParagraphEmbedder, align_monotonic, similarity_matrix, split_paragraphs,
)
from src.utils.config import add_common_args, ensure_dirs, load_config   # noqa: E402
from src.utils.io import write_json, write_jsonl               # noqa: E402
from src.utils.logging_utils import setup_logging              # noqa: E402

DEFAULT_OUT_DIR = "data/conversion"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------


def load_pages(db_path: str, limit: int = 0, largest: int = 0) -> List[Dict[str, str]]:
    """Load page pairs; ``largest`` picks the N biggest English documents.

    The biggest ones are what exercise the window splitter, so a targeted run on
    them validates multi-window packing without processing the whole corpus.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    columns = [row[1] for row in con.execute("PRAGMA table_info(items)")]
    if not {"slug", "en_text", "cn_text"} <= set(columns):
        raise SystemExit(f"unexpected schema in {db_path}: {columns}")
    order = "en_chars DESC" if largest else "num"
    rows = con.execute(
        "SELECT slug, en_text, cn_text FROM items "
        "WHERE en_text IS NOT NULL AND cn_text IS NOT NULL "
        "AND length(trim(en_text)) > 0 AND length(trim(cn_text)) > 0 "
        f"ORDER BY {order}"
    ).fetchall()
    con.close()
    if largest:
        rows = rows[:largest]
    elif limit:
        rows = rows[:limit]
    return [{"id": str(slug), "en": en, "zh": zh} for slug, en, zh in rows]


# ---------------------------------------------------------------------------
# regions and windows
# ---------------------------------------------------------------------------


def batch_lengths(tokenizer, texts: Sequence[str], step: int = 4096) -> np.ndarray:
    """Token counts for many texts, in batches.

    Calling the tokenizer once per paragraph is what made the window builder slow:
    two calls per unit over 774k paragraphs is 1.5M Python-level calls, each with
    its own overhead, and it ran while the GPU sat idle. Batched, the same work is
    a few hundred calls.
    """
    lengths = np.zeros(len(texts), dtype=np.int64)
    for start in range(0, len(texts), step):
        batch = [t for t in texts[start:start + step]]
        if not batch:
            continue
        for offset, ids in enumerate(
            tokenizer(batch, add_special_tokens=False)["input_ids"]
        ):
            lengths[start + offset] = len(ids)
    return lengths


def build_units(
    en_paragraphs: Sequence[str],
    zh_paragraphs: Sequence[str],
    pairs: Sequence[Tuple[int, int]],
) -> List[Tuple[str, str]]:
    """Ordered ``(en, zh)`` units, one per paragraph, covering the whole page.

    A matched pair is one unit. An unmatched paragraph becomes a unit with an
    empty counterpart, so nothing is dropped: paragraphs the Chinese translation
    legitimately omits (``[[module Rate]]``, the license footer) stay on the
    English side.

    Packing happens at *this* granularity rather than per matched region. Region
    granularity blew up whenever a page had few matches -- one "region" could
    swallow hundreds of paragraphs and produce a 182k-token window, eleven times
    the budget.
    """
    units: List[Tuple[str, str]] = []
    en_cursor = zh_cursor = 0
    for en_index, zh_index in pairs:
        for i in range(en_cursor, en_index):
            units.append((en_paragraphs[i], ""))
        for j in range(zh_cursor, zh_index):
            units.append(("", zh_paragraphs[j]))
        units.append((en_paragraphs[en_index], zh_paragraphs[zh_index]))
        en_cursor, zh_cursor = en_index + 1, zh_index + 1
    for i in range(en_cursor, len(en_paragraphs)):
        units.append((en_paragraphs[i], ""))
    for j in range(zh_cursor, len(zh_paragraphs)):
        units.append(("", zh_paragraphs[j]))
    return units


def join_paragraphs(paragraphs: Sequence[str]) -> str:
    return "\n\n".join(paragraphs)


def pack_windows(
    units: Sequence[Any],
    cost_of,
    budget: int,
    overlap: int = 0,
) -> List[Tuple[int, int]]:
    """Greedy windows ``[start, end)`` over ``units`` within ``budget`` tokens.

    ``cost_of(start, end)`` returns the token cost of the window, measured on the
    actual joined text so the accounting is exact rather than estimated. A single
    unit that already exceeds the budget is emitted alone and flagged, never
    dropped.
    """
    windows: List[Tuple[int, int]] = []
    total = len(units)
    start = 0
    while start < total:
        end = start
        while end < total and cost_of(start, end + 1) <= budget:
            end += 1
        if end == start:
            end = start + 1
        windows.append((start, end))
        if end >= total:
            break
        start = end - overlap if overlap else end
    return windows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the conversion training sets")
    add_common_args(parser)
    parser.add_argument("--db", default="data/raw/crawl.db")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", choices=["short", "long", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="0 = every page")
    parser.add_argument("--largest", type=int, default=0,
                        help="take the N documents with the most English characters "
                             "(the interesting case for window splitting)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="token budget for one long sample (prompt + EN + ZH)")
    parser.add_argument("--overlap", type=int, default=0,
                        help="overlap between long windows, in regions")
    parser.add_argument("--min-source-chars", type=int, default=24,
                        help="short set: skip pairs whose English side is shorter")
    parser.add_argument("--skip-identical", action="store_true", default=True,
                        help="short set: skip pairs where both sides are identical")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-cache", default=DEFAULT_CACHE_DIR,
                        help="per-document paragraph embedding cache directory")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore the embedding cache and recompute everything")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--gap-penalty", type=float, default=DEFAULT_GAP_PENALTY)
    parser.add_argument("--position-weight", type=float, default=DEFAULT_POSITION_WEIGHT,
                        help="diagonal prior: how strongly a pairing is pushed towards the "
                             "proportional position (0 = similarity only)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="hold out this share of *pages* as validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    ensure_dirs(cfg)
    log = setup_logging("INFO", Path(cfg["paths"]["log_dir"]) / "build_conversion.log",
                        name="scp.convert")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)      # visible from the first second
    log.info("output directory: %s", out_dir.resolve())

    fmt = dict(cfg.get("format") or {})
    system_prompt = str(fmt.get("system_prompt") or "")
    user_template = str(fmt.get("user_template") or "{source}")

    # ---- tokenizer for budgeting -----------------------------------------
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # we only ever *count* tokens, so lift the sequence-length warning that a very
    # long paragraph triggers
    tokenizer.model_max_length = 10 ** 9
    overhead = len(tokenizer(
        (system_prompt + "\n" + user_template.format(source="")) if system_prompt
        else user_template.format(source=""),
        add_special_tokens=False,
    )["input_ids"]) + 8          # chat scaffolding (role markers, turn ends)
    log.info("tokenizer %s: fixed overhead %d tokens, window budget %d",
             args.tokenizer, overhead, args.max_tokens)

    pages = load_pages(args.db, args.limit, args.largest)
    log.info("loaded %d page pairs%s", len(pages),
             f" (largest by en_chars)" if args.largest else "")

    # ---- validation split by whole page (never split inside a page) -------
    val_ids = set()
    if args.val_ratio > 0:
        import random
        rng = random.Random(args.seed)
        shuffled = [p["id"] for p in pages]
        rng.shuffle(shuffled)
        val_ids = set(shuffled[: int(len(shuffled) * args.val_ratio)])
        log.info("holding out %d pages for validation", len(val_ids))

    # ---- split + embed ----------------------------------------------------
    prepared: List[Dict[str, Any]] = []
    for page in pages:
        prepared.append({"id": page["id"],
                         "en": split_paragraphs(page["en"]),
                         "zh": split_paragraphs(page["zh"])})

    started = time.time()
    vectors_by_doc, cache_stats = embed_documents(
        prepared,
        lambda: ParagraphEmbedder(model_name=args.model, device=args.device,
                                  batch_size=args.batch_size, max_length=args.max_length,
                                  use_fp16=True, logger=log),
        cache_dir=args.embedding_cache, model=args.model,
        use_cache=not args.no_cache, logger=log,
    )
    embed_seconds = time.time() - started
    log.info("phase timing: embed %.1fs (cache: %d reused / %d embedded)",
             embed_seconds, cache_stats["hits"], cache_stats["misses"])
    if embed_seconds > 0 and cache_stats["misses"]:
        log.info("embedded %d paragraphs in %.1fs", cache_stats["embedded_paragraphs"]
                 if "embedded_paragraphs" in cache_stats else 0, embed_seconds)

    # ---- build both sets --------------------------------------------------
    short_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "pages": 0, "pages_with_pairs": 0, "pairs_total": 0, "pairs_kept": 0,
        "units": 0, "windows": 0, "windows_over_budget": 0,
        "unmatched_en": 0, "unmatched_zh": 0,
        "short_tokens": [], "long_tokens": [], "similarities": [],
        "align_seconds": 0.0, "embed_seconds": round(embed_seconds, 1),
        "cache_hits": cache_stats["hits"], "cache_misses": cache_stats["misses"],
    }
    align_started = time.time()
    _t = {"sim_dp": 0.0, "window": 0.0}

    align_log_every = max(1, len(prepared) // 40)
    for page_index, item in enumerate(prepared, 1):
        if page_index % align_log_every == 0:
            log.info("  aligned %d/%d pages (%.0f%%)", page_index, len(prepared),
                     100 * page_index / max(len(prepared), 1))
        en_paras, zh_paras = item["en"], item["zh"]
        if not en_paras or not zh_paras:
            continue
        en_vec, zh_vec = vectors_by_doc[item["id"]]
        _t0 = time.time()
        sim = similarity_matrix(en_vec, zh_vec)
        pairs, info = align_monotonic(sim, args.gap_penalty, args.position_weight)
        _t["sim_dp"] += time.time() - _t0
        stats["pages"] += 1
        stats["pairs_total"] += len(pairs)
        stats["unmatched_en"] += len(info["unmatched_en"])
        stats["unmatched_zh"] += len(info["unmatched_zh"])
        if pairs:
            stats["pages_with_pairs"] += 1
        split_name = "val" if item["id"] in val_ids else "train"

        # ---- set 1: one sample per matched paragraph pair ----------------
        if args.only in ("short", "both"):
            for en_index, zh_index in pairs:
                source, target = en_paras[en_index], zh_paras[zh_index]
                if len(source) < args.min_source_chars:
                    continue
                if args.skip_identical and source == target:
                    continue
                similarity = float(sim[en_index, zh_index])
                stats["similarities"].append(similarity)
                stats["pairs_kept"] += 1
                short_rows.append({
                    "id": f"{item['id']}_p{en_index:04d}",
                    "task": "convert",
                    "split": split_name,
                    "source": source,
                    "target": target,
                    "metadata": {
                        "scp_id": item["id"], "kind": "short",
                        "en_index": en_index, "zh_index": zh_index,
                        "similarity": round(similarity, 4),
                        "n_en": len(en_paras), "n_zh": len(zh_paras),
                    },
                })

        # ---- set 2: multi-paragraph windows ------------------------------
        if args.only in ("long", "both"):
            units = build_units(en_paras, zh_paras, pairs)
            stats["units"] += len(units)
            en_join = [u[0] for u in units]
            zh_join = [u[1] for u in units]
            # per-unit cost, with the leading separator counted so a window's
            # cost is the exact sum of its units' costs (batched, not per-paragraph)
            en_cost = batch_lengths(tokenizer, [
                ("\n\n" if i else "") + t for i, t in enumerate(en_join)])
            zh_cost = batch_lengths(tokenizer, [
                ("\n\n" if i else "") + t for i, t in enumerate(zh_join)])

            en_prefix = np.concatenate(([0], np.cumsum(en_cost)))
            zh_prefix = np.concatenate(([0], np.cumsum(zh_cost)))

            def cost(start: int, end: int) -> int:
                return int(overhead + en_prefix[end] - en_prefix[start]
                           + zh_prefix[end] - zh_prefix[start])

            _tw = time.time()
            windows = pack_windows(units, cost, args.max_tokens, args.overlap)
            _t["window"] += time.time() - _tw
            for index, (start, end) in enumerate(windows):
                source = join_paragraphs([u[0] for u in units[start:end] if u[0]])
                target = join_paragraphs([u[1] for u in units[start:end] if u[1]])
                tokens = cost(start, end)
                over = tokens > args.max_tokens
                if over:
                    stats["windows_over_budget"] += 1
                stats["windows"] += 1
                stats["long_tokens"].append(tokens)
                long_rows.append({
                    "id": f"{item['id']}_w{index:03d}",
                    "task": "convert",
                    "split": split_name,
                    "source": source,
                    "target": target,
                    "metadata": {
                        "scp_id": item["id"], "kind": "long",
                        "unit_start": start, "unit_end": end,
                        "paragraphs_en": sum(1 for u in units[start:end] if u[0]),
                        "paragraphs_zh": sum(1 for u in units[start:end] if u[1]),
                        "tokens": tokens, "over_budget": over,
                    },
                })

    stats["align_seconds"] = round(time.time() - align_started, 1)
    log.info("phase timing: align loop %.1fs (similarity+DP %.1fs, window cost %.1fs)",
             stats["align_seconds"], _t["sim_dp"], _t["window"])

    # ---- write ------------------------------------------------------------
    written: Dict[str, str] = {}
    for name, rows in (("short", short_rows), ("long", long_rows)):
        if not rows:
            continue
        for split in ("train", "val"):
            subset = [r for r in rows if r["split"] == split]
            if not subset:
                continue
            raw_path = out_dir / f"{name}.{split}.jsonl"
            chat_path = out_dir / f"{name}.{split}.chat.jsonl"
            write_jsonl(raw_path, subset)
            write_jsonl(chat_path, (
                {
                    "id": row["id"], "task": "convert",
                    "messages": (
                        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                        + [{"role": "user", "content": user_template.format(source=row["source"])},
                           {"role": "assistant", "content": row["target"]}]
                    ),
                    "metadata": row["metadata"],
                }
                for row in subset
            ))
            written[f"{name}.{split}"] = str(raw_path)
            log.info("wrote %s: %d samples (chat: %s)", raw_path.name, len(subset), chat_path.name)

    # ---- report -----------------------------------------------------------
    def describe(values: Sequence[int]) -> Dict[str, Any]:
        if not values:
            return {}
        ordered = sorted(values)
        def q(p: float) -> int:
            return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]
        return {"n": len(ordered), "min": ordered[0], "p50": q(.5), "p90": q(.9),
                "p99": q(.99), "max": ordered[-1], "mean": round(statistics.fmean(ordered), 1)}

    report = {
        "db": args.db,
        "pages": stats["pages"],
        "pages_with_pairs": stats["pages_with_pairs"],
        "val_pages": len(val_ids),
        "max_tokens": args.max_tokens,
        "tokenizer": args.tokenizer,
        "prompt_overhead_tokens": overhead,
        "alignment": {
            "pairs_total": stats["pairs_total"],
            "unmatched_en": stats["unmatched_en"],
            "unmatched_zh": stats["unmatched_zh"],
            "mean_similarity": round(statistics.fmean(stats["similarities"]), 4)
            if stats["similarities"] else None,
        },
        "short": {
            "samples": len(short_rows),
            "train": sum(1 for r in short_rows if r["split"] == "train"),
            "val": sum(1 for r in short_rows if r["split"] == "val"),
            "kept_rate": round(stats["pairs_kept"] / max(stats["pairs_total"], 1), 4),
            "source_chars": describe([len(r["source"]) for r in short_rows]),
        },
        "long": {
            "samples": len(long_rows),
            "train": sum(1 for r in long_rows if r["split"] == "train"),
            "val": sum(1 for r in long_rows if r["split"] == "val"),
            "units": stats["units"],
            "windows_over_budget": stats["windows_over_budget"],
            "tokens": describe(stats["long_tokens"]),
        },
        "cache": {"hits": stats["cache_hits"], "misses": stats["cache_misses"],
                  "dir": args.embedding_cache},
        "timing": {"embed_seconds": stats["embed_seconds"],
                   "align_and_build_seconds": stats["align_seconds"]},
        "files": written,
    }
    write_json(out_dir / "report.json", report)

    print(f"\n  页面            {report['pages']:,}  (有配对 {report['pages_with_pairs']:,})")
    print(f"  嵌入缓存        复用 {report['cache']['hits']:,} / 新算 {report['cache']['misses']:,}"
          f"   ({report['timing']['embed_seconds']}s)")
    print(f"  对齐            {report['alignment']['pairs_total']:,} 对，"
          f"未配对 EN {report['alignment']['unmatched_en']:,} / ZH {report['alignment']['unmatched_zh']:,}，"
          f"相似度 {report['alignment']['mean_similarity']}")
    print(f"\n  短训练集        {report['short']['samples']:,} 条  "
          f"(train {report['short']['train']:,} / val {report['short']['val']:,})")
    if report["short"]["source_chars"]:
        s = report["short"]["source_chars"]
        print(f"    源字符数      min {s['min']} / p50 {s['p50']} / p99 {s['p99']} / max {s['max']}")
    print(f"\n  长训练集        {report['long']['samples']:,} 条  "
          f"(train {report['long']['train']:,} / val {report['long']['val']:,})")
    if report["long"]["tokens"]:
        t = report["long"]["tokens"]
        print(f"    tokens        min {t['min']} / p50 {t['p50']} / p90 {t['p90']} / max {t['max']}  "
              f"(预算 {args.max_tokens}, 超预算 {report['long']['windows_over_budget']})")
    print(f"\n  输出目录        {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

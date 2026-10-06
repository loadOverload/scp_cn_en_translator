#!/usr/bin/env python
"""Build the conversion training sets from an alignment, with a sliding window.

    ./py scripts/build_datasets_from_alignment.py                     # both sets
    ./py scripts/build_datasets_from_alignment.py --only long         # windows only
    ./py scripts/build_datasets_from_alignment.py --max-tokens 16384 --overlap 2

The alignment comes from ``scripts/align_scp_paragraphs.py``: one JSON line per
move, ``1:1`` / ``1:2`` / ``2:1`` / ``2:2`` / ``1:0`` / ``0:1``. That file is what
makes the two sets line up, because a window boundary must fall between alignment
*moves*, never inside one.

**short** -- one sample per ``1:1`` move: a single English paragraph and the single
Chinese paragraph that translates it.

**long** -- consecutive moves are packed into windows bounded by ``--max-tokens``
(default 16,384: prompt plus both sides must fit the model's context). Filling
starts at the document's first paragraph and keeps adding moves while the sample
still fits, then the window slides on. Because the unit of packing is an alignment
move, a merged pair (``2:1``) is never split across two windows, and a paragraph the
Chinese side dropped stays on the English side instead of being silently lost.

Order is preserved on both sides, so the English window and the Chinese window of a
sample always cover the same stretch of the document.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import add_common_args, ensure_dirs, load_config   # noqa: E402
from src.utils.io import write_json                            # noqa: E402
from src.utils.logging_utils import setup_logging              # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from align_scp_paragraphs import load_pages                     # noqa: E402
from build_conversion_dataset import batch_lengths              # noqa: E402

DEFAULT_MAX_TOKENS = 16384
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ALIGNMENTS = "data/aligned/paragraph_alignments.jsonl"


def read_alignments(path: str, logger=None) -> Dict[str, List[Dict[str, Any]]]:
    """Group the alignment JSONL by document, keeping the file's order."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[row["doc"]].append(row)
    if logger:
        logger.info("alignments: %d documents from %s", len(grouped), path)
    return grouped


def alignment_units(
    moves: Sequence[Dict[str, Any]],
    paragraphs: Dict[str, List[str]],
) -> List[Tuple[List[str], List[str]]]:
    """Turn the moves of one document into ordered ``(english, chinese)`` units.

    A move with an empty side becomes a unit with an empty side, so no paragraph
    is dropped: the English paragraphs the Chinese translation omits (a
    ``[[module Rate]]``, a licence footer) stay on the English side of the window
    they belong to, which is exactly what the model should learn to drop.
    """
    units: List[Tuple[List[str], List[str]]] = []
    for move in moves:
        source = [paragraphs["en"][_index_of(i)] for i in move["source_ids"]]
        target = [paragraphs["zh"][_index_of(j)] for j in move["target_ids"]]
        units.append((source, target))
    return units


def _index_of(paragraph_id: str) -> int:
    """``scp-003#s0012`` -> 12."""
    return int(paragraph_id.rsplit("#", 1)[1][1:])


def pack_windows(lengths: np.ndarray, budget: int, overlap: int = 0) -> List[Tuple[int, int]]:
    """Greedy windows ``[start, end)`` over per-unit costs.

    ``lengths[i]`` is the cost of unit ``i`` (both sides plus the prompt share).
    Filling starts at the current position and keeps adding units while the total
    stays within ``budget``; a single unit that already exceeds the budget is
    emitted alone rather than dropped.
    """
    prefix = np.concatenate(([0], np.cumsum(lengths)))
    windows: List[Tuple[int, int]] = []
    total = len(lengths)
    start = 0
    while start < total:
        end = start
        while end < total and prefix[end + 1] - prefix[start] <= budget:
            end += 1
        if end == start:
            end = start + 1
        windows.append((start, end))
        if end >= total:
            break
        start = end - overlap if overlap else end
    return windows


def join(paragraphs: Sequence[str]) -> str:
    return "\n\n".join(paragraphs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build conversion sets from an alignment")
    add_common_args(parser)
    parser.add_argument("--alignments", default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--db", default="data/raw/crawl.db")
    parser.add_argument("--out-dir", default="data/conversion")
    parser.add_argument("--only", choices=["short", "long", "both"], default="both")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap", type=int, default=0,
                        help="units of overlap between consecutive windows")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--docs", default="", help="comma-separated document ids to build")
    parser.add_argument("--exclude-docs", default="",
                        help="comma-separated document ids to leave out; '*' is a wildcard")
    parser.add_argument("--largest", type=int, default=0,
                        help="take the N documents with the most English characters")
    parser.add_argument("--min-source-chars", type=int, default=24)
    parser.add_argument("--keep-over-budget", action="store_true",
                        help="keep documents whose windows exceed --max-tokens "
                             "(they are dropped by default)")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    ensure_dirs(cfg)
    log = setup_logging("INFO", Path(cfg["paths"]["log_dir"]) / "build_datasets.log",
                        name="scp.dataset")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)          # visible immediately
    log.info("output directory: %s", out_dir.resolve())

    fmt = dict(cfg.get("format") or {})
    system_prompt = str(fmt.get("system_prompt") or "")
    user_template = str(fmt.get("user_template") or "{source}")

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = 10 ** 9
    overhead = len(tokenizer(
        (system_prompt + "\n" + user_template.format(source="")) if system_prompt
        else user_template.format(source=""), add_special_tokens=False)["input_ids"]) + 8
    log.info("tokenizer %s: prompt overhead %d tokens, window budget %d",
             args.tokenizer, overhead, args.max_tokens)

    alignments = read_alignments(args.alignments, log)
    pages = {page["id"]: page for page in load_pages(args.db)}
    doc_ids = [doc_id for doc_id in alignments if doc_id in pages]
    if args.exclude_docs:
        import fnmatch
        patterns = [p.strip() for p in args.exclude_docs.split(",") if p.strip()]
        before = len(doc_ids)
        doc_ids = [d for d in doc_ids if not any(fnmatch.fnmatch(d, pat) for pat in patterns)]
        log.info("excluded %d documents matching %s", before - len(doc_ids), patterns)
    if args.docs:
        wanted = {d.strip() for d in args.docs.split(",") if d.strip()}
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in wanted]
    if args.largest:
        doc_ids = sorted(doc_ids, key=lambda d: len(pages[d]["en"]), reverse=True)[:args.largest]
    elif args.limit_docs:
        doc_ids = doc_ids[:args.limit_docs]
    log.info("%d documents have both an alignment and source text", len(doc_ids))

    val_ids: set = set()
    if args.val_ratio > 0:
        import random
        rng = random.Random(args.seed)
        shuffled = list(doc_ids)
        rng.shuffle(shuffled)
        val_ids = set(shuffled[: int(len(shuffled) * args.val_ratio)])

    counters: Counter = Counter()
    short_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    long_tokens: List[int] = []
    over_budget = 0
    dropped: List[Dict[str, Any]] = []
    started = time.time()

    for position, doc_id in enumerate(doc_ids, 1):
        page = pages[doc_id]
        from src.align.paragraph_align import split_paragraphs
        paragraphs = {"en": split_paragraphs(page["en"]), "zh": split_paragraphs(page["zh"])}
        moves = alignments[doc_id]
        units = alignment_units(moves, paragraphs)
        split_name = "val" if doc_id in val_ids else "train"

        # ---- set 1: single-paragraph pairs ---------------------------------
        if args.only in ("short", "both"):
            for move in moves:
                if move["type"] != "1:1":
                    continue
                source, target = paragraphs["en"][_index_of(move["source_ids"][0])], \
                    paragraphs["zh"][_index_of(move["target_ids"][0])]
                if len(source) < args.min_source_chars or source == target:
                    continue
                counters["short"] += 1
                short_rows.append({
                    "id": f"{doc_id}_p{_index_of(move['source_ids'][0]):04d}",
                    "task": "convert", "split": split_name,
                    "source": source, "target": target,
                    "metadata": {"scp_id": doc_id, "kind": "short",
                                 "similarity": move["similarity"]},
                })

        # ---- set 2: sliding windows ----------------------------------------
        if args.only in ("long", "both") and units:
            # Cost per unit, then a window cost that adds the prompt once.
            # The overhead is *per window*, not per document: charging it to unit
            # 0 only (the first thing I wrote) under-counted every window except
            # the first by the whole prompt length.
            en_texts = [("\n\n" if i else "") + join(u[0]) if u[0] else ""
                        for i, u in enumerate(units)]
            zh_texts = [("\n\n" if i else "") + join(u[1]) if u[1] else ""
                        for i, u in enumerate(units)]
            en_len = batch_lengths(tokenizer, en_texts)
            zh_len = batch_lengths(tokenizer, zh_texts)
            unit_len = en_len + zh_len

            def window_text(begin: int, finish: int) -> Tuple[str, str]:
                source = join([p for unit in units[begin:finish] for p in unit[0]])
                target = join([p for unit in units[begin:finish] for p in unit[1]])
                return source, target

            def exact_cost(begin: int, finish: int) -> int:
                """Real token count of the sample, tokenised as it will be trained on.

                Summing per-unit counts is only an estimate: BPE merges across a
                unit boundary make the whole differ from the parts, by up to ~100
                tokens on a long window in the tail. Since the estimate is what
                the greedy fill trusts, every window is re-measured exactly and
                shrunk until it truly fits.
                """
                source, target = window_text(begin, finish)
                cost = overhead
                if source:
                    cost += len(tokenizer(source, add_special_tokens=False)["input_ids"])
                if target:
                    cost += len(tokenizer(target, add_special_tokens=False)["input_ids"])
                return cost

            windows = pack_windows(unit_len, args.max_tokens - overhead, args.overlap)

            # Exact verification, once per window: the per-unit sum is only an
            # estimate, because BPE merges across a unit boundary make the whole
            # differ from its parts. Each window is re-measured on the text it
            # will actually be trained on and shrunk until it truly fits.
            packed: List[Tuple[int, int, int]] = []
            for begin, finish in windows:
                cost = exact_cost(begin, finish)
                while finish > begin + 1 and cost > args.max_tokens:
                    finish -= 1
                    cost = exact_cost(begin, finish)
                packed.append((begin, finish, cost))

            largest = max(cost for _, _, cost in packed) if packed else 0
            if largest > args.max_tokens and not args.keep_over_budget:
                # One alignment move can be bigger than the entire budget: here a
                # 92k-character CSS block with no blank line to split at, and
                # eleven more like it. Packing happens at move granularity so the
                # move cannot be cut, and a window that does not fit the context
                # cannot be trained on, so the document is left out of the long
                # set rather than producing an unusable sample.
                dropped.append({"doc": doc_id, "largest_window": largest,
                                "windows": len(packed), "units": len(units)})
                counters["long_dropped"] += 1
                continue

            for index, (start, end, tokens) in enumerate(packed):
                source = join([p for unit in units[start:end] for p in unit[0]])
                target = join([p for unit in units[start:end] for p in unit[1]])
                if tokens > args.max_tokens:
                    over_budget += 1
                counters["long"] += 1
                long_tokens.append(tokens)
                long_rows.append({
                    "id": f"{doc_id}_w{index:03d}",
                    "task": "convert", "split": split_name,
                    "source": source, "target": target,
                    "metadata": {
                        "scp_id": doc_id, "kind": "long",
                        "unit_start": start, "unit_end": end,
                        "paragraphs_en": sum(1 for u in units[start:end] if u[0]),
                        "paragraphs_zh": sum(1 for u in units[start:end] if u[1]),
                        "tokens": tokens, "over_budget": bool(tokens > args.max_tokens),
                    },
                })

        if position % 500 == 0 or position == len(doc_ids):
            log.info("  built %d/%d documents (%.0f%%), short %d / long %d",
                     position, len(doc_ids), 100 * position / len(doc_ids),
                     counters["short"], counters["long"])

    # ---- drop over-budget short samples -----------------------------------
    # The long set drops a whole document because a window cannot be trained on
    # half-way. A short sample is a single paragraph pair, so only the offending
    # pairs are dropped -- the remaining hundreds of pairs from the same document
    # are perfectly trainable.
    short_over = 0
    if short_rows and args.only in ("short", "both"):
        costs = batch_lengths(tokenizer, [r["source"] for r in short_rows]) + \
            batch_lengths(tokenizer, [r["target"] for r in short_rows]) + overhead
        keep = costs <= args.max_tokens
        short_over = int((~keep).sum())
        if short_over:
            short_rows = [row for row, ok in zip(short_rows, keep) if ok]

    elapsed = time.time() - started

    # ---- write ------------------------------------------------------------
    written: Dict[str, str] = {}
    for name, rows in (("short", short_rows), ("long", long_rows)):
        for split in ("train", "val"):
            subset = [row for row in rows if row["split"] == split]
            if not subset:
                continue
            raw_path = out_dir / f"{name}.{split}.jsonl"
            chat_path = out_dir / f"{name}.{split}.chat.jsonl"
            with raw_path.open("w", encoding="utf-8") as sink:
                for row in subset:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            with chat_path.open("w", encoding="utf-8") as sink:
                for row in subset:
                    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                        {"role": "user", "content": user_template.format(source=row["source"])},
                        {"role": "assistant", "content": row["target"]},
                    ]
                    sink.write(json.dumps({"id": row["id"], "task": "convert",
                                           "messages": messages,
                                           "metadata": row["metadata"]}, ensure_ascii=False) + "\n")
            written[f"{name}.{split}"] = str(raw_path)
            log.info("wrote %s: %d samples", raw_path.name, len(subset))

    def describe(values: Sequence[int]) -> Dict[str, Any]:
        if not values:
            return {}
        ordered = sorted(values)
        return {"n": len(ordered), "min": ordered[0],
                "p50": ordered[len(ordered) // 2],
                "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
                "max": ordered[-1], "mean": round(statistics.fmean(ordered), 1)}

    report = {
        "alignments": args.alignments, "db": args.db,
        "documents": len(doc_ids), "val_documents": len(val_ids),
        "excluded_docs": args.exclude_docs,
        "max_tokens": args.max_tokens, "overlap": args.overlap,
        "prompt_overhead_tokens": overhead,
        "short_dropped_over_budget": short_over,
        "short": {"samples": len(short_rows),
                  "train": sum(1 for r in short_rows if r["split"] == "train"),
                  "val": sum(1 for r in short_rows if r["split"] == "val"),
                  "source_chars": describe([len(r["source"]) for r in short_rows])},
        "long_dropped_documents": dropped,
        "long": {"samples": len(long_rows),
                 "train": sum(1 for r in long_rows if r["split"] == "train"),
                 "val": sum(1 for r in long_rows if r["split"] == "val"),
                 "over_budget": over_budget,
                 "tokens": describe(long_tokens)},
        "seconds": round(elapsed, 1), "files": written,
    }
    write_json(out_dir / "report.json", report)

    print(f"\n  文档            {report['documents']:,}")
    print(f"\n  短训练集        {report['short']['samples']:,} 条"
          f"  (train {report['short']['train']:,} / val {report['short']['val']:,})")
    if short_over:
        print(f"  短集剔除        {short_over} 条超预算样本（> {args.max_tokens:,} tokens）")
    if report["short"]["source_chars"]:
        s = report["short"]["source_chars"]
        print(f"    源字符        min {s['min']} / p50 {s['p50']} / p90 {s['p90']} / max {s['max']}")
    if dropped:
        print(f"\n  长集剔除        {len(dropped)} 篇文档（单个对齐移动就超过预算，"
              f"窗口无法满足上下文）")
        for item in sorted(dropped, key=lambda d: -d["largest_window"])[:6]:
            print(f"    {item['doc']:<12} 最大窗口 {item['largest_window']:>9,} tokens")
        if len(dropped) > 6:
            print(f"    ... 其余 {len(dropped) - 6} 篇见 report.json")
    print(f"\n  长训练集        {report['long']['samples']:,} 条"
          f"  (train {report['long']['train']:,} / val {report['long']['val']:,})")
    if report["long"]["tokens"]:
        t = report["long"]["tokens"]
        print(f"    tokens        min {t['min']:,} / p50 {t['p50']:,} / p90 {t['p90']:,} / max {t['max']:,}"
              f"   预算 {args.max_tokens:,}   超预算 {over_budget}")
    print(f"\n  耗时            {elapsed:.1f}s")
    print(f"  输出            {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

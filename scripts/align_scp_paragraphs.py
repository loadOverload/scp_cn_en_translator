#!/usr/bin/env python
"""Align every SCP page's English and Chinese paragraphs.

    ./py scripts/align_scp_paragraphs.py                      # whole corpus
    ./py scripts/align_scp_paragraphs.py --limit 100          # try it out
    ./py scripts/align_scp_paragraphs.py --with-text          # also store the text
    ./py scripts/align_scp_paragraphs.py --threshold 0.75     # stricter matching

Paragraphs come from splitting at blank lines; embeddings come from the
per-document cache (``data/embeddings``), so a run costs disk reads rather than a
GPU pass, and any document missing from the cache is embedded on the spot.

One JSON line per alignment move:

    {"doc": "scp-003", "type": "1:1",
     "source_ids": ["scp-003#s0002"], "target_ids": ["scp-003#t0001"],
     "similarity": 0.8988}

Results are appended and flushed **per document**, so an interrupted run keeps
everything it finished, and ``--resume`` (the default) skips documents that are
already in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.align.embedding_cache import DEFAULT_CACHE_DIR, load as cache_load, save as cache_save  # noqa: E402
from src.align.paragraph_align import DEFAULT_MAX_LENGTH, DEFAULT_MODEL, ParagraphEmbedder, split_paragraphs  # noqa: E402
from src.align.paragraph_aligner import (                       # noqa: E402
    DEFAULT_GAP_PENALTY, DEFAULT_MIN_SIMILARITY, DEFAULT_POSITION_WEIGHT,
    align_page,
)
from src.utils.config import add_common_args, ensure_dirs, load_config   # noqa: E402
from src.utils.io import write_json                            # noqa: E402
from src.utils.logging_utils import setup_logging              # noqa: E402


def load_pages(db_path: str, limit: int = 0) -> List[Dict[str, str]]:
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    columns = [row[1] for row in con.execute("PRAGMA table_info(items)")]
    if not {"slug", "en_text", "cn_text"} <= set(columns):
        raise SystemExit(f"unexpected schema in {db_path}: {columns}")
    rows = con.execute(
        "SELECT slug, en_text, cn_text FROM items "
        "WHERE en_text IS NOT NULL AND cn_text IS NOT NULL "
        "AND length(trim(en_text)) > 0 AND length(trim(cn_text)) > 0 ORDER BY num"
    ).fetchall()
    con.close()
    if limit:
        rows = rows[:limit]
    return [{"id": str(slug), "en": en, "zh": zh} for slug, en, zh in rows]


def already_done(out_path: Path) -> set:
    """Document ids already present in the output, for --resume."""
    done = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Monotonic paragraph alignment")
    add_common_args(parser)
    parser.add_argument("--db", default="data/raw/crawl.db")
    parser.add_argument("--out", default="data/aligned/paragraph_alignments.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--largest", type=int, default=0,
                        help="take the N documents with the most English characters")
    parser.add_argument("--embedding-cache", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_MIN_SIMILARITY,
                        help="cosine below which a pair may not be matched")
    parser.add_argument("--gap-penalty", type=float, default=DEFAULT_GAP_PENALTY)
    parser.add_argument("--position-weight", type=float, default=DEFAULT_POSITION_WEIGHT)
    parser.add_argument("--with-text", action="store_true",
                        help="also store the paragraph texts in each row")
    parser.add_argument("--no-resume", action="store_true",
                        help="re-align documents already present in the output")
    parser.add_argument("--fresh", action="store_true", help="truncate the output first")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    ensure_dirs(cfg)
    log = setup_logging("INFO", Path(cfg["paths"]["log_dir"]) / "align_scp_paragraphs.log",
                        name="scp.align")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and out_path.exists():
        out_path.unlink()
    done = set() if args.no_resume else already_done(out_path)
    if done:
        log.info("resuming: %d documents already aligned in %s", len(done), out_path)

    if args.largest:
        import sqlite3
        con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        order = "en_chars DESC"
        rows = con.execute(
            "SELECT slug, en_text, cn_text FROM items WHERE en_text IS NOT NULL AND cn_text IS NOT NULL "
            f"AND length(trim(en_text))>0 AND length(trim(cn_text))>0 ORDER BY {order}"
        ).fetchall()[:args.largest]
        con.close()
        pages = [{"id": str(s), "en": e, "zh": z} for s, e, z in rows]
    else:
        pages = load_pages(args.db, args.limit)
    log.info("loaded %d page pairs", len(pages))

    embedder: Optional[ParagraphEmbedder] = None

    def get_embedder() -> ParagraphEmbedder:
        nonlocal embedder
        if embedder is None:
            embedder = ParagraphEmbedder(
                model_name=args.model, device=args.device, batch_size=args.batch_size,
                max_length=args.max_length, use_fp16=True, logger=log,
            )
        return embedder

    started = time.time()
    type_counts: Counter = Counter()
    similarities: List[float] = []
    per_page: List[Dict[str, Any]] = []
    n_docs = n_skipped = n_embedded_docs = 0
    sink = out_path.open("a", encoding="utf-8")
    try:
        for index, page in enumerate(pages, 1):
            doc_id = page["id"]
            if doc_id in done:
                n_skipped += 1
                continue
            en_paras = split_paragraphs(page["en"])
            zh_paras = split_paragraphs(page["zh"])
            if not en_paras or not zh_paras:
                continue

            vectors = cache_load(args.embedding_cache, doc_id, en_paras, zh_paras, args.model)
            if vectors is None:
                en_vec = get_embedder().encode(en_paras)
                zh_vec = get_embedder().encode(zh_paras)
                cache_save(args.embedding_cache, doc_id, en_paras, zh_paras,
                           en_vec, zh_vec, args.model)
                n_embedded_docs += 1
            else:
                en_vec, zh_vec = vectors

            result = align_page(
                doc_id, en_vec, zh_vec,
                source_ids=[f"{doc_id}#s{i:04d}" for i in range(len(en_paras))],
                target_ids=[f"{doc_id}#t{j:04d}" for j in range(len(zh_paras))],
                gap_penalty=args.gap_penalty, position_weight=args.position_weight,
                min_similarity=args.threshold,
            )

            for alignment in result.alignments:
                row = alignment.to_dict(doc_id, result.source_ids, result.target_ids)
                if args.with_text:
                    row["source_text"] = [en_paras[i] for i in alignment.source]
                    row["target_text"] = [zh_paras[j] for j in alignment.target]
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                type_counts[alignment.type] += 1
                if alignment.target and alignment.source:
                    similarities.append(alignment.similarity)
            sink.flush()                       # a crash keeps everything written so far
            os.fsync(sink.fileno()) if index % 200 == 0 else None

            n_docs += 1
            counts = result.counts
            per_page.append({
                "doc": doc_id, "n_source": len(en_paras), "n_target": len(zh_paras),
                "score": round(result.score, 4), "counts": counts,
            })
            if index % 200 == 0 or index == len(pages):
                log.info("  aligned %d/%d pages (%.0f%%), %d moves so far",
                         index, len(pages), 100 * index / len(pages), sum(type_counts.values()))
    finally:
        sink.close()

    elapsed = time.time() - started
    matched = sum(v for k, v in type_counts.items() if k not in ("1:0", "0:1"))
    total = sum(type_counts.values())
    report = {
        "db": args.db, "output": str(out_path),
        "threshold": args.threshold, "gap_penalty": args.gap_penalty,
        "position_weight": args.position_weight,
        "pages_aligned": n_docs, "pages_skipped": n_skipped,
        "documents_embedded_on_the_fly": n_embedded_docs,
        "moves": total, "moves_by_type": dict(sorted(type_counts.items())),
        "matched_share": round(matched / max(total, 1), 4),
        "mean_similarity": round(sum(similarities) / len(similarities), 4) if similarities else None,
        "seconds": round(elapsed, 1),
        "pages": per_page,
    }
    write_json(Path(args.report) if args.report else out_path.with_suffix(".report.json"), report)

    print(f"\n  页面            {n_docs:,} 对齐  ({n_skipped:,} 跳过)")
    print(f"  对齐移动        {total:,}")
    for name in ("1:1", "1:2", "2:1", "2:2", "1:0", "0:1"):
        count = type_counts.get(name, 0)
        print(f"    {name:>3}           {count:>8,}  {100*count/max(total,1):5.1f}%")
    print(f"  匹配占比        {report['matched_share']:.1%}")
    print(f"  平均相似度      {report['mean_similarity']}")
    print(f"  耗时            {elapsed:.1f}s")
    print(f"  输出            {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

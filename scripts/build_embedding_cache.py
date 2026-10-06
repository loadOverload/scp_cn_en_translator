#!/usr/bin/env python
"""Warm the paragraph-embedding cache for the whole corpus.

    ./py scripts/build_embedding_cache.py                 # fill the cache
    ./py scripts/build_embedding_cache.py --limit 200     # try it out
    ./py scripts/build_embedding_cache.py --verify        # re-check what is cached

Writing the cache once costs one embedding pass (~6 min for 7,461 pages) and then
every later run -- changing the DP, the band width, the window budget -- reads
vectors from disk instead of recomputing them. Documents already cached are
skipped, so this is incremental and safe to re-run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.align.embedding_cache import (                       # noqa: E402
    DEFAULT_CACHE_DIR, embed_documents,
)
from src.align.paragraph_align import DEFAULT_BATCH_SIZE, DEFAULT_MAX_LENGTH, DEFAULT_MODEL  # noqa: E402
from src.utils.config import add_common_args, ensure_dirs, load_config   # noqa: E402
from src.utils.io import write_json                            # noqa: E402
from src.utils.logging_utils import setup_logging              # noqa: E402
from scripts.build_conversion_dataset import load_pages, split_paragraphs  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Warm the paragraph embedding cache")
    add_common_args(parser)
    parser.add_argument("--db", default="data/raw/crawl.db")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    ensure_dirs(cfg)
    log = setup_logging("INFO", Path(cfg["paths"]["log_dir"]) / "embedding_cache.log",
                        name="scp.cache")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("cache directory: %s", cache_dir.resolve())

    pages = load_pages(args.db, args.limit)
    log.info("loaded %d page pairs", len(pages))
    items = [{"id": page["id"],
              "en": split_paragraphs(page["en"]),
              "zh": split_paragraphs(page["zh"])} for page in pages]

    from src.align.paragraph_align import ParagraphEmbedder
    started = time.time()
    results, stats = embed_documents(
        items,
        lambda: ParagraphEmbedder(model_name=args.model, device=args.device,
                                  batch_size=args.batch_size, max_length=args.max_length,
                                  use_fp16=True, logger=log),
        cache_dir=cache_dir, model=args.model, logger=log,
    )
    elapsed = time.time() - started

    size_bytes = sum(p.stat().st_size for p in cache_dir.glob("*.npz"))
    report = {
        "db": args.db, "cache_dir": str(cache_dir), "model": args.model,
        "documents": stats["documents"], "reused": stats["hits"],
        "embedded": stats["misses"],
        "cached_paragraphs": stats["cached_paragraphs"],
        "embedded_paragraphs": stats["embedded_paragraphs"],
        "seconds": round(elapsed, 1),
        "files": len(list(cache_dir.glob("*.npz"))),
        "size_gb": round(size_bytes / 2**30, 2),
    }
    write_json(Path(args.report) if args.report else cache_dir / "report.json", report)

    print(f"\n  文档            {report['documents']:,}")
    print(f"  复用缓存        {report['reused']:,}")
    print(f"  本次嵌入        {report['embedded']:,}  ({report['embedded_paragraphs']:,} 段)")
    print(f"  缓存段落        {report['cached_paragraphs']:,}")
    print(f"  耗时            {report['seconds']}s")
    print(f"  缓存文件        {report['files']:,} 个, {report['size_gb']} GB")
    print(f"  目录            {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

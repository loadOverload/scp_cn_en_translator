"""Disk cache for paragraph embeddings.

Embedding the corpus is the fixed cost of every run: 774,547 paragraphs take
about 383 s on a 4090, and that is paid again on every invocation no matter how
small the change being tested. Nothing was persisted before, so five runs of the
pipeline meant five identical recomputations.

Layout -- one file per document, so a page can be re-embedded on its own::

    data/embeddings/scp-003.npz
        en       (n_en, 1024) float32   paragraph vectors, in paragraph order
        zh       (n_zh, 1024) float32
        en_hash  sha1 over the English paragraphs
        zh_hash  sha1 over the Chinese paragraphs
        model    the encoder that produced them

Invalidation is by content, not by timestamp: on reuse the paragraphs are split
again and their hashes compared. Change the splitting, the model, or the text and
the document is recomputed; otherwise the vectors are read straight from disk.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_CACHE_DIR = "data/embeddings"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(doc_id: str) -> str:
    return _SAFE_NAME.sub("_", str(doc_id)).strip("_") or "doc"


def side_hash(paragraphs: Sequence[str]) -> str:
    """Content hash of one side's paragraph sequence."""
    digest = hashlib.sha1()
    for paragraph in paragraphs:
        digest.update(paragraph.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def cache_file(cache_dir, doc_id: str) -> Path:
    return Path(cache_dir) / f"{_safe_name(doc_id)}.npz"


def load(
    cache_dir,
    doc_id: str,
    en_paragraphs: Sequence[str],
    zh_paragraphs: Sequence[str],
    model: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(en_vectors, zh_vectors)`` if the cache still matches, else None."""
    path = cache_file(cache_dir, doc_id)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model"]) != str(model):
                return None
            if str(data["en_hash"]) != side_hash(en_paragraphs):
                return None
            if str(data["zh_hash"]) != side_hash(zh_paragraphs):
                return None
            return data["en"].copy(), data["zh"].copy()
    except (OSError, ValueError, KeyError):
        # a truncated or foreign file must not stop a run: treat it as a miss
        return None


def save(
    cache_dir,
    doc_id: str,
    en_paragraphs: Sequence[str],
    zh_paragraphs: Sequence[str],
    en_vectors: np.ndarray,
    zh_vectors: np.ndarray,
    model: str,
) -> Path:
    path = cache_file(cache_dir, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # write to a temporary name first so an interrupted run cannot leave a
    # half-written file that a later run would read as valid
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as sink:
        np.savez(
            sink,
            en=np.ascontiguousarray(en_vectors, dtype=np.float32),
            zh=np.ascontiguousarray(zh_vectors, dtype=np.float32),
            en_hash=side_hash(en_paragraphs),
            zh_hash=side_hash(zh_paragraphs),
            model=str(model),
        )
    temporary.replace(path)
    return path


def embed_documents(
    items: Sequence[Dict[str, Any]],
    embedder_factory,
    cache_dir=DEFAULT_CACHE_DIR,
    model: str = "",
    use_cache: bool = True,
    logger=None,
    progress_every: int = 50000,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    """Embed every document, reusing the cache.

    ``items`` are dicts with ``id``, ``en`` (list of paragraphs) and ``zh``. Only
    cache misses are sent to the GPU, in a single batched stream, so a run that
    changes alignment parameters does no embedding at all. ``embedder_factory`` is
    called only if there is at least one miss, so a fully cached run does not even
    load the model.

    Returns ``({doc_id: (en_vectors, zh_vectors)}, stats)``.
    """
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    misses: List[Dict[str, Any]] = []
    for item in items:
        if not item["en"] or not item["zh"]:
            continue
        if use_cache:
            cached = load(cache_dir, item["id"], item["en"], item["zh"], model)
            if cached is not None:
                results[item["id"]] = cached
                continue
        misses.append(item)

    stats = {"documents": len(results) + len(misses), "hits": len(results),
             "misses": len(misses), "cached_paragraphs": 0, "embedded_paragraphs": 0}
    if not misses:
        if logger:
            logger.info("embedding cache: %d/%d documents reused, nothing to embed",
                        stats["hits"], stats["documents"])
        return results, stats

    texts: List[str] = []
    spans: List[Tuple[Dict[str, Any], int, int, int]] = []
    for item in misses:
        start = len(texts)
        texts.extend(item["en"])
        texts.extend(item["zh"])
        spans.append((item, start, len(item["en"]), len(item["zh"])))
    stats["embedded_paragraphs"] = len(texts)
    stats["cached_paragraphs"] = sum(
        len(r[0]) + len(r[1]) for r in results.values())

    if logger:
        logger.info("embedding %d paragraphs for %d documents (%d reused from cache)",
                    len(texts), len(misses), stats["hits"])
    vectors = embedder_factory().encode(texts, progress_every=progress_every)

    for item, start, n_en, n_zh in spans:
        en_vectors = vectors[start:start + n_en]
        zh_vectors = vectors[start + n_en:start + n_en + n_zh]
        results[item["id"]] = (en_vectors, zh_vectors)
        if use_cache:
            save(cache_dir, item["id"], item["en"], item["zh"],
                 en_vectors, zh_vectors, model)
    return results, stats


__all__ = ["DEFAULT_CACHE_DIR", "side_hash", "cache_file", "load", "save",
           "embed_documents"]

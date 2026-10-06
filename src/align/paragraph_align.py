"""Monotonic paragraph alignment between a Chinese and an English SCP page.

The two sides of an SCP page are translations of each other, so their paragraphs
appear in the **same order** even when the two languages split sentences and
merge paragraphs differently. That ordering constraint is what makes alignment
tractable: instead of searching arbitrary pairings, a monotonic dynamic program
finds the best order-preserving matching.

Pipeline
--------
1. split each side into paragraphs at blank lines
2. embed every paragraph with ``BAAI/bge-m3`` (CUDA, fp16, length-grouped batches)
3. cosine similarity between every English/Chinese paragraph pair
4. monotonic DP over the similarity matrix, with a gap penalty for skipped
   paragraphs (translators do merge, split and drop paragraphs)

The DP is the classic Needleman-Wunsch recurrence::

    dp[i][j] = max(
        dp[i-1][j-1] + sim[i-1][j-1],   # pair them
        dp[i-1][j]   - gap,             # English paragraph has no counterpart
        dp[i][j-1]   - gap,             # Chinese paragraph has no counterpart
    )

so every paragraph is either matched once or left unpaired, and the matches are
strictly increasing on both sides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"
# bge-m3 only ships pytorch_model.bin, which transformers 5.x refuses to load with
# torch < 2.6 (CVE-2025-32434). scripts/setup_bge_m3.py converts it to safetensors
# once; prefer that local directory when it exists.
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-m3"
# paragraphs are short; 512 tokens covers every one of them and keeps the batches
# cheap on a 4090 (bge-m3's own limit is 8192, which we do not need here)
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 128
DEFAULT_GAP_PENALTY = 0.25
# how strongly a pairing is pushed towards the proportional diagonal; small
# enough that content still decides, large enough to break ties among
# near-identical boilerplate paragraphs
DEFAULT_POSITION_WEIGHT = 0.15
# Half-width of the diagonal band, in paragraphs. Corresponding paragraphs of a
# translation sit at nearly the same relative position, so the alignment never
# needs the full n*m table: a band of this many cells either side of the
# proportional diagonal contains the optimum, and everything outside is
# unreachable. 32 tolerates a shift of 32 paragraphs in either direction, which
# is far more than real merges and omissions produce.
DEFAULT_BAND_WIDTH = 32

_BLANK_LINE = re.compile(r"\n[ \t]*\n[ \t\n]*")


# ---------------------------------------------------------------------------
# paragraph splitting
# ---------------------------------------------------------------------------


def split_paragraphs(text: str, min_chars: int = 1) -> List[str]:
    """Split on blank lines.

    Paragraph text is returned stripped, with internal newlines preserved, so a
    paragraph keeps its own line structure (lists, block quotes) without the
    leading/trailing whitespace that belongs to the layout between paragraphs.
    """
    if not text:
        return []
    parts = _BLANK_LINE.split(text)
    out: List[str] = []
    for part in parts:
        stripped = part.strip()
        if len(stripped) >= min_chars:
            out.append(stripped)
    return out


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------


class ParagraphEmbedder:
    """``BAAI/bge-m3`` paragraph encoder, tuned for a single 4090.

    * fp16 by default (bf16 is also fine; fp16 is what the BGE checkpoints ship)
    * batches are sorted by token length so padding wastes as little compute as
      possible, then results are restored to the caller's order
    * ``torch.inference_mode`` plus a single move to the GPU per batch
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        use_fp16: bool = True,
        logger=None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        if model_name == DEFAULT_MODEL and (LOCAL_MODEL_DIR / "model.safetensors").exists():
            model_name = str(LOCAL_MODEL_DIR)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.use_fp16 = bool(use_fp16) and self.device.startswith("cuda")
        self.logger = logger

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        if self.use_fp16:
            self.model.half()

    # -- one batch ---------------------------------------------------------
    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded)
        # BGE family convention: the [CLS] state is the sentence embedding
        cls = output.last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls.float(), p=2, dim=1)
        return cls.cpu().numpy()

    # -- many paragraphs ---------------------------------------------------
    def encode(self, texts: Sequence[str], progress_every: int = 0) -> np.ndarray:
        """Embed every text; the output rows follow the input order."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        # Tokenise in batches to measure lengths. Doing it one paragraph at a
        # time was the real cost of a full-corpus run: 774k separate tokenizer
        # calls, single-threaded, and the whole GPU sat idle while they ran.
        lengths = np.empty(len(texts), dtype=np.int32)
        measure_step = 4096
        for start in range(0, len(texts), measure_step):
            batch = texts[start:start + measure_step]
            for offset, ids in enumerate(
                self.tokenizer(batch, add_special_tokens=True)["input_ids"]
            ):
                lengths[start + offset] = len(ids)
        order = np.argsort(lengths, kind="stable")

        chunks: List[np.ndarray] = []
        done = 0
        for start in range(0, len(order), self.batch_size):
            index = order[start:start + self.batch_size]
            chunks.append(self._encode_batch([texts[i] for i in index]))
            done += len(index)
            if progress_every and self.logger and done % progress_every < self.batch_size:
                self.logger.info("    embedded %d/%d paragraphs", done, len(order))

        # undo the length sort: every chunk holds the batch that sat at
        # order[cursor:cursor+size], so scatter it back to those positions
        result = np.empty((len(texts), chunks[0].shape[1]), dtype=np.float32)
        cursor = 0
        for chunk in chunks:
            size = chunk.shape[0]
            result[order[cursor:cursor + size]] = chunk
            cursor += size
        return result


# ---------------------------------------------------------------------------
# similarity and alignment
# ---------------------------------------------------------------------------


def similarity_matrix(en_vectors: np.ndarray, zh_vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between every English and Chinese paragraph.

    The vectors are already L2-normalised, so this is a plain dot product.
    """
    if en_vectors.size == 0 or zh_vectors.size == 0:
        return np.zeros((len(en_vectors), len(zh_vectors)), dtype=np.float32)
    return (en_vectors @ zh_vectors.T).astype(np.float32)


def band_bounds(n_en: int, n_zh: int, width: int) -> List[Tuple[int, int]]:
    """1-based inclusive column range to compute for each English row.

    Row ``i`` is centred on the proportional position ``i * n_zh / n_en``: that is
    where the matching Chinese paragraph sits when the two sides are translations
    of the same document with the same ordering.
    """
    bounds: List[Tuple[int, int]] = []
    for i in range(1, n_en + 1):
        center = i * n_zh / n_en
        lo = max(1, int(center - width))
        hi = min(n_zh, int(center + width) + 1)
        bounds.append((lo, hi))
    return bounds


def align_monotonic(
    similarity: np.ndarray,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
    position_weight: float = DEFAULT_POSITION_WEIGHT,
    band_width: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Best order-preserving matching between the rows and columns.

    The recurrence is the classic Needleman-Wunsch one::

        dp[i][j] = max(dp[i-1][j-1] + sim[i-1][j-1],   # pair them
                       dp[i-1][j]   - gap,             # English left unmatched
                       dp[i][j-1]   - gap)             # Chinese left unmatched

    Written as a literal double loop it is O(n*m) *Python* iterations, which does
    not survive real pages: the largest SCP page has 1,861 paragraphs per side,
    i.e. 3.5M cells for one page, times 7,461 pages. Each row is therefore
    computed with vector operations.

    The ``dp[i][j-1] - gap`` term is a chain along the row, so it cannot be
    applied with a plain maximum. Substituting ``g[j] = dp[i][j] + gap*j`` turns
    the chain into a prefix maximum::

        g[j] = max(base[j] + gap*j, g[j-1]) = cumulative max over k <= j

    which ``np.maximum.accumulate`` computes in one pass. The traceback then walks
    the table backwards and re-tests the recurrence, so no backpointer array is
    needed and the result is identical to the literal version.
    """
    n_en, n_zh = similarity.shape
    if n_en == 0 or n_zh == 0:
        return [], {
            "score": 0.0,
            "unmatched_en": list(range(n_en)),
            "unmatched_zh": list(range(n_zh)),
        }

    gap = float(gap_penalty)

    # Positional prior. The two sides are translations of the same document, so
    # corresponding paragraphs sit at (roughly) the same *relative* position:
    # paragraph i of n_en lines up near paragraph j of n_zh with i/n_en ~ j/n_zh.
    # Similarity alone cannot express that, and it matters exactly where the
    # corpus is repetitive -- repeated notice boxes, per-instance headers, the
    # same boilerplate in several sections -- because there two Chinese
    # paragraphs look equally like one English paragraph and the choice is
    # decided by position. The deviation is subtracted from the match score, so
    # a pairing far off the diagonal has to be clearly better on content to win.
    # The positional prior is applied per band slice rather than up front. Doing it
    # globally materialised two full n*m float64 arrays (2 x 28 MB for the largest
    # page) and touched every cell, which cost more than the DP itself.
    weight = float(position_weight)
    en_position = ((np.arange(1, n_en + 1, dtype=np.float64)) / n_en) if weight else None
    zh_position = ((np.arange(1, n_zh + 1, dtype=np.float64)) / n_zh) if weight else None

    def score_slice(i: int, lo: int, hi: int) -> np.ndarray:
        """Match scores for row ``i`` (1-based), columns ``lo..hi`` (1-based)."""
        out = similarity[i - 1, lo - 1:hi].astype(np.float64)
        if weight:
            out -= weight * np.abs(en_position[i - 1] - zh_position[lo - 1:hi])
        return out

    def score_cell(i: int, j: int) -> float:
        value = float(similarity[i - 1, j - 1])
        if weight:
            value -= weight * abs(en_position[i - 1] - zh_position[j - 1])
        return value

    # With a band, only the strip around the proportional diagonal is computed and
    # everything else stays -inf, i.e. unreachable. The recurrence is unchanged,
    # so a band wide enough to contain the unbanded optimum returns exactly the
    # unbanded answer while touching n*w cells instead of n*m.
    bounds = band_bounds(n_en, n_zh, int(band_width)) if band_width else None

    neg_inf = -np.inf
    dp = np.full((n_en + 1, n_zh + 1), neg_inf, dtype=np.float64)
    dp[0, 0] = 0.0
    if n_en:
        dp[1:, 0] = -gap * np.arange(1, n_en + 1, dtype=np.float64)
    if n_zh:
        dp[0, 1:] = -gap * np.arange(1, n_zh + 1, dtype=np.float64)

    for i in range(1, n_en + 1):
        if bounds is None:
            lo, hi = 1, n_zh
        else:
            lo, hi = bounds[i - 1]
            if lo > hi:
                continue
        # columns j in [lo, hi]; match needs j-1, so the score slice starts at lo-1
        base = np.maximum(dp[i - 1, lo - 1:hi] + score_slice(i, lo, hi),
                          dp[i - 1, lo:hi + 1] - gap)
        column_index = np.arange(lo, hi + 1, dtype=np.float64)
        chain = base + gap * column_index
        np.maximum.accumulate(chain, out=chain)
        # dp[i][lo-1] is outside the band: -inf, so the chain restarts at lo
        dp[i, lo:hi + 1] = chain - gap * column_index

    # ---- traceback: re-test the recurrence instead of storing pointers ----
    tolerance = 1e-9
    pairs: List[Tuple[int, int]] = []
    i, j = n_en, n_zh
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(dp[i, j] - (dp[i - 1, j - 1] + score_cell(i, j))) <= tolerance:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(dp[i, j] - (dp[i - 1, j] - gap)) <= tolerance:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    matched_en = {a for a, _ in pairs}
    matched_zh = {b for _, b in pairs}
    return pairs, {
        "score": float(dp[n_en, n_zh]),
        "band_width": band_width,
        "cells": int(sum(hi - lo + 1 for lo, hi in bounds)) if bounds else n_en * n_zh,
        "matched_similarity": (
            float(np.mean([similarity[a, b] for a, b in pairs])) if pairs else 0.0
        ),
        "unmatched_en": [i for i in range(n_en) if i not in matched_en],
        "unmatched_zh": [j for j in range(n_zh) if j not in matched_zh],
    }


@dataclass
class AlignedPair:
    en_index: int
    zh_index: int
    en: str
    zh: str
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "en_index": self.en_index, "zh_index": self.zh_index,
            "en": self.en, "zh": self.zh, "similarity": round(self.similarity, 4),
        }


@dataclass
class DocumentAlignment:
    doc_id: str
    en_paragraphs: List[str] = field(default_factory=list)
    zh_paragraphs: List[str] = field(default_factory=list)
    pairs: List[AlignedPair] = field(default_factory=list)
    unmatched_en: List[int] = field(default_factory=list)
    unmatched_zh: List[int] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.doc_id,
            "n_en": len(self.en_paragraphs),
            "n_zh": len(self.zh_paragraphs),
            "n_pairs": len(self.pairs),
            "n_unmatched_en": len(self.unmatched_en),
            "n_unmatched_zh": len(self.unmatched_zh),
            "score": round(self.score, 4),
            "pairs": [p.to_dict() for p in self.pairs],
            "unmatched_en": [self.en_paragraphs[i] for i in self.unmatched_en],
            "unmatched_zh": [self.zh_paragraphs[i] for i in self.unmatched_zh],
        }


__all__ = [
    "DEFAULT_MODEL", "DEFAULT_MAX_LENGTH", "DEFAULT_BATCH_SIZE", "DEFAULT_GAP_PENALTY",
    "DEFAULT_POSITION_WEIGHT", "DEFAULT_BAND_WIDTH", "band_bounds",
    "split_paragraphs", "ParagraphEmbedder", "similarity_matrix", "align_monotonic",
    "AlignedPair", "DocumentAlignment",
]

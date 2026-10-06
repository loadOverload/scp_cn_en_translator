"""Monotonic paragraph alignment that allows merged spans.

Two aligned sides of an SCP page rarely pair one paragraph to one paragraph: a
translator merges two short English paragraphs into one Chinese one, splits a long
one, drops a ``[[module Rate]]`` line, or adds a translator's note. This aligner
models that directly, with six moves:

===================  =========================================================
``1:1``              one English paragraph to one Chinese paragraph
``1:2``              one English paragraph to two consecutive Chinese ones
``2:1``              two consecutive English paragraphs to one Chinese one
``2:2``              two consecutive English paragraphs to two Chinese ones
``1:0``              English paragraph with no Chinese counterpart (dropped)
``0:1``              Chinese paragraph with no English counterpart (added)
===================  =========================================================

Merged spans are compared by **mean embedding**: the vectors of the paragraphs in
the span are averaged and the mean is L2-normalised, then cosine similarity is
taken. That is why a merge is not free -- averaging two unlike paragraphs pulls
the mean away from either of them, so a merge only wins when the two English
paragraphs really do correspond to the two Chinese ones.

Order is preserved by construction: the alignment is a monotonic path through the
table, so a paragraph can never match something earlier than a paragraph already
matched.

Low similarity does not force a match. Any pair whose cosine falls below
``min_similarity`` is made unreachable, so the program is obliged to emit ``1:0``
or ``0:1`` (a deletion or an addition) instead of pairing unrelated text.

Implementation notes
--------------------
The four match scores are matmuls. With ``E2`` / ``Z2`` the normalised means of
consecutive pairs::

    S11 = E  @ Z.T      S12 = E  @ Z2.T
    S21 = E2 @ Z.T      S22 = E2 @ Z2.T

and the dynamic program is evaluated one row at a time with NumPy, so a page with
1,861 paragraphs per side takes milliseconds rather than the minutes a literal
Python double loop needs. A small positional prior subtracts the distance from the
proportional diagonal, which stops repetitive boilerplate from matching across the
page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Cosine below which a pair may not be matched at all, so the program is forced
# to emit a deletion/addition instead of pairing unrelated text.
#
# Calibrated against synthetic cases with a known answer, because the value is
# what decides whether a contaminated merge is allowed:
#     genuine 1:1 with bge-m3           ~0.98
#     genuine 2:1 / 1:2 (mean diluted)  ~0.94
#     a merge polluted by an unrelated paragraph ~0.67
#     0.55 lets the polluted merge through (2:1 with an unrelated English
#     paragraph swallowed into it); 0.70 rejects it and the paragraph is
#     correctly emitted as 1:0 / 0:1. On real data the matched pairs sit at
#     mean 0.87 / p10 0.79, so 0.70 only reclassifies the bottom couple of
#     percent as unmatched -- exactly the ones that were dubious anyway.
DEFAULT_MIN_SIMILARITY = 0.70
# Cost of leaving one paragraph unmatched on either side.
DEFAULT_GAP_PENALTY = 0.30
# Weight of the proportional-position prior.
DEFAULT_POSITION_WEIGHT = 0.10
# The largest span either side may merge (1:2, 2:1, 2:2 all use 2).
MAX_SPAN = 2

# move codes, stored as int8 backpointers
M11, M12, M21, M22, M10, M01 = 0, 1, 2, 3, 4, 5
MOVE_TYPES = {M11: "1:1", M12: "1:2", M21: "2:1", M22: "2:2", M10: "1:0", M01: "0:1"}
# how many paragraphs each move consumes on each side
MOVE_COST = {M11: (1, 1), M12: (1, 2), M21: (2, 1), M22: (2, 2), M10: (1, 0), M01: (0, 1)}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (a no-op for vectors straight from the cache)."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


def span_means(vectors: np.ndarray, span: int) -> np.ndarray:
    """Normalised means of ``span`` consecutive paragraphs.

    Row ``k`` of the result is the normalised mean of ``vectors[k:k+span]``, i.e.
    the embedding of a merged span starting at ``k``.
    """
    if span <= 1 or len(vectors) < span:
        return l2_normalize(vectors) if span <= 1 else np.zeros((0, vectors.shape[1]), np.float32)
    total = np.zeros((len(vectors) - span + 1, vectors.shape[1]), dtype=np.float32)
    for offset in range(span):
        total += vectors[offset:offset + len(total)]
    return l2_normalize(total / span)


@dataclass
class Alignment:
    """One move of the alignment path."""

    type: str
    source: List[int]
    target: List[int]
    similarity: float

    def to_dict(self, doc_id: str, source_ids: Sequence[str],
                target_ids: Sequence[str]) -> Dict[str, Any]:
        return {
            "doc": doc_id,
            "type": self.type,
            "source_ids": [source_ids[i] for i in self.source],
            "target_ids": [target_ids[j] for j in self.target],
            "similarity": round(float(self.similarity), 4),
        }


@dataclass
class PageAlignment:
    doc_id: str
    source_ids: List[str] = field(default_factory=list)
    target_ids: List[str] = field(default_factory=list)
    alignments: List[Alignment] = field(default_factory=list)
    score: float = 0.0

    @property
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for alignment in self.alignments:
            out[alignment.type] = out.get(alignment.type, 0) + 1
        return out

    def to_rows(self) -> List[Dict[str, Any]]:
        return [a.to_dict(self.doc_id, self.source_ids, self.target_ids)
                for a in self.alignments]


def emission_scores(
    en_vectors: np.ndarray,
    zh_vectors: np.ndarray,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> Dict[str, np.ndarray]:
    """Cosine similarity for each move, with unacceptably low pairs masked out.

    ``S12[i, k]`` is "English ``i`` against Chinese ``k`` and ``k+1``", ``S21`` is
    the mirror, ``S22`` covers two and two.
    """
    en = l2_normalize(np.asarray(en_vectors, dtype=np.float32))
    zh = l2_normalize(np.asarray(zh_vectors, dtype=np.float32))
    en_pair = span_means(en, 2)
    zh_pair = span_means(zh, 2)

    scores = {
        "s11": en @ zh.T,
        "s12": en @ zh_pair.T if len(zh_pair) else np.zeros((len(en), 0), np.float32),
        "s21": en_pair @ zh.T if len(en_pair) else np.zeros((0, len(zh)), np.float32),
        "s22": (en_pair @ zh_pair.T if len(en_pair) and len(zh_pair)
                else np.zeros((len(en_pair), len(zh_pair)), np.float32)),
    }
    # A pair below the threshold is not a candidate at all, which is what forces
    # the program to choose a deletion or an addition instead of a bad match.
    for name, matrix in scores.items():
        matrix = np.asarray(matrix, dtype=np.float32)
        matrix[matrix < float(min_similarity)] = -np.inf
        scores[name] = matrix
    return scores


def align_vectors(
    en_vectors: np.ndarray,
    zh_vectors: np.ndarray,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
    position_weight: float = DEFAULT_POSITION_WEIGHT,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> Tuple[List[Alignment], Dict[str, Any]]:
    """Align two paragraph sequences; returns the moves and a small info dict."""
    n_en = int(len(en_vectors))
    n_zh = int(len(zh_vectors))
    if n_en == 0 and n_zh == 0:
        return [], {"score": 0.0}
    if n_en == 0 or n_zh == 0:
        # one side empty: everything on the other side is an addition/deletion
        moves = [Alignment("0:1", [], [j], 0.0) for j in range(n_zh)] if n_en == 0 \
            else [Alignment("1:0", [i], [], 0.0) for i in range(n_en)]
        return moves, {"score": 0.0}

    scores = emission_scores(en_vectors, zh_vectors, min_similarity)
    s11, s12, s21, s22 = scores["s11"], scores["s12"], scores["s21"], scores["s22"]
    gap = float(gap_penalty)
    weight = float(position_weight)

    neg = -np.inf
    dp = np.full((n_en + 1, n_zh + 1), neg, dtype=np.float64)
    back = np.full((n_en + 1, n_zh + 1), -1, dtype=np.int8)
    dp[0, 0] = 0.0
    for i in range(1, n_en + 1):
        dp[i, 0] = dp[i - 1, 0] - gap
        back[i, 0] = M10
    for j in range(1, n_zh + 1):
        dp[0, j] = dp[0, j - 1] - gap
        back[0, j] = M01

    en_pos = np.arange(1, n_en + 1, dtype=np.float64) / n_en
    zh_pos = np.arange(1, n_zh + 1, dtype=np.float64) / n_zh
    columns = np.arange(1, n_zh + 1)

    for i in range(1, n_en + 1):
        # the six candidates, as vectors over the Chinese index j = 1..n_zh
        candidates = np.full((6, n_zh), neg, dtype=np.float64)

        # --- 1:1 ---
        prior = np.abs(en_pos[i - 1] - zh_pos)
        candidates[M11] = dp[i - 1, 0:n_zh] + s11[i - 1] - weight * prior

        # --- 1:2 (Chinese j-1, j) ---
        if n_zh >= 2:
            prior = np.abs(en_pos[i - 1] - (columns[1:] - 0.5) / n_zh)
            candidates[M12, 1:] = (dp[i - 1, 0:n_zh - 1] + s12[i - 1]
                                   - weight * prior)

        # --- 2:1 (English i-1, i) ---
        if i >= 2:
            prior = np.abs((en_pos[i - 1] - 0.5 / n_en) - zh_pos)
            candidates[M21] = dp[i - 2, 0:n_zh] + s21[i - 2] - weight * prior

            # --- 2:2 (English i-1, i against Chinese j-1, j) ---
            if n_zh >= 2:
                prior = np.abs((en_pos[i - 1] - 0.5 / n_en) - (columns[1:] - 0.5) / n_zh)
                candidates[M22, 1:] = (dp[i - 2, 0:n_zh - 1] + s22[i - 2]
                                       - weight * prior)

        # --- 1:0 (English i unmatched) ---
        candidates[M10] = dp[i - 1, 1:n_zh + 1] - gap

        best_move = np.argmax(candidates, axis=0)
        base = candidates[best_move, np.arange(n_zh)]

        # --- 0:1 is a chain along the row: dp[i][j] = max(base[j], dp[i][j-1]-gap)
        # substituting h[j] = dp[i][j] + gap*j turns it into a prefix maximum
        chain = np.concatenate(([dp[i, 0]], base + gap * columns))
        np.maximum.accumulate(chain, out=chain)
        row = chain[1:] - gap * columns
        dp[i, 1:] = row

        # backpointer: the chain won wherever it beat the base value
        inserted = np.abs(row - base) > 1e-12
        back[i, 1:] = np.where(inserted, M01, best_move).astype(np.int8)

    # ---- traceback -------------------------------------------------------
    alignments: List[Alignment] = []
    i, j = n_en, n_zh
    while i > 0 or j > 0:
        move = int(back[i, j])
        if move < 0:
            move = M10 if i > 0 else M01
        take_en, take_zh = MOVE_COST[move]
        source = list(range(i - take_en, i))
        target = list(range(j - take_zh, j))
        similarity = _pair_similarity(en_vectors, zh_vectors, source, target) if source and target else 0.0
        alignments.append(Alignment(MOVE_TYPES[move], source, target, similarity))
        i -= take_en
        j -= take_zh
    alignments.reverse()

    return alignments, {"score": float(dp[n_en, n_zh])}


def _pair_similarity(
    en_vectors: np.ndarray,
    zh_vectors: np.ndarray,
    source: Sequence[int],
    target: Sequence[int],
) -> float:
    """Cosine similarity of the two spans, by mean embedding."""
    en = l2_normalize(np.asarray(en_vectors, dtype=np.float32))
    zh = l2_normalize(np.asarray(zh_vectors, dtype=np.float32))
    left = l2_normalize(en[list(source)].mean(axis=0, keepdims=True))
    right = l2_normalize(zh[list(target)].mean(axis=0, keepdims=True))
    return float((left @ right.T).ravel()[0])


def align_page(
    doc_id: str,
    en_vectors: np.ndarray,
    zh_vectors: np.ndarray,
    source_ids: Optional[Sequence[str]] = None,
    target_ids: Optional[Sequence[str]] = None,
    **kwargs,
) -> PageAlignment:
    """Convenience wrapper producing a :class:`PageAlignment` with paragraph ids."""
    alignments, info = align_vectors(en_vectors, zh_vectors, **kwargs)
    page = PageAlignment(
        doc_id=doc_id,
        source_ids=list(source_ids) if source_ids is not None
        else [f"{doc_id}#p{i:04d}" for i in range(len(en_vectors))],
        target_ids=list(target_ids) if target_ids is not None
        else [f"{doc_id}#p{j:04d}" for j in range(len(zh_vectors))],
        alignments=alignments,
        score=info["score"],
    )
    return page


__all__ = [
    "DEFAULT_MIN_SIMILARITY", "DEFAULT_GAP_PENALTY", "DEFAULT_POSITION_WEIGHT",
    "MAX_SPAN", "MOVE_TYPES", "Alignment", "PageAlignment", "l2_normalize",
    "span_means", "emission_scores", "align_vectors", "align_page",
]

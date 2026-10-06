"""Evaluation metrics for SCP translation.

Implemented (Phase 6 requirement: never BLEU alone):

1. BLEU                     sacrebleu, ``tokenize='zh'``
2. chrF                     sacrebleu, character n-grams
3. ROUGE-L                  character-tokenised for Chinese
5. Wikidot syntax preservation  via :mod:`src.wikidot.validator`
6. length anomaly detection char ratios hypothesis/reference and source/reference

Metrics are computed both on the **raw** Wikidot text (structure + prose) and on
the **plain** text (structure removed), because a model can score well on one
and badly on the other.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..wikidot.validator import cjk_ratio, plain_text, validate_pair


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def char_tokenize(text: str) -> str:
    """Whitespace-separated characters: makes ROUGE meaningful for Chinese."""
    return " ".join(text)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def bleu(hyps: Sequence[str], refs: Sequence[str], tokenize: str = "zh") -> Dict[str, Any]:
    try:
        import sacrebleu
    except Exception as exc:  # pragma: no cover
        return {"error": f"sacrebleu unavailable: {exc}"}
    if not hyps:
        return {"score": 0.0, "n": 0}
    result = sacrebleu.corpus_bleu(list(hyps), [list(refs)], tokenize=tokenize)
    return {
        "score": round(float(result.score), 4),
        "n": len(hyps),
        "precisions": [round(p, 3) for p in result.precisions],
        "bp": round(float(result.bp), 4),
        "sys_len": int(result.sys_len),
        "ref_len": int(result.ref_len),
    }


def chrf(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, Any]:
    try:
        import sacrebleu
    except Exception as exc:  # pragma: no cover
        return {"error": f"sacrebleu unavailable: {exc}"}
    if not hyps:
        return {"score": 0.0, "n": 0}
    result = sacrebleu.corpus_chrf(list(hyps), [list(refs)], word_order=0)
    return {"score": round(float(result.score), 4), "n": len(hyps)}


def rouge_l(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, Any]:
    try:
        from rouge_score import rouge_scorer
    except Exception as exc:  # pragma: no cover
        return {"error": f"rouge_score unavailable: {exc}"}
    if not hyps:
        return {"score": 0.0, "n": 0}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(char_tokenize(ref), char_tokenize(hyp))["rougeL"].fmeasure
        for hyp, ref in zip(hyps, refs)
    ]
    return {
        "score": round(100 * sum(scores) / max(len(scores), 1), 4),
        "n": len(scores),
    }


# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Wikidot preservation
# ---------------------------------------------------------------------------


def wikidot_preservation(
    sources: Sequence[str],
    hyps: Sequence[str],
    cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not sources:
        return {"score": 0.0, "n": 0}
    check_totals: Dict[str, List[float]] = {}
    scores: List[float] = []
    errors: List[int] = []
    clean = 0
    issue_counter: Counter = Counter()

    for source, hyp in zip(sources, hyps):
        report = validate_pair(source, hyp, cfg)
        scores.append(report.score)
        errors.append(report.error_count())
        if report.error_count() == 0:
            clean += 1
        for name, outcome in report.outcomes.items():
            check_totals.setdefault(name, []).append(outcome.rate)
        for issue in report.issues:
            issue_counter[issue.code] += 1

    return {
        "score": round(sum(scores) / len(scores), 4),
        "perfect_ratio": round(clean / len(scores), 4),
        "errors_total": int(sum(errors)),
        "errors_mean": round(sum(errors) / len(errors), 4),
        "samples_with_errors": sum(1 for e in errors if e),
        "per_check": {name: round(sum(vals) / len(vals), 4) for name, vals in sorted(check_totals.items())},
        "top_issues": dict(issue_counter.most_common(15)),
        "n": len(scores),
    }


# ---------------------------------------------------------------------------
# length anomalies
# ---------------------------------------------------------------------------


def length_analysis(
    sources: Sequence[str],
    refs: Sequence[str],
    hyps: Sequence[str],
    ratio_min: float = 0.3,
    ratio_max: float = 2.0,
) -> Dict[str, Any]:
    """Detect length / language anomalies.

    All decisions are made on the **prose** length (:func:`plain_text`) and on
    the CJK ratio of the prose. Measuring the raw Wikidot source would flag
    perfectly good translations: markup (``[[include]]`` parameters, styles,
    file names) is Latin and can easily dominate a short Chinese page.
    """
    anomalies: List[Dict[str, Any]] = []
    ratios: List[float] = []
    for index, (source, ref, hyp) in enumerate(zip(sources, refs, hyps)):
        hyp_plain = plain_text(hyp)
        ref_plain = plain_text(ref)
        hyp_len, ref_len, src_len = len(hyp), len(ref), len(source)
        hyp_plain_len, ref_plain_len = len(hyp_plain), len(ref_plain)
        ratio = _safe_div(hyp_plain_len, ref_plain_len or hyp_len)
        raw_ratio = _safe_div(hyp_len, ref_len)
        ratios.append(ratio)
        hyp_cjk = cjk_ratio(hyp_plain)

        reasons = []
        if not hyp.strip():
            reasons.append("empty_hypothesis")
        elif ratio < ratio_min:
            reasons.append("too_short")
        elif ratio > ratio_max:
            reasons.append("too_long")
        if hyp_plain_len > 50 and hyp_cjk < 0.15:
            reasons.append("not_chinese")
        if reasons:
            anomalies.append(
                {
                    "index": index,
                    "reasons": reasons,
                    "hyp_chars": hyp_len,
                    "ref_chars": ref_len,
                    "src_chars": src_len,
                    "hyp_plain_chars": hyp_plain_len,
                    "ref_plain_chars": ref_plain_len,
                    "hyp_ref_ratio": round(ratio, 4),
                    "hyp_ref_ratio_raw": round(raw_ratio, 4),
                    "hyp_cjk_ratio": round(hyp_cjk, 4),
                }
            )
    ratios_sorted = sorted(ratios) if ratios else [0.0]

    def pct(q: float) -> float:
        idx = min(len(ratios_sorted) - 1, max(0, int(round(q * (len(ratios_sorted) - 1)))))
        return round(ratios_sorted[idx], 4)

    return {
        "n": len(hyps),
        "anomalies": len(anomalies),
        "anomaly_ratio": round(_safe_div(len(anomalies), len(hyps)), 4),
        "reason_counts": dict(Counter(r for a in anomalies for r in a["reasons"]).most_common()),
        "hyp_ref_ratio_p05": pct(0.05),
        "hyp_ref_ratio_p50": pct(0.50),
        "hyp_ref_ratio_p95": pct(0.95),
        "examples": anomalies[:20],
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    id: str
    source: str
    reference: str
    hypothesis: str
    source_plain: str = ""
    reference_plain: str = ""
    hypothesis_plain: str = ""
    wikidot_score: float = 0.0
    wikidot_errors: int = 0
    hyp_ref_ratio: float = 0.0
    cjk_ratio: float = 0.0
    issues: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_plain_len": len(self.source_plain),
            "reference_plain_len": len(self.reference_plain),
            "hypothesis_plain_len": len(self.hypothesis_plain),
            "wikidot_score": round(self.wikidot_score, 4),
            "wikidot_errors": self.wikidot_errors,
            "hyp_ref_ratio": round(self.hyp_ref_ratio, 4),
            "cjk_ratio": round(self.cjk_ratio, 4),
            "issues": self.issues,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
        }


def evaluate(
    records: Sequence[Mapping[str, Any]],
    cfg: Optional[Mapping[str, Any]] = None,
    logger=None,
) -> Tuple[Dict[str, Any], List[EvalSample]]:
    """``records`` must contain ``id``, ``source``, ``target`` (=reference), ``hypothesis``."""
    cfg = dict(cfg or {})
    metrics_cfg = list(cfg.get("metrics") or ["bleu", "chrf", "rouge_l", "wikidot", "length"])
    wikidot_cfg = cfg.get("wikidot") or {}

    sources = [str(r.get("source") or "") for r in records]
    refs = [str(r.get("target") or r.get("reference") or "") for r in records]
    hyps = [str(r.get("hypothesis") or "") for r in records]
    ids = [str(r.get("id") or f"row-{i}") for i, r in enumerate(records)]

    report: Dict[str, Any] = {"n_samples": len(records), "metrics": {}}
    samples: List[EvalSample] = []

    plain_src = [plain_text(s) for s in sources]
    plain_ref = [plain_text(r) for r in refs]
    plain_hyp = [plain_text(h) for h in hyps]

    if "bleu" in metrics_cfg:
        report["metrics"]["bleu_raw"] = bleu(hyps, refs, tokenize="zh")
        report["metrics"]["bleu_plain"] = bleu(plain_hyp, plain_ref, tokenize="zh")
    if "chrf" in metrics_cfg:
        report["metrics"]["chrf_raw"] = chrf(hyps, refs)
        report["metrics"]["chrf_plain"] = chrf(plain_hyp, plain_ref)
    if "rouge_l" in metrics_cfg:
        report["metrics"]["rouge_l_raw"] = rouge_l(hyps, refs)
        report["metrics"]["rouge_l_plain"] = rouge_l(plain_hyp, plain_ref)
    if "wikidot" in metrics_cfg:
        report["metrics"]["wikidot"] = wikidot_preservation(sources, hyps, wikidot_cfg)
    if "length" in metrics_cfg:
        report["metrics"]["length"] = length_analysis(
            sources, refs, hyps,
            ratio_min=float(cfg.get("hyp_ref_ratio_min", 0.3)),
            ratio_max=float(cfg.get("hyp_ref_ratio_max", 2.0)),
        )

    for index, (pid, source, ref, hyp) in enumerate(zip(ids, sources, refs, hyps)):
        vreport = validate_pair(source, hyp, wikidot_cfg) if "wikidot" in metrics_cfg else None
        sample = EvalSample(
            id=pid,
            source=source,
            reference=ref,
            hypothesis=hyp,
            source_plain=plain_src[index],
            reference_plain=plain_ref[index],
            hypothesis_plain=plain_hyp[index],
            wikidot_score=vreport.score if vreport else 0.0,
            wikidot_errors=vreport.error_count() if vreport else 0,
            hyp_ref_ratio=_safe_div(len(hyp), len(ref)),
            cjk_ratio=cjk_ratio(hyp),
            issues=[i.to_dict() for i in (vreport.issues if vreport else [])][:20],
        )
        samples.append(sample)

    if logger:
        m = report["metrics"]
        logger.info(
            "BLEU(raw)=%s chrF(raw)=%s ROUGE-L(raw)=%s wikidot=%s term_acc=%s",
            m.get("bleu_raw", {}).get("score"),
            m.get("chrf_raw", {}).get("score"),
            m.get("rouge_l_raw", {}).get("score"),
            m.get("wikidot", {}).get("score"),
        )
    return report, samples

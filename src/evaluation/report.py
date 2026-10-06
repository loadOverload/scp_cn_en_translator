"""Human-readable evaluation reports (Markdown + JSON + per-sample JSONL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..utils.io import write_json, write_jsonl


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown(report: Mapping[str, Any], title: str = "SCP translation evaluation") -> str:
    metrics = report.get("metrics") or {}
    lines: List[str] = [f"# {title}", ""]
    lines.append(f"- samples: **{report.get('n_samples', 0)}**")
    if report.get("model"):
        lines.append(f"- model: `{report['model']}`")
    if report.get("adapter"):
        lines.append(f"- adapter: `{report['adapter']}`")
    if report.get("generated_at"):
        lines.append(f"- generated: {report['generated_at']}")
    lines.append("")

    def section(name: str, payload: Mapping[str, Any]) -> None:
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key, value in payload.items():
            if isinstance(value, dict):
                lines.append(f"| {key} | `{ {k: v for k, v in list(value.items())[:6]} }` |")
            elif isinstance(value, list):
                lines.append(f"| {key} | `{value[:6]}` |")
            else:
                lines.append(f"| {key} | {_fmt(value)} |")
        lines.append("")

    if "bleu_raw" in metrics:
        section("BLEU", {"raw (Wikidot source)": metrics["bleu_raw"], "plain text": metrics.get("bleu_plain", {})})
    if "chrf_raw" in metrics:
        section("chrF", {"raw": metrics["chrf_raw"], "plain": metrics.get("chrf_plain", {})})
    if "rouge_l_raw" in metrics:
        section("ROUGE-L", {"raw": metrics["rouge_l_raw"], "plain": metrics.get("rouge_l_plain", {})})
    if "wikidot" in metrics:
        wiki = dict(metrics["wikidot"])
        per_check = wiki.pop("per_check", None)
        top_issues = wiki.pop("top_issues", None)
        section("Wikidot syntax preservation", wiki)
        if per_check:
            lines.append("### Per-check pass rate")
            lines.append("")
            lines.append("| check | pass rate |")
            lines.append("| --- | --- |")
            for name, rate in per_check.items():
                lines.append(f"| {name} | {rate:.4f} |")
            lines.append("")
        if top_issues:
            lines.append("### Most frequent structural issues")
            lines.append("")
            for code, count in list(top_issues.items())[:15]:
                lines.append(f"- `{code}`: {count}")
            lines.append("")
    if "length" in metrics:
        length = dict(metrics["length"])
        examples = length.pop("examples", None)
        section("Length anomalies", length)
        if examples:
            lines.append("### Examples")
            lines.append("")
            lines.append("| # | reasons | hyp chars | ref chars | ratio |")
            lines.append("| --- | --- | --- | --- | --- |")
            for ex in examples[:10]:
                lines.append(
                    f"| {ex['index']} | {','.join(ex['reasons'])} | {ex['hyp_chars']} | {ex['ref_chars']} | {ex['hyp_ref_ratio']:.3f} |"
                )
            lines.append("")

    worst = report.get("worst_samples") or []
    if worst:
        lines.append("## Worst samples by Wikidot score")
        lines.append("")
        lines.append("| id | score | errors | hyp/ref |")
        lines.append("| --- | --- | --- | --- |")
        for row in worst:
            lines.append(f"| {row['id']} | {row['wikidot_score']:.3f} | {row['wikidot_errors']} | {row['hyp_ref_ratio']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def write_reports(
    report: Dict[str, Any],
    samples: Sequence[Any],
    output_dir: str | Path,
    prefix: str = "eval",
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # surface the worst samples in the JSON/Markdown report
    ranked = sorted(samples, key=lambda s: (s.wikidot_score, s.hyp_ref_ratio))
    report["worst_samples"] = [s.to_dict() for s in ranked[:20]]

    json_path = out / f"{prefix}_report.json"
    md_path = out / f"{prefix}_report.md"
    samples_path = out / f"{prefix}_per_sample.jsonl"

    write_json(json_path, report)
    md_path.write_text(build_markdown(report), encoding="utf-8")
    write_jsonl(samples_path, (s.to_dict() for s in samples))
    return {"report_json": str(json_path), "report_md": str(md_path), "per_sample": str(samples_path)}

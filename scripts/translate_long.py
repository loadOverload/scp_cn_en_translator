#!/usr/bin/env python
"""Translate an English Wikidot document of any length.

    ./py scripts/translate_long.py input.wikidot \
        --adapter outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter \
        -o output.wikidot

``scripts/translate.py`` truncates at ``--max-input-tokens`` (default 6144) and
sets a flag, so a document longer than the model's context silently loses its
tail. This driver windows instead, using the *same* paragraph splitter and the
*same* greedy packer that built the training data, so a long document is handled
exactly the way the model was trained to see one:

    split_paragraphs()   blank-line paragraphs          (src/align/paragraph_align.py)
    pack_windows()       greedy fill to a token budget  (scripts/build_datasets_from_alignment.py)
    "\\n\\n".join()       the join used when building samples

The prompt budget is measured at runtime rather than hard-coded: the ChatML
wrappers add a fixed number of tokens, and the training build under-counted them
by 7, which is why the earlier 8192-budget dataset silently dropped 24% of its
samples. Measuring the rendered empty prompt removes that whole class of error.

Each window is verified individually with the Wikidot validator, so a failure is
reported against the paragraph range that caused it instead of appearing as one
opaque score for a 40,000-token document.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Sentence-ish boundaries for splitting a paragraph that cannot fit on its own.
_SENTENCE_RE = None


def _sentence_split(text: str, limit: int, measure) -> List[str]:
    """Break ``text`` into pieces of at most ``limit`` tokens.

    A single alignment unit can exceed the whole budget: Wikidot markup blocks
    (``[[div]]``, ``[[include]]``) happily run for tens of thousands of
    characters with no blank line, which is exactly what made the training build
    drop 47 documents outright. Sentence boundaries are the least damaging place
    to cut, with a hard character split as the fallback when a "sentence" is
    itself oversized (minified CSS, base64 blobs).
    """
    global _SENTENCE_RE
    if _SENTENCE_RE is None:
        import re
        _SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")

    pieces: List[str] = []
    current = ""
    for chunk in _SENTENCE_RE.split(text):
        if chunk is None:
            continue
        candidate = f"{current} {chunk}".strip() if current else chunk
        if measure(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if measure(chunk) <= limit:
            current = chunk
            continue
        # Still too big: cut on characters, halving until it fits.
        start = 0
        while start < len(chunk):
            step = len(chunk) - start
            while step > 1 and measure(chunk[start:start + step]) > limit:
                step = max(1, step // 2)
            pieces.append(chunk[start:start + step])
            start += step
        current = ""
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


def plan_windows(
    source: str,
    measure,
    budget: int,
    overlap: int = 0,
) -> Tuple[List[str], List[Tuple[int, int]], List[List[str]]]:
    """Return (window texts, window spans, per-window paragraph lists)."""
    from src.align.paragraph_align import split_paragraphs
    from scripts.build_datasets_from_alignment import join



    paragraphs = split_paragraphs(source)
    if not paragraphs:
        return [], [], []

    # Split any paragraph that cannot fit alone; remember nothing else about it.
    expanded: List[str] = []
    for paragraph in paragraphs:
        cost = measure(paragraph)
        if cost <= budget:
            expanded.append(paragraph)
        else:
            expanded.extend(_sentence_split(paragraph, budget, measure))

    # Floating-point-free exact packing. ``pack_windows`` trusts the sum of the
    # per-unit costs, but BPE merges across a paragraph boundary make the joined
    # text longer than its parts -- on scp-7243 the estimate produced a 8116-token
    # window against a 8074 budget. The training builder hit exactly this and
    # fixed it by re-measuring each window; the same fix is applied here. The
    # cumulative sum is used only to find where to start checking, so the number
    # of (expensive) exact measurements stays proportional to the window count.
    unit = [measure(p) for p in expanded]
    cumulative = [0]
    for value in unit:
        cumulative.append(cumulative[-1] + value)

    total = len(expanded)
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        end = start + 1
        while end + 1 <= total and cumulative[end + 1] - cumulative[start] <= budget:
            end += 1
        while end > start + 1 and measure(join(expanded[start:end])) > budget:
            end -= 1
        spans.append((start, end))
        if end >= total:
            break
        start = end - overlap if overlap else end

    texts: List[str] = []
    groups: List[List[str]] = []
    for begin, end in spans:
        group = expanded[begin:end]
        groups.append(group)
        texts.append(join(group))
    return texts, spans, groups


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Translate a Wikidot document of any length")
    parser.add_argument("input", nargs="?", help="input .wikidot file (or pipe via stdin)")
    parser.add_argument("--text", help="translate this string instead of a file")
    parser.add_argument("-o", "--output", help="output file (default: stdout)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", help="base model name/path")
    parser.add_argument("--adapter", help="LoRA adapter directory")
    parser.add_argument("--base-model", help="explicit base model for the adapter")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=8192,
                        help="context the model was trained with (default 8192)")
    parser.add_argument("--safety-margin", type=int, default=8,
                        help="tokens reserved below the context")
    parser.add_argument("--overlap", type=int, default=0,
                        help="paragraphs shared between consecutive windows (training used 0)")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--plan-only", action="store_true", help="print the window plan and exit")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", default="", help="write a JSON report here")
    args = parser.parse_args(argv)

    from src.utils.config import ensure_dirs, load_config
    from src.utils.io import read_text, write_text
    from src.utils.logging_utils import setup_logging

    # --config handling mirrors the other CLI entry points
    cfg = load_config(args.config, None)
    ensure_dirs(cfg)
    log = setup_logging("WARNING" if args.quiet else "INFO",
                        Path(cfg["paths"]["log_dir"]) / "translate_long.log",
                        name="scp.translate_long")

    if args.text is not None:
        source = args.text
    elif args.input:
        source = read_text(Path(args.input))
    else:
        source = sys.stdin.read()
    if not source.strip():
        print("error: empty input", file=sys.stderr)
        return 2

    # ---- planning needs only the tokenizer ----------------------------------
    # Loading a 4-bit 7B model to count tokens would make --plan-only take a
    # minute; the tokenizer answers in a second and the model is loaded below
    # only when there is something to generate.
    base_model = args.base_model or str((cfg.get("model") or {}).get("name_or_path", "Qwen/Qwen2.5-7B-Instruct"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    fmt_cfg = dict(cfg.get("format") or {})
    system_prompt = str(fmt_cfg.get("system_prompt") or "")
    from src.data.to_chat import build_user_prompt

    def measure(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def render_prompt(source_text: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": build_user_prompt(source_text, fmt_cfg)})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Measured, not assumed: the training build under-counted the ChatML
    # wrappers by 7 tokens, which silently cost it 24% of its samples.
    overhead = measure(render_prompt(""))

    # The window is sized so that source + generation + overhead fits the
    # context. Getting this wrong is not a small error: an earlier version gave
    # the source the whole 8074-token budget, so a full window left ~200 tokens
    # for the output and generation was cut off mid-document (scp-643 lost 7 of
    # its 11 blockquote blocks that way, which looked exactly like a model that
    # drops markup).
    #
    # Training filled source and target from the SAME budget -- the builder's
    # exact_cost is overhead + source + target -- and the observed sources
    # (p50 1709, max 6929 tokens) stayed well under half of it. Reserving
    # max_new_tokens for the answer reproduces that split and makes truncation
    # impossible by construction.
    budget = args.max_seq_length - overhead - args.max_new_tokens - args.safety_margin
    if budget <= 0:
        print(f"error: prompt overhead {overhead} leaves no room in {args.max_seq_length}", file=sys.stderr)
        return 2

    n_source = measure(source)
    windows, spans, groups = plan_windows(source, measure, budget, args.overlap)

    print(f"  源码         {n_source:,} token", file=sys.stderr)
    print(f"  prompt 开销  {overhead} token（实测）", file=sys.stderr)
    print(f"  源码预算     {budget} token（= {args.max_seq_length} - {overhead} 开销 - "
          f"{args.max_new_tokens} 输出上限 - {args.safety_margin} 余量）", file=sys.stderr)
    print(f"  窗口数       {len(windows)}", file=sys.stderr)
    if len(windows) > 1:
        sizes = [measure(w) for w in windows]
        print(f"  窗口大小     min {min(sizes)} / max {max(sizes)}", file=sys.stderr)
    print(f"  单窗可达     {'是' if len(windows) <= 1 else '否（已切窗）'}", file=sys.stderr)

    if args.plan_only:
        for index, ((begin, end), group) in enumerate(zip(spans, groups)):
            print(f"    窗口 {index}: 段落 {begin}~{end}（{len(group)} 段，{measure(windows[index])} token）",
                  file=sys.stderr)
        return 0

    # ---- generate ----
    from src.inference.translator import GenerationSettings, SCPTranslator

    adapter = args.adapter
    model = args.model
    if not adapter and not model:
        for candidate in ("qwen2.5-7b-scp-convert-stage3-8k", "qwen2.5-7b-scp-qlora"):
            path = Path(cfg["paths"]["output_dir"]) / candidate / "final_adapter"
            if path.exists():
                adapter = str(path)
                break
        else:
            model = base_model
            log.warning("no adapter found -- translating with the base model")

    generation = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_seq_length,
        do_sample=False,
        repetition_penalty=args.repetition_penalty,
    )
    translator = SCPTranslator.from_config(
        cfg, model_path=model, adapter_path=adapter, base_model=args.base_model,
        load_in_4bit=not args.no_4bit, generation=generation, logger=log,
    )

    outputs: List[str] = []
    window_reports: List[Dict[str, Any]] = []
    started = time.time()
    for index, window in enumerate(windows):
        begin, end = spans[index]
        t0 = time.time()
        result = translator.translate(window)
        dt = time.time() - t0
        outputs.append(result.text)
        entry: Dict[str, Any] = {
            "window": index,
            "paragraphs": [begin, end],
            "paragraph_count": len(groups[index]),
            "source_tokens": measure(window),
            "output_tokens": measure(result.text),
            "seconds": round(dt, 1),
            "truncated_input": bool(getattr(result, "truncated_input", False)),
        }
        try:
            from src.wikidot.validator import validate_pair
            report = validate_pair(window, result.text)
            entry["wikidot_score"] = report.score
            entry["wikidot_errors"] = report.error_count()
            entry["wikidot_issues"] = [f"{i.code}: {i.detail}" for i in report.issues[:8]]
        except Exception as exc:  # validation must never lose the translation
            entry["wikidot_error"] = f"{type(exc).__name__}: {exc}"
        window_reports.append(entry)
        print(f"\r    窗口 {index + 1}/{len(windows)}  {dt:.0f}s  "
              f"标记分 {entry.get('wikidot_score', float('nan')):.4f}",
              end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)

    from scripts.build_datasets_from_alignment import join
    translation = join(outputs)

    overall: Dict[str, Any] = {}
    try:
        from src.wikidot.validator import cjk_ratio, validate_pair
        report = validate_pair(source, translation)
        overall = {
            "wikidot_score": report.score,
            "wikidot_errors": report.error_count(),
            "per_check": {k: round(v.rate, 4) for k, v in report.outcomes.items()},
            "top_issues": [f"{i.code}: {i.detail}" for i in report.issues[:15]],
            "cjk_ratio": round(cjk_ratio(translation), 4),
        }
    except Exception as exc:
        overall = {"wikidot_error": f"{type(exc).__name__}: {exc}"}

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_text(Path(args.output), translation)
        print(f"  已写出       {args.output}", file=sys.stderr)
    else:
        print(translation)

    print(file=sys.stderr)
    print(f"  总耗时       {time.time() - started:.0f}s", file=sys.stderr)
    if overall.get("wikidot_score") is not None:
        print(f"  整体标记分   {overall['wikidot_score']:.4f}   错误 {overall['wikidot_errors']} 处",
              file=sys.stderr)
        print(f"  分类         {overall['per_check']}", file=sys.stderr)
        print(f"  中文占比     {overall['cjk_ratio']:.4f}", file=sys.stderr)
        if overall["top_issues"]:
            print("  问题:", file=sys.stderr)
            for issue in overall["top_issues"][:8]:
                print(f"    {issue}", file=sys.stderr)
    bad = [w for w in window_reports if w.get("wikidot_errors")]
    if bad:
        print(f"  有缺失的窗口 {[w['window'] for w in bad]}（段落区间 "
              f"{[w['paragraphs'] for w in bad]}）", file=sys.stderr)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(
            {"source_tokens": n_source, "overhead": overhead, "budget": budget,
             "windows": window_reports, "overall": overall},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  报告         {args.report}", file=sys.stderr)

    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

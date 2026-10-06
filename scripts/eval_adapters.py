#!/usr/bin/env python
"""Score one or more LoRA adapters on a conversion validation set.

    ./py scripts/eval_adapters.py --split data/conversion_8k/long.val.chat.jsonl \
        --limit 20 --strata 3 \
        --run base --run stage1=outputs/qwen2.5-7b-scp-convert-stage1/final_adapter \
        --run stage3=outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter

Generation is greedy and each configuration is loaded in turn, so the GPU holds
only one model at a time. Samples are drawn stratified by token length: the short
ones say nothing about long-document handling and the long ones dominate the
average, so both ends are sampled deliberately and the per-stratum scores are
reported separately rather than only as one pooled number.

Metrics come from ``src/evaluation/metrics.py`` (sacrebleu BLEU/chrF, ROUGE-L,
Wikidot syntax preservation), so the numbers are comparable with the rest of the
project. Wikidot preservation matters more than BLEU here: the task is defined as
"keep every marker, component name, parameter and URL byte-identical", which a
translation metric cannot see.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_rows(path: Path, limit: int, strata: int) -> List[Dict[str, Any]]:
    """Load the val rows, sampling evenly across the token-length distribution.

    Taking the first N rows would sample whatever order the builder happened to
    write; the builder writes documents in id order, so the first N are all short
    early-series articles.
    """
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["_tokens"] = int(row.get("metadata", {}).get("tokens") or 0)
            rows.append(row)
    if limit <= 0 or limit >= len(rows):
        return rows
    rows.sort(key=lambda r: r["_tokens"])
    picked: List[Dict[str, Any]] = []
    for i in range(strata):
        lo = i * len(rows) // strata
        hi = (i + 1) * len(rows) // strata
        block = rows[lo:hi]
        take = limit // strata + (1 if i < limit % strata else 0)
        if take >= len(block):
            picked.extend(block)
        else:
            step = len(block) / take
            picked.extend(block[int(j * step)] for j in range(take))
    return picked


def stratum_of(tokens: int, bounds) -> str:
    for name, hi in bounds:
        if tokens <= hi:
            return name
    return bounds[-1][0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score LoRA adapters on a conversion val set")
    parser.add_argument("--split", default="data/conversion_8k/long.val.chat.jsonl")
    parser.add_argument("--limit", type=int, default=18, help="total samples generated per adapter")
    parser.add_argument("--strata", type=int, default=3, help="length strata to sample evenly from")
    parser.add_argument("--run", action="append", default=[],
                        help="name or name=adapter_path; repeatable. 'base' means no adapter.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--report", default="outputs/eval_adapters.json")
    parser.add_argument("--dump", default="", help="write per-sample generations here (JSONL)")
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from src.evaluation.metrics import bleu, chrf, length_analysis, rouge_l, wikidot_preservation

    rows = read_rows(Path(args.split), args.limit, args.strata)
    toks = sorted(r["_tokens"] for r in rows)
    print(f"  split   {args.split}")
    print(f"  samples {len(rows)}   token  min {toks[0]} / median {toks[len(toks)//2]} / max {toks[-1]}")
    print()

    runs: List[tuple] = []
    for spec in (args.run or ["base"]):
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = spec, None
        runs.append((name, path))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: List[Dict[str, Any]] = []
    dumps: List[Dict[str, Any]] = []
    dump_handle = None
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        dump_handle = Path(args.dump).open("w", encoding="utf-8")
    for name, adapter in runs:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=quant, device_map="cuda")
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()

        hyps: List[str] = []
        refs: List[str] = []
        srcs: List[str] = []
        per_stratum: Dict[str, List[str]] = {}
        started = time.time()
        for index, row in enumerate(rows):
            messages = row["messages"]
            prompt = [m for m in messages if m.get("role") != "assistant"]
            reference = next(m["content"] for m in messages if m.get("role") == "assistant")
            source = next(m["content"] for m in messages if m.get("role") == "user")
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     repetition_penalty=args.repetition_penalty,
                                     pad_token_id=tokenizer.pad_token_id)
            hyp = tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            hyps.append(hyp); refs.append(reference); srcs.append(source)
            if dump_handle is not None:
                # flush per sample: the run is ~40 minutes and a crash at the
                # metrics step used to throw away every generation.
                dump_handle.write(json.dumps(
                    {"run": name, "id": row.get("id"), "tokens": row["_tokens"],
                     "hypothesis": hyp, "reference": reference,
                     "source": source}, ensure_ascii=False) + "\n")
                dump_handle.flush()
            print(f"\r    {name}: {index + 1}/{len(rows)}", end="", flush=True)
        elapsed = time.time() - started

        bounds = [("short", 2000), ("medium", 6000), ("long", 10 ** 9)]
        for hyp, ref, row in zip(hyps, refs, rows):
            per_stratum.setdefault(stratum_of(row["_tokens"], bounds), []).append((hyp, ref))

        entry: Dict[str, Any] = {
            "run": name,
            "adapter": adapter or "(none)",
            "n": len(hyps),
            "seconds": round(elapsed, 1),
            "seconds_per_sample": round(elapsed / max(len(hyps), 1), 1),
            "bleu": bleu(hyps, refs).get("score"),
            "chrf": chrf(hyps, refs).get("score"),
            "rouge_l": rouge_l(hyps, refs).get("score"),
            "wikidot": wikidot_preservation(srcs, hyps),
            "length": length_analysis(srcs, refs, hyps),
            "per_stratum": {
                key: {"n": len(v), "chrf": chrf([h for h, _ in v], [r for _, r in v]).get("score")}
                for key, v in sorted(per_stratum.items())
            },
        }
        results.append(entry)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\r    {name}: {len(rows)}/{len(rows)}  done in {elapsed:.0f}s "
              f"({elapsed/len(rows):.1f}s/条){' ' * 20}")

        del model
        torch.cuda.empty_cache()

    print()
    print("=" * 92)
    header = f"  {'配置':<10}{'BLEU':>8}{'chrF':>8}{'ROUGE-L':>9}{'Wikidot':>9}{'秒/条':>8}   分段 chrF"
    print(header)
    print("=" * 92)
    for entry in results:
        w = entry["wikidot"]
        wscore = w.get("score") if isinstance(w, dict) else None
        seg = "  ".join(f"{k} {v['chrf']}" for k, v in entry["per_stratum"].items())
        print(f"  {entry['run']:<10}{entry['bleu']:>8}{entry['chrf']:>8}{entry['rouge_l']:>9}"
              f"{wscore if wscore is not None else '-':>9}{entry['seconds_per_sample']:>8}   {seg}")
    print("=" * 92)

    for entry in results:
        w = entry.get("wikidot")
        if isinstance(w, dict) and w.get("n"):
            print(f"  {entry['run']:<10} Wikidot 细节: {json.dumps({k: v for k, v in w.items() if k != 'n'}, ensure_ascii=False)[:150]}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告: {args.report}")
    if dump_handle is not None:
        dump_handle.close()
        print(f"  生成样本: {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

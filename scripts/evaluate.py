#!/usr/bin/env python
"""Phase 6 -- evaluate the SCP translator.

Two ways to obtain hypotheses:

1. generate them with the model
   ``python scripts/evaluate.py --adapter outputs/qwen2.5-7b-scp-qlora/final_adapter``

2. score an existing prediction file (``{id, hypothesis}`` per line)
   ``python scripts/evaluate.py --predictions outputs/eval/predictions.jsonl``

Metrics: BLEU, chrF, ROUGE-L, Wikidot syntax
preservation, length-anomaly detection (see ``src/evaluation/metrics.py``).

Reports are written to ``outputs/eval/``: ``eval_report.json``,
``eval_report.md``, ``eval_per_sample.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate                # noqa: E402
from src.evaluation.report import write_reports            # noqa: E402
from src.utils.config import add_common_args, ensure_dirs, load_config  # noqa: E402
from src.utils.io import load_jsonl, write_jsonl           # noqa: E402
from src.utils.logging_utils import setup_logging          # noqa: E402

HYP_FIELDS = ["hypothesis", "prediction", "output", "translation", "hyp", "target_pred"]


def pick_hypothesis(row: dict) -> str:
    for field in HYP_FIELDS:
        if field in row and row[field] is not None:
            return str(row[field])
    if "messages" in row and isinstance(row["messages"], list):
        for message in reversed(row["messages"]):
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
    raise ValueError(f"cannot find a hypothesis field in {list(row)[:8]}")


def build_records(predictions_path: Path, test_path: Path) -> list:
    """Join predictions with references from the test split (by id)."""
    refs = {}
    if test_path.exists():
        for row in load_jsonl(test_path):
            refs[str(row["id"])] = row
            refs.setdefault(str(row.get("id", "")).split("::")[0], row)
    records = []
    for row in load_jsonl(predictions_path):
        pid = str(row.get("id", ""))
        hypothesis = pick_hypothesis(row)
        reference = row.get("reference") or row.get("target")
        source = row.get("source") or ""
        if reference is None:
            match = refs.get(pid) or refs.get(pid.split("::")[0])
            if match:
                reference = match.get("target")
                source = source or match.get("source", "")
        if reference is None:
            raise ValueError(f"no reference found for prediction id={pid!r}; pass a test split with matching ids")
        records.append({"id": pid, "source": source, "target": reference, "hypothesis": hypothesis})
    return records


def generate_records(cfg, args, log) -> list:
    from src.inference.translator import GenerationSettings, SCPTranslator

    test_path = Path(args.test_file) if args.test_file else Path(cfg["paths"]["splits_dir"]) / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found -- run scripts/prepare_data.py first")
    rows = list(load_jsonl(test_path))
    if args.limit:
        rows = rows[: args.limit]
    log.info("generating translations for %d test pages", len(rows))

    generation = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        do_sample=False,
        repetition_penalty=args.repetition_penalty,
    )
    translator = SCPTranslator.from_config(
        cfg,
        model_path=args.model,
        adapter_path=args.adapter,
        base_model=args.base_model,
        load_in_4bit=not args.no_4bit,
        generation=generation,
        logger=log,
    )

    out_dir = Path((cfg.get("evaluation") or {}).get("output_dir", "outputs/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(args.batch_size or (cfg.get("evaluation") or {}).get("batch_size", 8)))

    records = []
    predict_path = out_dir / "predictions.jsonl"
    started = time.time()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        results = translator.translate_batch([str(r["source"]) for r in batch])
        for row, result in zip(batch, results):
            records.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "target": row["target"],
                    "hypothesis": result.text,
                    "n_input_tokens": result.n_input_tokens,
                    "n_output_tokens": result.n_output_tokens,
                }
            )
        done = min(start + batch_size, len(rows))
        log.info("[%d/%d] generated (%.1fs elapsed)", done, len(rows), time.time() - started)

    write_jsonl(predict_path, records)
    log.info("wrote %s", predict_path)
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SCP translation quality")
    add_common_args(parser)
    parser.add_argument("--predictions", help="JSONL with {id, hypothesis} to score instead of generating")
    parser.add_argument("--model", help="base model name/path")
    parser.add_argument("--adapter", help="LoRA adapter directory")
    parser.add_argument("--base-model", help="explicit base model for the adapter")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--tag", default="eval", help="file name prefix for the reports")
    parser.add_argument("--test-file", default=None,
                        help="evaluate this file instead of splits/test.jsonl (e.g. data/processed/test.chunks.jsonl)")
    parser.add_argument("--metrics", help="comma separated subset: bleu,chrf,rouge_l,wikidot,length")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    ensure_dirs(cfg)
    log = setup_logging("INFO", Path(cfg["paths"]["log_dir"]) / "evaluate.log", name="scp.evaluate")

    eval_cfg = dict(cfg.get("evaluation") or {})
    if args.metrics:
        eval_cfg["metrics"] = [m.strip() for m in args.metrics.split(",") if m.strip()]

    test_path = Path(args.test_file) if args.test_file else Path(cfg["paths"]["splits_dir"]) / "test.jsonl"

    if args.predictions:
        records = build_records(Path(args.predictions), test_path)
        log.info("scoring %d predictions from %s", len(records), args.predictions)
        model_label, adapter_label = args.model, args.adapter
    else:
        if not args.adapter and not args.model:
            default_adapter = Path(cfg["paths"]["output_dir"]) / "qwen2.5-7b-scp-qlora" / "final_adapter"
            if default_adapter.exists():
                args.adapter = str(default_adapter)
            else:
                args.model = str((cfg.get("model") or {}).get("name_or_path", "Qwen/Qwen2.5-7B-Instruct"))
        if args.limit:
            pass  # generate_records applies the limit itself
        records = generate_records(cfg, args, log)
        model_label, adapter_label = args.model, args.adapter

    report, samples = evaluate(records, cfg=eval_cfg, logger=log)
    report.update(
        {
            "model": model_label,
            "adapter": adapter_label,
            "config": args.config,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": cfg["project_root"],
        }
    )

    out_dir = Path(eval_cfg.get("output_dir", "outputs/eval"))
    paths = write_reports(report, samples, out_dir, prefix=args.tag)
    log.info("reports: %s", json.dumps(paths, indent=2))

    print("\n" + "=" * 64)
    print(f"samples: {report['n_samples']}")
    metrics = report["metrics"]
    for key in ("bleu_raw", "bleu_plain", "chrf_raw", "chrf_plain", "rouge_l_raw", "rouge_l_plain"):
        if key in metrics:
            print(f"  {key:16s} {metrics[key].get('score')}")
    if "wikidot" in metrics:
        w = metrics["wikidot"]
        print(f"  wikidot score    {w.get('score')} (perfect={w.get('perfect_ratio')}, errors={w.get('errors_total')})")
    if "length" in metrics:
        l = metrics["length"]
        print(f"  length anomalies {l.get('anomalies')}/{l.get('n')} ({l.get('reason_counts')})")
    print("=" * 64)
    print(f"markdown report: {paths['report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

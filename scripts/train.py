#!/usr/bin/env python
"""Phase 2 -- QLoRA fine-tuning of a Qwen 7B model on SCP translation.

Usage
-----
    # sanity check the GPU + the actual module names, without training
    python scripts/train.py --config configs/train.yaml --inspect-model

    # 6-step smoke test on a 0.5B model (proves the whole path works)
    python scripts/train.py --config configs/train.smoke.yaml

    # the real run
    python scripts/train.py --config configs/train.yaml

    # resume
    python scripts/train.py --config configs/train.yaml --set training.resume_from_checkpoint=auto

Useful overrides
----------------
    --set training.max_seq_length=8192
    --set lora.r=32 --set lora.alpha=64
    --set training.per_device_train_batch_size=2
    --set training.max_steps=200            # short trial run
    --set model.name_or_path=Qwen/Qwen2.5-14B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.config import TrainSettings, gpu_summary   # noqa: E402
from src.training.trainer import inspect_model, train       # noqa: E402
from src.utils.config import add_common_args, ensure_dirs, load_config  # noqa: E402
from src.utils.io import write_json                          # noqa: E402
from src.utils.logging_utils import setup_logging            # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for SCP translation")
    add_common_args(parser)
    parser.add_argument("--inspect-model", action="store_true", help="print module names + LoRA target decision, then exit")
    parser.add_argument("--dry-run", action="store_true", help="load model + datasets but do not train")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="resume from a checkpoint ('auto' = newest)")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N training samples")
    args = parser.parse_args(argv)

    overrides = list(args.overrides or [])
    if args.resume is not None:
        overrides.append(f"training.resume_from_checkpoint={json.dumps(args.resume)}")
    if args.limit:
        overrides.append(f"training.limit={int(args.limit)}")

    # the train configs inherit paths/data from configs/default.yaml
    cfg = load_config(args.config, overrides)
    ensure_dirs(cfg)
    ts = TrainSettings.from_config(cfg)
    Path(ts.log_dir).mkdir(parents=True, exist_ok=True)
    log = setup_logging("INFO", Path(ts.log_dir) / "train.log", name="scp.train")

    log.info("=" * 70)
    log.info("SCP translator -- QLoRA training")
    log.info("config      : %s", Path(args.config).resolve())
    log.info("output_dir  : %s", ts.output_dir)
    log.info("base model  : %s", ts.model.name_or_path)
    log.info("quantization: 4bit=%s type=%s double_quant=%s compute=%s",
             ts.quantization.enabled, ts.quantization.bnb_4bit_quant_type,
             ts.quantization.bnb_4bit_use_double_quant, ts.quantization.bnb_4bit_compute_dtype)
    log.info("lora        : r=%s alpha=%s dropout=%s target=%s",
             ts.lora.r, ts.lora.alpha, ts.lora.dropout, ts.lora.target_modules)
    log.info("gpu         : %s", gpu_summary())
    log.info("=" * 70)

    if args.inspect_model:
        report = inspect_model(ts, log)
        out = ts.output_dir / "model_inspection.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, report)
        log.info("wrote %s", out)
        return 0

    result = train(ts, log, dry_run=args.dry_run)
    log.info("result: %s", json.dumps(result, ensure_ascii=False)[:800])
    if not args.dry_run:
        log.info("next: python scripts/translate.py --model %s --text '**Item #:** SCP-173'", result.get("adapter_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

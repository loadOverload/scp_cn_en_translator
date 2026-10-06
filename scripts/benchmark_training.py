#!/usr/bin/env python
"""A/B benchmark of training throughput on a fixed sample subset.

    ./py scripts/benchmark_training.py                    # all configurations
    ./py scripts/benchmark_training.py --only baseline,bs2
    ./py scripts/benchmark_training.py --steps 4 --limit 64

Every configuration runs on **the same first N samples of the same file**, so the
token count is identical and the wall-clock times are directly comparable. The
point is to locate the bottleneck before touching the dataset: end-to-end
throughput covers the data pipeline, tokenisation, the forward and backward pass,
attention, gradient-checkpoint recomputation and the optimizer, and a single
tokens/second number cannot say which of those is responsible.

Reported per configuration: seconds per optimizer step (the first step is listed
separately because it carries the CUDA warm-up), tokens/second, peak VRAM, and
mean GPU utilisation and power sampled while it runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (label, overrides). Order follows the debugging order: attention first, then
# batching, then gradient checkpointing, then the optional extras.
CONFIGURATIONS: List[Tuple[str, Dict[str, Any]]] = [
    # Every batch/accumulation setting is stated explicitly. Relying on the
    # config file made an earlier comparison meaningless: the "8k bs1" entry
    # inherited bs4 from configs/train.convert.stage2.yaml and ran as bs4,
    # while the same entry on another machine (whose config still said bs1)
    # ran as bs1. Same label, different experiment.
    ("16k bs1", {"training.max_seq_length": 16384, "training.use_liger_kernel": False,
                 "training.per_device_train_batch_size": 1,
                 "training.gradient_accumulation_steps": 16}),
    ("16k bs2", {"training.max_seq_length": 16384, "training.use_liger_kernel": False,
                 "training.per_device_train_batch_size": 2,
                 "training.gradient_accumulation_steps": 8}),
    ("8k bs1", {"training.max_seq_length": 8192, "training.use_liger_kernel": False,
                "training.per_device_train_batch_size": 1,
                "training.gradient_accumulation_steps": 16}),
    ("8k bs2", {"training.max_seq_length": 8192, "training.use_liger_kernel": False,
                "training.per_device_train_batch_size": 2,
                "training.gradient_accumulation_steps": 8}),
    ("8k bs4", {"training.max_seq_length": 8192, "training.use_liger_kernel": False,
                "training.per_device_train_batch_size": 4,
                "training.gradient_accumulation_steps": 4}),
    ("16k bs4", {"training.max_seq_length": 16384, "training.use_liger_kernel": False,
                 "training.per_device_train_batch_size": 4,
                 "training.gradient_accumulation_steps": 4}),
]


def _nvidia_smi(fields: str) -> Optional[List[str]]:
    for binary in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        try:
            out = subprocess.run([binary, f"--query-gpu={fields}", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return [x.strip() for x in out.stdout.strip().split("\n")[0].split(",")]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


class GpuSampler(threading.Thread):
    """Poll utilisation and power while a run is in flight."""

    def __init__(self, interval: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.util: List[float] = []
        self.power: List[float] = []
        self.vram: List[float] = []

    def run(self) -> None:
        while not self.stop_flag.is_set():
            row = _nvidia_smi("utilization.gpu,power.draw,memory.used")
            if row and len(row) == 3:
                try:
                    self.util.append(float(row[0].replace("%", "").strip()))
                    self.power.append(float(row[1].replace("W", "").strip()))
                    self.vram.append(float(row[2].replace("MiB", "").strip()))
                except ValueError:
                    pass
            self.stop_flag.wait(self.interval)

    def stop(self) -> None:
        self.stop_flag.set()
        self.join(timeout=5)


STEP_RE = re.compile(r"(\d+)/(\d+) \[([0-9:]+)<([0-9:]+),\s+([0-9.]+)s/it\]")
METRICS_RE = re.compile(r"\{'train_runtime'.*?\}")


def parse_log(text: str) -> Dict[str, Any]:
    """Per-step seconds (the bar's own running average, converted to deltas)."""
    seen: Dict[int, float] = {}
    for match in STEP_RE.finditer(text.replace("\r", "\n")):
        step = int(match.group(1))
        # the bar prints MM:SS below an hour and HH:MM:SS above it
        parts = [int(x) for x in reversed(match.group(3).split(":"))]
        seen[step] = sum(value * 60 ** index for index, value in enumerate(parts))
    steps = sorted(seen)
    deltas = [seen[steps[i]] - seen[steps[i - 1]] for i in range(1, len(steps))]
    metrics: Dict[str, Any] = {}
    found = METRICS_RE.findall(text)
    if found:
        try:
            metrics = json.loads(found[-1].replace("'", '"'))
        except json.JSONDecodeError:
            pass
    return {"steps": steps, "deltas": deltas, "metrics": metrics,
            "last_elapsed": seen[steps[-1]] if steps else 0}


def run_configuration(label: str, overrides: Dict[str, Any], args) -> Dict[str, Any]:
    out_dir = Path("/tmp/bench") / label.replace(" ", "_").replace("/", "_")
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)
    log_path = Path("/tmp/bench") / f"{out_dir.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [str(Path(args.python)), "-u", "scripts/train.py",
               "--config", args.config,
               "--limit", str(args.limit),
               "--set", f"training.max_steps={args.steps}",
               "--set", 'training.eval_strategy="no"',
               "--set", 'training.save_strategy="no"',
               "--set", f"training.output_dir={out_dir}"]
    for key, value in overrides.items():
        command += ["--set", f"{key}={value}"]

    sampler = GpuSampler()
    sampler.start()
    started = time.time()
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with log_path.open("w", encoding="utf-8") as sink:
        process = subprocess.run(command, stdout=sink, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), env=environment)
    wall = time.time() - started
    sampler.stop()

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_log(text)
    failed = process.returncode != 0
    error_hint = ""
    if failed:
        for line in text.split("\n"):
            if any(k in line for k in ("Error", "error:", "RuntimeError", "OutOfMemory",
                                       "ImportError", "ValueError")):
                error_hint = line.strip()[:140]
                break

    num_tokens = parsed["metrics"].get("num_tokens")
    train_runtime = parsed["metrics"].get("train_runtime")
    tokens_per_second = (round(num_tokens / train_runtime, 1)
                         if num_tokens and train_runtime else None)
    return {
        "label": label,
        "overrides": overrides,
        "tokens_per_second": tokens_per_second,
        "ok": not failed,
        "error": error_hint,
        "wall_seconds": round(wall, 1),
        "steps_done": len(parsed["steps"]),
        # drop the first step: it carries CUDA warm-up and kernel selection
        "per_step": round(statistics.fmean(parsed["deltas"][1:]), 1) if len(parsed["deltas"]) > 1 else None,
        "first_step": round(parsed["deltas"][0], 1) if parsed["deltas"] else None,
        "train_runtime": parsed["metrics"].get("train_runtime"),
        "num_tokens": parsed["metrics"].get("num_tokens"),
        "mean_util": round(statistics.fmean(sampler.util), 1) if sampler.util else None,
        "mean_power": round(statistics.fmean(sampler.power), 1) if sampler.power else None,
        "peak_vram_gb": round(max(sampler.vram) / 1024, 2) if sampler.vram else None,
        "log": str(log_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark of training throughput")
    parser.add_argument("--config", default="configs/train.convert.stage2.yaml")
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps per configuration")
    parser.add_argument("--limit", type=int, default=48, help="samples (steps x effective batch)")
    parser.add_argument("--only", default="", help="comma-separated labels to run")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--report", default="outputs/benchmark.json")
    args = parser.parse_args(argv)

    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    configs = [(l, o) for l, o in CONFIGURATIONS if not wanted or l in wanted]

    print(f"  subset: first {args.limit} samples of the long set, {args.steps} optimizer steps each")
    print(f"  {len(configs)} configurations\n")
    results: List[Dict[str, Any]] = []
    for label, overrides in configs:
        print(f"  --- {label} ---", flush=True)
        result = run_configuration(label, overrides, args)
        results.append(result)
        if result["ok"]:
            print(f"      per-step {result['per_step']}s  first {result['first_step']}s  "
                  f"util {result['mean_util']}%  power {result['mean_power']}W  "
                  f"VRAM {result['peak_vram_gb']}GB", flush=True)
        else:
            print(f"      FAILED: {result['error']}", flush=True)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"  {'配置':<14}{'每步s':>7}{'tok/s':>8}{'util%':>7}{'功率W':>7}{'VRAM_GB':>9}{'相对吞吐':>10}")
    print("=" * 88)
    baseline = next((r for r in results if r["label"] == "baseline" and r["ok"]), None)
    for r in results:
        if not r["ok"]:
            print(f"  {r['label']:<16}{'失败':>9}   {r['error'][:50]}")
            continue
        # the meaningful comparison is tokens/second: a longer step that carries
        # more tokens can still be the faster configuration
        speed = "-"
        if baseline and baseline.get("tokens_per_second") and r.get("tokens_per_second"):
            speed = f"{r['tokens_per_second'] / baseline['tokens_per_second']:.2f}x"
        tps = r.get("tokens_per_second")
        print(f"  {r['label']:<14}{r['per_step']:>7}{tps if tps else '-':>8}"
              f"{r['mean_util']:>7}{r['mean_power']:>7}{r['peak_vram_gb']:>9}{speed:>10}")
    print("=" * 88)
    print(f"  报告: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

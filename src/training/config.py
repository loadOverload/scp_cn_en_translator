"""Typed training settings derived from the YAML config.

Everything the trainer needs comes from here, so no magic numbers are buried in
the training code. ``TrainSettings.from_config`` also resolves the
"auto" values (dtype, bf16, optimizer) against the *actual* machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _torch():
    import torch  # local import: this module is imported by non-GPU scripts too

    return torch


# ---------------------------------------------------------------------------
# capability probes
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


def bf16_supported() -> bool:
    try:
        torch = _torch()
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def gpu_summary() -> Dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        return {"cuda": False, "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "vram_gb": round(props.total_memory / 1024 ** 3, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "bf16": bf16_supported(),
        "device_count": torch.cuda.device_count(),
    }


def resolve_dtype(name: str, allow_fp16: bool = True):
    torch = _torch()
    name = str(name or "auto").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    # auto
    if torch.cuda.is_available():
        return torch.bfloat16 if bf16_supported() else (torch.float16 if allow_fp16 else torch.float32)
    return torch.float32


def resolve_optim(name: str) -> str:
    """Fall back when paged/8-bit optimizers are unavailable."""
    name = str(name or "adamw_torch")
    if name.startswith(("paged_", "adamw_bnb", "adamw_8bit")):
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            return "adamw_torch"
    return name


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class ModelSettings:
    name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    attn_implementation: str = "sdpa"
    use_cache: bool = False


@dataclass
class QuantizationSettings:
    enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "auto"


@dataclass
class LoraSettings:
    enabled: bool = True
    target_modules: Any = "auto"
    target_module_candidates: List[str] = field(default_factory=list)
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    modules_to_save: Optional[List[str]] = None


@dataclass
class TrainingSettings:
    output_dir: str = "outputs/qwen-scp-qlora"
    trainer: str = "auto"
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    max_seq_length: int = 4096
    overlong_policy: str = "drop"
    train_on_assistant_only: bool = True
    # Tokenized samples are cached here and reused across runs. The key
    # includes the data file's size and mtime, so rebuilding the dataset
    # invalidates the cache automatically. Empty string disables it.
    tokenized_cache_dir: str = "data/tokenized_cache"
    # Threads used to tokenize on load. Threads, not processes: the Rust
    # tokenizer releases the GIL, so this scales without pickling anything.
    # 0 = choose from the machine, negative = strictly serial.
    tokenize_workers: int = 0
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict[str, Any] = field(default_factory=lambda: {"use_reentrant": False})
    bf16: Any = "auto"
    fp16: bool = False
    tf32: bool = True
    logging_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 200
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    # transformers 5.x renamed this: ``group_by_length`` no longer exists on
    # TrainingArguments, and the old kwarg was silently dropped by the
    # "unsupported kwargs" filter -- so the setting looked applied and did
    # nothing. "random" | "group_by_length" | "batch_rebalance".
    train_sampling_strategy: str = "random"
    seed: int = 42
    resume_from_checkpoint: Any = None
    # Start from an existing LoRA adapter instead of a fresh one. This is what
    # chains the two stages: stage 2 continues the adapter stage 1 produced, so
    # the paragraph-level mapping is already learned when long context starts.
    # (``resume_from_checkpoint`` is a different thing -- it restores a Trainer
    # checkpoint, optimizer state included, which is wrong across a data change.)
    init_adapter: Any = None
    # Liger's fused linear cross-entropy. Qwen2.5's vocabulary is 151,936 tokens, so
    # the LM head output is [seq_len, 151936]: at 16k context that is 4.6 GB in bf16
    # and 9.3 GB once upcast for the loss, ~14 GB of the 24 GB card, which pushed
    # VRAM to 98% and left the run memory-thrashing instead of computing. Liger
    # computes the loss in chunks without ever materialising those logits.
    use_liger_kernel: bool = False
    packing: bool = False
    # debug helper: only use the first N training samples (0 = all)
    limit: int = 0
    # resolved runtime values
    resolved_dtype: Any = None
    resolved_optim: str = ""
    resolved_bf16: bool = False
    resolved_fp16: bool = False


@dataclass
class DataFiles:
    train: str = "data/splits/train.jsonl"
    val: str = "data/splits/val.jsonl"
    test: str = "data/splits/test.jsonl"
    chat_train: str = "data/splits/train.chat.jsonl"
    chat_val: str = "data/splits/val.chat.jsonl"


@dataclass
class TrainSettings:
    model: ModelSettings
    quantization: QuantizationSettings
    lora: LoraSettings
    training: TrainingSettings
    data_files: DataFiles
    project_root: str = "."
    log_dir: str = "logs"
    hf_home: str = "data/hf_cache"
    format: Dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], logger=None) -> "TrainSettings":
        model = ModelSettings(**_filter(ModelSettings, cfg.get("model") or {}))
        quant = QuantizationSettings(**_filter(QuantizationSettings, cfg.get("quantization") or {}))
        lora = LoraSettings(**_filter(LoraSettings, cfg.get("lora") or {}))
        training = TrainingSettings(**_filter(TrainingSettings, cfg.get("training") or {}))
        data_files = DataFiles(**_filter(DataFiles, cfg.get("data_files") or {}))

        training.resolved_dtype = resolve_dtype(model.dtype)
        training.resolved_optim = resolve_optim(training.optim)

        want_bf16 = training.bf16
        if isinstance(want_bf16, str) and want_bf16.lower() == "auto":
            # prefer bf16 (RTX 4090 supports it); fall back to fp16 on older GPUs
            training.resolved_bf16 = bf16_supported()
            training.resolved_fp16 = (not training.resolved_bf16) and cuda_available()
        else:
            training.resolved_bf16 = bool(want_bf16)
            training.resolved_fp16 = bool(training.fp16) and not training.resolved_bf16

        training.overlong_policy = str(training.overlong_policy).lower()
        if training.overlong_policy not in {"drop", "truncate"}:
            raise ValueError(f"training.overlong_policy must be 'drop' or 'truncate', got {training.overlong_policy!r}")

        paths = cfg.get("paths") or {}
        return cls(
            model=model,
            quantization=quant,
            lora=lora,
            training=training,
            data_files=data_files,
            project_root=str(cfg.get("project_root", ".")),
            log_dir=str(paths.get("log_dir", "logs")),
            hf_home=str(paths.get("hf_home", "data/hf_cache")),
            format=dict(cfg.get("format") or {}),
        )

    # -- derived paths -----------------------------------------------------
    @property
    def output_dir(self) -> Path:
        return Path(self.training.output_dir)

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / "final_adapter"

    def resolve_data_file(self, name: str) -> str:
        value = getattr(self.data_files, name)
        p = Path(value)
        return str(p if p.is_absolute() else Path(self.project_root) / p)

    def to_dict(self) -> Dict[str, Any]:
        def as_dict(obj) -> Dict[str, Any]:
            out = {}
            for f in fields(obj):
                value = getattr(obj, f.name)
                if hasattr(value, "isoformat"):
                    value = value.isoformat()
                elif value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                    # torch.dtype etc. -> readable string
                    value = str(value)
                out[f.name] = value
            return out

        return {
            "model": as_dict(self.model),
            "quantization": as_dict(self.quantization),
            "lora": as_dict(self.lora),
            "training": as_dict(self.training),
            "data_files": as_dict(self.data_files),
        }


def _filter(dataclass_type, mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only keys the dataclass declares, ignoring config extras."""
    valid = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in (mapping or {}).items() if k in valid}

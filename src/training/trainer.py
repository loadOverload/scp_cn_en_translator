"""QLoRA training for the SCP translator.

Pipeline: ``load_tokenizer`` -> ``load_model`` (4-bit NF4) -> ``apply_lora``
(target modules detected from the real model) -> ``build_trainer`` -> ``train``.

Two trainer back ends are supported:

``native``
    ``transformers.Trainer`` with this project's own chat dataset/collator.
    Predictable across library versions; this is the default.
``trl``
    ``trl.SFTTrainer`` when the installed TRL exposes a compatible API.
    Selected via ``training.trainer``; falls back to ``native`` with a warning.

Everything is driven by :class:`src.training.config.TrainSettings`, so no
hyper-parameter is hard-coded here.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..utils.io import write_json
from .config import TrainSettings, gpu_summary
from .data import ChatSFTDataset, DataCollatorForChatSFT
from .lora_utils import detect_target_modules, format_report, trainable_parameter_report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _signature_params(obj) -> List[str]:
    try:
        return list(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return []


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets", "tokenizers"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "missing"
    return versions


def _dtype_kwarg(model_cls) -> str:
    """transformers >= 4.56 renamed ``torch_dtype`` to ``dtype``."""
    params = _signature_params(model_cls.from_pretrained)
    return "dtype" if "dtype" in params else "torch_dtype"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_tokenizer(ts: TrainSettings, logger=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ts.model.name_or_path,
        revision=ts.model.revision,
        trust_remote_code=ts.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if logger:
        logger.info("tokenizer: %s | vocab=%d | pad=%r | chat_template=%s",
                    ts.model.name_or_path, len(tokenizer), tokenizer.pad_token,
                    "yes" if tokenizer.chat_template else "NO")
    if not tokenizer.chat_template:
        raise ValueError(
            f"{ts.model.name_or_path} has no chat template; use an Instruct checkpoint "
            "(e.g. Qwen/Qwen2.5-7B-Instruct)"
        )
    return tokenizer


def load_model(ts: TrainSettings, logger=None):
    import torch
    from transformers import AutoModelForCausalLM

    from .config import resolve_dtype

    kwargs: Dict[str, Any] = {
        "revision": ts.model.revision,
        "trust_remote_code": ts.model.trust_remote_code,
        "attn_implementation": ts.model.attn_implementation,
    }
    dtype_key = _dtype_kwarg(AutoModelForCausalLM)
    compute_dtype = resolve_dtype(ts.quantization.bnb_4bit_compute_dtype)

    if ts.quantization.enabled:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=ts.quantization.load_in_4bit,
            bnb_4bit_quant_type=ts.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=ts.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs[dtype_key] = compute_dtype
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            kwargs["device_map"] = {"": local_rank}
    else:
        kwargs[dtype_key] = ts.training.resolved_dtype

    started = time.time()
    if logger:
        logger.info("loading base model %s (4bit=%s, %s=%s)...",
                    ts.model.name_or_path, ts.quantization.enabled, dtype_key, kwargs[dtype_key])
    model = AutoModelForCausalLM.from_pretrained(ts.model.name_or_path, **kwargs)
    model.config.use_cache = bool(ts.model.use_cache) and not ts.training.gradient_checkpointing

    if logger:
        logger.info("model loaded in %.1fs", time.time() - started)
        logger.info("module names (first block): %s",
                    [n for n, _ in list(model.named_modules())[:14]])
    return model


def apply_lora(model, ts: TrainSettings, logger=None):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if ts.quantization.enabled and ts.training.gradient_checkpointing:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs=ts.training.gradient_checkpointing_kwargs,
        )
    elif ts.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ts.training.gradient_checkpointing_kwargs)

    if not ts.lora.enabled:
        if logger:
            logger.warning("lora.enabled=false -> full fine-tuning of every parameter")
        return model, {"requested": "disabled", "resolved": "all-parameters"}

    candidates = ts.lora.target_module_candidates or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    targets, report = detect_target_modules(model, candidates, ts.lora.target_modules)
    if logger:
        logger.info("LoRA target-module detection:\n%s", format_report(report))

    lora_config = LoraConfig(
        r=int(ts.lora.r),
        lora_alpha=int(ts.lora.alpha),
        lora_dropout=float(ts.lora.dropout),
        bias=str(ts.lora.bias),
        task_type=str(ts.lora.task_type),
        target_modules=targets,
        use_rslora=bool(ts.lora.use_rslora),
        modules_to_save=list(ts.lora.modules_to_save) if ts.lora.modules_to_save else None,
    )
    init_adapter = ts.training.init_adapter
    if init_adapter:
        from peft import PeftModel

        adapter_path = Path(str(init_adapter))
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"training.init_adapter points at {adapter_path}, which does not exist; "
                "stage 2 must start from stage 1's final_adapter")
        # The adapter's own adapter_config.json governs from here on -- the LoraConfig
        # built above is discarded, so anything set in this run's ``lora:`` block is
        # silently ignored. That is the right precedence for a continuation, but it
        # has to be said out loud: otherwise editing lora.r or target_modules on the
        # second stage produces no change and nothing anywhere says so.
        import json as _json

        saved_config = _json.loads((adapter_path / "adapter_config.json").read_text())
        effective = {
            "r": saved_config.get("r"),
            "lora_alpha": saved_config.get("lora_alpha"),
            "lora_dropout": saved_config.get("lora_dropout"),
            "bias": saved_config.get("bias"),
            "use_rslora": saved_config.get("use_rslora"),
            "modules_to_save": saved_config.get("modules_to_save"),
        }
        requested = {
            "r": int(ts.lora.r),
            "lora_alpha": int(ts.lora.alpha),
            "lora_dropout": float(ts.lora.dropout),
            "bias": str(ts.lora.bias),
            "use_rslora": bool(ts.lora.use_rslora),
            "modules_to_save": list(ts.lora.modules_to_save) if ts.lora.modules_to_save else None,
        }
        differences = []
        for key, wanted in requested.items():
            got = effective.get(key)
            if isinstance(wanted, float) or isinstance(got, float):
                same = got is not None and abs(float(got) - float(wanted)) < 1e-9
            else:
                same = got == wanted
            if not same:
                differences.append(f"{key}: config says {wanted!r}, adapter has {got!r}")
        saved_targets = set(saved_config.get("target_modules") or [])
        if saved_targets != set(targets):
            differences.append(f"target_modules: config resolves {sorted(targets)}, "
                               f"adapter has {sorted(saved_targets)}")

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        if logger:
            logger.info("LoRA initialised from existing adapter: %s", adapter_path)
            logger.info("effective LoRA config (from the adapter): r=%s alpha=%s dropout=%s "
                        "bias=%s rslora=%s target_modules=%s",
                        effective["r"], effective["lora_alpha"], effective["lora_dropout"],
                        effective["bias"], effective["use_rslora"], sorted(saved_targets))
            if differences:
                logger.warning(
                    "the lora: block of this run disagrees with the adapter being continued, "
                    "and the ADAPTER WINS (its config is what gets loaded): %s", differences)
            else:
                logger.info("lora: block matches the adapter -- nothing is being overridden")
    else:
        model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    report.update(trainable_parameter_report(model))
    if logger:
        logger.info("trainable parameters: %s / %s (%.4f%%)",
                    f"{report['trainable_params']:,}", f"{report['total_params']:,}", report["trainable_pct"])
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    return model, report


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


def build_datasets(ts: TrainSettings, tokenizer, logger=None) -> Tuple[ChatSFTDataset, Optional[ChatSFTDataset]]:
    candidates = [ts.resolve_data_file("chat_train"), ts.resolve_data_file("train")]
    train_path = next((p for p in candidates if Path(p).exists()), candidates[0])
    candidates_val = [ts.resolve_data_file("chat_val"), ts.resolve_data_file("val")]
    val_path = next((p for p in candidates_val if Path(p).exists()), candidates_val[0])

    train_ds = ChatSFTDataset(
        train_path, tokenizer,
        max_seq_length=ts.training.max_seq_length,
        overlong_policy=ts.training.overlong_policy,
        train_on_assistant_only=ts.training.train_on_assistant_only,
        # --limit is a debugging aid for the training set only; applying it to
        # validation as well silently shrank every evaluation to the first N
        # rows of the file, which are all from the same few documents.
        limit=int(ts.training.limit or 0),
        packing=bool(ts.training.packing),
        cache_dir=(ts.training.tokenized_cache_dir or None),
        tokenize_workers=int(ts.training.tokenize_workers),
        name="train", logger=logger,
    )
    val_ds = None
    if Path(val_path).exists():
        val_ds = ChatSFTDataset(
            val_path, tokenizer,
            max_seq_length=ts.training.max_seq_length,
            overlong_policy=ts.training.overlong_policy,
            train_on_assistant_only=ts.training.train_on_assistant_only,
            limit=int(ts.training.limit or 0),
            cache_dir=(ts.training.tokenized_cache_dir or None),
            tokenize_workers=int(ts.training.tokenize_workers),
            name="val", logger=logger,
        )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# training arguments
# ---------------------------------------------------------------------------


def build_training_arguments(ts: TrainSettings, logger=None):
    from transformers import TrainingArguments

    supported = set(_signature_params(TrainingArguments.__init__))
    report_to = [r for r in (ts.training.report_to or []) if r != "tensorboard" or importlib.util.find_spec("tensorboard")]

    kwargs: Dict[str, Any] = {
        "output_dir": str(ts.output_dir),
        "num_train_epochs": float(ts.training.num_train_epochs),
        "per_device_train_batch_size": int(ts.training.per_device_train_batch_size),
        "per_device_eval_batch_size": int(ts.training.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(ts.training.gradient_accumulation_steps),
        "learning_rate": float(ts.training.learning_rate),
        "lr_scheduler_type": ts.training.lr_scheduler_type,
        "warmup_ratio": float(ts.training.warmup_ratio),
        "weight_decay": float(ts.training.weight_decay),
        "max_grad_norm": float(ts.training.max_grad_norm),
        "optim": ts.training.resolved_optim or ts.training.optim,
        "bf16": bool(ts.training.resolved_bf16),
        "fp16": bool(ts.training.resolved_fp16),
        "logging_steps": int(ts.training.logging_steps),
        "save_strategy": ts.training.save_strategy,
        "save_steps": int(ts.training.save_steps),
        "save_total_limit": int(ts.training.save_total_limit),
        "report_to": report_to,
        "dataloader_num_workers": int(ts.training.dataloader_num_workers),
        "dataloader_pin_memory": bool(ts.training.dataloader_pin_memory),
        "train_sampling_strategy": str(ts.training.train_sampling_strategy),
        "seed": int(ts.training.seed),
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "gradient_checkpointing": bool(ts.training.gradient_checkpointing),
        "gradient_checkpointing_kwargs": ts.training.gradient_checkpointing_kwargs,
        "log_level": "info",
    }
    if int(ts.training.max_steps) > 0:
        kwargs["max_steps"] = int(ts.training.max_steps)
    if ts.training.tf32:
        kwargs["tf32"] = True

    eval_key = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    kwargs[eval_key] = ts.training.eval_strategy
    # eval_steps must be passed explicitly. When it is omitted, TrainingArguments
    # falls back to ``logging_steps`` -- here 20 -- so the run evaluated every 20
    # steps: 92 full passes over the validation set, ~11 minutes each, which is
    # what turned a 1.2-hour training job into an 11.6-hour one. The configuration
    # said 1000 the whole time; it simply never reached the trainer.
    if str(ts.training.eval_strategy) == "steps":
        kwargs["eval_steps"] = int(ts.training.eval_steps)
    if str(ts.training.eval_strategy) == "steps" and int(ts.training.eval_steps) < int(ts.training.logging_steps):
        if logger:
            logger.warning(
                "eval_steps (%s) is smaller than logging_steps (%s); evaluation will "
                "dominate the run", ts.training.eval_steps, ts.training.logging_steps)

    if "save_only_model" in supported:
        kwargs["save_only_model"] = False

    dropped = {k: v for k, v in kwargs.items() if k not in supported}
    kwargs = {k: v for k, v in kwargs.items() if k in supported}
    # This filter is how two settings disappeared without a trace: `group_by_length`
    # (renamed to train_sampling_strategy in transformers 5) and `warmup_ratio`
    # (removed entirely -- only warmup_steps exists now). Both looked applied in the
    # config and were silently discarded here, so the run behaved differently from
    # what the file said. Anything dropped is now reported.
    if dropped and logger:
        logger.warning("these settings are not supported by this transformers version "
                       "and were DROPPED: %s", sorted(dropped))
    if logger and dropped:
        logger.info("TrainingArguments does not accept: %s (ignored)", sorted(dropped))
    args = TrainingArguments(**kwargs)
    if logger:
        logger.info("effective batch size: %d x %d grad-accum = %d sequences/step",
                    args.per_device_train_batch_size, args.gradient_accumulation_steps,
                    args.per_device_train_batch_size * args.gradient_accumulation_steps)
        logger.info("precision: bf16=%s fp16=%s | optim=%s | max_steps=%s",
                    args.bf16, args.fp16, args.optim, getattr(args, "max_steps", -1))
    return args


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------


def make_callbacks(logger=None):
    from transformers import TrainerCallback

    class MemoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logger:
                return
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024 ** 3
                    reserved = torch.cuda.memory_reserved() / 1024 ** 3
                    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                    logger.info("GPU memory: allocated=%.2fGB reserved=%.2fGB peak=%.2fGB", allocated, reserved, peak)
            except Exception:
                pass

    return [MemoryCallback()]


# ---------------------------------------------------------------------------
# trainer construction
# ---------------------------------------------------------------------------


def resolve_resume(ts: TrainSettings, logger=None) -> Optional[str]:
    value = ts.training.resume_from_checkpoint
    if value in (None, "", False):
        return None
    if isinstance(value, str) and value.lower() == "auto":
        ckpts = sorted(
            (p for p in ts.output_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        if not ckpts:
            if logger:
                logger.info("resume=auto but no checkpoint found in %s -- starting fresh", ts.output_dir)
            return None
        if logger:
            logger.info("resuming from %s", ckpts[-1])
        return str(ckpts[-1])
    return str(value)


def build_trainer(model, tokenizer, args, train_ds, val_ds, ts: TrainSettings, logger=None):
    from transformers import Trainer

    collator = DataCollatorForChatSFT(tokenizer)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
    }
    params = set(_signature_params(Trainer.__init__))
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    if val_ds is not None and "compute_metrics" in params:
        pass
    class _Trainer(_LengthGroupedMixin, Trainer):
        pass

    trainer = _Trainer(**trainer_kwargs)
    for cb in make_callbacks(logger):
        trainer.add_callback(cb)
    return trainer


class _LengthGroupedMixin:
    """Make ``train_sampling_strategy="group_by_length"`` work on our dataset.

    transformers 5.x builds the length-grouped sampler only when the training
    dataset is a ``datasets.Dataset``, reading a length column from it; for any
    other map-style dataset it sets ``lengths = None`` and quietly keeps example
    order. That is exactly what happened here, and it is expensive: batches then
    mix 30-token and 6,000-token samples, the collator pads to the batch maximum,
    and measurement showed **47% of every batch is padding**. Each sample cost
    0.30 s as a result, against ~0.04 s when the batch is length-homogeneous.

    Our samples are already tokenised, so the lengths are free -- the sampler
    just has to be handed them.
    """

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        strategy = getattr(self.args, "train_sampling_strategy", None)
        if strategy == "group_by_length" and hasattr(dataset, "lengths"):
            from src.training.data import LengthBucketedBatchSampler

            # Sorted globally, not inside transformers' mega-batch window: the
            # windowed version kept 47%-ish padding and was worth only ~15% speed.
            return LengthBucketedBatchSampler(
                dataset.lengths,
                self.args.train_batch_size,
                seed=int(getattr(self.args, "seed", 42) or 42),
            )
        return super()._get_train_sampler(train_dataset)


def build_trl_trainer(model, tokenizer, args, train_ds, val_ds, ts: TrainSettings, logger=None):
    """Optional TRL back end. Returns ``None`` when TRL is unusable here."""
    try:
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        if logger:
            logger.warning("TRL unavailable (%s) -- using the native transformers trainer", exc)
        return None

    # Hand TRL an SFTConfig we built ourselves.
    #
    # SFTConfig carries its own ``max_length`` (default 1024) and filters the
    # dataset by it. If we pass a plain TrainingArguments, TRL does this:
    #
    #     elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
    #         dict_args = args.to_dict()
    #         args = SFTConfig(**dict_args)          # <-- rebuilt, max_length = 1024
    #
    # and our ``max_seq_length`` never reaches it. The effect was severe and silent:
    # stage 2 handed 8,184 long-context samples to TRL and TRL kept **890**, dropping
    # everything over 1024 tokens, so the 16k-context stage trained on no long context
    # at all. Stage 1 lost only 356 of 30,000 (its p99 is 1,375), which is why it went
    # unnoticed for a whole run. Building the SFTConfig here means the isinstance
    # check passes and nothing is rebuilt.
    if not isinstance(args, SFTConfig):
        config_args = args.to_dict()
        config_args["hub_token"] = getattr(args, "hub_token", None)
        config_args["max_length"] = int(ts.training.max_seq_length)
        config_args["use_liger_kernel"] = bool(ts.training.use_liger_kernel)
        args = SFTConfig(**config_args)
        if logger:
            logger.info("converted TrainingArguments -> SFTConfig with max_length=%d "
                        "(TRL's own default would have been 1024)",
                        args.max_length)
    else:
        args.max_length = int(ts.training.max_seq_length)
        if hasattr(args, "use_liger_kernel"):
            args.use_liger_kernel = bool(ts.training.use_liger_kernel)

    collator = DataCollatorForChatSFT(tokenizer)
    supported = set(_signature_params(SFTTrainer.__init__))
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
    }
    if "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    if "data_collator" in supported:
        kwargs["data_collator"] = collator
    if logger:
        logger.info("using TRL SFTTrainer (accepted kwargs: %s)", sorted(kwargs))
    try:
        class _SFTTrainer(_LengthGroupedMixin, SFTTrainer):
            pass

        return _SFTTrainer(**kwargs)
    except TypeError as exc:
        # TRL >= 1.x insists on a datasets.Dataset; our map-style dataset is fine
        # for the native trainer, so materialise it only for this call.
        if "Dataset" not in str(exc):
            if logger:
                logger.warning("SFTTrainer construction failed (%s) -- using the native trainer", exc)
            return None
        try:
            if logger:
                logger.info("materialising the tokenised split into a datasets.Dataset for TRL")
            kwargs["train_dataset"] = train_ds.to_hf_dataset()
            if val_ds is not None:
                kwargs["eval_dataset"] = val_ds.to_hf_dataset()
            # an HF datasets.Dataset carries a length column, so the stock path works
            return SFTTrainer(**kwargs)
        except Exception as exc2:
            if logger:
                logger.warning("SFTTrainer still failed (%s) -- using the native trainer", exc2)
            return None
    except Exception as exc:
        if logger:
            logger.warning("SFTTrainer construction failed (%s) -- using the native trainer", exc)
        return None


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def train(ts: TrainSettings, logger=None, dry_run: bool = False) -> Dict[str, Any]:
    ts.output_dir.mkdir(parents=True, exist_ok=True)
    versions = package_versions()
    gpu = gpu_summary()
    if logger:
        logger.info("packages: %s", versions)
        logger.info("gpu: %s", gpu)
        if not gpu.get("cuda"):
            logger.warning("CUDA is not available -- 4-bit QLoRA on CPU will be extremely slow")

    if ts.training.tf32 and gpu.get("cuda"):
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    tokenizer = load_tokenizer(ts, logger)
    model = load_model(ts, logger)
    model, lora_report = apply_lora(model, ts, logger)

    train_ds, val_ds = build_datasets(ts, tokenizer, logger)
    if len(train_ds) == 0:
        raise RuntimeError(
            "the training split is empty after length filtering -- raise training.max_seq_length "
            "or switch training.overlong_policy to 'truncate'"
        )

    write_json(ts.output_dir / "lora_targets.json", lora_report)
    write_json(
        ts.output_dir / "run_metadata.json",
        {
            "settings": ts.to_dict(),
            "packages": versions,
            "gpu": gpu,
            "lora": lora_report,
            "datasets": {"train": train_ds.stats(), "val": val_ds.stats() if val_ds else None},
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    if dry_run:
        if logger:
            logger.info("dry run: model + datasets are ready, skipping trainer.train()")
        return {"dry_run": True, "lora": lora_report, "train_dataset": train_ds.stats()}

    args = build_training_arguments(ts, logger)

    # `warmup_ratio` no longer exists on TrainingArguments (transformers 5 keeps
    # only `warmup_steps`), so the configured ratio has to be turned into a step
    # count here, where the dataset size is known.
    ratio = float(getattr(ts.training, "warmup_ratio", 0.0) or 0.0)
    if ratio > 0 and not getattr(args, "warmup_steps", 0):
        import math
        steps_per_epoch = math.ceil(
            len(train_ds) / max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps))
        total_steps = (int(ts.training.max_steps) if int(ts.training.max_steps) > 0
                       else steps_per_epoch * max(1, int(ts.training.num_train_epochs)))
        args.warmup_steps = max(1, int(round(ratio * total_steps)))
        if logger:
            logger.info("warmup_ratio %.4g -> warmup_steps %d (of %d total steps)",
                        ratio, args.warmup_steps, total_steps)
    trainer = None
    backend = str(ts.training.trainer or "auto").lower()
    if backend in {"auto", "trl"}:
        trainer = build_trl_trainer(model, tokenizer, args, train_ds, val_ds, ts, logger)
        if trainer is not None:
            backend = "trl"
        elif backend == "trl":
            if logger:
                logger.warning("trainer=trl requested but unusable -- falling back to native")
            backend = "native"
    if trainer is None:
        trainer = build_trainer(model, tokenizer, args, train_ds, val_ds, ts, logger)
        backend = "native"
    if logger:
        logger.info("training with the %s back end", backend)

    resume = resolve_resume(ts, logger)
    started = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - started

    ts.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ts.adapter_dir))
    tokenizer.save_pretrained(str(ts.adapter_dir))
    # keep the exact target modules next to the adapter for later inference
    write_json(ts.adapter_dir / "lora_targets.json", lora_report)

    metrics = dict(getattr(result, "metrics", {}) or {})
    # TrainOutput.metrics carries only the training numbers, so eval_loss never
    # reached train_metrics.json -- the only place the validation result lived was
    # stdout and the TensorBoard event file, both of which are easy to lose. Pull
    # the last evaluation out of the trainer's own history instead of running
    # another pass over the validation set.
    history = getattr(getattr(trainer, "state", None), "log_history", None) or []
    evaluations = [entry for entry in history
                   if any(str(key).startswith("eval_") for key in entry)]
    if evaluations:
        last = evaluations[-1]
        for key, value in last.items():
            metrics.setdefault(key, value)
        metrics["eval_count"] = len(evaluations)
    metrics["train_seconds"] = round(elapsed, 1)
    metrics["backend"] = backend
    write_json(ts.output_dir / "train_metrics.json", metrics)
    if logger:
        logger.info("training finished in %.1fs | metrics: %s", elapsed, metrics)
        logger.info("adapter saved to %s", ts.adapter_dir)
    return {"metrics": metrics, "lora": lora_report, "adapter_dir": str(ts.adapter_dir), "backend": backend}


def inspect_model(ts: TrainSettings, logger=None) -> Dict[str, Any]:
    """``--inspect-model``: report module names without training."""
    from .lora_utils import linear_module_inventory

    model = load_model(ts, logger)
    inventory = linear_module_inventory(model)
    candidates = ts.lora.target_module_candidates or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    _, report = detect_target_modules(model, candidates, ts.lora.target_modules)
    if logger:
        logger.info("linear layer inventory:\n%s", json.dumps(inventory, indent=2, ensure_ascii=False)[:4000])
        logger.info("target module decision:\n%s", format_report(report))
    return {"inventory": inventory, "target_modules": report}

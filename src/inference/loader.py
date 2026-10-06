"""Model/adapter loading for inference (kept separate so importing
:mod:`src.inference.translator` stays cheap).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _adapter_base_model(adapter_path: Path) -> Optional[str]:
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        return None
    with config_file.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("base_model_name_or_path")


def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    base_model: Optional[str] = None,
    load_in_4bit: bool = True,
    logger=None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load a base model and (optionally) a LoRA adapter on top of it.

    ``model_path`` may be either a plain model or a directory that contains an
    adapter; ``adapter_path`` points explicitly at a LoRA adapter directory.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info: Dict[str, Any] = {}
    adapter = Path(adapter_path) if adapter_path else None
    if adapter and not adapter.exists():
        raise FileNotFoundError(f"adapter not found: {adapter}")

    if adapter and not base_model:
        base_model = _adapter_base_model(adapter)
        info["base_model_from_adapter"] = base_model

    base = base_model or model_path
    if not base:
        raise ValueError("provide --model (base model or adapter dir) or --adapter")

    tokenizer_source = str(adapter) if adapter and (adapter / "tokenizer_config.json").exists() else str(base)

    kwargs: Dict[str, Any] = {"trust_remote_code": False}
    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        kwargs["device_map"] = {"": 0}
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if logger:
        logger.info("loading base model %s (4bit=%s)", base, bool(load_in_4bit and torch.cuda.is_available()))

    model = AutoModelForCausalLM.from_pretrained(str(base), **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter:
        from peft import PeftModel

        if logger:
            logger.info("attaching LoRA adapter %s", adapter)
        model = PeftModel.from_pretrained(model, str(adapter))
        info["adapter"] = str(adapter)

    model.eval()
    info.update({"base_model": str(base), "tokenizer": tokenizer_source, "load_in_4bit": bool(load_in_4bit and torch.cuda.is_available())})
    return model, tokenizer, info

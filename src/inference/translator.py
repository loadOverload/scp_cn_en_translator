"""Inference: English SCP Wikidot source -> Chinese SCP Wikidot source.

Works with the QLoRA adapter produced by ``scripts/train.py`` or with a plain
HF model (``adapter_path=None``).

The prompt is built by :func:`src.data.to_chat.build_user_prompt`, i.e. the very
same function used to build the training data, so training and inference
prompts can never drift apart.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..data.to_chat import build_user_prompt

DEFAULT_SYSTEM_PROMPT = (
    "你是 SCP 基金会中文维基（SCP-CN）的资深翻译。你负责把英文 SCP Wikidot "
    "源码翻译成符合 SCP-CN 规范的简体中文 Wikidot 源码。"
)


@dataclass
class GenerationSettings:
    max_new_tokens: int = 4096
    max_input_tokens: int = 6144
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    num_beams: int = 1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


@dataclass
class TranslationResult:
    text: str
    prompt: str
    n_input_tokens: int
    n_output_tokens: int
    truncated_input: bool
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
            "truncated_input": self.truncated_input,
            "extra": self.extra,
        }


class SCPTranslator:
    """Loaded model + tokenizer + prompt policy."""

    def __init__(
        self,
        model,
        tokenizer,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        format_cfg: Optional[Mapping[str, Any]] = None,
        generation: Optional[GenerationSettings] = None,
        device: Optional[str] = None,
        logger=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.format_cfg = dict(format_cfg or {})
        self.format_cfg.setdefault("system_prompt", system_prompt)
        self.generation = generation or GenerationSettings()
        self.logger = logger
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) if not getattr(self.model, "is_loaded_in_4bit", False) else None
        self.model.eval()

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        base_model: Optional[str] = None,
        load_in_4bit: bool = True,
        generation: Optional[GenerationSettings] = None,
        logger=None,
    ) -> "SCPTranslator":
        from .loader import load_model_and_tokenizer

        model, tokenizer, info = load_model_and_tokenizer(
            model_path=model_path,
            adapter_path=adapter_path,
            base_model=base_model,
            load_in_4bit=load_in_4bit,
            logger=logger,
        )

        fmt_cfg = dict(cfg.get("format") or {})

        translator = cls(
            model=model,
            tokenizer=tokenizer,
            system_prompt=str(fmt_cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT),
            format_cfg=fmt_cfg,
            generation=generation,
            logger=logger,
        )
        translator.info = info  # type: ignore[attr-defined]
        return translator

    # -- prompting ---------------------------------------------------------
    def build_messages(self, source: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        system = self.format_cfg.get("system_prompt")
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": build_user_prompt(source, self.format_cfg)})
        return messages

    def _render_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)

    # -- generation --------------------------------------------------------
    def translate(
        self,
        source: str,
        generation: Optional[GenerationSettings] = None,
    ) -> TranslationResult:
        return self.translate_batch([source], generation=generation)[0]

    def translate_batch(
        self,
        sources: Sequence[str],
        generation: Optional[GenerationSettings] = None,
    ) -> List[TranslationResult]:
        import torch

        gen = generation or self.generation
        results: List[TranslationResult] = []
        prompts, n_input_tokens, truncated = [], [], []

        for source in sources:
            messages = self.build_messages(source)
            prompt = self._render_prompt(messages)
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            was_truncated = False
            if len(ids) > gen.max_input_tokens:
                ids = ids[: gen.max_input_tokens]
                prompt = self.tokenizer.decode(ids, skip_special_tokens=False)
                was_truncated = True
                if self.logger:
                    self.logger.warning("input truncated to %d tokens (source had %d)", gen.max_input_tokens, len(ids))
            prompts.append(prompt)
            n_input_tokens.append(len(ids))
            truncated.append(was_truncated)

        pad_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            self.tokenizer.padding_side = pad_side
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[1])

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": bool(gen.do_sample),
            "num_beams": int(gen.num_beams),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if gen.do_sample:
            generate_kwargs.update(
                temperature=float(gen.temperature),
                top_p=float(gen.top_p),
                top_k=int(gen.top_k),
            )
        if gen.repetition_penalty and gen.repetition_penalty != 1.0:
            generate_kwargs["repetition_penalty"] = float(gen.repetition_penalty)
        if gen.no_repeat_ngram_size:
            generate_kwargs["no_repeat_ngram_size"] = int(gen.no_repeat_ngram_size)
        if gen.num_beams > 1:
            generate_kwargs["length_penalty"] = float(gen.length_penalty)

        with torch.inference_mode():
            output = self.model.generate(**encoded, **generate_kwargs)

        for i in range(len(prompts)):
            new_tokens = output[i][prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = self.postprocess(text)
            results.append(
                TranslationResult(
                    text=text,
                    prompt=prompts[i],
                    n_input_tokens=n_input_tokens[i],
                    n_output_tokens=int(new_tokens.shape[0]),
                    truncated_input=truncated[i],
                )
            )
        return results

    # -- output hygiene ----------------------------------------------------
    @staticmethod
    def postprocess(text: str) -> str:
        """Remove wrappers a chat model may add around the translation."""
        text = text.strip()
        # strip a single fenced code block
        fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        # strip a leading "翻译如下：" style preamble only when it is one line
        text = re.sub(r"^(以下是|下面是)?(翻译|译文)[^\n]{0,20}[:：]\s*\n", "", text)
        return text.strip()

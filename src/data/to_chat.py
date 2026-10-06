"""Convert cleaned pairs into Qwen chat/instruction format for SFT.

The whole Wikidot source goes in and the whole Wikidot source comes out; markup
is never removed. The output is plain JSONL with a ``messages`` list, which both
the TRL and the native trainer consume directly.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List, Mapping, Sequence

from .clean import CleanRecord

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_user_prompt(source: str, fmt_cfg: Mapping[str, Any]) -> str:
    return str(fmt_cfg.get("user_template", "{source}")).format(source=source)


def to_messages(
    source: str,
    target: str,
    fmt_cfg: Mapping[str, Any],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    system = system_prompt if system_prompt is not None else fmt_cfg.get("system_prompt")
    if system:
        messages.append({"role": SYSTEM_ROLE, "content": str(system)})
    messages.append({"role": USER_ROLE, "content": build_user_prompt(source, fmt_cfg)})
    messages.append({"role": ASSISTANT_ROLE, "content": target})
    return messages


def translation_sample(
    record: CleanRecord,
    fmt_cfg: Mapping[str, Any],
    source_field: str = "source",
    target_field: str = "target",
) -> Dict[str, Any]:
    source = record.source if source_field == "source" else record.target
    target = record.target if target_field == "target" else record.source
    return {
        "id": record.id,
        "task": "translate",
        "source": source,
        "target": target,
        "messages": to_messages(source, target, fmt_cfg),
    }


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def build_translation_dataset(
    records: Sequence[CleanRecord],
    fmt_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    return [translation_sample(r, fmt_cfg) for r in records]


def build_sft_dataset(
    records: Sequence[CleanRecord],
    fmt_cfg: Mapping[str, Any],
    seed: int = 42,
    logger=None,
) -> Dict[str, Any]:
    """Return ``{'samples': [...], 'stats': {...}}`` for one split."""
    samples = build_translation_dataset(records, fmt_cfg)
    stats = {
        "translation_samples": len(samples),
        "total": len(samples),
        "pages": len(records),
    }
    if logger:
        logger.info("chat samples: %d translation samples", len(samples))
    return {"samples": samples, "stats": stats}

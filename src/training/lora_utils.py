"""LoRA target-module discovery.

Never assume the module names of a Qwen checkpoint. Before training we walk the
*actually loaded* model, list every ``nn.Linear`` leaf, and pick the LoRA
targets from the intersection of the candidate list and reality. The result is
printed and saved to ``output_dir/lora_targets.json`` for auditing.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def linear_module_inventory(model) -> Dict[str, Any]:
    """Histogram of leaf module names for every Linear-like layer."""
    import torch.nn as nn

    histogram: Counter = Counter()
    full_names: List[str] = []
    example_by_leaf: Dict[str, str] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or (
            hasattr(module, "weight") and module.__class__.__name__ in {"Linear4bit", "Linear8bitLt", "Conv1D"}
        ):
            leaf = name.split(".")[-1]
            histogram[leaf] += 1
            full_names.append(name)
            example_by_leaf.setdefault(leaf, name)
    return {
        "n_linear_layers": len(full_names),
        "leaf_histogram": dict(histogram.most_common()),
        "example_names": example_by_leaf,
        "first_block": [n for n in full_names[:24]],
    }


def layer_block_names(model, limit: int = 4) -> List[str]:
    """Full names of the Linear layers inside the first N transformer blocks."""
    import torch.nn as nn

    names: List[str] = []
    blocks = 0
    seen_block_prefix = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prefix = name.split(".")[:-1]
            if len(prefix) >= 2:
                block_prefix = ".".join(prefix[:-1])
                if block_prefix != seen_block_prefix:
                    seen_block_prefix = block_prefix
                    blocks += 1
                    if blocks > limit:
                        break
            names.append(name)
    return names


def detect_target_modules(
    model,
    candidates: Sequence[str],
    requested: Any = "auto",
) -> Tuple[Any, Dict[str, Any]]:
    """Resolve the LoRA ``target_modules`` value against the loaded model.

    ``requested`` may be:
      * ``"auto"``        -> intersect ``candidates`` with the real module names
      * ``"all-linear"``  -> let PEFT expand every linear layer
      * a list/tuple      -> validate the names exist and use them as given
    """
    inventory = linear_module_inventory(model)
    present = set(inventory["leaf_histogram"])

    report: Dict[str, Any] = {
        "requested": requested,
        "n_linear_layers": inventory["n_linear_layers"],
        "leaf_histogram": inventory["leaf_histogram"],
        "example_names": inventory["example_names"],
    }

    if isinstance(requested, str) and requested.lower() == "all-linear":
        report["resolved"] = "all-linear"
        report["matched"] = sorted(present)
        report["missing"] = []
        return "all-linear", report

    if isinstance(requested, (list, tuple)) and requested:
        wanted = [str(m) for m in requested]
        missing = [m for m in wanted if m not in present]
        matched = [m for m in wanted if m in present]
        if not matched:
            raise ValueError(
                f"none of the configured lora.target_modules exist in the model: {wanted}. "
                f"available leaf modules: {sorted(present)}"
            )
        report.update({"resolved": matched, "matched": matched, "missing": missing})
        return matched, report

    # auto
    matched = [c for c in candidates if c in present]
    missing = [c for c in candidates if c not in present]
    if not matched:
        # last resort: the most duplicated linear leaf, excluding the LM head
        ordered = [leaf for leaf, _ in inventory["leaf_histogram"].items() if leaf not in {"lm_head", "score"}]
        if not ordered:
            raise ValueError("could not find any linear layer to attach LoRA to")
        matched = ordered[:1]
        report["fallback_used"] = matched
    report.update({"resolved": matched, "matched": matched, "missing": missing})
    return matched, report


def format_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"linear layers in model : {report.get('n_linear_layers')}",
        f"leaf module histogram  : {report.get('leaf_histogram')}",
        f"requested target_modules: {report.get('requested')}",
        f"resolved target_modules : {report.get('resolved')}",
    ]
    if report.get("missing"):
        lines.append(f"candidates not in model: {report['missing']} (ignored)")
    if report.get("fallback_used"):
        lines.append(f"WARNING: no candidate matched, fell back to {report['fallback_used']}")
    examples = report.get("example_names") or {}
    if examples:
        shown = ", ".join(f"{k} -> {v}" for k, v in list(examples.items())[:8])
        lines.append(f"example full names     : {shown}")
    return "\n".join(lines)


def trainable_parameter_report(model) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / max(total, 1), 4),
    }

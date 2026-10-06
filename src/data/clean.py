"""Conservative cleaning / quality filtering for SCP page pairs.

Design rules
------------
* the raw files are NEVER modified; cleaned output goes to ``data/cleaned/``
* Wikidot markup is NEVER stripped -- only whitespace/BOM normalisation
* every rejection is recorded with a machine-readable reason so the filter
  thresholds can be tuned later without re-deriving anything
* suspicious-but-usable samples are *flagged*, not dropped
  (``drop_on_ratio_extreme: false`` by default)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CleanRecord:
    id: str
    source: str
    target: str
    source_len: int
    target_len: int
    length_ratio: float
    source_plain_len: int
    target_plain_len: int
    source_cjk_ratio: float
    target_cjk_ratio: float
    source_latin_ratio: float
    target_latin_ratio: float
    source_sha1: str
    target_sha1: str
    flags: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    source_struct: Dict[str, Any] = field(default_factory=dict)
    target_struct: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "CleanRecord":
        """Rebuild from a ``pairs.jsonl`` row (so cleaning need not be redone)."""
        meta = row.get("meta") or {}
        source = row.get("source") or ""
        target = row.get("target") or ""
        return cls(
            id=str(row.get("id", "")),
            source=source,
            target=target,
            source_len=int(meta.get("source_len", len(source))),
            target_len=int(meta.get("target_len", len(target))),
            length_ratio=float(meta.get("length_ratio", len(target) / max(len(source), 1))),
            source_plain_len=int(meta.get("source_plain_len", 0)),
            target_plain_len=int(meta.get("target_plain_len", 0)),
            source_cjk_ratio=float(meta.get("source_cjk_ratio", 0.0)),
            target_cjk_ratio=float(meta.get("target_cjk_ratio", 0.0)),
            source_latin_ratio=float(meta.get("source_latin_ratio", 0.0)),
            target_latin_ratio=float(meta.get("target_latin_ratio", 0.0)),
            source_sha1=str(meta.get("source_sha1") or ""),
            target_sha1=str(meta.get("target_sha1") or ""),
            flags=list(meta.get("flags") or []),
            source_path=meta.get("source_path"),
            target_path=meta.get("target_path"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "meta": {
                "source_len": self.source_len,
                "target_len": self.target_len,
                "length_ratio": round(self.length_ratio, 4),
                "source_plain_len": self.source_plain_len,
                "target_plain_len": self.target_plain_len,
                "source_cjk_ratio": round(self.source_cjk_ratio, 4),
                "target_cjk_ratio": round(self.target_cjk_ratio, 4),
                "source_latin_ratio": round(self.source_latin_ratio, 4),
                "target_latin_ratio": round(self.target_latin_ratio, 4),
                "source_sha1": self.source_sha1,
                "target_sha1": self.target_sha1,
                "flags": self.flags,
                "source_path": self.source_path,
                "target_path": self.target_path,
                "source_struct": self.source_struct,
                "target_struct": self.target_struct,
            },
        }




# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_BOM = "\ufeff"


def normalize_text(text: str, cfg: Mapping[str, Any]) -> str:
    if text is None:
        return ""
    if cfg.get("strip_bom", True):
        text = text.lstrip(_BOM)
    if cfg.get("normalize_line_endings", True):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if cfg.get("strip_outer_whitespace", True):
        text = text.strip("\n \t")
    return text





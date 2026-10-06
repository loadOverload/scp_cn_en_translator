"""Wikidot syntax preservation checks.

The task is defined as "convert the prose between the two languages while every
marker, component name, parameter name, code block, link target and URL stays
byte-identical". A translation metric cannot see that: a fluent Chinese version
of a page whose ``[[include component:image-block name=...]]`` lost its ``name=``
parameter scores well on BLEU and renders as a broken page.

This module compares the Wikidot constructs of a source text with those of a
generated text and reports how many survived. It is deliberately structural
rather than a full parser: it counts constructs of each kind and checks that the
hypothesis still contains them, which is what "preserved" means here.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ``[[include component:image-block | caption=...]]`` / ``[[module Rate]]`` /
# ``[[div_ style=...]]`` / ``[[/=]]`` -- capture the keyword after the brackets.
TAG_RE = re.compile(r"\[\[\s*/?\s*([A-Za-z_][A-Za-z0-9_:.\-]*)")
URL_RE = re.compile(r"https?://[^\s\]\|<>\"']+")
# ``name=value`` inside a ``[[...]]`` construct.
PARAM_RE = re.compile(r"\|?\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*=")
CODE_RE = re.compile(r"\{\{\{.*?\}\}\}", re.S)
HEADING_RE = re.compile(r"^\s*(\+{1,6})\s", re.M)

# Inline formatting that must survive as a pair, not necessarily in order.
INLINE_MARKERS = ("**", "//", "__", "--", "##")

INCLUDE_RE = re.compile(r"\[\[\s*include\s+([^\]\|]+)", re.I)
MODULE_RE = re.compile(r"\[\[\s*module\s+([^\]\|]+)", re.I)
TAG_WORD_RE = re.compile(r"\[\[\s*(/?)\s*([A-Za-z_][\w:-]*)")
CLOSE_RE = re.compile(r"\[\[\s*/\s*([A-Za-z_][\w:-]*)\s*\]\]")
PARAM_EQ_RE = re.compile(r"([^\s\|\]=]+)\s*=")
CODE_TAGGED_RE = re.compile(r"\[\[\s*code[^\]]*\]\](.*?)\[\[\s*/\s*code\s*\]\]", re.S)
CODE_BRACES_RE = re.compile(r"\{\{\{(.*?)\}\}\}", re.S)

# Wikidot tags that must be explicitly closed. Container-only tags such as
# ``include`` / ``module`` / ``image`` / ``footnoteblock`` have no closing form
# and must not be counted here.
PAIRED_TAGS = frozenset({
    "div", "span", "collapsible", "tabview", "tab", "footnote", "code", "size",
    "quote", "html", "raw", "table", "row", "cell", "li", "ul", "ol", "align",
    "center", "bold", "italic", "underline", "strikethrough", "sub", "sup",
    "color", "hidden", "iframe", "math", "module641", "bibcite", "toc",
})


def _canon_include(target: str) -> str:
    """Collapse the EN/CN component variants so localisation is not a loss.

    The Chinese branch rewrites ``:scp-wiki:component:foo`` to
    ``:scp-wiki-cn:component:foo`` and drops variant suffixes such as
    ``-source`` / ``-standalone``. Comparing raw targets would report every one
    of those correct rewrites as a missing include.
    """
    key = target.strip().lower()
    key = re.sub(r"^:?scp-wiki(?:-cn)?(?:-[a-z]+)?:", "", key)
    key = re.sub(r"-(?:standalone|source|dev|base|cn|en)$", "", key)
    return key


def _protected_tokens(text: str) -> List[Tuple[str, str]]:
    """The parts of a ``[[...]]`` construct that must stay ASCII.

    Only the tag keyword, the include/module target and the parameter *names*
    are checked. Parameter **values** are prose and are expected to be Chinese
    (``|caption=收容区域内的SCP-002``), so they are deliberately excluded --
    flagging those would make the check useless on real pages.
    """
    out: List[Tuple[str, str]] = []
    for match in TAG_WORD_RE.finditer(text):
        out.append(("tag", match.group(2)))
    for match in INCLUDE_RE.finditer(text):
        out.append(("include", match.group(1).strip()))
    for match in MODULE_RE.finditer(text):
        out.append(("module", match.group(1).strip()))
    for construct in re.findall(r"\[\[.*?\]\]", text, flags=re.S):
        for match in PARAM_EQ_RE.finditer(construct):
            out.append(("param", match.group(1)))
    return out


@dataclass
class Outcome:
    """One kind of construct: how many were present and how many survived."""

    checked: int = 0
    preserved: int = 0

    @property
    def rate(self) -> float:
        return self.preserved / self.checked if self.checked else 1.0


@dataclass
class Issue:
    code: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        """Rendered form used by :mod:`src.evaluation.metrics`."""
        return {"code": self.code, "detail": self.detail}


@dataclass
class Report:
    score: float = 1.0
    outcomes: Dict[str, Outcome] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    source_len: int = 0
    hypothesis_len: int = 0

    def error_count(self) -> int:
        return len(self.issues)

    @property
    def ok(self) -> bool:
        """True when nothing was lost: the CLI uses this as its exit code."""
        return not self.issues and self.score >= 1.0

    def warning_count(self) -> int:
        """Issues that are worth reporting but do not mean data was lost.

        Currently empty: everything this module detects is a real absence. It
        exists because the CLI prints it and because a future check (for example
        "the model added a construct the source did not have") should not be
        counted as an error.
        """
        return 0

    def summary(self) -> str:
        """One-line-per-check rendering, for the CLI."""
        lines = [f"保留率 {self.score:.4f}   错误 {self.error_count()} 处"]
        for name, outcome in sorted(self.outcomes.items()):
            if outcome.checked:
                flag = "" if outcome.rate >= 1.0 else "   <-- 有丢失"
                lines.append(f"  {name:<8} {outcome.preserved}/{outcome.checked}"
                             f"  ({outcome.rate:.4f}){flag}")
        if self.issues:
            lines.append("  问题:")
            for issue in self.issues[:10]:
                lines.append(f"    {issue.code}: {issue.detail}")
            if len(self.issues) > 10:
                lines.append(f"    ... 其余 {len(self.issues) - 10} 处")
        return "\n".join(lines)


def plain_text(text: str) -> str:
    """Strip Wikidot structure, leaving the prose.

    Length ratios have to be computed on the prose: markup is Latin script, so a
    short Chinese page wrapped in a long ``[[include]]`` block would otherwise
    look like a catastrophic under-translation.
    """
    if not text:
        return ""
    out = CODE_RE.sub(" ", text)
    out = re.sub(r"\[\[/?[^\]]*\]\]", " ", out)          # all [[...]] constructs
    out = URL_RE.sub(" ", out)
    out = HEADING_RE.sub(" ", out)
    out = re.sub(r"^\s*[>#*]\s+", " ", out, flags=re.M)   # quotes / lists
    for marker in INLINE_MARKERS:
        out = out.replace(marker, "")
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()


# CJK ideographs (incl. extension A and compatibility) plus CJK punctuation.
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f]")


def cjk_ratio(text: str) -> float:
    """Fraction of characters that are CJK.

    Catches the failure where the model answers in English, or leaves the source
    untouched, while still looking plausible by length alone.
    """
    if not text:
        return 0.0
    return len(CJK_RE.findall(text)) / len(text)


def _multiset(text: str, pattern: re.Pattern) -> Counter:
    return Counter(m.group(0) if pattern.groups == 0 else m.group(1).lower()
                   for m in pattern.finditer(text))


def _tag_names(text: str) -> Counter:
    return Counter(m.group(1).lower().rstrip("_") for m in TAG_RE.finditer(text))


# The Chinese branch runs on its own domain, and the reference translations
# rewrite links accordingly (scp-wiki.wikidot.com -> scp-wiki-cn.wikidot.com).
# Comparing raw URLs therefore reports a correct localisation as a loss, so the
# known wiki hosts collapse to one token before comparing.
_WIKI_HOSTS = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:scp-wiki-cn\.wikidot\.com|scp-wiki\.wikidot\.com"
    r"|scpwiki\.com|scp-wiki\.net)(?P<rest>/.*)?$", re.I)


def _canonical_url(url: str) -> str:
    match = _WIKI_HOSTS.match(url)
    if match:
        return "<wiki>" + (match.group("rest") or "")
    return url


def _urls(text: str) -> Counter:
    # trailing punctuation is not part of the target
    return Counter(_canonical_url(m.group(0).rstrip(".,;:)")) for m in URL_RE.finditer(text))


def _params(text: str) -> Counter:
    # only parameters inside a [[...]] construct count
    names: Counter = Counter()
    for construct in re.findall(r"\[\[.*?\]\]", text, flags=re.S):
        names.update(m.group(1).lower() for m in PARAM_RE.finditer(construct))
    return names


def _inline_counts(text: str) -> Dict[str, int]:
    return {marker: text.count(marker) for marker in INLINE_MARKERS}


def validate_pair(source: str, hypothesis: str, cfg: Optional[Mapping[str, Any]] = None) -> Report:
    """Compare the Wikidot constructs of ``source`` and ``hypothesis``.

    Ordering is irrelevant; what matters is that nothing went missing. Extra
    constructs in the hypothesis are not penalised here (a checked code block may
    legitimately be added), but they are reported as issues because in practice
    they usually mean the model duplicated a block.
    """
    source = source or ""
    hypothesis = hypothesis or ""
    outcomes: Dict[str, Outcome] = {}
    issues: List[Issue] = []

    checks: Sequence[Tuple[str, Counter, Counter]] = (
        ("tag", _tag_names(source), _tag_names(hypothesis)),
        ("url", _urls(source), _urls(hypothesis)),
        ("param", _params(source), _params(hypothesis)),
    )
    for name, want, got in checks:
        outcome = Outcome()
        for key, count in want.items():
            outcome.checked += count
            outcome.preserved += min(count, got.get(key, 0))
            if got.get(key, 0) < count:
                issues.append(Issue(f"{name}_missing",
                                    f"{key!r} {count}->{got.get(key, 0)}"))
        outcomes[name] = outcome

    # Code blocks and headings are counted rather than matched: their bodies are
    # usually left untranslated, so a count is the meaningful check.
    for name, pattern in (("code", CODE_RE), ("heading", HEADING_RE)):
        want = len(pattern.findall(source))
        got = len(pattern.findall(hypothesis))
        outcomes[name] = Outcome(checked=want, preserved=min(want, got))
        if got < want:
            issues.append(Issue(f"{name}_missing", f"{want}->{got}"))

    # Inline markers must pair up: an odd count means an unclosed **bold**.
    src_inline, hyp_inline = _inline_counts(source), _inline_counts(hypothesis)
    inline = Outcome()
    for marker in INLINE_MARKERS:
        need = src_inline[marker] // 2
        if not need:
            continue
        inline.checked += need
        inline.preserved += min(need, hyp_inline[marker] // 2)
        if hyp_inline[marker] // 2 < need:
            issues.append(Issue("inline_missing", f"{marker} {need}->{hyp_inline[marker] // 2}"))
        if hyp_inline[marker] % 2:
            issues.append(Issue("inline_unbalanced", f"{marker} count={hyp_inline[marker]}"))
    outcomes["inline"] = inline

    # ---- include targets, with EN/CN variants collapsed ------------------
    # A raw comparison would report every correct ``:scp-wiki:`` ->
    # ``:scp-wiki-cn:`` rewrite as a loss, which is how an earlier version of
    # this module produced false positives on 17 of 18 samples.
    want_inc = Counter(_canon_include(t) for t in INCLUDE_RE.findall(source))
    got_inc = Counter(_canon_include(t) for t in INCLUDE_RE.findall(hypothesis))
    include = Outcome()
    for key, count in want_inc.items():
        include.checked += count
        include.preserved += min(count, got_inc.get(key, 0))
        if got_inc.get(key, 0) < count:
            issues.append(Issue("include_missing", f"{key!r} {count}->{got_inc.get(key, 0)}"))
    for key, count in got_inc.items():
        if got_inc.get(key, 0) > want_inc.get(key, 0):
            issues.append(Issue("include_added", f"{key!r} ->{count}"))
    outcomes["include"] = include

    # Module names get their own code: ``[[module Rate]]`` is legitimately
    # absent from some SCP-CN pages, so a caller may want to treat these
    # differently from a lost ``[[include]]``.
    want_mod = Counter(m.strip().lower() for m in MODULE_RE.findall(source))
    got_mod = Counter(m.strip().lower() for m in MODULE_RE.findall(hypothesis))
    for key, count in want_mod.items():
        if got_mod.get(key, 0) < count:
            issues.append(Issue("module_missing", f"{key!r} {count}->{got_mod.get(key, 0)}"))
    for key, count in got_mod.items():
        if want_mod.get(key, 0) < count:
            issues.append(Issue("module_added", f"{key!r} ->{count}"))

    # ---- protected regions must stay ASCII --------------------------------
    # Translating a tag name (``[[footnote]]`` -> ``[[脚注]]``) or a component
    # target is the single most destructive failure this task has: the page
    # stops rendering. Parameter values are excluded on purpose -- those are
    # prose and are supposed to become Chinese.
    protected = Outcome()
    got_tokens = Counter(_protected_tokens(hypothesis))
    seen: Counter = Counter()
    for kind, token in _protected_tokens(source):
        seen[(kind, token)] += 1
        protected.checked += 1
        if CJK_RE.search(token):
            protected.preserved += 1          # the source itself is suspect
            continue
        if got_tokens[(kind, token)] >= seen[(kind, token)]:
            protected.preserved += 1
        else:
            issues.append(Issue("cjk_in_protected", f"{kind} {token!r} not kept ASCII"))
    outcomes["protected"] = protected

    # ---- open/close balance ----------------------------------------------
    src_open = Counter(m.group(2).lower() for m in TAG_WORD_RE.finditer(source) if m.group(1) != "/")
    hyp_open = Counter(m.group(2).lower() for m in TAG_WORD_RE.finditer(hypothesis) if m.group(1) != "/")
    src_close = Counter(m.group(1).lower() for m in CLOSE_RE.finditer(source))
    hyp_close = Counter(m.group(1).lower() for m in CLOSE_RE.finditer(hypothesis))
    balance = Outcome()
    for tag in sorted(PAIRED_TAGS):
        want = src_open[tag]
        if not want:
            continue
        balance.checked += want
        # A tag counts as balanced only when the hypothesis closes as many as it
        # opens, which is stricter than "as many as the source had".
        balance.preserved += min(want, hyp_open[tag], hyp_close[tag])
        if hyp_close[tag] < hyp_open[tag]:
            issues.append(Issue("unclosed_wikidot",
                                f"[[{tag}]] opened {hyp_open[tag]}, closed {hyp_close[tag]}"))
    for tag in sorted(PAIRED_TAGS):
        if src_close[tag] and not hyp_close[tag]:
            issues.append(Issue("unclosed_wikidot", f"[[/{tag}]] lost"))
    outcomes["balance"] = balance

    # ---- code block bodies must not be translated -------------------------
    def _code_bodies(text: str) -> Counter:
        bodies = CODE_TAGGED_RE.findall(text) + CODE_BRACES_RE.findall(text)
        return Counter(re.sub(r"\s+", " ", b).strip() for b in bodies if b.strip())

    want_code = _code_bodies(source)
    got_code = _code_bodies(hypothesis)
    code = Outcome()
    for body, count in want_code.items():
        code.checked += count
        code.preserved += min(count, got_code.get(body, 0))
        if got_code.get(body, 0) < count:
            issues.append(Issue("code_block_modified", f"{body[:40]!r} changed or lost"))
    outcomes["code_body"] = code

    total = sum(o.checked for o in outcomes.values())
    kept = sum(o.preserved for o in outcomes.values())
    score = kept / total if total else 1.0
    return Report(score=round(score, 4), outcomes=outcomes, issues=issues,
                  source_len=len(source), hypothesis_len=len(hypothesis))

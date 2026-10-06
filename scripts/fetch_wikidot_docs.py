#!/usr/bin/env python3
"""Download the official Wikidot "Wiki Syntax" documentation and convert it to text.

The documentation is only available as HTML, and the HTML mixes the wiki source
examples (inside ``<tt><span style="white-space: pre-wrap">``) with the rendered
result.  A parser author needs the *source* side intact, so this script keeps
whitespace inside pre-formatted elements and converts everything else to plain
markdown.

Output: ``docs/wikidot-syntax/raw/<slug>.md`` — one file per documentation page,
with the source URL in the header.  Re-running is safe and idempotent.

Usage:
    python3 scripts/fetch_wikidot_docs.py                 # fetch everything
    python3 scripts/fetch_wikidot_docs.py --only links tables
    python3 scripts/fetch_wikidot_docs.py --out docs/wikidot-syntax/raw
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

SYNTAX_PAGES = [
    "inline-formatting", "text-size", "paragraphs-and-newline", "typography",
    "literal-text", "universal-escaping", "comments", "headings",
    "table-of-contents", "horizontal-rules", "lists", "definition-lists",
    "block-quotes", "collapsible-blocks", "foldable-list", "links", "images",
    "notes", "html-blocks", "code-blocks", "tables", "block-formatting-elements",
    "math", "footnotes", "bibliography", "date", "include", "embedding",
    "embedding-code", "iftags", "attachment", "users", "social-bookmarking",
    "buttons", "tag-buttons", "layout",
]

# Pages outside the doc-wiki-syntax namespace that a parser still needs.
EXTRA_PAGES = {
    "doc__quick-reference": "https://www.wikidot.com/doc:quick-reference",
    "doc-modules__start": "https://www.wikidot.com/doc-modules:start",
    "doc__embedding": "https://www.wikidot.com/doc:embedding",
    "doc__templates": "https://www.wikidot.com/doc:templates",
}

PAGES: Dict[str, str] = {s: f"https://www.wikidot.com/doc-wiki-syntax:{s}" for s in SYNTAX_PAGES}
PAGES.update(EXTRA_PAGES)

USER_AGENT = "Mozilla/5.0 (compatible; wikidot-syntax-corpus/1.0; documentation archival)"

# --------------------------------------------------------------------------
# Minimal DOM
# --------------------------------------------------------------------------

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
PREFORMATTED = {"pre", "code", "tt"}


class Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str, attrs: Dict[str, str]):
        self.tag = tag
        self.attrs = attrs
        self.children: List[Union["Node", str]] = []

    def attr(self, name: str) -> str:
        return self.attrs.get(name, "")

    def find(self, tag: str) -> Optional["Node"]:
        if self.tag == tag:
            return self
        for child in self.children:
            if isinstance(child, Node):
                got = child.find(tag)
                if got is not None:
                    return got
        return None


class TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", {})
        self.stack: List[Node] = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k: (v or "") for k, v in attrs})
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, {k: (v or "") for k, v in attrs}))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------

BLOCK_TAGS = {"p", "div", "table", "tr", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "hr", "br"}


def normalize(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text.replace("\n", " "))


def render_inline(node: Node) -> str:
    """Text of a node with markup kept as plain text (used for table cells)."""
    out: List[str] = []

    def walk(n: Union[Node, str], pre: bool) -> None:
        if isinstance(n, str):
            out.append(n if pre else normalize(n))
            return
        pre = pre or n.tag in PREFORMATTED or "pre-wrap" in n.attr("style") or "pre" in n.attr("class")
        if n.tag == "br":
            out.append("\n")
            return
        if n.tag == "hr":
            out.append("\n----\n")
            return
        if n.tag == "a":
            inner = "".join(render_inline_children(n, pre))
            href = n.attr("href")
            out.append(f"[{inner}](https://www.wikidot.com{href})" if href.startswith("/") else (f"[{inner}]({href})" if href else inner))
            return
        for child in n.children:
            walk(child, pre)

    walk(node, False)
    return "".join(out)


def render_inline_children(node: Node, pre: bool) -> List[str]:
    parts: List[str] = []
    for child in node.children:
        parts.append(render_inline(child) if isinstance(child, Node) else (child if pre else normalize(child)))
    return parts


def cell_text(node: Node) -> str:
    text = render_inline(node)
    text = text.strip()
    # Newlines inside a markdown table cell would break the table; make them explicit.
    text = text.replace("\r", "")
    text = "\\n".join(re.split(r"\n\s*", text))
    text = text.replace("|", "\\|")
    return text


def render_block(node: Node, out: List[str]) -> None:
    tag = node.tag
    if tag in ("script", "style", "noscript"):
        return
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(tag[1])
        out.append("\n" + "#" * level + " " + render_inline(node).strip() + "\n")
        return
    if tag == "p":
        out.append("\n" + render_inline(node).strip() + "\n")
        return
    if tag == "hr":
        out.append("\n----\n")
        return
    if tag in ("pre", "code") and node.tag == "pre":
        body = raw_text(node).rstrip("\n")
        fence = fence_for(body)
        out.append("\n" + fence + "\n" + body + "\n" + fence + "\n")
        return
    if tag == "div" and "code" in node.attr("class").split():
        pre = node.find("pre")
        body = raw_text(pre if pre is not None else node).rstrip("\n")
        fence = fence_for(body)
        out.append("\n" + fence + "\n" + body + "\n" + fence + "\n")
        return
    if tag == "table":
        render_table(node, out)
        return
    if tag in ("ul", "ol"):
        for li in node.children:
            if isinstance(li, Node) and li.tag == "li":
                out.append("- " + render_inline(li).strip())
        out.append("")
        return
    if tag == "blockquote":
        for line in render_inline(node).strip().split("\n"):
            out.append("> " + line)
        out.append("")
        return
    if tag == "br":
        out.append("\n")
        return
    for child in node.children:
        if isinstance(child, Node):
            render_block(child, out)
        elif child.strip():
            out.append(normalize(child).strip())


def fence_for(body: str) -> str:
    """A tilde fence longer than any tilde run in the body, so it cannot be closed early."""
    runs = [len(m.group(0)) for m in re.finditer(r"~{3,}", body)]
    return "~" * (max(3, max(runs) + 1) if runs else 3)


def raw_text(node: Optional[Node]) -> str:
    """Verbatim text, preserving whitespace exactly (for source examples)."""
    if node is None:
        return ""
    parts: List[str] = []

    def walk(n: Union[Node, str]) -> None:
        if isinstance(n, str):
            parts.append(n)
            return
        if n.tag == "br":
            parts.append("\n")
            return
        for child in n.children:
            walk(child)

    walk(node)
    return "".join(parts)


def render_table(node: Node, out: List[str]) -> None:
    rows: List[List[str]] = []
    for tr in node.children:
        if not isinstance(tr, Node) or tr.tag != "tr":
            continue
        cells = [cell_text(c) for c in tr.children if isinstance(c, Node) and c.tag in ("td", "th")]
        if cells:
            rows.append(cells)
    if not rows:
        return
    width = max(len(r) for r in rows)
    out.append("")
    for i, row in enumerate(rows):
        out.append("| " + " | ".join(row + [""] * (width - len(row))) + " |")
        if i == 0:
            out.append("|" + " --- |" * width)
    out.append("")


def convert(html_text: str) -> str:
    parser = TreeBuilder()
    parser.feed(html_text)
    parser.close()

    content = None
    for node in parser.root.children:
        if isinstance(node, Node):
            found = _find_by_id(node, "page-content")
            if found is not None:
                content = found
                break
    if content is None:
        content = parser.root.find("body") or parser.root

    out: List[str] = []
    for child in content.children:
        if isinstance(child, Node):
            render_block(child, out)
        elif child.strip():
            out.append(normalize(child).strip())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _find_by_id(node: Node, wanted: str) -> Optional[Node]:
    if node.attr("id") == wanted:
        return node
    for child in node.children:
        if isinstance(child, Node):
            got = _find_by_id(child, wanted)
            if got is not None:
                return got
    return None


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def fetch(url: str, retries: int = 3) -> str:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/wikidot-syntax/raw")
    ap.add_argument("--only", nargs="*", default=None, help="subset of slugs")
    ap.add_argument("--sleep", type=float, default=0.4, help="delay between requests")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    slugs = args.only or list(PAGES)
    failures: List[Tuple[str, str]] = []
    for slug in slugs:
        url = PAGES[slug]
        dest = out_dir / f"{slug}.md"
        try:
            body = convert(fetch(url))
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((slug, str(exc)))
            print(f"FAIL {slug}: {exc}", file=sys.stderr)
            continue
        header = (
            f"<!-- source: {url} -->\n"
            f"<!-- retrieved by scripts/fetch_wikidot_docs.py; "
            f"source-code examples preserve whitespace, table cells encode newlines as \\n -->\n\n"
        )
        dest.write_text(header + body, encoding="utf-8")
        print(f"ok   {slug}  ({len(body):,} chars)")
        time.sleep(args.sleep)

    print(f"\nwrote {len(slugs) - len(failures)}/{len(slugs)} pages to {out_dir}", file=sys.stderr)
    if failures:
        print("failures: " + ", ".join(s for s, _ in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

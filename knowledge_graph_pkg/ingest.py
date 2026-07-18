"""
Document ingestion: turn files of various formats into plain text.

``load_text(path)`` dispatches on file extension:

* ``.txt`` -- read as-is.
* ``.md`` / ``.markdown`` -- strip light Markdown markup.
* ``.html`` / ``.htm`` -- extract readable body text, including
  Substack-style posts whose body lives in an embedded ``body_html`` JSON
  blob (``window._preloads``).
* ``.pdf`` -- extract text via the optional ``pymupdf`` extra
  (``pip install knowledgereduce[pdf]``).

The HTML and Markdown converters are pure-Python (stdlib only), keeping
the core dependency-free. PDF is the only format that needs an extra.
"""

import html as _html
import json
import os
import re
from typing import List


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def markdown_to_text(md: str) -> str:
    """Strip common Markdown markup, returning readable plain text."""
    text = md
    # Fenced code blocks -> keep inner text, drop the fences.
    text = re.sub(r"```[a-zA-Z0-9]*\n?", "", text)
    # Images ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Headings ###, blockquotes >, list bullets -, *, +
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}\d+\.\s+", "", text)
    # Emphasis / bold / inline code markers
    text = re.sub(r"[*_`~]", "", text)
    # Collapse excess blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _strip_tags(fragment: str) -> str:
    """Convert an HTML fragment to text, separating block elements."""
    # Drop script/style entirely.
    fragment = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", fragment)
    # Block-level close tags / <br> become newlines.
    fragment = re.sub(r"(?i)</(p|div|li|h[1-6]|blockquote|tr|section|article)\s*>", "\n", fragment)
    fragment = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    # Drop all remaining tags.
    text = re.sub(r"<[^>]+>", "", fragment)
    text = _html.unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _extract_substack_body(raw: str) -> str:
    """Return the largest ``body_html`` string from a window._preloads blob.

    Substack pages render the post body client-side; the source HTML carries
    it as an escaped JSON string assigned to ``window._preloads``. Returns an
    empty string when no such blob is present.
    """
    m = re.search(r"window\._preloads\s*=\s*JSON\.parse\(", raw)
    if not m:
        return ""
    i = m.end()
    if i >= len(raw) or raw[i] != '"':
        return ""
    # Walk the double-quoted JS string literal, respecting escapes.
    j = i + 1
    buf = []
    while j < len(raw):
        c = raw[j]
        if c == "\\":
            buf.append(raw[j:j + 2])
            j += 2
            continue
        if c == '"':
            break
        buf.append(c)
        j += 1
    try:
        inner = json.loads('"' + "".join(buf) + '"')  # JS string -> JSON text
        data = json.loads(inner)                       # JSON text -> object
    except (ValueError, json.JSONDecodeError):
        return ""

    bodies: List[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "body_html" and isinstance(v, str):
                    bodies.append(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    if not bodies:
        return ""
    return max(bodies, key=len)


def html_to_text(html_str: str) -> str:
    """Extract readable text from an HTML document.

    Prefers a Substack ``body_html`` blob when present (it holds the real
    article text); otherwise falls back to stripping tags from the whole
    document body.
    """
    body = _extract_substack_body(html_str)
    if body:
        return _strip_tags(body)

    # Fall back to the <body>...</body> region, else the whole document.
    m = re.search(r"(?is)<body[^>]*>(.*?)</body>", html_str)
    region = m.group(1) if m else html_str
    return _strip_tags(region)


# --------------------------------------------------------------------------- #
# PDF (optional)
# --------------------------------------------------------------------------- #
def pdf_to_text(path: str) -> str:
    """Extract text from a PDF using pymupdf (optional ``[pdf]`` extra)."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "PDF ingestion requires the 'pdf' extra: pip install knowledgereduce[pdf]"
        ) from exc
    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts).strip()


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def load_text(path: str, encoding: str = "utf-8") -> str:
    """Load a document of any supported format as plain text.

    Dispatches on file extension. Raises FileNotFoundError if the path does
    not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return pdf_to_text(path)

    with open(path, "r", encoding=encoding, errors="ignore") as fh:
        raw = fh.read()

    if ext in (".html", ".htm"):
        return html_to_text(raw)
    if ext in (".md", ".markdown"):
        return markdown_to_text(raw)
    # Default: treat as plain text.
    return raw

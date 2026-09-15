"""Strip Project Gutenberg license boilerplate, keep the actual book body."""
from __future__ import annotations

import re

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)


def strip_boilerplate(raw_text: str) -> str:
    text = raw_text
    start_match = START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
    end_match = END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]
    return text.strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse runs of 3+ blank lines to a paragraph boundary marker
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

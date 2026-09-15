"""Structure-aware chunking: detect chapter/section headings from plain-text
layout, then pack paragraphs into overlapping word-budget chunks within each
section. This is what "graph structure based on the structure of the text"
means concretely: Book -HAS_SECTION-> Section -HAS_CHUNK-> Chunk, plus
Chunk -NEXT_CHUNK-> Chunk for reading order.
"""
from __future__ import annotations

import re

import config
from graphbuild.node_schema import BookNode, ChunkNode, SectionNode, new_id

HEADING_RE = re.compile(
    r"^\s*(CHAPTER|PART|SECTION|BOOK)\s+[IVXLCDM\d]+\.?\s*[:.\-]?\s*(.*)$",
    re.IGNORECASE,
)
ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,'\-:]{3,70}$")


def _looks_like_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 90:
        return None
    m = HEADING_RE.match(line)
    if m:
        title = m.group(0).strip()
        return title
    if ALLCAPS_HEADING_RE.match(line) and len(line.split()) <= 12:
        return line
    return None


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Returns [(heading_or_default, section_body_text), ...]."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_title = "Full Text"
    current_body: list[str] = []
    found_any_heading = False
    for line in lines:
        heading = _looks_like_heading(line)
        if heading:
            if current_body:
                sections.append((current_title, current_body))
            current_title = heading
            current_body = []
            found_any_heading = True
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_title, current_body))
    if not found_any_heading:
        return [("Full Text", text)]
    return [(title, "\n".join(body)) for title, body in sections]


def _pack_paragraphs(paragraphs: list[str], target_words: int, overlap_words: int) -> list[str]:
    chunks: list[str] = []
    current_words: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        current_words.extend(para.split())
        if len(current_words) >= target_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_words:] if overlap_words else []
    if current_words and (not chunks or len(current_words) > overlap_words):
        chunks.append(" ".join(current_words))
    return chunks


def chunk_book(text: str, book: BookNode) -> tuple[list[SectionNode], list[ChunkNode]]:
    section_specs = split_into_sections(text)
    sections: list[SectionNode] = []
    chunks: list[ChunkNode] = []
    prev_chunk_id: str | None = None
    order_index = 0

    for sec_idx, (title, body) in enumerate(section_specs):
        section = SectionNode(
            id=new_id("sec"),
            type=None,  # set in __post_init__
            label=title[:120],
            topic=book.topic,
            book_id=book.id,
            level=1,
            order_index=sec_idx,
            created_by="pipeline",
        )
        sections.append(section)

        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        packed = _pack_paragraphs(paragraphs, config.CHUNK_TARGET_WORDS, config.CHUNK_OVERLAP_WORDS)
        for chunk_text in packed:
            if len(chunk_text.split()) < 15:
                continue
            chunk = ChunkNode(
                id=new_id("chunk"),
                type=None,
                label=(chunk_text[:80] + "...") if len(chunk_text) > 80 else chunk_text,
                topic=book.topic,
                book_id=book.id,
                section_id=section.id,
                text=chunk_text,
                order_index=order_index,
                prev_chunk_id=prev_chunk_id,
                created_by="pipeline",
            )
            if prev_chunk_id and chunks:
                chunks[-1].next_chunk_id = chunk.id
            chunks.append(chunk)
            prev_chunk_id = chunk.id
            order_index += 1

    return sections, chunks

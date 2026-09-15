"""End-to-end ingestion orchestration: fetch -> clean -> structure-aware
chunk -> corpus-wide TF-IDF fit -> dense embed -> write Neo4j + Qdrant ->
statistical similarity pass -> LLM connection-finder pass.

This is intentionally a single-shot batch pipeline (see README): each run
refits TF-IDF over the *entire* accumulated corpus so IDF stays globally
consistent, rather than trying to incrementally patch sparse vectors.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import config
from graph_ops.edge_ops import run_connection_finder_pass
from graphbuild.embeddings import embed_texts
from graphbuild.node_schema import BookNode, ChunkNode, EdgeMethod, RelationType, SectionNode, make_edge, new_id
from graphbuild.sparse import SparseIndex, SparseVector
from ingestion import gutenberg_fetch, text_cleaner
from ingestion.chunker import chunk_book
from store import neo4j_store, qdrant_store


@dataclass
class IngestReport:
    books: list[dict] = field(default_factory=list)
    topics_with_no_hits: list[str] = field(default_factory=list)
    total_chunks: int = 0
    connection_pass: dict[str, dict] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def _load_and_structure_book(meta: dict) -> tuple[BookNode, list[SectionNode], list[ChunkNode]]:
    raw_text = open(meta["raw_path"], encoding="utf-8").read()
    clean_text = text_cleaner.normalize_whitespace(text_cleaner.strip_boilerplate(raw_text))
    book = BookNode(
        id=new_id("book"),
        type=None,
        label=meta["title"][:150],
        topic=meta["topic"],
        gutenberg_id=meta["gutenberg_id"],
        source_url=f"https://www.gutenberg.org/ebooks/{meta['gutenberg_id']}",
        created_by="pipeline",
        metadata={"query_used": meta.get("query_used", "")},
    )
    sections, chunks = chunk_book(clean_text, book)
    for chunk in chunks:
        chunk.metadata["book_title"] = book.label
    return book, sections, chunks


def run_full_ingest(
    topics: list[str] | None = None,
    max_books_per_topic: int = 1,
    run_connection_finder: bool = True,
) -> IngestReport:
    start = time.time()
    report = IngestReport()
    neo4j_store.init_schema()

    topics = topics or list(gutenberg_fetch.TOPIC_BOOK_IDS.keys())

    all_books: list[BookNode] = []
    all_sections: list[SectionNode] = []
    all_chunks: list[ChunkNode] = []
    structural_edges = []

    for topic in topics:
        fetched = gutenberg_fetch.fetch_topic(topic, max_books=max_books_per_topic)
        if not fetched:
            report.topics_with_no_hits.append(topic)
            continue
        for meta in fetched:
            book, sections, chunks = _load_and_structure_book(meta)
            all_books.append(book)
            all_sections.extend(sections)
            all_chunks.extend(chunks)
            report.books.append({
                "topic": topic, "title": book.label, "gutenberg_id": book.gutenberg_id,
                "sections": len(sections), "chunks": len(chunks),
            })

            for section in sections:
                structural_edges.append(make_edge(
                    book.id, section.id, RelationType.HAS_SECTION, EdgeMethod.STRUCTURAL,
                ))
            for chunk in chunks:
                structural_edges.append(make_edge(
                    chunk.section_id, chunk.id, RelationType.HAS_CHUNK, EdgeMethod.STRUCTURAL,
                ))
                if chunk.next_chunk_id:
                    structural_edges.append(make_edge(
                        chunk.id, chunk.next_chunk_id, RelationType.NEXT_CHUNK, EdgeMethod.STRUCTURAL,
                    ))

    report.total_chunks = len(all_chunks)
    if not all_chunks:
        report.elapsed_seconds = time.time() - start
        return report

    neo4j_store.upsert_nodes(all_books)
    neo4j_store.upsert_nodes(all_sections)
    neo4j_store.upsert_nodes(all_chunks)
    neo4j_store.upsert_edges(structural_edges)

    texts = [c.text for c in all_chunks]
    sparse_index = SparseIndex()
    sparse_matrix = sparse_index.fit_transform(texts)
    sparse_index.save()

    dense_vecs = embed_texts(texts, show_progress=True)
    sparse_vecs = [
        SparseVector(indices=sparse_matrix.getrow(i).indices.tolist(), values=sparse_matrix.getrow(i).data.tolist())
        for i in range(len(texts))
    ]
    payloads = []
    for chunk in all_chunks:
        payload = chunk.to_dict()
        payload.pop("metadata", None)
        payload.update(chunk.metadata)
        payloads.append(payload)

    qdrant_store.upsert_points(
        node_ids=[c.id for c in all_chunks],
        dense_vecs=[v.tolist() for v in dense_vecs],
        sparse_vecs=sparse_vecs,
        payloads=payloads,
    )

    if run_connection_finder:
        for topic in topics:
            if topic in report.topics_with_no_hits:
                continue
            report.connection_pass[topic] = run_connection_finder_pass(topic=topic)

    report.elapsed_seconds = time.time() - start
    return report

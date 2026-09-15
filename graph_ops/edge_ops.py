"""Edge-level AI/user operations: manual relationship creation, and the
on-demand AI relationship-finder pass (statistical candidates -> LLM
judgment -> typed edges + possible new Concept nodes) that the Flask UI
can trigger against whatever is currently in the graph, not just at
initial-ingest time.
"""
from __future__ import annotations

import config
from graphbuild.node_schema import ChunkNode, GraphEdge, RelationType
from graphbuild.similarity import hybrid_similarity, top_k_candidates
from graphbuild.sparse import SparseIndex
from llm.connection_finder import process_candidates
from store import neo4j_store, qdrant_store


def create_relationship(
    source_id: str,
    target_id: str,
    rel_type: str,
    explanation: str = "",
    created_by: str = "user",
    weight: float = 1.0,
) -> GraphEdge | None:
    """Validates `rel_type` against the fixed RelationType enum before it ever
    reaches Cypher -- invalid input is rejected here, not string-interpolated."""
    try:
        rel_type_enum = RelationType(rel_type)
    except ValueError:
        return None
    return neo4j_store.create_relationship(
        source_id, target_id, rel_type_enum, weight=weight, explanation=explanation, created_by=created_by,
    )


def run_connection_finder_pass(
    topic: str | None = None,
    max_chunks: int = 150,
    max_pairs: int = config.CONNECTION_FINDER_MAX_PAIRS,
) -> dict:
    """Pulls chunks currently in the graph (optionally scoped to one topic),
    recomputes hybrid similarity among them, and runs the LLM connection-finder
    over the top candidate pairs -- writing typed edges and any new Concept
    nodes it proposes into both Neo4j and Qdrant."""
    raw_nodes = neo4j_store.list_nodes(node_type="Chunk", topic=topic, limit=max_chunks)
    if len(raw_nodes) < 2:
        return {"pairs_judged": 0, "edges_created": 0, "concepts_created": 0}

    chunks = [
        ChunkNode(
            id=n["id"], type=None, label=n.get("label", ""), topic=n.get("topic", ""),
            book_id=n.get("book_id", ""), section_id=n.get("section_id", ""), text=n.get("text", ""),
            order_index=n.get("order_index", 0),
            metadata={"book_title": n.get("meta_book_title", "")},
        )
        for n in raw_nodes
    ]
    texts = [c.text for c in chunks]

    from graphbuild.embeddings import embed_texts

    sparse_index = SparseIndex.load()
    dense = embed_texts(texts)
    sparse_matrix = sparse_index.transform(texts)

    sim = hybrid_similarity(dense, sparse_matrix)
    ids = [c.id for c in chunks]
    id_to_chunk = {c.id: c for c in chunks}
    candidate_pairs = top_k_candidates(sim, ids, k=4)
    candidate_pairs.sort(key=lambda t: -t[2])
    candidate_pairs = candidate_pairs[:max_pairs]

    triples = [(id_to_chunk[a], id_to_chunk[b], score) for a, b, score in candidate_pairs]
    typed_edges, new_concepts, concept_edges = process_candidates(triples)

    # Concept nodes must exist in Neo4j before concept_edges are written --
    # upsert_edges MATCHes both endpoints and silently drops an edge whose
    # target node isn't there yet.
    if new_concepts:
        from graph_ops.node_ops import _embed_and_upsert
        _embed_and_upsert(new_concepts, [f"{c.label}. {c.description}" for c in new_concepts])

    neo4j_store.upsert_edges(typed_edges + concept_edges)

    return {
        "pairs_judged": len(triples),
        "edges_created": len(typed_edges),
        "concepts_created": len(new_concepts),
        "concept_edges_created": len(concept_edges),
    }

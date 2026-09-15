"""GraphRAG query path: hybrid-search Qdrant for seed nodes, traverse the
Neo4j graph outward from them to pull in connected chunks/concepts (the
"Graph" half of GraphRAG -- not just a vector-search top-k), assemble a
grounded context bundle, and ask the answer-role LLM to respond citing
node ids. Uses a distinct model/role from the connection-finder LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from graphbuild.embeddings import embed_one
from graphbuild.sparse import SparseIndex
from llm.base import LLMProvider, get_provider
from store import neo4j_store, qdrant_store

SYSTEM_PROMPT = """You are a technical assistant answering questions about flight systems \
engineering (flight rules/operations, GNC, propulsion, electrical, mechanical, communications, \
thermal, data systems, life support, etc.) using ONLY the graph context provided below.

Rules:
- Ground every claim in the provided context; do not use outside knowledge.
- Cite the node id in square brackets after each claim it comes from, e.g. [chunk_ab12cd34].
- If the context does not answer the question, say so plainly instead of guessing.
- Be concise: a few sentences to a short paragraph, not an essay.
"""


@dataclass
class GraphRAGResult:
    answer: str
    seed_node_ids: list[str] = field(default_factory=list)
    context_node_ids: list[str] = field(default_factory=list)
    context_preview: list[dict] = field(default_factory=list)


def _format_context(seed_payloads: list[dict], neighborhoods: list[dict]) -> tuple[str, list[str], list[dict]]:
    seen: set[str] = set()
    blocks: list[str] = []
    preview: list[dict] = []

    def add_node_block(node_id: str, label: str, text: str, extra: str = "") -> None:
        if node_id in seen or not text:
            return
        seen.add(node_id)
        blocks.append(f"[{node_id}] {label}{(' - ' + extra) if extra else ''}\n{text[:700]}")
        preview.append({"node_id": node_id, "label": label, "snippet": text[:200]})

    for payload in seed_payloads:
        node_id = payload.get("node_id")
        add_node_block(node_id, payload.get("label", ""), payload.get("text") or payload.get("description", ""))

    for nb in neighborhoods:
        center = nb.get("node") or {}
        add_node_block(center.get("id"), center.get("label", ""), center.get("text") or center.get("description", ""))
        for rel in nb.get("relationships", []):
            neighbor = rel["neighbor"]
            edge = rel["edge"]
            extra = f"{edge.get('rel_type')} (via {edge.get('explanation', '')[:120]})"
            add_node_block(
                neighbor.get("id"), neighbor.get("label", ""),
                neighbor.get("text") or neighbor.get("description", ""), extra,
            )

    return "\n\n---\n\n".join(blocks), list(seen), preview


def answer(
    query: str,
    top_k_seed: int = 5,
    hops: int = 1,
    topic: str | None = None,
    provider: LLMProvider | None = None,
) -> GraphRAGResult:
    provider = provider or get_provider("answer")
    sparse_index = SparseIndex.load()

    dense_vec = embed_one(query).tolist()
    sparse_vec = sparse_index.query_to_sparse_vector(query)

    seeds = qdrant_store.hybrid_search(dense_vec, sparse_vec, top_k=top_k_seed, topic=topic)
    seed_ids = [s["node_id"] for s in seeds if s.get("node_id")]

    neighborhoods = [neo4j_store.get_node_neighborhood(nid, hops=hops) for nid in seed_ids]

    context_text, context_ids, preview = _format_context(seeds, neighborhoods)
    if not context_text:
        return GraphRAGResult(
            answer="No relevant graph context was found for this query. Try ingesting more books "
                   "for this topic, or rephrase the question.",
            seed_node_ids=seed_ids,
        )

    prompt = f"QUESTION: {query}\n\nGRAPH CONTEXT:\n\n{context_text}\n\nAnswer the question using only this context."
    reply = provider.complete(prompt, system=SYSTEM_PROMPT)

    return GraphRAGResult(
        answer=reply,
        seed_node_ids=seed_ids,
        context_node_ids=context_ids,
        context_preview=preview,
    )

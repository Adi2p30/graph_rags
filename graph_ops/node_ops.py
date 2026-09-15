"""Node-level AI/user operations: create and merge. Both write through to
Neo4j and Qdrant together so the two stores never drift apart -- there is
no path in this codebase that touches one without the other.
"""
from __future__ import annotations

from graphbuild.embeddings import embed_texts
from graphbuild.node_schema import ChunkNode, ConceptNode, GraphNode, new_id
from graphbuild.sparse import SparseIndex
from store import neo4j_store, qdrant_store


def _embed_and_upsert(nodes: list[GraphNode], texts: list[str]) -> None:
    if not nodes:
        return
    sparse_index = SparseIndex.load()
    dense_vecs = embed_texts(texts)
    sparse_matrix = sparse_index.transform(texts)
    payloads = []
    for node, text in zip(nodes, texts):
        payload = node.to_dict()
        payload["text"] = text
        payloads.append(payload)
    from graphbuild.sparse import SparseVector
    sparse_vecs = [
        SparseVector(indices=sparse_matrix.getrow(i).indices.tolist(), values=sparse_matrix.getrow(i).data.tolist())
        for i in range(len(texts))
    ]
    qdrant_store.upsert_points(
        node_ids=[n.id for n in nodes],
        dense_vecs=[v.tolist() for v in dense_vecs],
        sparse_vecs=sparse_vecs,
        payloads=payloads,
    )
    neo4j_store.upsert_nodes(nodes)


def create_concept_node(label: str, description: str, topic: str, created_by: str) -> ConceptNode:
    concept = ConceptNode(
        id=new_id("concept"),
        type=None,
        label=label.strip(),
        topic=topic,
        description=description.strip(),
        created_by=created_by,
    )
    _embed_and_upsert([concept], [f"{label}. {description}"])
    return concept


def create_manual_relationship_node_text(label: str, text: str, topic: str, created_by: str) -> ChunkNode:
    """Lets a user add a free-standing text node (e.g. a manual note/observation)
    that participates in the graph like any ingested chunk."""
    chunk = ChunkNode(
        id=new_id("chunk"),
        type=None,
        label=label.strip(),
        topic=topic,
        text=text.strip(),
        created_by=created_by,
    )
    _embed_and_upsert([chunk], [text])
    return chunk


def merge_nodes(keep_id: str, absorb_ids: list[str]) -> dict:
    """Merges `absorb_ids` into `keep_id`: Neo4j relationships are rewired,
    absorbed Qdrant points are deleted, and the kept node's text/label is
    widened with the absorbed nodes' text so retrieval still surfaces it."""
    absorb_ids = [a for a in absorb_ids if a != keep_id]
    if not absorb_ids:
        return {"keep_id": keep_id, "absorbed": []}

    keep_node = neo4j_store.get_node(keep_id)
    absorbed_nodes = [neo4j_store.get_node(a) for a in absorb_ids]
    absorbed_nodes = [n for n in absorbed_nodes if n]

    result = neo4j_store.merge_nodes(keep_id, absorb_ids)
    qdrant_store.delete_points(absorb_ids)

    if keep_node:
        combined_text = " ".join(
            filter(None, [keep_node.get("text") or keep_node.get("description", "")] +
                   [n.get("text") or n.get("description", "") for n in absorbed_nodes])
        )
        if combined_text.strip():
            sparse_index = SparseIndex.load()
            dense_vec = embed_texts([combined_text])[0]
            sparse_row = sparse_index.transform([combined_text])
            from graphbuild.sparse import SparseVector
            sparse_vec = SparseVector(indices=sparse_row.getrow(0).indices.tolist(), values=sparse_row.getrow(0).data.tolist())
            payload = dict(keep_node)
            payload.pop("_labels", None)
            payload["text"] = combined_text
            qdrant_store.upsert_points([keep_id], [dense_vec.tolist()], [sparse_vec], [payload])

    return result

"""Qdrant adapter: one collection, two named vectors per point (dense
sentence-transformer + sparse TF-IDF), fused at query time via RRF. Points
are keyed by a deterministic UUID5 of the graph node id (Qdrant point ids
must be int or UUID) -- the real id lives in the payload as `node_id` and
that's what everything else in this codebase keys off of.
"""
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, Optional

from qdrant_client import QdrantClient, models

import config
from graphbuild.embeddings import dense_dim
from graphbuild.sparse import SparseVector

DENSE_VEC = "dense"
SPARSE_VEC = "sparse"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=60)


def point_id_for(node_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"graph-rags:{node_id}"))


def ensure_collection() -> None:
    client = get_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={DENSE_VEC: models.VectorParams(size=dense_dim(), distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VEC: models.SparseVectorParams()},
    )


def upsert_points(
    node_ids: list[str],
    dense_vecs: list[list[float]],
    sparse_vecs: list[SparseVector],
    payloads: list[dict[str, Any]],
    batch_size: int = 128,
) -> None:
    """Batches upserts -- a single request carrying thousands of points (dense +
    sparse vectors each) reliably write-times-out against the local Qdrant server."""
    ensure_collection()
    client = get_client()
    points = []
    for node_id, dense, sparse, payload in zip(node_ids, dense_vecs, sparse_vecs, payloads):
        full_payload = {**payload, "node_id": node_id}
        points.append(
            models.PointStruct(
                id=point_id_for(node_id),
                vector={
                    DENSE_VEC: list(dense),
                    SPARSE_VEC: models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload=full_payload,
            )
        )
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points[i:i + batch_size])


def delete_points(node_ids: list[str]) -> None:
    ids = [point_id_for(n) for n in node_ids]
    get_client().delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=models.PointIdsList(points=ids),
    )


def hybrid_search(
    dense_vec: list[float],
    sparse_vec: SparseVector,
    top_k: int = 10,
    node_type: Optional[str] = None,
    topic: Optional[str] = None,
) -> list[dict]:
    ensure_collection()
    query_filter = None
    conditions = []
    if node_type:
        conditions.append(models.FieldCondition(key="type", match=models.MatchValue(value=node_type)))
    if topic:
        conditions.append(models.FieldCondition(key="topic", match=models.MatchValue(value=topic)))
    if conditions:
        query_filter = models.Filter(must=conditions)

    result = get_client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using=DENSE_VEC, limit=top_k * 4, filter=query_filter),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using=SPARSE_VEC,
                limit=top_k * 4,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [{"score": p.score, **p.payload} for p in result.points]


def get_point_payload(node_id: str) -> Optional[dict]:
    points = get_client().retrieve(
        collection_name=config.QDRANT_COLLECTION,
        ids=[point_id_for(node_id)],
        with_payload=True,
    )
    return points[0].payload if points else None


def collection_stats() -> dict:
    ensure_collection()
    info = get_client().get_collection(config.QDRANT_COLLECTION)
    return {"points_count": info.points_count, "status": str(info.status)}

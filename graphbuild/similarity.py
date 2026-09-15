"""Hybrid dense+sparse similarity -> candidate SIMILAR_TO edges between chunks.

This is the purely-statistical pass (no LLM): cosine similarity on dense
sentence-transformer embeddings, cosine similarity on TF-IDF sparse vectors,
combined by a weighted sum. Candidate edges above threshold/top-k are what
the LLM connection-finder later looks at to decide a *typed* relationship
(or discard as noise).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import config
from graphbuild.sparse import cosine_sim_matrix


def hybrid_similarity(dense: np.ndarray, sparse: sp.csr_matrix) -> np.ndarray:
    """dense: (n, d) L2-normalized. sparse: (n, v) TF-IDF rows. Returns (n, n) hybrid score."""
    dense_sim = dense @ dense.T  # already normalized -> cosine
    sparse_sim = cosine_sim_matrix(sparse, sparse)
    return config.DENSE_WEIGHT * dense_sim + config.SPARSE_WEIGHT * sparse_sim


def top_k_candidates(
    sim_matrix: np.ndarray,
    ids: list[str],
    k: int = config.SIMILARITY_TOP_K,
    threshold: float = config.SIMILARITY_EDGE_THRESHOLD,
    exclude_adjacent: bool = True,
) -> list[tuple[str, str, float]]:
    """Returns (source_id, target_id, score) triples, one direction only (i<j),
    top-k neighbors per node that clear `threshold`."""
    n = len(ids)
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        row = sim_matrix[i].copy()
        row[i] = -1.0
        if exclude_adjacent and i + 1 < n:
            row[i + 1] = -1.0
        if exclude_adjacent and i - 1 >= 0:
            row[i - 1] = -1.0
        top_idx = np.argpartition(-row, min(k, n - 1) - 1)[:k]
        for j in top_idx:
            j = int(j)
            score = float(row[j])
            if score < threshold:
                continue
            key = (min(i, j), max(i, j))
            if key not in edges or score > edges[key]:
                edges[key] = score
    return [(ids[i], ids[j], score) for (i, j), score in edges.items()]

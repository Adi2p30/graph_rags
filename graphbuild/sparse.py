"""Sparse (TF-IDF) representation of the chunk corpus.

Fit once per ingestion batch over all chunk texts, corpus-wide, so IDF
reflects the whole topic set rather than a single book. Both the sparse
similarity edges (structural graph pass) and Qdrant's sparse named vector
read from the same fitted vectorizer.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import config

VECTORIZER_PATH = config.PROCESSED_DIR / "tfidf_vectorizer.pkl"


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseIndex:
    def __init__(self, max_features: int = 50_000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix: sp.csr_matrix | None = None

    def fit_transform(self, texts: list[str]) -> sp.csr_matrix:
        self.matrix = self.vectorizer.fit_transform(texts)
        return self.matrix

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        return self.vectorizer.transform(texts)

    def row_to_sparse_vector(self, row_idx: int) -> SparseVector:
        row = self.matrix.getrow(row_idx)
        return SparseVector(indices=row.indices.tolist(), values=row.data.tolist())

    def query_to_sparse_vector(self, text: str) -> SparseVector:
        row = self.vectorizer.transform([text]).tocsr()
        return SparseVector(indices=row.indices.tolist(), values=row.data.tolist())

    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def save(self, path: Path = VECTORIZER_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, path: Path = VECTORIZER_PATH) -> "SparseIndex":
        idx = cls()
        with open(path, "rb") as f:
            idx.vectorizer = pickle.load(f)
        return idx


def cosine_sim_matrix(a: sp.csr_matrix | np.ndarray, b: sp.csr_matrix | np.ndarray) -> np.ndarray:
    """Cosine similarity between two matrices of row vectors (sparse or dense, already
    non-negative TF-IDF or L2-normalized dense embeddings)."""
    if sp.issparse(a):
        a = a.toarray()
    if sp.issparse(b):
        b = b.toarray()
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

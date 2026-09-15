"""Dense embeddings via sentence-transformers."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_dense_model() -> SentenceTransformer:
    return SentenceTransformer(config.DENSE_MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """Returns an (n, dim) float32 array of L2-normalized dense embeddings."""
    if not texts:
        return np.zeros((0, get_dense_model().get_embedding_dimension()), dtype=np.float32)
    model = get_dense_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def dense_dim() -> int:
    return get_dense_model().get_embedding_dimension()

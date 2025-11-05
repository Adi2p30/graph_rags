"""Vector Store - Manages embeddings and vector similarity search for entities"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path


class VectorStore:
    """Manages vector embeddings and similarity search for graph entities"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_faiss: bool = True):
        """
        Initialize vector store

        Args:
            model_name: SentenceTransformer model name
            use_faiss: Whether to use FAISS for fast similarity search
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.use_faiss = use_faiss

        # Storage for embeddings
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.entity_texts: Dict[str, str] = {}

        # FAISS index
        self.index = None
        self.index_to_entity_id = {}

        if use_faiss:
            try:
                import faiss
                self.faiss = faiss
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(self.dimension)
            except ImportError:
                print("FAISS not available, using numpy for similarity search")
                self.use_faiss = False
                self.faiss = None

    def add_entity(self, entity_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        Add an entity and compute its embedding

        Args:
            entity_id: Unique entity identifier
            text: Text to embed (entity description)
            metadata: Additional metadata
        """
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Store
        self.entity_embeddings[entity_id] = embedding
        self.entity_texts[entity_id] = text

        # Update FAISS index
        if self.use_faiss and self.index is not None:
            idx = len(self.index_to_entity_id)
            self.index.add(embedding.reshape(1, -1))
            self.index_to_entity_id[idx] = entity_id

    def add_entities_batch(self, entities: List[Dict[str, Any]]):
        """
        Add multiple entities in batch for efficiency

        Args:
            entities: List of dicts with 'id' and 'text' keys
        """
        texts = [e['text'] for e in entities]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        for entity, embedding in zip(entities, embeddings):
            entity_id = entity['id']
            self.entity_embeddings[entity_id] = embedding
            self.entity_texts[entity_id] = entity['text']

        # Rebuild FAISS index
        if self.use_faiss and len(self.entity_embeddings) > 0:
            self._rebuild_faiss_index()

    def search_similar(self, query: str, k: int = 10,
                      filter_ids: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Search for similar entities using vector similarity

        Args:
            query: Query text
            k: Number of results to return
            filter_ids: Optional list of entity IDs to filter

        Returns:
            List of (entity_id, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        if self.use_faiss and self.index is not None and self.index.ntotal > 0:
            # Use FAISS for fast search
            distances, indices = self.index.search(query_embedding.reshape(1, -1), min(k * 2, self.index.ntotal))

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.index_to_entity_id):
                    continue

                entity_id = self.index_to_entity_id[idx]

                # Apply filter if provided
                if filter_ids and entity_id not in filter_ids:
                    continue

                # Convert L2 distance to similarity score
                similarity = 1 / (1 + dist)
                results.append((entity_id, float(similarity)))

                if len(results) >= k:
                    break

            return results
        else:
            # Fallback to numpy cosine similarity
            return self._numpy_similarity_search(query_embedding, k, filter_ids)

    def _numpy_similarity_search(self, query_embedding: np.ndarray, k: int,
                                 filter_ids: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Fallback similarity search using numpy"""
        if not self.entity_embeddings:
            return []

        # Filter embeddings if needed
        if filter_ids:
            entity_ids = [eid for eid in self.entity_embeddings.keys() if eid in filter_ids]
        else:
            entity_ids = list(self.entity_embeddings.keys())

        if not entity_ids:
            return []

        # Compute cosine similarities
        embeddings_matrix = np.array([self.entity_embeddings[eid] for eid in entity_ids])

        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_norm = embeddings_matrix / (np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = [(entity_ids[idx], float(similarities[idx])) for idx in top_indices]
        return results

    def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Get embedding for an entity"""
        return self.entity_embeddings.get(entity_id)

    def compute_similarity(self, entity_id1: str, entity_id2: str) -> float:
        """Compute similarity between two entities"""
        emb1 = self.entity_embeddings.get(entity_id1)
        emb2 = self.entity_embeddings.get(entity_id2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return float(similarity)

    def get_entity_clusters(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster entities using k-means

        Args:
            n_clusters: Number of clusters

        Returns:
            Dict mapping cluster_id to list of entity_ids
        """
        if not self.entity_embeddings:
            return {}

        from sklearn.cluster import KMeans

        entity_ids = list(self.entity_embeddings.keys())
        embeddings = np.array([self.entity_embeddings[eid] for eid in entity_ids])

        # K-means clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(entity_ids)), random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Group by cluster
        clusters = {}
        for entity_id, label in zip(entity_ids, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(entity_id)

        return clusters

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from scratch"""
        if not self.use_faiss or self.faiss is None:
            return

        entity_ids = list(self.entity_embeddings.keys())
        embeddings = np.array([self.entity_embeddings[eid] for eid in entity_ids])

        # Create new index
        self.index = self.faiss.IndexFlatL2(self.dimension)
        self.index_to_entity_id = {}

        # Add all embeddings
        if len(embeddings) > 0:
            self.index.add(embeddings.astype('float32'))
            for idx, entity_id in enumerate(entity_ids):
                self.index_to_entity_id[idx] = entity_id

    def save(self, filepath: str):
        """Save vector store to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'model_name': self.model_name,
            'entity_embeddings': self.entity_embeddings,
            'entity_texts': self.entity_texts,
            'index_to_entity_id': self.index_to_entity_id
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.model_name = data['model_name']
        self.entity_embeddings = data['entity_embeddings']
        self.entity_texts = data['entity_texts']
        self.index_to_entity_id = data.get('index_to_entity_id', {})

        # Reload model if different
        if self.model_name != self.model.config_name:
            self.model = SentenceTransformer(self.model_name)

        # Rebuild FAISS index
        if self.use_faiss:
            self._rebuild_faiss_index()

    def clear(self):
        """Clear all embeddings"""
        self.entity_embeddings = {}
        self.entity_texts = {}
        self.index_to_entity_id = {}

        if self.use_faiss and self.faiss is not None:
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.index = self.faiss.IndexFlatL2(self.dimension)

    def __len__(self):
        """Return number of entities in the store"""
        return len(self.entity_embeddings)

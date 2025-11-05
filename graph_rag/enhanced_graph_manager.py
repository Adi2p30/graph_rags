"""Enhanced Graph Manager - Integrates vector search, LLM extraction, and community detection"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .graph_store import GraphStore
from .vector_store import VectorStore
from .llm_extractor import GeminiExtractor
from .community_detector import CommunityDetector
from .query_engine import QueryEngine


class EnhancedGraphManager:
    """
    Manages knowledge graphs with vector search, LLM-based extraction,
    and hierarchical community detection (GraphRAG architecture)
    """

    def __init__(self, storage_path: str = "./data/graphs", gemini_api_key: Optional[str] = None):
        """
        Initialize enhanced graph manager

        Args:
            storage_path: Base path for storing graphs
            gemini_api_key: Gemini API key for LLM features
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Graph stores
        self.graphs: Dict[str, GraphStore] = {}
        self.vector_stores: Dict[str, VectorStore] = {}
        self.community_detectors: Dict[str, CommunityDetector] = {}
        self.query_engines: Dict[str, QueryEngine] = {}

        # LLM extractor (shared across graphs)
        self.llm_extractor = GeminiExtractor(api_key=gemini_api_key)

        # Load existing graphs
        self._load_existing_graphs()

    def create_graph(self, graph_name: str) -> GraphStore:
        """
        Create a new graph with vector search support

        Args:
            graph_name: Name for the new graph

        Returns:
            New GraphStore instance
        """
        if graph_name in self.graphs:
            raise ValueError(f"Graph '{graph_name}' already exists")

        # Create graph and supporting structures
        graph = GraphStore(graph_name)
        vector_store = VectorStore(model_name="all-MiniLM-L6-v2")
        community_detector = CommunityDetector(graph)
        query_engine = QueryEngine(graph)

        self.graphs[graph_name] = graph
        self.vector_stores[graph_name] = vector_store
        self.community_detectors[graph_name] = community_detector
        self.query_engines[graph_name] = query_engine

        # Save
        self._save_graph(graph_name)

        return graph

    def add_document(
        self,
        graph_name: str,
        text: str,
        document_id: str = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Add a document using LLM-based extraction and vector embeddings

        Args:
            graph_name: Target graph name
            text: Document text
            document_id: Optional document identifier
            use_llm: Whether to use LLM for extraction (Gemini)

        Returns:
            Extraction results with statistics
        """
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")

        vector_store = self.vector_stores[graph_name]

        # Extract entities and relationships
        if use_llm and self.llm_extractor.model:
            entities, relationships = self.llm_extractor.extract_entities_and_relationships(
                text, document_id
            )
        else:
            from .entity_extractor import EntityExtractor
            extractor = EntityExtractor(use_advanced_nlp=False)
            entities, relationships = extractor.extract_entities_and_relationships(
                text, document_id
            )

        # Add to graph
        for entity in entities:
            graph.add_entity(
                entity['id'],
                entity.get('type', 'ENTITY'),
                {k: v for k, v in entity.items() if k not in ['id', 'type']}
            )

        # Add to vector store
        entity_texts = []
        for entity in entities:
            text_repr = graph.get_entity_text(entity['id'])
            entity_texts.append({
                'id': entity['id'],
                'text': text_repr
            })

        if entity_texts:
            vector_store.add_entities_batch(entity_texts)

        # Add relationships
        for rel in relationships:
            graph.add_relationship(
                rel['source'],
                rel['target'],
                rel['type'],
                {k: v for k, v in rel.items() if k not in ['source', 'target', 'type']}
            )

        # Save
        self._save_graph(graph_name)

        return {
            'entities_added': len(entities),
            'relationships_added': len(relationships),
            'entities': entities,
            'relationships': relationships
        }

    def query_graph(
        self,
        graph_name: str,
        query: str,
        use_vector_search: bool = True,
        use_communities: bool = False,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Query a graph using vector similarity and/or community summaries

        Args:
            graph_name: Graph to query
            query: Query text
            use_vector_search: Use vector similarity for entity retrieval
            use_communities: Include community summaries in results
            max_results: Maximum results

        Returns:
            Query results with entities, relationships, and optional summaries
        """
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")

        vector_store = self.vector_stores[graph_name]
        query_engine = self.query_engines[graph_name]

        results = {
            'query': query,
            'entities': [],
            'relationships': [],
            'community_summaries': [],
            'llm_answer': None
        }

        # Vector search for relevant entities
        if use_vector_search and len(vector_store) > 0:
            similar_entities = vector_store.search_similar(query, k=max_results)
            entity_ids = [eid for eid, _ in similar_entities]

            # Get entity details
            for entity_id, score in similar_entities:
                entity = graph.get_entity(entity_id)
                if entity:
                    entity['relevance_score'] = score
                    results['entities'].append(entity)

            # Get relationships for these entities
            for entity_id in entity_ids:
                rels = graph.get_relationships(entity_id)
                results['relationships'].extend(rels)

        else:
            # Fallback to traditional query
            query_result = query_engine.query(query, max_results=max_results)
            results['entities'] = query_result['entities']
            results['relationships'] = query_result['relationships']

        # Add community summaries if requested
        if use_communities:
            community_detector = self.community_detectors[graph_name]

            # Get communities for relevant entities
            community_ids = set()
            for entity in results['entities'][:5]:  # Top 5 entities
                comm_id = community_detector.get_community_for_entity(entity['id'])
                if comm_id is not None:
                    community_ids.add(comm_id)

            # Get summaries
            for comm_id in community_ids:
                if comm_id in community_detector.community_summaries:
                    results['community_summaries'].append(
                        community_detector.community_summaries[comm_id]
                    )

        # Generate LLM answer if available
        if self.llm_extractor.model and results['entities']:
            try:
                answer = self.llm_extractor.answer_query(
                    query,
                    results['entities'][:10],
                    results['relationships'][:20],
                    results['community_summaries'][:3]
                )
                results['llm_answer'] = answer
            except Exception as e:
                print(f"Error generating LLM answer: {e}")

        # Remove duplicate relationships
        unique_rels = []
        seen = set()
        for rel in results['relationships']:
            key = (rel['source'], rel['target'], rel['type'])
            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)
        results['relationships'] = unique_rels[:max_results]

        results['total_results'] = len(results['entities'])
        return results

    def detect_communities(self, graph_name: str, generate_summaries: bool = True) -> Dict[str, Any]:
        """
        Detect communities and optionally generate summaries

        Args:
            graph_name: Graph name
            generate_summaries: Whether to generate LLM summaries

        Returns:
            Community detection results
        """
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")

        community_detector = self.community_detectors[graph_name]

        # Detect communities
        communities = community_detector.detect_communities(resolution=1.0)

        # Generate summaries
        if generate_summaries:
            for comm_id in communities.keys():
                try:
                    community_detector.generate_community_summary(
                        comm_id,
                        llm_extractor=self.llm_extractor if self.llm_extractor.model else None
                    )
                except Exception as e:
                    print(f"Error generating summary for community {comm_id}: {e}")

        # Save
        self._save_graph(graph_name)

        stats = community_detector.get_community_stats()

        return {
            'communities': communities,
            'stats': stats,
            'summaries': community_detector.community_summaries
        }

    def get_similar_entities(
        self,
        graph_name: str,
        entity_id: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar entities using vector embeddings

        Args:
            graph_name: Graph name
            entity_id: Entity to find similar entities for
            k: Number of similar entities to return

        Returns:
            List of similar entities with scores
        """
        graph = self.graphs.get(graph_name)
        vector_store = self.vector_stores.get(graph_name)

        if not graph or not vector_store:
            return []

        # Get entity text
        entity_text = graph.get_entity_text(entity_id)
        if not entity_text:
            return []

        # Search for similar
        similar = vector_store.search_similar(entity_text, k=k + 1)  # +1 to exclude self

        results = []
        for sim_id, score in similar:
            if sim_id == entity_id:
                continue

            entity = graph.get_entity(sim_id)
            if entity:
                entity['similarity_score'] = score
                results.append(entity)

            if len(results) >= k:
                break

        return results

    def rebuild_vector_index(self, graph_name: str):
        """
        Rebuild vector index for a graph

        Args:
            graph_name: Graph name
        """
        graph = self.graphs.get(graph_name)
        vector_store = self.vector_stores.get(graph_name)

        if not graph or not vector_store:
            return

        # Clear and rebuild
        vector_store.clear()

        # Add all entities
        entities = graph.get_all_entities()
        entity_texts = []

        for entity in entities:
            text = graph.get_entity_text(entity['id'])
            entity_texts.append({
                'id': entity['id'],
                'text': text
            })

        if entity_texts:
            vector_store.add_entities_batch(entity_texts)

        # Save
        self._save_graph(graph_name)

    def get_graph_statistics(self, graph_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive statistics for a graph"""
        graph = self.graphs.get(graph_name)
        vector_store = self.vector_stores.get(graph_name)
        community_detector = self.community_detectors.get(graph_name)

        if not graph:
            return None

        stats = graph.get_statistics()

        # Add vector store stats
        stats['vector_embeddings_count'] = len(vector_store) if vector_store else 0

        # Add community stats
        if community_detector and community_detector.communities:
            stats['community_stats'] = community_detector.get_community_stats()

        return stats

    def _save_graph(self, graph_name: str):
        """Save graph, vector store, and communities to disk"""
        graph = self.graphs.get(graph_name)
        vector_store = self.vector_stores.get(graph_name)
        community_detector = self.community_detectors.get(graph_name)

        if not graph:
            return

        # Save graph
        graph_file = self.storage_path / f"{graph_name}.json"
        graph.save(str(graph_file))

        # Save vector store
        if vector_store:
            vector_file = self.storage_path / f"{graph_name}_vectors.pkl"
            vector_store.save(str(vector_file))

        # Save communities
        if community_detector and community_detector.communities:
            communities_file = self.storage_path / f"{graph_name}_communities.json"
            with open(communities_file, 'w') as f:
                json.dump({
                    'communities': {str(k): v for k, v in community_detector.communities.items()},
                    'summaries': {str(k): v for k, v in community_detector.community_summaries.items()}
                }, f, indent=2)

    def _load_existing_graphs(self):
        """Load all existing graphs from storage"""
        if not self.storage_path.exists():
            return

        for graph_file in self.storage_path.glob("*.json"):
            # Skip community files
            if "_communities" in graph_file.stem:
                continue

            graph_name = graph_file.stem

            try:
                # Load graph
                graph = GraphStore(graph_name)
                graph.load(str(graph_file))
                self.graphs[graph_name] = graph

                # Load vector store
                vector_store = VectorStore(model_name="all-MiniLM-L6-v2")
                vector_file = self.storage_path / f"{graph_name}_vectors.pkl"
                if vector_file.exists():
                    vector_store.load(str(vector_file))
                self.vector_stores[graph_name] = vector_store

                # Load communities
                community_detector = CommunityDetector(graph)
                communities_file = self.storage_path / f"{graph_name}_communities.json"
                if communities_file.exists():
                    with open(communities_file, 'r') as f:
                        data = json.load(f)
                        community_detector.communities = {
                            int(k): v for k, v in data.get('communities', {}).items()
                        }
                        community_detector.community_summaries = {
                            int(k): v for k, v in data.get('summaries', {}).items()
                        }
                self.community_detectors[graph_name] = community_detector

                # Create query engine
                query_engine = QueryEngine(graph)
                self.query_engines[graph_name] = query_engine

            except Exception as e:
                print(f"Error loading graph '{graph_name}': {e}")

    def get_graph(self, graph_name: str) -> Optional[GraphStore]:
        """Get a graph by name"""
        return self.graphs.get(graph_name)

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all available graphs"""
        graphs_info = []

        for name, graph in self.graphs.items():
            stats = self.get_graph_statistics(name)
            graphs_info.append({
                "name": name,
                "statistics": stats,
                "metadata": graph.metadata
            })

        return graphs_info

    def delete_graph(self, graph_name: str):
        """Delete a graph and all associated data"""
        if graph_name in self.graphs:
            del self.graphs[graph_name]
            del self.vector_stores[graph_name]
            del self.community_detectors[graph_name]
            del self.query_engines[graph_name]

            # Delete files
            for pattern in [f"{graph_name}.json", f"{graph_name}_vectors.pkl", f"{graph_name}_communities.json"]:
                file_path = self.storage_path / pattern
                if file_path.exists():
                    file_path.unlink()

"""Graph Manager - Manages multiple knowledge graphs"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from .graph_store import GraphStore
from .entity_extractor import EntityExtractor
from .query_engine import QueryEngine


class GraphManager:
    """Manages multiple knowledge graphs and provides high-level operations"""

    def __init__(self, storage_path: str = "./data/graphs"):
        """
        Initialize graph manager

        Args:
            storage_path: Base path for storing graphs
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.graphs: Dict[str, GraphStore] = {}
        self.query_engines: Dict[str, QueryEngine] = {}
        self.entity_extractor = EntityExtractor(use_advanced_nlp=False)

        # Load existing graphs
        self._load_existing_graphs()

    def create_graph(self, graph_name: str) -> GraphStore:
        """
        Create a new graph

        Args:
            graph_name: Name for the new graph

        Returns:
            New GraphStore instance
        """
        if graph_name in self.graphs:
            raise ValueError(f"Graph '{graph_name}' already exists")

        graph = GraphStore(graph_name)
        self.graphs[graph_name] = graph
        self.query_engines[graph_name] = QueryEngine(graph)

        # Save immediately
        self._save_graph(graph_name)

        return graph

    def get_graph(self, graph_name: str) -> Optional[GraphStore]:
        """Get a graph by name"""
        return self.graphs.get(graph_name)

    def get_query_engine(self, graph_name: str) -> Optional[QueryEngine]:
        """Get query engine for a graph"""
        return self.query_engines.get(graph_name)

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all available graphs"""
        graphs_info = []

        for name, graph in self.graphs.items():
            stats = graph.get_statistics()
            graphs_info.append({
                "name": name,
                "statistics": stats,
                "metadata": graph.metadata
            })

        return graphs_info

    def delete_graph(self, graph_name: str):
        """Delete a graph"""
        if graph_name in self.graphs:
            del self.graphs[graph_name]
            del self.query_engines[graph_name]

            # Delete file
            graph_file = self.storage_path / f"{graph_name}.json"
            if graph_file.exists():
                graph_file.unlink()

    def add_document(self, graph_name: str, text: str,
                    document_id: str = None) -> Dict[str, Any]:
        """
        Add a document to a graph by extracting entities and relationships

        Args:
            graph_name: Target graph name
            text: Document text
            document_id: Optional document identifier

        Returns:
            Extraction results
        """
        graph = self.get_graph(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")

        # Extract entities and relationships
        entities, relationships = self.entity_extractor.extract_entities_and_relationships(
            text, document_id
        )

        # Add to graph
        for entity in entities:
            graph.add_entity(
                entity['id'],
                entity.get('type', 'ENTITY'),
                {k: v for k, v in entity.items() if k not in ['id', 'type']}
            )

        for rel in relationships:
            graph.add_relationship(
                rel['source'],
                rel['target'],
                rel['type'],
                {k: v for k, v in rel.items() if k not in ['source', 'target', 'type']}
            )

        # Save graph
        self._save_graph(graph_name)

        return {
            "entities_added": len(entities),
            "relationships_added": len(relationships),
            "entities": entities,
            "relationships": relationships
        }

    def add_structured_data(self, graph_name: str, data: Any,
                           id_field: str = "id") -> Dict[str, Any]:
        """
        Add structured data (JSON/dict) to a graph

        Args:
            graph_name: Target graph name
            data: Structured data
            id_field: Field to use as entity ID

        Returns:
            Extraction results
        """
        graph = self.get_graph(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")

        # Extract entities and relationships
        entities, relationships = self.entity_extractor.extract_from_structured_data(
            data, id_field
        )

        # Add to graph
        for entity in entities:
            graph.add_entity(
                entity['id'],
                entity.get('type', 'ENTITY'),
                entity.get('properties', {})
            )

        for rel in relationships:
            graph.add_relationship(
                rel['source'],
                rel['target'],
                rel['type'],
                rel.get('properties', {})
            )

        # Save graph
        self._save_graph(graph_name)

        return {
            "entities_added": len(entities),
            "relationships_added": len(relationships),
            "entities": entities,
            "relationships": relationships
        }

    def query_graph(self, graph_name: str, query: str,
                   max_results: int = 10) -> Dict[str, Any]:
        """
        Query a graph

        Args:
            graph_name: Graph to query
            query: Query text
            max_results: Maximum results

        Returns:
            Query results
        """
        engine = self.get_query_engine(graph_name)
        if not engine:
            raise ValueError(f"Graph '{graph_name}' not found")

        return engine.query(query, max_results=max_results)

    def merge_graphs(self, source_graph: str, target_graph: str):
        """
        Merge one graph into another

        Args:
            source_graph: Source graph name
            target_graph: Target graph name
        """
        source = self.get_graph(source_graph)
        target = self.get_graph(target_graph)

        if not source or not target:
            raise ValueError("Source or target graph not found")

        # Copy all entities
        for entity_id, entity_data in source.graph.nodes(data=True):
            target.add_entity(
                entity_id,
                entity_data.get('entity_type', 'ENTITY'),
                {k: v for k, v in entity_data.items() if k != 'entity_type'}
            )

        # Copy all relationships
        for source_id, target_id, edge_data in source.graph.edges(data=True):
            target.add_relationship(
                source_id,
                target_id,
                edge_data.get('relationship_type', 'RELATED_TO'),
                {k: v for k, v in edge_data.items() if k != 'relationship_type'}
            )

        # Save target graph
        self._save_graph(target_graph)

    def export_graph(self, graph_name: str, format: str = "json") -> Any:
        """
        Export a graph in various formats

        Args:
            graph_name: Graph to export
            format: Export format ('json', 'cypher', 'graphml')

        Returns:
            Exported data
        """
        graph = self.get_graph(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")

        if format == "json":
            return {
                "entities": graph.get_all_entities(),
                "relationships": graph.get_all_relationships(),
                "metadata": graph.metadata,
                "statistics": graph.get_statistics()
            }
        elif format == "cypher":
            return self._export_to_cypher(graph)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_to_cypher(self, graph: GraphStore) -> List[str]:
        """Export graph to Cypher statements (for Neo4j)"""
        statements = []

        # Create nodes
        for entity in graph.get_all_entities():
            props = []
            for key, value in entity.items():
                if key not in ['id', 'entity_type']:
                    if isinstance(value, str):
                        props.append(f'{key}: "{value}"')
                    else:
                        props.append(f'{key}: {value}')

            props_str = ', '.join(props)
            stmt = f"CREATE (n:{entity.get('entity_type', 'Entity')} {{id: '{entity['id']}', {props_str}}})"
            statements.append(stmt)

        # Create relationships
        for rel in graph.get_all_relationships():
            stmt = f"MATCH (a {{id: '{rel['source']}'}}), (b {{id: '{rel['target']}'}}) CREATE (a)-[:{rel['type']}]->(b)"
            statements.append(stmt)

        return statements

    def _save_graph(self, graph_name: str):
        """Save a graph to disk"""
        graph = self.graphs.get(graph_name)
        if graph:
            filepath = self.storage_path / f"{graph_name}.json"
            graph.save(str(filepath))

    def _load_existing_graphs(self):
        """Load all existing graphs from storage"""
        if not self.storage_path.exists():
            return

        for graph_file in self.storage_path.glob("*.json"):
            graph_name = graph_file.stem
            try:
                graph = GraphStore(graph_name)
                graph.load(str(graph_file))
                self.graphs[graph_name] = graph
                self.query_engines[graph_name] = QueryEngine(graph)
            except Exception as e:
                print(f"Error loading graph '{graph_name}': {e}")

    def get_graph_statistics(self, graph_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific graph"""
        graph = self.get_graph(graph_name)
        if graph:
            return graph.get_statistics()
        return None

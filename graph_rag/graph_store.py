"""Graph Store - Manages the knowledge graph using NetworkX"""

import networkx as nx
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class GraphStore:
    """Manages storage and retrieval of entities and relationships in a knowledge graph"""

    def __init__(self, graph_name: str = "default"):
        """
        Initialize a graph store

        Args:
            graph_name: Name identifier for this graph
        """
        self.graph_name = graph_name
        self.graph = nx.MultiDiGraph()
        self.metadata = {
            "name": graph_name,
            "created_at": None,
            "updated_at": None,
            "entity_count": 0,
            "relationship_count": 0
        }

    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any] = None):
        """
        Add an entity (node) to the graph

        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type/category of the entity
            properties: Additional properties for the entity
        """
        properties = properties or {}
        self.graph.add_node(
            entity_id,
            entity_type=entity_type,
            **properties
        )
        self.metadata["entity_count"] = self.graph.number_of_nodes()

    def get_entity_text(self, entity_id: str) -> str:
        """
        Generate text description for entity (for embedding)

        Args:
            entity_id: Entity identifier

        Returns:
            Text description
        """
        if entity_id not in self.graph:
            return ""

        entity_data = dict(self.graph.nodes[entity_id])
        entity_type = entity_data.get('entity_type', 'ENTITY')

        # Build text representation
        parts = [f"{entity_id.replace('_', ' ')} is a {entity_type}"]

        # Add description if available
        if 'description' in entity_data:
            parts.append(entity_data['description'])
        elif 'name' in entity_data:
            parts.append(f"named {entity_data['name']}")

        # Add other properties
        for key, value in entity_data.items():
            if key not in ['entity_type', 'description', 'name', 'source_document']:
                if isinstance(value, str) and len(value) < 100:
                    parts.append(f"{key}: {value}")

        return ". ".join(parts) + "."

    def add_relationship(self, source_id: str, target_id: str,
                        relationship_type: str, properties: Dict[str, Any] = None):
        """
        Add a relationship (edge) between two entities

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            properties: Additional properties for the relationship
        """
        properties = properties or {}
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type,
            **properties
        )
        self.metadata["relationship_count"] = self.graph.number_of_edges()

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity and its properties"""
        if entity_id in self.graph:
            node_data = dict(self.graph.nodes[entity_id])
            node_data['id'] = entity_id
            return node_data
        return None

    def get_relationships(self, entity_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity

        Args:
            entity_id: Entity to get relationships for
            direction: 'in', 'out', or 'both'
        """
        relationships = []

        if direction in ["out", "both"]:
            for target in self.graph.successors(entity_id):
                for key, edge_data in self.graph[entity_id][target].items():
                    relationships.append({
                        "source": entity_id,
                        "target": target,
                        "type": edge_data.get("relationship_type", "unknown"),
                        "properties": {k: v for k, v in edge_data.items()
                                     if k != "relationship_type"}
                    })

        if direction in ["in", "both"]:
            for source in self.graph.predecessors(entity_id):
                for key, edge_data in self.graph[source][entity_id].items():
                    relationships.append({
                        "source": source,
                        "target": entity_id,
                        "type": edge_data.get("relationship_type", "unknown"),
                        "properties": {k: v for k, v in edge_data.items()
                                     if k != "relationship_type"}
                    })

        return relationships

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for entities by text query

        Args:
            query: Text to search for
            entity_type: Optional filter by entity type
        """
        query_lower = query.lower()
        results = []

        for node_id, node_data in self.graph.nodes(data=True):
            # Check entity type filter
            if entity_type and node_data.get('entity_type') != entity_type:
                continue

            # Search in node ID and properties
            if query_lower in node_id.lower():
                result = dict(node_data)
                result['id'] = node_id
                result['match_score'] = 1.0
                results.append(result)
                continue

            # Search in text properties
            for key, value in node_data.items():
                if isinstance(value, str) and query_lower in value.lower():
                    result = dict(node_data)
                    result['id'] = node_id
                    result['match_score'] = 0.8
                    results.append(result)
                    break

        return results

    def get_subgraph(self, entity_ids: List[str], depth: int = 1) -> nx.MultiDiGraph:
        """
        Get a subgraph around specified entities

        Args:
            entity_ids: Center entities
            depth: How many hops to include
        """
        nodes_to_include = set(entity_ids)

        for _ in range(depth):
            new_nodes = set()
            for node in nodes_to_include:
                if node in self.graph:
                    new_nodes.update(self.graph.successors(node))
                    new_nodes.update(self.graph.predecessors(node))
            nodes_to_include.update(new_nodes)

        return self.graph.subgraph(nodes_to_include).copy()

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all entities in the graph"""
        entities = []
        for node_id, node_data in self.graph.nodes(data=True):
            entity = dict(node_data)
            entity['id'] = node_id
            entities.append(entity)
        return entities

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships in the graph"""
        relationships = []
        for source, target, edge_data in self.graph.edges(data=True):
            relationships.append({
                "source": source,
                "target": target,
                "type": edge_data.get("relationship_type", "unknown"),
                "properties": {k: v for k, v in edge_data.items()
                             if k != "relationship_type"}
            })
        return relationships

    def delete_entity(self, entity_id: str):
        """Delete an entity and all its relationships"""
        if entity_id in self.graph:
            self.graph.remove_node(entity_id)
            self.metadata["entity_count"] = self.graph.number_of_nodes()
            self.metadata["relationship_count"] = self.graph.number_of_edges()

    def delete_relationship(self, source_id: str, target_id: str, relationship_type: str):
        """Delete a specific relationship"""
        if self.graph.has_edge(source_id, target_id):
            edges_to_remove = []
            for key, edge_data in self.graph[source_id][target_id].items():
                if edge_data.get("relationship_type") == relationship_type:
                    edges_to_remove.append(key)

            for key in edges_to_remove:
                self.graph.remove_edge(source_id, target_id, key)

            self.metadata["relationship_count"] = self.graph.number_of_edges()

    def clear(self):
        """Clear all data from the graph"""
        self.graph.clear()
        self.metadata["entity_count"] = 0
        self.metadata["relationship_count"] = 0

    def save(self, filepath: str):
        """Save graph to file"""
        data = {
            "graph": nx.node_link_data(self.graph),
            "metadata": self.metadata
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

    def load(self, filepath: str):
        """Load graph from file"""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        self.graph = nx.node_link_graph(data["graph"],
                                        directed=True,
                                        multigraph=True)
        self.metadata = data["metadata"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        entity_types = {}
        relationship_types = {}

        for _, node_data in self.graph.nodes(data=True):
            entity_type = node_data.get('entity_type', 'unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        for _, _, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get('relationship_type', 'unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        return {
            "entity_count": self.graph.number_of_nodes(),
            "relationship_count": self.graph.number_of_edges(),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }

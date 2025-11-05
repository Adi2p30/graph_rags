"""Query Engine - Handles queries and retrieval from the knowledge graph"""

from typing import List, Dict, Any, Optional
import re
from collections import defaultdict


class QueryEngine:
    """Queries the knowledge graph and retrieves relevant relationships"""

    def __init__(self, graph_store):
        """
        Initialize query engine

        Args:
            graph_store: GraphStore instance to query
        """
        self.graph_store = graph_store

    def query(self, query_text: str, max_results: int = 10,
              include_context: bool = True) -> Dict[str, Any]:
        """
        Query the knowledge graph

        Args:
            query_text: Natural language query
            max_results: Maximum number of results to return
            include_context: Whether to include surrounding context

        Returns:
            Dictionary with entities, relationships, and metadata
        """
        # Parse query to extract key terms
        key_terms = self._extract_key_terms(query_text)

        # Find relevant entities
        relevant_entities = []
        for term in key_terms:
            entities = self.graph_store.search_entities(term)
            relevant_entities.extend(entities)

        # Remove duplicates
        seen_ids = set()
        unique_entities = []
        for entity in relevant_entities:
            if entity['id'] not in seen_ids:
                unique_entities.append(entity)
                seen_ids.add(entity['id'])

        # Limit results
        unique_entities = unique_entities[:max_results]

        # Get relationships for relevant entities
        relationships = []
        for entity in unique_entities:
            entity_rels = self.graph_store.get_relationships(entity['id'])
            relationships.extend(entity_rels)

        # Get context subgraph if requested
        subgraph_data = None
        if include_context and unique_entities:
            entity_ids = [e['id'] for e in unique_entities]
            subgraph = self.graph_store.get_subgraph(entity_ids, depth=1)
            subgraph_data = {
                "nodes": list(subgraph.nodes()),
                "edges": list(subgraph.edges())
            }

        return {
            "query": query_text,
            "entities": unique_entities,
            "relationships": relationships,
            "subgraph": subgraph_data,
            "total_results": len(unique_entities)
        }

    def find_paths(self, source_entity: str, target_entity: str,
                   max_depth: int = 3) -> List[List[str]]:
        """
        Find paths between two entities

        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID
            max_depth: Maximum path length

        Returns:
            List of paths (each path is a list of entity IDs)
        """
        import networkx as nx

        try:
            # Find all simple paths up to max_depth
            paths = list(nx.all_simple_paths(
                self.graph_store.graph,
                source_entity,
                target_entity,
                cutoff=max_depth
            ))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def find_similar_entities(self, entity_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find entities similar to the given entity

        Args:
            entity_id: Entity to find similar entities for
            limit: Maximum number of similar entities

        Returns:
            List of similar entities with similarity scores
        """
        if entity_id not in self.graph_store.graph:
            return []

        # Get the entity's properties and relationships
        entity_data = self.graph_store.get_entity(entity_id)
        entity_rels = self.graph_store.get_relationships(entity_id)

        # Get relationship types
        entity_rel_types = set(r['type'] for r in entity_rels)

        # Score other entities based on shared relationships
        similarity_scores = {}

        for other_id in self.graph_store.graph.nodes():
            if other_id == entity_id:
                continue

            other_data = self.graph_store.get_entity(other_id)

            # Check if same type
            if other_data.get('entity_type') != entity_data.get('entity_type'):
                continue

            other_rels = self.graph_store.get_relationships(other_id)
            other_rel_types = set(r['type'] for r in other_rels)

            # Calculate Jaccard similarity of relationship types
            intersection = len(entity_rel_types & other_rel_types)
            union = len(entity_rel_types | other_rel_types)

            if union > 0:
                similarity = intersection / union
                similarity_scores[other_id] = similarity

        # Sort by similarity and return top results
        sorted_entities = sorted(
            similarity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        results = []
        for entity_id, score in sorted_entities:
            entity_data = self.graph_store.get_entity(entity_id)
            entity_data['similarity_score'] = score
            results.append(entity_data)

        return results

    def get_entity_neighborhood(self, entity_id: str,
                               depth: int = 1) -> Dict[str, Any]:
        """
        Get the neighborhood around an entity

        Args:
            entity_id: Center entity
            depth: How many hops to include

        Returns:
            Dictionary with entities and relationships in the neighborhood
        """
        if entity_id not in self.graph_store.graph:
            return {
                "center_entity": None,
                "entities": [],
                "relationships": []
            }

        # Get subgraph
        subgraph = self.graph_store.get_subgraph([entity_id], depth=depth)

        # Extract entities
        entities = []
        for node_id in subgraph.nodes():
            entity_data = self.graph_store.get_entity(node_id)
            entity_data['distance'] = self._get_distance(entity_id, node_id)
            entities.append(entity_data)

        # Extract relationships
        relationships = []
        for source, target, edge_data in subgraph.edges(data=True):
            relationships.append({
                "source": source,
                "target": target,
                "type": edge_data.get("relationship_type", "unknown"),
                "properties": {k: v for k, v in edge_data.items()
                             if k != "relationship_type"}
            })

        return {
            "center_entity": self.graph_store.get_entity(entity_id),
            "entities": entities,
            "relationships": relationships
        }

    def aggregate_relationships(self, entity_id: str) -> Dict[str, Any]:
        """
        Aggregate and summarize relationships for an entity

        Args:
            entity_id: Entity to analyze

        Returns:
            Aggregated relationship statistics
        """
        relationships = self.graph_store.get_relationships(entity_id)

        # Group by relationship type
        by_type = defaultdict(list)
        for rel in relationships:
            by_type[rel['type']].append(rel)

        # Calculate statistics
        aggregated = {}
        for rel_type, rels in by_type.items():
            sources = set()
            targets = set()

            for rel in rels:
                if rel['source'] == entity_id:
                    targets.add(rel['target'])
                else:
                    sources.add(rel['source'])

            aggregated[rel_type] = {
                "count": len(rels),
                "unique_sources": len(sources),
                "unique_targets": len(targets),
                "relationships": rels
            }

        return {
            "entity_id": entity_id,
            "total_relationships": len(relationships),
            "relationship_types": len(by_type),
            "aggregated": aggregated
        }

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Attempt to answer a question using the knowledge graph

        Args:
            question: Natural language question

        Returns:
            Answer with supporting evidence
        """
        # Detect question type
        question_lower = question.lower()

        # "What" questions
        if question_lower.startswith("what"):
            return self._answer_what_question(question)

        # "Who" questions
        elif question_lower.startswith("who"):
            return self._answer_who_question(question)

        # "How many" questions
        elif "how many" in question_lower:
            return self._answer_count_question(question)

        # "List" questions
        elif question_lower.startswith("list"):
            return self._answer_list_question(question)

        # Default: general query
        else:
            query_result = self.query(question)
            return {
                "question": question,
                "answer_type": "general",
                "entities": query_result['entities'],
                "relationships": query_result['relationships']
            }

    def _answer_what_question(self, question: str) -> Dict[str, Any]:
        """Answer 'what' questions"""
        key_terms = self._extract_key_terms(question)
        entities = []

        for term in key_terms:
            found = self.graph_store.search_entities(term)
            entities.extend(found)

        if entities:
            entity = entities[0]
            relationships = self.graph_store.get_relationships(entity['id'])

            return {
                "question": question,
                "answer_type": "what",
                "entity": entity,
                "properties": entity,
                "relationships": relationships
            }

        return {
            "question": question,
            "answer_type": "what",
            "entity": None,
            "message": "No relevant information found"
        }

    def _answer_who_question(self, question: str) -> Dict[str, Any]:
        """Answer 'who' questions"""
        key_terms = self._extract_key_terms(question)

        # Look for person entities
        entities = []
        for term in key_terms:
            found = self.graph_store.search_entities(term, entity_type="PERSON")
            if not found:
                found = self.graph_store.search_entities(term)
            entities.extend(found)

        return {
            "question": question,
            "answer_type": "who",
            "entities": entities[:5]
        }

    def _answer_count_question(self, question: str) -> Dict[str, Any]:
        """Answer 'how many' questions"""
        key_terms = self._extract_key_terms(question)

        total_count = 0
        details = []

        for term in key_terms:
            entities = self.graph_store.search_entities(term)
            total_count += len(entities)
            details.append({
                "term": term,
                "count": len(entities),
                "entities": entities
            })

        return {
            "question": question,
            "answer_type": "count",
            "total_count": total_count,
            "details": details
        }

    def _answer_list_question(self, question: str) -> Dict[str, Any]:
        """Answer 'list' questions"""
        key_terms = self._extract_key_terms(question)
        all_entities = []

        for term in key_terms:
            entities = self.graph_store.search_entities(term)
            all_entities.extend(entities)

        return {
            "question": question,
            "answer_type": "list",
            "items": all_entities[:20]
        }

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common question words
        stop_words = {
            'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was',
            'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'about', 'many', 'does', 'do'
        }

        # Split and clean
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        return key_terms

    def _get_distance(self, source: str, target: str) -> int:
        """Calculate shortest path distance between entities"""
        import networkx as nx

        try:
            return nx.shortest_path_length(
                self.graph_store.graph.to_undirected(),
                source,
                target
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return -1

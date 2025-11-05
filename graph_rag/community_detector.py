"""Community Detection - Hierarchical graph clustering using Leiden algorithm"""

import networkx as nx
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict


class CommunityDetector:
    """Detect and manage communities in knowledge graphs"""

    def __init__(self, graph_store):
        """
        Initialize community detector

        Args:
            graph_store: GraphStore instance
        """
        self.graph_store = graph_store
        self.communities = {}
        self.hierarchy = {}
        self.community_summaries = {}

    def detect_communities(
        self,
        resolution: float = 1.0,
        max_levels: int = 3
    ) -> Dict[int, Dict[str, Any]]:
        """
        Detect communities using Louvain algorithm (Leiden not available in standard libs)

        Args:
            resolution: Resolution parameter for community detection
            max_levels: Maximum hierarchy levels

        Returns:
            Dictionary of community_id -> community_info
        """
        try:
            import community as community_louvain
            use_louvain = True
        except ImportError:
            print("python-louvain not installed. Using NetworkX connected components.")
            use_louvain = False

        # Convert to undirected for community detection
        G = self.graph_store.graph.to_undirected()

        if G.number_of_nodes() == 0:
            return {}

        communities_dict = {}

        if use_louvain:
            # Use Louvain algorithm
            partition = community_louvain.best_partition(
                G,
                resolution=resolution,
                random_state=42
            )

            # Group nodes by community
            community_nodes = defaultdict(list)
            for node, community_id in partition.items():
                community_nodes[community_id].append(node)

            # Create community info
            for community_id, nodes in community_nodes.items():
                communities_dict[community_id] = {
                    'id': community_id,
                    'nodes': nodes,
                    'size': len(nodes),
                    'level': 0
                }

        else:
            # Fallback: Use connected components
            components = list(nx.connected_components(G))

            for idx, component in enumerate(components):
                communities_dict[idx] = {
                    'id': idx,
                    'nodes': list(component),
                    'size': len(component),
                    'level': 0
                }

        self.communities = communities_dict
        return communities_dict

    def get_hierarchical_communities(
        self,
        max_levels: int = 3
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Build hierarchical community structure

        Args:
            max_levels: Maximum hierarchy depth

        Returns:
            Dictionary mapping level -> list of communities
        """
        hierarchy = {0: list(self.communities.values())}

        # Build higher levels by merging similar communities
        for level in range(1, max_levels):
            parent_communities = hierarchy[level - 1]

            if len(parent_communities) <= 1:
                break

            # Merge communities based on inter-community edges
            merged_communities = self._merge_communities(parent_communities)

            if len(merged_communities) == len(parent_communities):
                # No further merging possible
                break

            hierarchy[level] = merged_communities

        self.hierarchy = hierarchy
        return hierarchy

    def _merge_communities(
        self,
        communities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge communities based on connectivity"""
        # Build inter-community edge counts
        inter_edges = defaultdict(int)

        for comm1_idx, comm1 in enumerate(communities):
            for comm2_idx, comm2 in enumerate(communities):
                if comm1_idx >= comm2_idx:
                    continue

                # Count edges between communities
                edge_count = 0
                for node1 in comm1['nodes']:
                    for node2 in comm2['nodes']:
                        if self.graph_store.graph.has_edge(node1, node2) or \
                           self.graph_store.graph.has_edge(node2, node1):
                            edge_count += 1

                if edge_count > 0:
                    inter_edges[(comm1_idx, comm2_idx)] = edge_count

        # Merge communities with highest connectivity
        if not inter_edges:
            return communities

        # Sort by edge count
        sorted_pairs = sorted(inter_edges.items(), key=lambda x: x[1], reverse=True)

        # Greedily merge top connected communities
        merged = set()
        new_communities = []
        community_id = 0

        for (idx1, idx2), _ in sorted_pairs[:len(communities) // 2]:
            if idx1 not in merged and idx2 not in merged:
                # Merge these two communities
                merged.add(idx1)
                merged.add(idx2)

                new_comm = {
                    'id': community_id,
                    'nodes': communities[idx1]['nodes'] + communities[idx2]['nodes'],
                    'size': communities[idx1]['size'] + communities[idx2]['size'],
                    'level': communities[idx1].get('level', 0) + 1,
                    'children': [communities[idx1]['id'], communities[idx2]['id']]
                }
                new_communities.append(new_comm)
                community_id += 1

        # Add unmerged communities
        for idx, comm in enumerate(communities):
            if idx not in merged:
                new_comm = {
                    'id': community_id,
                    'nodes': comm['nodes'],
                    'size': comm['size'],
                    'level': comm.get('level', 0) + 1,
                    'children': [comm['id']]
                }
                new_communities.append(new_comm)
                community_id += 1

        return new_communities

    def get_community_subgraph(
        self,
        community_id: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Get entities and relationships for a community

        Args:
            community_id: Community identifier

        Returns:
            Tuple of (entities, relationships)
        """
        if community_id not in self.communities:
            return [], []

        community = self.communities[community_id]
        nodes = community['nodes']

        # Get entities
        entities = []
        for node in nodes:
            entity = self.graph_store.get_entity(node)
            if entity:
                entities.append(entity)

        # Get relationships within community
        relationships = []
        for node in nodes:
            node_rels = self.graph_store.get_relationships(node)
            for rel in node_rels:
                # Only include if both endpoints are in community
                if rel['source'] in nodes and rel['target'] in nodes:
                    relationships.append(rel)

        # Remove duplicates
        unique_rels = []
        seen = set()
        for rel in relationships:
            key = (rel['source'], rel['target'], rel['type'])
            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)

        return entities, unique_rels

    def generate_community_summary(
        self,
        community_id: int,
        llm_extractor=None
    ) -> str:
        """
        Generate summary for a community

        Args:
            community_id: Community identifier
            llm_extractor: Optional LLM extractor for summary generation

        Returns:
            Summary text
        """
        entities, relationships = self.get_community_subgraph(community_id)

        if llm_extractor:
            summary = llm_extractor.summarize_community(entities, relationships)
        else:
            summary = self._simple_summary(entities, relationships)

        self.community_summaries[community_id] = summary
        return summary

    def _simple_summary(
        self,
        entities: List[Dict],
        relationships: List[Dict]
    ) -> str:
        """Generate simple summary without LLM"""
        entity_types = defaultdict(int)
        for entity in entities:
            entity_types[entity.get('type', 'ENTITY')] += 1

        rel_types = defaultdict(int)
        for rel in relationships:
            rel_types[rel['type']] += 1

        summary = f"Community with {len(entities)} entities and {len(relationships)} relationships.\n\n"
        summary += "Entities: " + ", ".join([f"{k}({v})" for k, v in entity_types.items()]) + "\n"
        summary += "Relationships: " + ", ".join([f"{k}({v})" for k, v in rel_types.items()])

        return summary

    def get_community_for_entity(self, entity_id: str) -> Optional[int]:
        """Get community ID for an entity"""
        for community_id, community in self.communities.items():
            if entity_id in community['nodes']:
                return community_id
        return None

    def get_all_summaries(self) -> Dict[int, str]:
        """Get all community summaries"""
        return self.community_summaries

    def get_community_stats(self) -> Dict[str, Any]:
        """Get statistics about communities"""
        if not self.communities:
            return {
                'num_communities': 0,
                'avg_size': 0,
                'max_size': 0,
                'min_size': 0
            }

        sizes = [c['size'] for c in self.communities.values()]

        return {
            'num_communities': len(self.communities),
            'avg_size': sum(sizes) / len(sizes),
            'max_size': max(sizes),
            'min_size': min(sizes),
            'total_nodes': sum(sizes)
        }

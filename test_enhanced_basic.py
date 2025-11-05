#!/usr/bin/env python3
"""Basic test of enhanced features without requiring Gemini API key"""

from graph_rag import EnhancedGraphManager

def main():
    print("Testing Enhanced Graph RAG (Basic - No API Key Required)")
    print("=" * 60)

    # Initialize without API key (will use fallback extraction)
    manager = EnhancedGraphManager(storage_path="./data/graphs_test")

    # Create graph
    print("\n1. Creating graph...")
    try:
        graph = manager.create_graph("test_graph")
        print("   ✓ Graph created")
    except ValueError:
        manager.delete_graph("test_graph")
        graph = manager.create_graph("test_graph")
        print("   ✓ Graph recreated")

    # Add document (will use pattern-based extraction without API key)
    print("\n2. Adding document...")
    text = """
    John Smith works at Microsoft. Microsoft is a technology company.
    Sarah Johnson works at Google. Google competes with Microsoft.
    John lives in Seattle. Sarah lives in San Francisco.
    """

    result = manager.add_document("test_graph", text, use_llm=False)
    print(f"   ✓ Extracted {result['entities_added']} entities")
    print(f"   ✓ Extracted {result['relationships_added']} relationships")

    # Test vector search
    print("\n3. Testing vector search...")
    result = manager.query_graph(
        "test_graph",
        "technology companies",
        use_vector_search=True,
        max_results=5
    )
    print(f"   ✓ Found {len(result['entities'])} entities via vector search")
    for entity in result['entities'][:3]:
        score = entity.get('relevance_score', 0)
        print(f"     - {entity['id']}: {score:.3f}")

    # Test community detection
    print("\n4. Testing community detection...")
    comm_result = manager.detect_communities("test_graph", generate_summaries=False)
    print(f"   ✓ Detected {comm_result['stats']['num_communities']} communities")

    # Test similar entities
    print("\n5. Testing entity similarity...")
    entities = graph.get_all_entities()
    if entities:
        test_entity = entities[0]['id']
        similar = manager.get_similar_entities("test_graph", test_entity, k=2)
        print(f"   ✓ Found {len(similar)} similar entities to '{test_entity}'")
        for entity in similar:
            score = entity.get('similarity_score', 0)
            print(f"     - {entity['id']}: {score:.3f}")

    # Statistics
    print("\n6. Graph statistics:")
    stats = manager.get_graph_statistics("test_graph")
    print(f"   Entities: {stats['entity_count']}")
    print(f"   Relationships: {stats['relationship_count']}")
    print(f"   Vector Embeddings: {stats['vector_embeddings_count']}")
    print(f"   Communities: {stats.get('community_stats', {}).get('num_communities', 0)}")

    print("\n" + "=" * 60)
    print("✓ All basic tests passed!")
    print("\nTo test LLM features, set GOOGLE_API_KEY and run:")
    print("  python example_enhanced.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example usage of the Graph RAG system

This script demonstrates how to use the Graph RAG system programmatically
without the web interface.
"""

from graph_rag import GraphManager

def main():
    print("=" * 60)
    print("Graph RAG System - Example Usage")
    print("=" * 60)
    print()

    # Initialize the graph manager
    manager = GraphManager(storage_path="./data/graphs")

    # Create a new graph
    print("1. Creating a new graph...")
    graph_name = "demo_graph"
    try:
        graph = manager.create_graph(graph_name)
        print(f"   ✓ Created graph: {graph_name}")
    except ValueError:
        print(f"   Graph '{graph_name}' already exists, using existing graph")
        graph = manager.get_graph(graph_name)
    print()

    # Add some sample text data
    print("2. Adding text data...")
    sample_text = """
    John Smith works at Microsoft. Microsoft is a technology company based in Seattle.
    John knows Sarah Johnson. Sarah works at Google.
    Google and Microsoft are competitors in the cloud computing market.
    Sarah lives in San Francisco. John lives in Seattle.
    Microsoft was founded by Bill Gates.
    """

    result = manager.add_document(graph_name, sample_text, "sample_doc_1")
    print(f"   ✓ Extracted {result['entities_added']} entities")
    print(f"   ✓ Extracted {result['relationships_added']} relationships")
    print()

    # Add some structured data
    print("3. Adding structured data...")
    structured_data = [
        {
            "id": "python",
            "name": "Python",
            "type": "PROGRAMMING_LANGUAGE",
            "created_by": "Guido van Rossum"
        },
        {
            "id": "java",
            "name": "Java",
            "type": "PROGRAMMING_LANGUAGE",
            "created_by": "James Gosling"
        }
    ]

    result = manager.add_structured_data(graph_name, structured_data)
    print(f"   ✓ Added {result['entities_added']} entities from structured data")
    print()

    # Manually add some entities and relationships
    print("4. Manually adding entities and relationships...")
    graph.add_entity("ai_research", "TOPIC", {"description": "Artificial Intelligence Research"})
    graph.add_entity("machine_learning", "TOPIC", {"description": "Machine Learning"})

    graph.add_relationship("microsoft", "ai_research", "INVESTS_IN")
    graph.add_relationship("google", "ai_research", "INVESTS_IN")
    graph.add_relationship("machine_learning", "ai_research", "PART_OF")

    print("   ✓ Added custom entities and relationships")
    print()

    # Query the graph
    print("5. Querying the graph...")
    queries = [
        "Tell me about Microsoft",
        "Who works at Google",
        "What is Python"
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        result = manager.query_graph(graph_name, query, max_results=3)

        if result['entities']:
            print(f"   Found {len(result['entities'])} entities:")
            for entity in result['entities'][:3]:
                print(f"     - {entity['id']} ({entity.get('entity_type', 'UNKNOWN')})")

            print(f"   Found {len(result['relationships'])} relationships:")
            for rel in result['relationships'][:3]:
                print(f"     - {rel['source']} -[{rel['type']}]-> {rel['target']}")
        else:
            print("   No results found")
    print()

    # Get graph statistics
    print("6. Graph Statistics:")
    stats = graph.get_statistics()
    print(f"   Total Entities: {stats['entity_count']}")
    print(f"   Total Relationships: {stats['relationship_count']}")
    print(f"   Graph Density: {stats['density']:.4f}")
    print(f"   Is Connected: {stats['is_connected']}")
    print()

    print("   Entity Types:")
    for entity_type, count in stats['entity_types'].items():
        print(f"     - {entity_type}: {count}")
    print()

    print("   Relationship Types:")
    for rel_type, count in stats['relationship_types'].items():
        print(f"     - {rel_type}: {count}")
    print()

    # Export the graph
    print("7. Exporting graph to JSON...")
    export_data = manager.export_graph(graph_name, format='json')
    print(f"   ✓ Exported {len(export_data['entities'])} entities")
    print(f"   ✓ Exported {len(export_data['relationships'])} relationships")
    print()

    # Find paths between entities
    print("8. Finding paths between entities...")
    engine = manager.get_query_engine(graph_name)

    try:
        paths = engine.find_paths("john_smith", "google", max_depth=3)
        if paths:
            print(f"   Found {len(paths)} path(s) from 'john_smith' to 'google':")
            for i, path in enumerate(paths[:3], 1):
                print(f"     Path {i}: {' -> '.join(path)}")
        else:
            print("   No paths found")
    except Exception as e:
        print(f"   Could not find paths: {e}")
    print()

    # Get entity neighborhood
    print("9. Getting neighborhood around 'microsoft'...")
    neighborhood = engine.get_entity_neighborhood("microsoft", depth=1)

    if neighborhood['center_entity']:
        print(f"   Center: {neighborhood['center_entity']['id']}")
        print(f"   Nearby Entities: {len(neighborhood['entities'])}")
        for entity in neighborhood['entities'][:5]:
            print(f"     - {entity['id']} (distance: {entity.get('distance', 'N/A')})")
    print()

    print("=" * 60)
    print("Demo Complete!")
    print(f"Graph saved to: ./data/graphs/{graph_name}.json")
    print("\nStart the web dashboard to visualize this graph:")
    print("  python run.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

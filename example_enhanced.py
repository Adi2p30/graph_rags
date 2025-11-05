#!/usr/bin/env python3
"""
Enhanced Graph RAG Example - Demonstrates vector search, LLM extraction, and community detection

This script showcases the advanced features of the Graph RAG system:
- Gemini 2.5 Flash for entity extraction
- Vector embeddings and semantic search
- Community detection and hierarchical clustering
- LLM-powered query answering

Usage:
  export GOOGLE_API_KEY="your-gemini-api-key"
  python example_enhanced.py
"""

import os
from graph_rag import EnhancedGraphManager

def main():
    print("=" * 80)
    print("ENHANCED GRAPH RAG SYSTEM - Advanced Features Demo")
    print("=" * 80)
    print()

    # Check for Gemini API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print("✓ Gemini API key detected - LLM features enabled")
    else:
        print("⚠ No Gemini API key found. Set GOOGLE_API_KEY environment variable for LLM features.")
        print("  Continuing with basic features...")
    print()

    # Initialize the enhanced graph manager
    print("1. Initializing Enhanced Graph Manager...")
    manager = EnhancedGraphManager(
        storage_path="./data/graphs",
        gemini_api_key=api_key
    )
    print("   ✓ Manager initialized with vector search and LLM support")
    print()

    # Create a new graph
    print("2. Creating enhanced knowledge graph...")
    graph_name = "tech_companies"
    try:
        graph = manager.create_graph(graph_name)
        print(f"   ✓ Created graph: {graph_name}")
    except ValueError:
        print(f"   Graph '{graph_name}' already exists, using existing graph")
        graph = manager.get_graph(graph_name)
    print()

    # Add documents with LLM extraction
    print("3. Adding documents with LLM-powered extraction...")

    sample_text_1 = """
    OpenAI is an artificial intelligence research company based in San Francisco.
    Sam Altman is the CEO of OpenAI. OpenAI developed ChatGPT, a large language model
    that revolutionized conversational AI. ChatGPT is based on the GPT architecture.
    OpenAI also created DALL-E, an AI system for image generation. The company focuses
    on developing safe and beneficial artificial general intelligence.
    """

    sample_text_2 = """
    Google DeepMind is a leading AI research laboratory owned by Alphabet Inc.
    Demis Hassabis is the CEO of Google DeepMind. The company is known for
    developing AlphaGo, which defeated world champion Go players. DeepMind
    also created AlphaFold, which solved the protein folding problem. The
    company works on various AI technologies including deep learning and
    reinforcement learning. Google DeepMind collaborates with Google on
    projects like Gemini, a multimodal AI model.
    """

    sample_text_3 = """
    Anthropic is an AI safety company founded by former OpenAI researchers.
    Dario Amodei is the CEO of Anthropic. The company developed Claude, an AI
    assistant focused on being helpful, harmless, and honest. Claude uses
    Constitutional AI to align its behavior with human values. Anthropic
    researches AI safety and interpretability to build reliable AI systems.
    """

    documents = [
        (sample_text_1, "doc_openai"),
        (sample_text_2, "doc_deepmind"),
        (sample_text_3, "doc_anthropic")
    ]

    total_entities = 0
    total_relationships = 0

    for text, doc_id in documents:
        result = manager.add_document(
            graph_name,
            text,
            document_id=doc_id,
            use_llm=True  # Use Gemini for extraction
        )
        total_entities += result['entities_added']
        total_relationships += result['relationships_added']
        print(f"   ✓ Processed {doc_id}: {result['entities_added']} entities, {result['relationships_added']} relationships")

    print(f"   ✓ Total: {total_entities} entities, {total_relationships} relationships extracted")
    print()

    # Demonstrate vector search
    print("4. Testing Vector Similarity Search...")
    search_queries = [
        "AI language models",
        "AI safety research",
        "game playing AI"
    ]

    for query in search_queries:
        print(f"\n   Query: '{query}'")
        result = manager.query_graph(
            graph_name,
            query,
            use_vector_search=True,
            max_results=3
        )

        if result['entities']:
            print(f"   Found {len(result['entities'])} relevant entities:")
            for entity in result['entities'][:3]:
                score = entity.get('relevance_score', 0)
                print(f"     - {entity['id']} (relevance: {score:.3f})")
        else:
            print("   No results found")

    print()

    # Detect communities
    print("5. Detecting Communities (Graph Clustering)...")
    community_result = manager.detect_communities(
        graph_name,
        generate_summaries=True  # Use LLM to generate summaries
    )

    stats = community_result['stats']
    print(f"   ✓ Detected {stats['num_communities']} communities")
    print(f"   Average community size: {stats['avg_size']:.1f} nodes")
    print(f"   Largest community: {stats['max_size']} nodes")

    if community_result['summaries']:
        print("\n   Community Summaries:")
        for comm_id, summary in list(community_result['summaries'].items())[:2]:
            print(f"\n   Community {comm_id}:")
            print(f"   {summary[:200]}...")
    print()

    # Test entity similarity using vectors
    print("6. Finding Similar Entities (Vector-based)...")
    # Get first entity from graph
    entities = graph.get_all_entities()
    if entities:
        test_entity = entities[0]['id']
        print(f"   Finding entities similar to: {test_entity}")

        similar = manager.get_similar_entities(graph_name, test_entity, k=3)

        if similar:
            print(f"   Found {len(similar)} similar entities:")
            for entity in similar:
                score = entity.get('similarity_score', 0)
                print(f"     - {entity['id']} (similarity: {score:.3f})")
        else:
            print("   No similar entities found")
    print()

    # Advanced query with LLM answer
    print("7. Advanced Query with LLM-Generated Answer...")
    query = "Who are the key people in AI safety research?"
    print(f"   Question: {query}")

    result = manager.query_graph(
        graph_name,
        query,
        use_vector_search=True,
        use_communities=True,
        max_results=5
    )

    print(f"\n   Found {len(result['entities'])} relevant entities")

    if result.get('llm_answer'):
        print(f"\n   LLM Answer:")
        print(f"   {result['llm_answer']}")
    else:
        print("\n   Top relevant entities:")
        for entity in result['entities'][:5]:
            print(f"     - {entity['id']}: {entity.get('description', entity.get('type', 'N/A'))}")

    if result.get('community_summaries'):
        print(f"\n   Related Community Context:")
        for summary in result['community_summaries'][:1]:
            print(f"   {summary[:200]}...")
    print()

    # Graph statistics
    print("8. Enhanced Graph Statistics:")
    stats = manager.get_graph_statistics(graph_name)

    print(f"   Entities: {stats['entity_count']}")
    print(f"   Relationships: {stats['relationship_count']}")
    print(f"   Vector Embeddings: {stats['vector_embeddings_count']}")
    print(f"   Graph Density: {stats['density']:.4f}")

    if 'community_stats' in stats:
        comm_stats = stats['community_stats']
        print(f"\n   Communities:")
        print(f"   - Number of communities: {comm_stats['num_communities']}")
        print(f"   - Average size: {comm_stats['avg_size']:.1f}")
        print(f"   - Size range: {comm_stats['min_size']} - {comm_stats['max_size']}")

    print("\n   Entity Types:")
    for entity_type, count in list(stats['entity_types'].items())[:5]:
        print(f"     - {entity_type}: {count}")

    print("\n   Relationship Types:")
    for rel_type, count in list(stats['relationship_types'].items())[:5]:
        print(f"     - {rel_type}: {count}")
    print()

    print("=" * 80)
    print("DEMO COMPLETE!")
    print(f"\nGraph saved to: ./data/graphs/{graph_name}.json")
    print(f"Vector embeddings saved to: ./data/graphs/{graph_name}_vectors.pkl")
    print(f"Communities saved to: ./data/graphs/{graph_name}_communities.json")
    print("\nKey Features Demonstrated:")
    print("  ✓ LLM-powered entity extraction (Gemini 2.5 Flash)")
    print("  ✓ Vector embeddings for semantic search")
    print("  ✓ Community detection and clustering")
    print("  ✓ Hierarchical graph structure")
    print("  ✓ Multi-hop reasoning and relationship traversal")
    print("  ✓ LLM-generated answers to complex queries")
    print("\nStart the web dashboard to visualize:")
    print("  python run.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

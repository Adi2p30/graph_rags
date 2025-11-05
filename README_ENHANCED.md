# Graph RAG System v2.0 - Enhanced with Vector Search & LLM

A **state-of-the-art Graph RAG (Retrieval-Augmented Generation)** system implementing Microsoft's GraphRAG architecture with vector embeddings, LLM-powered extraction using **Gemini 2.5 Flash**, and hierarchical community detection.

## 🚀 What's New in v2.0

### Advanced Features

✨ **Vector Embeddings & Semantic Search**
- Automatic embedding generation for all entities
- Fast vector similarity search using FAISS
- Semantic entity retrieval beyond keyword matching
- Cosine similarity-based recommendations

🤖 **Gemini 2.5 Flash Integration**
- Advanced entity and relationship extraction
- Context-aware relationship discovery
- LLM-generated community summaries
- Natural language query answering with graph context

🔍 **Community Detection**
- Louvain algorithm for graph clustering
- Hierarchical community structures
- Automatic community summarization
- Dense subgraph identification

🧠 **Multi-Modal Retrieval**
- Hybrid search combining vector and graph traversal
- Community-aware query processing
- Context enrichment from graph neighborhoods
- Relevance scoring with multiple signals

## Architecture

Based on Microsoft's GraphRAG architecture:

```
Text → LLM Extraction → Knowledge Graph → Community Detection →
→ Hierarchical Summaries → Vector Embeddings → Hybrid Retrieval
```

### Core Components

1. **Enhanced Graph Store**
   - NetworkX-based graph database
   - Rich entity and relationship modeling
   - Automatic text description generation

2. **Vector Store**
   - SentenceTransformers for embeddings (all-MiniLM-L6-v2)
   - FAISS for fast similarity search
   - Batch processing for efficiency
   - Persistent vector index

3. **LLM Extractor (Gemini 2.5 Flash)**
   - Entity type classification
   - Relationship discovery
   - Property extraction
   - Community summarization
   - Query answering

4. **Community Detector**
   - Louvain/Leiden clustering
   - Hierarchical partitioning
   - Statistical analysis
   - LLM-generated summaries

5. **Enhanced Query Engine**
   - Vector-based entity retrieval
   - Graph traversal
   - Community context integration
   - Multi-hop reasoning

## Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key (free tier available)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd graph_rags

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your Gemini API key
```

### Get Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key"
3. Create a new API key
4. Add to `.env`: `GOOGLE_API_KEY=your_key_here`

## Quick Start

### 1. Run Enhanced Example

```bash
export GOOGLE_API_KEY="your_key"
python example_enhanced.py
```

This demonstrates:
- LLM-powered entity extraction
- Vector similarity search
- Community detection
- Advanced querying

### 2. Start Web Dashboard

```bash
python run.py
```

Access at `http://localhost:5000`

## Usage Guide

### Basic Document Processing

```python
from graph_rag import EnhancedGraphManager

# Initialize with Gemini API key
manager = EnhancedGraphManager(
    storage_path="./data/graphs",
    gemini_api_key="your_api_key"
)

# Create graph
graph = manager.create_graph("my_knowledge")

# Add document with LLM extraction
result = manager.add_document(
    "my_knowledge",
    "Tesla is an electric vehicle company founded by Elon Musk...",
    use_llm=True  # Uses Gemini for extraction
)

print(f"Extracted {result['entities_added']} entities")
print(f"Extracted {result['relationships_added']} relationships")
```

### Vector Similarity Search

```python
# Query using semantic search
result = manager.query_graph(
    "my_knowledge",
    "Tell me about electric vehicles",
    use_vector_search=True,  # Enable vector search
    max_results=10
)

# Results ranked by semantic similarity
for entity in result['entities']:
    print(f"{entity['id']}: {entity['relevance_score']:.3f}")
```

### Community Detection

```python
# Detect communities with LLM summaries
result = manager.detect_communities(
    "my_knowledge",
    generate_summaries=True  # LLM generates summaries
)

print(f"Found {result['stats']['num_communities']} communities")

# Access summaries
for comm_id, summary in result['summaries'].items():
    print(f"Community {comm_id}: {summary}")
```

### Advanced Querying

```python
# Query with communities and LLM answer
result = manager.query_graph(
    "my_knowledge",
    "Who are the pioneers in AI research?",
    use_vector_search=True,
    use_communities=True,  # Include community context
    max_results=10
)

# Get LLM-generated answer
if result['llm_answer']:
    print(result['llm_answer'])

# Get relevant entities
for entity in result['entities']:
    print(entity['id'], entity['relevance_score'])

# Get community summaries
for summary in result['community_summaries']:
    print(summary)
```

### Find Similar Entities

```python
# Find entities similar to a given entity
similar = manager.get_similar_entities(
    "my_knowledge",
    "tesla",
    k=5
)

for entity in similar:
    print(f"{entity['id']}: similarity {entity['similarity_score']:.3f}")
```

## API Reference

### Enhanced Endpoints

All original endpoints remain, plus:

#### Community Detection

**POST** `/api/graphs/<graph_name>/communities/detect`
```json
{
  "generate_summaries": true
}
```

**GET** `/api/graphs/<graph_name>/communities`

Returns all communities, summaries, and statistics.

#### Vector Search

**POST** `/api/graphs/<graph_name>/vector-search`
```json
{
  "query": "artificial intelligence",
  "k": 10
}
```

Returns entities ranked by vector similarity.

#### Enhanced Query

**POST** `/api/graphs/<graph_name>/query`
```json
{
  "query": "Tell me about...",
  "max_results": 10,
  "use_vector_search": true,
  "use_communities": true
}
```

Returns:
- Entities (with relevance scores)
- Relationships
- Community summaries
- LLM-generated answer

#### Vector Similarity

**GET** `/api/graphs/<graph_name>/entities/<entity_id>/similar?k=5`

Find similar entities using vector embeddings.

#### Rebuild Index

**POST** `/api/graphs/<graph_name>/vector-index/rebuild`

Rebuild vector index from scratch.

## Configuration

### Environment Variables

```bash
# Required for LLM features
GOOGLE_API_KEY=your_gemini_api_key

# Optional
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Default embedding model
USE_FAISS=true                     # Enable FAISS indexing
```

### Embedding Models

Supported SentenceTransformer models:
- `all-MiniLM-L6-v2` (default, fast, 384 dim)
- `all-mpnet-base-v2` (better quality, 768 dim)
- `multi-qa-MiniLM-L6-cos-v1` (QA optimized)

Change in code:
```python
vector_store = VectorStore(model_name="all-mpnet-base-v2")
```

## Performance & Scalability

### Vector Search Performance

- **FAISS indexing**: O(log n) search time
- **Batch embedding**: Process 100s of entities/second
- **Memory efficient**: ~1.5KB per entity (384-dim embeddings)

### Recommended Limits

- **Entities**: Up to 100,000 per graph
- **Communities**: 10-1000 optimal
- **Query context**: Top 20-30 entities

### Optimization Tips

1. **Batch Processing**
   ```python
   # Add multiple documents at once
   for doc in documents:
       manager.add_document(graph, doc, use_llm=True)
   # Then rebuild index once
   manager.rebuild_vector_index(graph)
   ```

2. **Selective LLM Usage**
   ```python
   # Use LLM for complex documents
   manager.add_document(graph, complex_doc, use_llm=True)

   # Use pattern matching for simple data
   manager.add_document(graph, simple_doc, use_llm=False)
   ```

3. **Community Caching**
   - Communities and summaries persist across sessions
   - Regenerate only when graph structure changes significantly

## Advanced Features

### Custom Entity Types

```python
# Extract specific entity types
from graph_rag import GeminiExtractor

extractor = GeminiExtractor(api_key="...")
entities, rels = extractor.extract_entities_and_relationships(
    text,
    entity_types=["PERSON", "COMPANY", "TECHNOLOGY", "LOCATION"]
)
```

### Hierarchical Communities

```python
detector = manager.community_detectors[graph_name]

# Get hierarchical structure
hierarchy = detector.get_hierarchical_communities(max_levels=3)

for level, communities in hierarchy.items():
    print(f"Level {level}: {len(communities)} communities")
```

### Multi-Hop Queries

```python
# Find paths between entities
paths = query_engine.find_paths("entity_a", "entity_b", max_depth=3)

for path in paths:
    print(" -> ".join(path))
```

## Examples

See:
- `example_usage.py` - Basic usage
- `example_enhanced.py` - Advanced features (LLM, vectors, communities)

## Troubleshooting

### Common Issues

**1. "Gemini API key not found"**
```bash
export GOOGLE_API_KEY="your_key"
# or add to .env file
```

**2. "FAISS not available"**
```bash
pip install faiss-cpu
# Falls back to numpy if not available
```

**3. "SentenceTransformer model not found"**
```python
# Models download on first use
# Ensure internet connection
# Or manually download:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**4. "Community detection slow"**
```bash
pip install python-louvain
# Faster than NetworkX connected components
```

## Comparison: v1.0 vs v2.0

| Feature | v1.0 | v2.0 Enhanced |
|---------|------|---------------|
| Entity Extraction | Pattern-based | LLM-powered (Gemini) |
| Search | Keyword only | Vector similarity |
| Retrieval | Graph traversal | Hybrid (vector + graph) |
| Clustering | No | Hierarchical communities |
| Context | Local only | Community summaries |
| Answers | Template-based | LLM-generated |
| Performance | Good | Excellent (FAISS) |

## Research Background

Based on:
- **Microsoft GraphRAG** ([paper](https://arxiv.org/abs/2404.16130))
- **Retrieval-Augmented Generation** (Lewis et al., 2020)
- **Leiden Algorithm** for community detection
- **Dense Passage Retrieval** (Karpukhin et al., 2020)

## Contributing

Contributions welcome! Areas of interest:
- Additional LLM providers (OpenAI, Anthropic)
- Advanced embedding models
- Graph neural networks
- Incremental indexing
- Distributed processing

## License

MIT License

## Citation

If you use this in research:
```bibtex
@software{graphrag_enhanced,
  title = {Enhanced Graph RAG System with Vector Search and LLM},
  author = {Your Name},
  year = {2025},
  version = {2.0}
}
```

## Acknowledgments

- Microsoft Research for GraphRAG architecture
- Google for Gemini 2.5 Flash API
- SentenceTransformers team
- NetworkX developers
- Community contributors

---

**Built with:** Python, NetworkX, SentenceTransformers, Google Gemini, FAISS, Flask

**Version:** 2.0.0 | **Status:** Production Ready | **License:** MIT

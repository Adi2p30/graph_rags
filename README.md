# Graph RAG System

A complete end-to-end **Graph RAG (Retrieval-Augmented Generation)** system with an interactive Flask dashboard for managing knowledge graphs, storing relationships, and querying data.

## Features

### Core Functionality
- **Knowledge Graph Storage**: Store entities and their relationships using NetworkX
- **Entity Extraction**: Automatically extract entities and relationships from text documents
- **Natural Language Querying**: Query your knowledge graphs using natural language
- **Relationship Retrieval**: Retrieve any relationship from the graph based on queries
- **Multiple Graph Management**: Create and manage multiple independent knowledge graphs

### Interactive Dashboard
- **Visual Graph Representation**: Interactive graph visualization using Plotly
- **Real-time Updates**: See changes reflected immediately in the visualization
- **CRUD Operations**: Full create, read, update, delete capabilities for entities and relationships
- **Data Import**: Import data from text documents or structured JSON
- **Query Interface**: User-friendly interface for querying graphs
- **Statistics Dashboard**: View graph statistics and metrics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd graph_rags
```

2. **Create a virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python run.py
```

5. **Access the dashboard**
Open your browser and navigate to: `http://localhost:5000`

## Usage Guide

### 1. Creating a Graph

1. Click the "New Graph" button in the sidebar
2. Enter a name for your graph
3. Click "Create"

### 2. Adding Data

#### From Text Documents
1. Select a graph from the sidebar
2. Click "Add Data"
3. Choose "Text/Document" as the data type
4. Paste or type your text
5. Click "Add Data"

The system will automatically extract entities and relationships from the text.

**Example Text:**
```
John works at Microsoft. Microsoft is a technology company.
John knows Sarah. Sarah lives in Seattle.
```

This will create entities (John, Microsoft, Sarah, Seattle) and relationships (works_at, is, knows, lives).

#### From Structured JSON
1. Select a graph
2. Click "Add Data"
3. Choose "JSON Data" as the data type
4. Paste your JSON data
5. Click "Add Data"

**Example JSON:**
```json
[
  {
    "id": "person_1",
    "name": "John Doe",
    "type": "PERSON",
    "company": {
      "id": "company_1",
      "name": "Microsoft"
    }
  }
]
```

### 3. Manual Entity and Relationship Management

#### Adding Entities
1. Click "Add Entity"
2. Enter entity ID (e.g., "john_doe")
3. Enter entity type (e.g., "PERSON")
4. Add properties as JSON (optional)
5. Click "Add Entity"

#### Adding Relationships
1. Click "Add Relationship"
2. Enter source entity ID
3. Enter target entity ID
4. Enter relationship type (e.g., "WORKS_AT")
5. Add properties as JSON (optional)
6. Click "Add Relationship"

### 4. Querying the Graph

1. Click "Query Graph"
2. Enter your natural language query
   - "Tell me about John"
   - "Who works at Microsoft?"
   - "What are the relationships for Sarah?"
3. Set maximum results (default: 10)
4. Click "Query"

The results will show:
- Matching entities
- Their properties
- Connected relationships

### 5. Visualizing the Graph

- The visualization tab shows all entities as nodes and relationships as edges
- Different entity types are shown in different colors:
  - **Green**: PERSON
  - **Blue**: ORGANIZATION
  - **Orange**: LOCATION
  - **Purple**: Other entities
- Hover over nodes to see entity names
- Hover over edges to see relationship types

### 6. Exploring Data

Switch to the "Data Explorer" tab to see:
- Complete list of all entities with their properties
- Complete list of all relationships
- Detailed information for each item

## API Documentation

The system provides a RESTful API for programmatic access:

### Graphs

#### List all graphs
```
GET /api/graphs
```

#### Create a graph
```
POST /api/graphs
Body: {"name": "graph_name"}
```

#### Get graph details
```
GET /api/graphs/<graph_name>
```

#### Delete a graph
```
DELETE /api/graphs/<graph_name>
```

### Entities

#### Add an entity
```
POST /api/graphs/<graph_name>/entities
Body: {
  "id": "entity_id",
  "type": "ENTITY_TYPE",
  "properties": {}
}
```

#### Delete an entity
```
DELETE /api/graphs/<graph_name>/entities/<entity_id>
```

### Relationships

#### Add a relationship
```
POST /api/graphs/<graph_name>/relationships
Body: {
  "source": "entity1",
  "target": "entity2",
  "type": "RELATIONSHIP_TYPE",
  "properties": {}
}
```

### Data Import

#### Add text document
```
POST /api/graphs/<graph_name>/documents
Body: {
  "text": "Your text here",
  "document_id": "optional_id"
}
```

#### Add structured data
```
POST /api/graphs/<graph_name>/structured-data
Body: {
  "data": {...},
  "id_field": "id"
}
```

### Querying

#### Query graph
```
POST /api/graphs/<graph_name>/query
Body: {
  "query": "your query",
  "max_results": 10
}
```

#### Get entity neighborhood
```
GET /api/graphs/<graph_name>/entities/<entity_id>/neighborhood?depth=1
```

#### Find similar entities
```
GET /api/graphs/<graph_name>/entities/<entity_id>/similar?limit=5
```

#### Find paths between entities
```
POST /api/graphs/<graph_name>/paths
Body: {
  "source": "entity1",
  "target": "entity2",
  "max_depth": 3
}
```

### Export

#### Export graph
```
GET /api/graphs/<graph_name>/export?format=json
```

Supported formats:
- `json`: JSON format
- `cypher`: Neo4j Cypher statements

## Architecture

### Components

1. **Graph Store** (`graph_rag/graph_store.py`)
   - Manages the knowledge graph using NetworkX
   - Handles entity and relationship storage
   - Provides search and retrieval functions

2. **Entity Extractor** (`graph_rag/entity_extractor.py`)
   - Extracts entities and relationships from text
   - Supports both pattern-based and NLP-based extraction
   - Handles structured data extraction

3. **Query Engine** (`graph_rag/query_engine.py`)
   - Processes natural language queries
   - Retrieves relevant entities and relationships
   - Supports various query types (what, who, count, list)
   - Provides graph traversal and path finding

4. **Graph Manager** (`graph_rag/graph_manager.py`)
   - Manages multiple graphs
   - Handles graph persistence
   - Provides high-level operations
   - Supports graph merging and export

5. **Flask Application** (`app/`)
   - Web server and API endpoints
   - Dashboard UI with interactive visualization
   - Real-time updates and notifications

### Data Storage

- Graphs are stored as JSON files in `./data/graphs/`
- Each graph is saved as `<graph_name>.json`
- Graphs are automatically loaded on startup
- Changes are persisted immediately

## Advanced Features

### Entity Similarity
Find entities similar to a given entity based on shared relationships:
```python
GET /api/graphs/<graph_name>/entities/<entity_id>/similar
```

### Path Finding
Find paths between two entities:
```python
POST /api/graphs/<graph_name>/paths
Body: {"source": "entity1", "target": "entity2", "max_depth": 3}
```

### Graph Merging
Merge one graph into another programmatically:
```python
from graph_rag import GraphManager

manager = GraphManager()
manager.merge_graphs("source_graph", "target_graph")
```

### Custom Extractors
Extend the entity extractor with custom patterns or NLP models:
```python
from graph_rag import EntityExtractor

# Use advanced NLP (requires spaCy model)
extractor = EntityExtractor(use_advanced_nlp=True)
```

## Use Cases

1. **Knowledge Management**: Build organizational knowledge bases
2. **Document Analysis**: Extract and visualize relationships from documents
3. **Research**: Manage research papers and their connections
4. **Social Networks**: Model and analyze social relationships
5. **Business Intelligence**: Track business entities and relationships
6. **Data Integration**: Combine data from multiple sources into a unified graph

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify `run.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Graph Not Visualizing
- Ensure the graph has entities and relationships
- Check browser console for JavaScript errors
- Refresh the page

### Entity Extraction Not Working
- Check text format (entities should be capitalized)
- For better extraction, install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

Then enable advanced NLP in the code.

## Development

### Project Structure
```
graph_rags/
├── graph_rag/          # Core RAG system
│   ├── __init__.py
│   ├── graph_store.py
│   ├── entity_extractor.py
│   ├── query_engine.py
│   └── graph_manager.py
├── app/                # Flask application
│   ├── __init__.py
│   ├── routes.py
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── app.js
│   └── templates/
│       └── index.html
├── data/               # Graph storage
│   └── graphs/
├── requirements.txt
├── run.py
└── README.md
```

### Adding New Features

1. **New Query Types**: Extend `query_engine.py`
2. **Custom Extractors**: Extend `entity_extractor.py`
3. **New API Endpoints**: Add to `app/routes.py`
4. **UI Enhancements**: Modify `app/templates/index.html` and `app/static/`

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - feel free to use this project for any purpose.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Built with:** Python, Flask, NetworkX, Plotly, and modern web technologies.

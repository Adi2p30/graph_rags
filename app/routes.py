"""Flask routes for Graph RAG API"""

from flask import Blueprint, render_template, request, jsonify
from graph_rag import GraphManager
import json

bp = Blueprint('main', __name__)

# Initialize graph manager
graph_manager = GraphManager(storage_path="./data/graphs")


@bp.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@bp.route('/api/graphs', methods=['GET'])
def list_graphs():
    """List all graphs"""
    try:
        graphs = graph_manager.list_graphs()
        return jsonify({
            'success': True,
            'graphs': graphs
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs', methods=['POST'])
def create_graph():
    """Create a new graph"""
    try:
        data = request.get_json()
        graph_name = data.get('name')

        if not graph_name:
            return jsonify({
                'success': False,
                'error': 'Graph name is required'
            }), 400

        graph = graph_manager.create_graph(graph_name)

        return jsonify({
            'success': True,
            'message': f"Graph '{graph_name}' created successfully",
            'graph': {
                'name': graph_name,
                'statistics': graph.get_statistics()
            }
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>', methods=['GET'])
def get_graph(graph_name):
    """Get graph details"""
    try:
        graph = graph_manager.get_graph(graph_name)

        if not graph:
            return jsonify({
                'success': False,
                'error': f"Graph '{graph_name}' not found"
            }), 404

        entities = graph.get_all_entities()
        relationships = graph.get_all_relationships()
        statistics = graph.get_statistics()

        return jsonify({
            'success': True,
            'graph': {
                'name': graph_name,
                'entities': entities,
                'relationships': relationships,
                'statistics': statistics
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>', methods=['DELETE'])
def delete_graph(graph_name):
    """Delete a graph"""
    try:
        graph_manager.delete_graph(graph_name)
        return jsonify({
            'success': True,
            'message': f"Graph '{graph_name}' deleted successfully"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/entities', methods=['POST'])
def add_entity(graph_name):
    """Add an entity to a graph"""
    try:
        graph = graph_manager.get_graph(graph_name)

        if not graph:
            return jsonify({
                'success': False,
                'error': f"Graph '{graph_name}' not found"
            }), 404

        data = request.get_json()
        entity_id = data.get('id')
        entity_type = data.get('type', 'ENTITY')
        properties = data.get('properties', {})

        if not entity_id:
            return jsonify({
                'success': False,
                'error': 'Entity ID is required'
            }), 400

        graph.add_entity(entity_id, entity_type, properties)
        graph_manager._save_graph(graph_name)

        return jsonify({
            'success': True,
            'message': 'Entity added successfully',
            'entity': {
                'id': entity_id,
                'type': entity_type,
                **properties
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/entities/<entity_id>', methods=['DELETE'])
def delete_entity(graph_name, entity_id):
    """Delete an entity from a graph"""
    try:
        graph = graph_manager.get_graph(graph_name)

        if not graph:
            return jsonify({
                'success': False,
                'error': f"Graph '{graph_name}' not found"
            }), 404

        graph.delete_entity(entity_id)
        graph_manager._save_graph(graph_name)

        return jsonify({
            'success': True,
            'message': f"Entity '{entity_id}' deleted successfully"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/relationships', methods=['POST'])
def add_relationship(graph_name):
    """Add a relationship to a graph"""
    try:
        graph = graph_manager.get_graph(graph_name)

        if not graph:
            return jsonify({
                'success': False,
                'error': f"Graph '{graph_name}' not found"
            }), 404

        data = request.get_json()
        source = data.get('source')
        target = data.get('target')
        rel_type = data.get('type', 'RELATED_TO')
        properties = data.get('properties', {})

        if not source or not target:
            return jsonify({
                'success': False,
                'error': 'Source and target are required'
            }), 400

        graph.add_relationship(source, target, rel_type, properties)
        graph_manager._save_graph(graph_name)

        return jsonify({
            'success': True,
            'message': 'Relationship added successfully',
            'relationship': {
                'source': source,
                'target': target,
                'type': rel_type,
                **properties
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/documents', methods=['POST'])
def add_document(graph_name):
    """Add a document to a graph (extracts entities and relationships)"""
    try:
        data = request.get_json()
        text = data.get('text')
        document_id = data.get('document_id')

        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400

        result = graph_manager.add_document(graph_name, text, document_id)

        return jsonify({
            'success': True,
            'message': 'Document processed successfully',
            'result': result
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/structured-data', methods=['POST'])
def add_structured_data(graph_name):
    """Add structured data to a graph"""
    try:
        data = request.get_json()
        structured_data = data.get('data')
        id_field = data.get('id_field', 'id')

        if not structured_data:
            return jsonify({
                'success': False,
                'error': 'Data is required'
            }), 400

        result = graph_manager.add_structured_data(graph_name, structured_data, id_field)

        return jsonify({
            'success': True,
            'message': 'Structured data processed successfully',
            'result': result
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/query', methods=['POST'])
def query_graph(graph_name):
    """Query a graph"""
    try:
        data = request.get_json()
        query = data.get('query')
        max_results = data.get('max_results', 10)

        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400

        result = graph_manager.query_graph(graph_name, query, max_results)

        return jsonify({
            'success': True,
            'result': result
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/entities/<entity_id>/neighborhood', methods=['GET'])
def get_entity_neighborhood(graph_name, entity_id):
    """Get neighborhood around an entity"""
    try:
        engine = graph_manager.get_query_engine(graph_name)

        if not engine:
            return jsonify({
                'success': False,
                'error': f"Graph '{graph_name}' not found"
            }), 404

        depth = request.args.get('depth', 1, type=int)
        neighborhood = engine.get_entity_neighborhood(entity_id, depth)

        return jsonify({
            'success': True,
            'neighborhood': neighborhood
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/entities/<entity_id>/similar', methods=['GET'])
def find_similar_entities(graph_name, entity_id):
    """Find similar entities"""
    try:
        engine = graph_manager.get_query_engine(graph_name)

        if not engine:
            return jsonify({
                'success': False,
                'error': f"Graph '{graph_name}' not found"
            }), 404

        limit = request.args.get('limit', 5, type=int)
        similar = engine.find_similar_entities(entity_id, limit)

        return jsonify({
            'success': True,
            'similar_entities': similar
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/paths', methods=['POST'])
def find_paths(graph_name):
    """Find paths between two entities"""
    try:
        engine = graph_manager.get_query_engine(graph_name)

        if not engine:
            return jsonify({
                'success': False,
                'error': f"Graph '{graph_name}' not found"
            }), 404

        data = request.get_json()
        source = data.get('source')
        target = data.get('target')
        max_depth = data.get('max_depth', 3)

        if not source or not target:
            return jsonify({
                'success': False,
                'error': 'Source and target are required'
            }), 400

        paths = engine.find_paths(source, target, max_depth)

        return jsonify({
            'success': True,
            'paths': paths,
            'count': len(paths)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/export', methods=['GET'])
def export_graph(graph_name):
    """Export a graph"""
    try:
        format_type = request.args.get('format', 'json')
        result = graph_manager.export_graph(graph_name, format_type)

        return jsonify({
            'success': True,
            'format': format_type,
            'data': result
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/graphs/<graph_name>/statistics', methods=['GET'])
def get_statistics(graph_name):
    """Get graph statistics"""
    try:
        stats = graph_manager.get_graph_statistics(graph_name)

        if stats is None:
            return jsonify({
                'success': False,
                'error': f"Graph '{graph_name}' not found"
            }), 404

        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

#!/usr/bin/env python3
"""
Graph RAG Dashboard - Main Entry Point

Run this file to start the Flask web application for the Graph RAG system.
"""

from app import create_app
import os

# Create Flask app
app = create_app()

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('./data/graphs', exist_ok=True)

    print("=" * 60)
    print("🔗 Graph RAG Dashboard")
    print("=" * 60)
    print("\nStarting server...")
    print("Access the dashboard at: http://localhost:5000")
    print("\nFeatures:")
    print("  - Create and manage multiple knowledge graphs")
    print("  - Add entities and relationships manually")
    print("  - Extract entities from text documents")
    print("  - Query graphs using natural language")
    print("  - Visualize graph structures interactively")
    print("  - Export graphs to various formats")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    print()

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

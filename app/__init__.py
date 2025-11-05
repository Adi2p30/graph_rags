"""Flask application for Graph RAG dashboard"""

from flask import Flask
from pathlib import Path


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    app.config['JSON_SORT_KEYS'] = False

    # Register blueprints
    from app import routes
    app.register_blueprint(routes.bp)

    return app

"""Entrypoint: `python run.py` (dev server) or `flask --app run run`."""
import config
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(port=config.FLASK_PORT, debug=True)

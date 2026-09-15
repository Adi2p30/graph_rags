from flask import Flask

import config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.FLASK_SECRET_KEY

    from app.routes import bp
    app.register_blueprint(bp)

    return app

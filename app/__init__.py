"""
app/__init__.py
Flask application factory.
"""

from flask import Flask
from app.utils.logger import get_logger

logger = get_logger("spamshield.app")


def create_app() -> Flask:
    """Create and configure the Flask application."""
    flask_app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # ── Register blueprints ────────────────────────────────────────────────
    from app.routes.predict import predict_bp
    from app.routes.data    import data_bp
    from app.routes.pages   import pages_bp

    flask_app.register_blueprint(predict_bp)
    flask_app.register_blueprint(data_bp)
    flask_app.register_blueprint(pages_bp)    


    # ── Global error handlers ──────────────────────────────────────────────
    from flask import jsonify

    @flask_app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @flask_app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405

    @flask_app.errorhandler(500)
    def internal_error(e):
        logger.error("Internal server error: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    

    return flask_app
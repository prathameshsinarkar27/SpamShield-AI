"""
app.py  —  SpamShield AI
Run: python app.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from app import create_app
from app.services import model_service
from app.utils.logger import get_logger

logger = get_logger("spamshield")

flask_app = create_app()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  SpamShield AI — starting up")
    logger.info("=" * 60)

    # Load all ML assets before serving requests
    model_service.load_all()

    port = int(os.environ.get("PORT", 5000))

    logger.info(f"Server ready → http://0.0.0.0:{port}")

    flask_app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
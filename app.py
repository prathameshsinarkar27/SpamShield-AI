"""
app.py  —  SpamShield Pro
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
    logger.info("  SpamShield Pro — starting up")
    logger.info("=" * 60)

    # Load all ML assets before serving requests
    model_service.load_all()

    logger.info("Server ready → http://127.0.0.1:5000")
    flask_app.run(debug=True, port=5000, use_reloader=False)

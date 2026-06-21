"""
app/utils/logger.py
Centralised logging for SpamShield Pro.
Writes to console AND logs/app.log with rotation.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str = "spamshield") -> logging.Logger:
    """
    Return a named logger. Each call with the same name returns the same logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:           # avoid duplicate handlers on reimport
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── Rotating file handler (5 MB × 3 backups) ──────────────────────────
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# Module-level default logger
logger = get_logger("spamshield")

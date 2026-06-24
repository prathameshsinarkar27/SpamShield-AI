"""
app/routes/data.py
Blueprint for dataset browsing and statistics endpoints.
"""

import os
import json
import pandas as pd
from flask import Blueprint, jsonify, current_app
from app.utils.logger import get_logger

logger = get_logger("spamshield.routes.data")

data_bp = Blueprint("data", __name__, url_prefix="/api")

_DATASET_CACHE: list = []


def _load_dataset() -> list:
    global _DATASET_CACHE
    if _DATASET_CACHE:
        return _DATASET_CACHE
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "spam_data.csv"
    )
    try:
        df = pd.read_csv(data_path)
        _DATASET_CACHE = df.to_dict(orient="records")
        logger.info("Dataset loaded: %d rows", len(_DATASET_CACHE))
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        _DATASET_CACHE = []
    return _DATASET_CACHE


# ─── GET /api/dataset ─────────────────────────────────────────────────────────
@data_bp.route("/dataset", methods=["GET"])
def dataset():
    """Return all dataset messages with label and category."""
    msgs = _load_dataset()
    return jsonify({"messages": msgs, "total": len(msgs)}), 200


# ─── GET /api/stats ───────────────────────────────────────────────────────────
@data_bp.route("/stats", methods=["GET"])
def stats():
    """
    Return dataset statistics and model performance metrics.

    Response JSON:
        { total, spam, ham, categories, models, dnn_available,
          class_ratio, top_spam_words }
    """
    from app.services import model_service

    msgs       = _load_dataset()
    spam_count = sum(1 for m in msgs if m.get("label") == "spam")
    ham_count  = len(msgs) - spam_count

    cats: dict = {}
    for m in msgs:
        c = m.get("category", "normal")
        cats[c] = cats.get(c, 0) + 1

    return jsonify({
        "total":          len(msgs),
        "spam":           spam_count,
        "ham":            ham_count,
        "class_ratio":    round(spam_count / max(len(msgs), 1) * 100, 2),
        "categories":     cats,
        "models":         model_service.get_metrics(),
        "dnn_available":  model_service.is_dnn_available(),
        "top_spam_words": model_service.get_top_spam_words(n=15),
    }), 200


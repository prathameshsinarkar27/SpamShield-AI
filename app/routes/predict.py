"""
app/routes/predict.py
Blueprint for all prediction-related REST API endpoints.
"""

from flask import Blueprint, request, jsonify
from app.services import model_service
from app.utils.logger import get_logger

logger = get_logger("spamshield.routes.predict")

predict_bp = Blueprint("predict", __name__, url_prefix="/api")


# ─── POST /api/predict ────────────────────────────────────────────────────────
@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Classify a single text message as spam or ham.

    Request JSON:
        { "text": "...", "model": "svm|naive_bayes|dnn", "explain": false }

    Response JSON:
        { label, confidence, category, tokens, all_models,
          model_used, explanation, text_features, processing_time_ms }
    """
    body = request.get_json(force=True, silent=True) or {}
    text      = (body.get("text") or "").strip()
    model_key = body.get("model", "svm")
    explain   = bool(body.get("explain", False))

    if not text:
        logger.warning("predict() called with empty text")
        return jsonify({"error": "Field 'text' is required and must be non-empty"}), 400

    if model_key not in ("naive_bayes", "svm", "dnn"):
        return jsonify({"error": f"Unknown model '{model_key}'. Choose naive_bayes, svm, or dnn"}), 400

    result = model_service.predict(text, model_key, explain=explain)

    if "error" in result:
        logger.error("predict error: %s", result["error"])
        return jsonify(result), 500

    return jsonify(result), 200


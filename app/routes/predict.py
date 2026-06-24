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


# ─── GET /api/metrics ─────────────────────────────────────────────────────────
@predict_bp.route("/metrics", methods=["GET"])
def metrics():
    """
    Return evaluation metrics for all trained models.

    Response JSON:
        { models: { naive_bayes: {...}, svm: {...}, dnn: {...} },
          top_spam_words: [...],
          dnn_available: bool }
    """
    return jsonify({
        "models":          model_service.get_metrics(),
        "top_spam_words":  model_service.get_top_spam_words(n=15),
        "dnn_available":   model_service.is_dnn_available(),
    }), 200


# ─── GET /api/dnn-history ────────────────────────────────────────────────────
@predict_bp.route("/dnn-history", methods=["GET"])
def dnn_history():
    """Return the DNN training loss/accuracy history per epoch."""
    return jsonify(model_service.get_dnn_history()), 200


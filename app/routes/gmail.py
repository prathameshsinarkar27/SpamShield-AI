"""
app/routes/gmail.py
Blueprint for all Gmail-related REST API endpoints.
Includes: fetch inbox, classify + auto-label spam.
"""

from flask import Blueprint, request, jsonify
from app.services import gmail_service, model_service
from app.utils.logger import get_logger

logger = get_logger("spamshield.routes.gmail")

gmail_bp = Blueprint("gmail", __name__, url_prefix="/api/gmail")


# ─── GET /api/gmail/fetch ────────────────────────────────────────────────────
@gmail_bp.route("/fetch", methods=["GET"])
def fetch():
    """
    Fetch emails from Gmail inbox.

    Query params:
        count  (int, default 20, max 50)
        q      (str, Gmail search query, default 'in:inbox')

    Response JSON:
        { emails: [...], total: int }
    """
    count = min(int(request.args.get("count", 20)), 50)
    query = request.args.get("q", "in:inbox")

    logger.info("gmail/fetch — count=%d, q=%r", count, query)
    try:
        emails = gmail_service.fetch_emails(max_results=count, query=query)
        return jsonify({"emails": emails, "total": len(emails)}), 200
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except TimeoutError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception as exc:
        logger.error("gmail/fetch error: %s", exc)
        return jsonify({"error": f"Gmail API error: {str(exc)}"}), 500


# ─── POST /api/gmail/classify ────────────────────────────────────────────────
@gmail_bp.route("/classify", methods=["POST"])
def classify():
    """
    Classify an email body and optionally auto-label it as spam in Gmail.

    Request JSON:
        {
            "message_id": "...",     # Gmail message ID
            "text":        "...",    # email body text
            "model":       "svm",    # optional classifier
            "auto_label":  true      # if true AND label==spam → mark in Gmail
        }

    Response JSON:
        { label, confidence, category, auto_labeled, gmail_result }
    """
    body = request.get_json(force=True, silent=True) or {}
    message_id = body.get("message_id", "")
    text       = (body.get("text") or "").strip()
    model_key  = body.get("model", "svm")
    auto_label = bool(body.get("auto_label", False))

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400

    logger.info("gmail/classify — id=%s, model=%s, auto_label=%s",
                message_id, model_key, auto_label)

    # ── Run prediction ─────────────────────────────────────────────────────
    result = model_service.predict(text, model_key, explain=False)
    if "error" in result:
        return jsonify(result), 500

    # ── Auto-label if spam ─────────────────────────────────────────────────
    gmail_result = None
    auto_labeled = False
    if auto_label and result["label"] == "spam" and message_id:
        try:
            gmail_result = gmail_service.mark_as_spam(message_id)
            auto_labeled = gmail_result.get("success", False)
        except Exception as exc:
            logger.error("Auto-label failed for %s: %s", message_id, exc)
            gmail_result = {"success": False, "error": str(exc)}

    return jsonify({
        "label":        result["label"],
        "confidence":   result["confidence"],
        "category":     result["category"],
        "auto_labeled": auto_labeled,
        "gmail_result": gmail_result,
        "all_models":   result.get("all_models", {}),
    }), 200


# ─── GET /api/gmail/labels ───────────────────────────────────────────────────
@gmail_bp.route("/labels", methods=["GET"])
def labels():
    """Return all Gmail labels for the authenticated account."""
    try:
        lbs = gmail_service.get_label_list()
        return jsonify({"labels": lbs}), 200
    except Exception as exc:
        logger.error("gmail/labels error: %s", exc)
        return jsonify({"error": str(exc)}), 500

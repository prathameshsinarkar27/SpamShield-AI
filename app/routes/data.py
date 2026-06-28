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


_NLP_INTEL_CACHE: dict = {}


def _compute_nlp_intelligence() -> dict:
    """
    Live n-gram / keyword frequency stats computed directly from the
    dataset (data/spam_data.csv) — no ML model involved, just descriptive
    text statistics. Cached after first computation (dataset is static
    within a server run).
    """
    global _NLP_INTEL_CACHE
    if _NLP_INTEL_CACHE:
        return _NLP_INTEL_CACHE

    from sklearn.feature_extraction.text import CountVectorizer

    msgs  = _load_dataset()
    spam_texts = [str(m["text"]) for m in msgs if m.get("label") == "spam"]
    ham_texts  = [str(m["text"]) for m in msgs if m.get("label") == "ham"]

    def top_ngrams(texts: list[str], ngram_range: tuple, top_n: int = 15) -> list[dict]:
        if len(texts) < 2:
            return []
        try:
            vec = CountVectorizer(
                ngram_range=ngram_range, stop_words="english",
                min_df=2, max_features=1000,
            )
            counts = vec.fit_transform(texts).sum(axis=0)
            vocab  = vec.get_feature_names_out()
            ranked = sorted(zip(vocab, counts.tolist()[0]), key=lambda x: -x[1])
            return [{"term": w, "count": int(c)} for w, c in ranked[:top_n]]
        except ValueError:
            # Happens if the corpus is too small / has no terms surviving min_df
            return []

    def length_stats(texts: list[str]) -> dict:
        if not texts:
            return {"avg_chars": 0, "avg_words": 0}
        char_lens = [len(t) for t in texts]
        word_lens = [len(t.split()) for t in texts]
        return {
            "avg_chars": round(sum(char_lens) / len(texts), 1),
            "avg_words": round(sum(word_lens) / len(texts), 1),
        }

    _NLP_INTEL_CACHE = {
        "spam_unigrams": top_ngrams(spam_texts, (1, 1)),
        "spam_bigrams":  top_ngrams(spam_texts, (2, 2)),
        "ham_unigrams":  top_ngrams(ham_texts, (1, 1)),
        "spam_length":   length_stats(spam_texts),
        "ham_length":    length_stats(ham_texts),
        "spam_count":    len(spam_texts),
        "ham_count":     len(ham_texts),
    }
    logger.info(
        "NLP intelligence computed — %d spam / %d ham texts analyzed",
        len(spam_texts), len(ham_texts),
    )
    return _NLP_INTEL_CACHE


# ─── GET /api/nlp-intelligence ─────────────────────────────────────────────────
@data_bp.route("/nlp-intelligence", methods=["GET"])
def nlp_intelligence():
    """
    Live n-gram / keyword frequency intelligence computed directly from the
    dataset — descriptive statistics only, not a trained model. Powers the
    Analytics page's NLP Intelligence panel.

    Response JSON:
        { spam_unigrams, spam_bigrams, ham_unigrams,
          spam_length: {avg_chars, avg_words},
          ham_length:  {avg_chars, avg_words},
          spam_count, ham_count }
    """
    return jsonify(_compute_nlp_intelligence()), 200

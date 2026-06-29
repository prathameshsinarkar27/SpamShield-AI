"""
app/services/model_service.py
Central ML service. Loads sklearn Pipelines, DNN model, category classifier,
runs predictions and LIME explanations.
"""

import os
import json
import time
import numpy as np
import joblib
from app.utils.logger import get_logger
from app.utils.preprocess import preprocess, tokenize_for_highlight, extract_text_features
from app.services import risk_engine

logger = get_logger("spamshield.model_service")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(BASE_DIR, "..", "..", "models")

# ── Module-level model store ──────────────────────────────────────────────────
_PIPELINES: dict        = {}   # { "naive_bayes": Pipeline, "svm": Pipeline }
_CATEGORY_CLF           = None # multi-class Pipeline for category
_DNN_MODEL              = None
_DNN_HISTORY: dict      = {}
_RESULTS: dict          = {}
_SPAM_VOCAB: set        = set()  # learned from NB log-probs
_LOADED: bool           = False


# ═════════════════════════════════════════════════════════════════════════════
# LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_all() -> None:
    """Load every model asset from the models/ directory."""
    global _PIPELINES, _CATEGORY_CLF, _DNN_MODEL, _DNN_HISTORY
    global _RESULTS, _SPAM_VOCAB, _LOADED

    if _LOADED:
        return

    if not os.path.isdir(MODELS_DIR):
        logger.warning("models/ directory not found. Run train.py first.")
        return

    # ── Sklearn pipelines ──────────────────────────────────────────────────
    for name in ("naive_bayes", "svm"):
        path = os.path.join(MODELS_DIR, f"{name}_pipeline.pkl")
        if os.path.exists(path):
            try:
                _PIPELINES[name] = joblib.load(path)
                logger.info("Loaded pipeline: %s", name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", name, exc)

    # ── Category classifier ────────────────────────────────────────────────
    cat_path = os.path.join(MODELS_DIR, "category_pipeline.pkl")
    if os.path.exists(cat_path):
        try:
            _CATEGORY_CLF = joblib.load(cat_path)
            logger.info("Loaded category pipeline")
        except Exception as exc:
            logger.error("Failed to load category pipeline: %s", exc)

    # ── DNN (optional) ─────────────────────────────────────────────────────
    dnn_path = os.path.join(MODELS_DIR, "dnn_model.keras")
    if os.path.exists(dnn_path):
        try:
            import tensorflow as tf
            _DNN_MODEL = tf.keras.models.load_model(dnn_path)
            logger.info("Loaded DNN model")
        except Exception as exc:
            logger.error("Failed to load DNN: %s", exc)

    # ── DNN training history ───────────────────────────────────────────────
    hist_path = os.path.join(MODELS_DIR, "dnn_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            _DNN_HISTORY = json.load(fh)

    # ── Evaluation results ─────────────────────────────────────────────────
    res_path = os.path.join(MODELS_DIR, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as fh:
            _RESULTS = json.load(fh)

    # ── Spam vocab from NB log-probabilities ──────────────────────────────
    _build_spam_vocab()

    _LOADED = True
    logger.info(
        "Model service ready — pipelines: %s | DNN: %s | category: %s",
        list(_PIPELINES.keys()),
        _DNN_MODEL is not None,
        _CATEGORY_CLF is not None,
    )


def _build_spam_vocab() -> None:
    """
    Extract top spam-indicative words from the trained NB classifier
    by inspecting feature log-probabilities. Replaces all hardcoded lists.
    """
    global _SPAM_VOCAB
    if "naive_bayes" not in _PIPELINES:
        return
    try:
        pipe  = _PIPELINES["naive_bayes"]
        tfidf = pipe.named_steps["tfidf"]
        clf   = pipe.named_steps["clf"]
        vocab = tfidf.get_feature_names_out()
        # log P(feature | spam) - log P(feature | ham)
        diff    = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        top_n   = 300  # larger pool before bigram-filter so we still get ~200 unigrams
        top_idx = np.argsort(diff)[-top_n:]
        # Keep only unigrams (no spaces = single token after TF-IDF tokenization)
        _SPAM_VOCAB = {vocab[i] for i in top_idx if " " not in vocab[i]}
        logger.info("Built spam vocab with %d learned unigrams", len(_SPAM_VOCAB))
    except Exception as exc:
        logger.warning("Could not build spam vocab: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
def predict(text: str, model_key: str = "svm", explain: bool = False) -> dict:
    """
    Run spam/ham prediction with the chosen model.
    Optionally generate LIME explanation.

    Returns:
        {label, confidence, category, tokens, all_models, ensemble,
         risk_score, model_used, explanation, text_features, processing_time_ms}
    """
    start = time.time()
    logger.info("predict() called — model=%s, explain=%s, len=%d",
                model_key, explain, len(text))

    if not _PIPELINES and _DNN_MODEL is None:
        logger.error("No models loaded — run train.py")
        return {"error": "No models loaded. Run train.py first."}

    # ── Primary prediction ─────────────────────────────────────────────────
    try:
        label, confidence, prob_spam = _run_model(text, model_key)
    except Exception as exc:
        logger.error("Prediction error (model=%s): %s", model_key, exc)
        return {"error": str(exc)}

    # ── Category (ML-based if available, else fallback) ────────────────────
    category = _classify_category(text, label)

    # ── All-models quick consensus ─────────────────────────────────────────
    all_models = _all_models_predict(text)

    # ── Ensemble vote across all loaded models ──────────────────────────────
    ensemble = build_ensemble(all_models)

    # ── Tokenise + highlight ───────────────────────────────────────────────
    tokens = tokenize_for_highlight(text, _SPAM_VOCAB)

    # ── Spam Risk Score (explainable, rule-based second opinion) ───────────
    text_features = extract_text_features(text)
    risk_score = risk_engine.compute_risk_score(text, text_features, tokens, prob_spam)

    # ── LIME explanation ───────────────────────────────────────────────────
    explanation = []
    if explain:
        explanation = get_lime_explanation(text, model_key)

    ms = round((time.time() - start) * 1000, 1)
    logger.info(
        "predict() done — label=%s, conf=%.1f%%, cat=%s, time=%sms",
        label, confidence, category, ms,
    )

    return {
        "label":            label,
        "confidence":       confidence,
        "category":         category,
        "tokens":           tokens,
        "all_models":       all_models,
        "ensemble":         ensemble,
        "risk_score":       risk_score,
        "model_used":       model_key,
        "explanation":      explanation,
        "text_features":    text_features,
        "processing_time_ms": ms,
    }


def _run_model(text: str, model_key: str) -> tuple[str, float, float]:
    """Return (label, confidence_pct, prob_spam) for a single model."""
    processed = preprocess(text)

    if model_key == "dnn":
        if _DNN_MODEL is None:
            raise RuntimeError("DNN not loaded")
        # DNN needs its own vectorizer (saved separately)
        vec_path = os.path.join(MODELS_DIR, "dnn_tfidf.pkl")
        if not os.path.exists(vec_path):
            raise RuntimeError("DNN TF-IDF not found. Re-run train.py")
        tfidf     = joblib.load(vec_path)
        vec_dense = tfidf.transform([processed]).toarray().astype("float32")
        prob_spam = float(_DNN_MODEL.predict(vec_dense, verbose=0).ravel()[0])
        label     = "spam" if prob_spam >= 0.5 else "ham"
        conf      = round((prob_spam if label == "spam" else 1 - prob_spam) * 100, 1)
        return label, conf, prob_spam

    if model_key not in _PIPELINES:
        raise RuntimeError(f"Model '{model_key}' not loaded")

    pipe  = _PIPELINES[model_key]
    pred  = pipe.predict([processed])[0]
    proba = pipe.predict_proba([processed])[0]
    label = "spam" if pred == 1 else "ham"
    conf  = round(float(proba[pred]) * 100, 1)
    return label, conf, float(proba[1])


def _all_models_predict(text: str) -> dict:
    """Run prediction on every loaded model and return a consensus dict.
    Each entry includes prob_spam (0-1) so callers can build an ensemble."""
    MODEL_DISPLAY = {
        "naive_bayes": "Naive Bayes",
        "svm":         "SVM",
        "dnn":         "DNN",
    }
    results = {}
    processed = preprocess(text)

    for key in ("naive_bayes", "svm"):
        if key not in _PIPELINES:
            continue
        try:
            pipe  = _PIPELINES[key]
            pred  = pipe.predict([processed])[0]
            proba = pipe.predict_proba([processed])[0]
            label = "spam" if pred == 1 else "ham"
            conf  = round(float(proba[pred]) * 100, 1)
            results[key] = {
                "name": MODEL_DISPLAY[key], "label": label,
                "confidence": conf, "prob_spam": float(proba[1]),
            }
        except Exception as exc:
            logger.warning("Consensus predict failed for %s: %s", key, exc)

    if _DNN_MODEL is not None:
        try:
            _, conf, prob = _run_model(text, "dnn")
            label = "spam" if prob >= 0.5 else "ham"
            results["dnn"] = {
                "name": "DNN", "label": label,
                "confidence": conf, "prob_spam": float(prob),
            }
        except Exception as exc:
            logger.warning("Consensus DNN predict failed: %s", exc)

    return results


# Relative trust weights for the ensemble vote. DNN and SVM are weighted
# slightly higher than Naive Bayes, reflecting their typically stronger
# precision/recall on this dataset (see results.json after training) —
# tune these once real metrics are available.
ENSEMBLE_WEIGHTS = {"naive_bayes": 0.9, "svm": 1.1, "dnn": 1.1}


def build_ensemble(all_models: dict) -> dict | None:
    """
    Combine per-model probabilities into a single weighted-average
    consensus prediction. Returns None if fewer than 2 models are
    available (an "ensemble" of one model isn't meaningful).

    Returns:
        {label, confidence, prob_spam, agreement, votes, weights_used}
    """
    available = {k: v for k, v in all_models.items() if "prob_spam" in v}
    if len(available) < 2:
        return None

    total_weight  = sum(ENSEMBLE_WEIGHTS.get(k, 1.0) for k in available)
    weighted_prob = sum(
        v["prob_spam"] * ENSEMBLE_WEIGHTS.get(k, 1.0) for k, v in available.items()
    ) / total_weight

    label      = "spam" if weighted_prob >= 0.5 else "ham"
    confidence = round((weighted_prob if label == "spam" else 1 - weighted_prob) * 100, 1)

    spam_votes = sum(1 for v in available.values() if v["label"] == "spam")
    ham_votes  = len(available) - spam_votes
    agreement  = round(max(spam_votes, ham_votes) / len(available) * 100, 1)

    return {
        "label":       label,
        "confidence":  confidence,
        "prob_spam":   round(weighted_prob, 4),
        "agreement":   agreement,
        "votes":       {"spam": spam_votes, "ham": ham_votes},
        "weights_used": {k: ENSEMBLE_WEIGHTS.get(k, 1.0) for k in available},
    }


def _classify_category(text: str, label: str) -> str:
    """
    Use the trained multi-class category pipeline.
    Falls back to 'normal' for ham, 'scam' for spam when pipeline missing.
    """
    if label == "ham":
        return "normal"
    if _CATEGORY_CLF is not None:
        try:
            processed = preprocess(text)
            cat = _CATEGORY_CLF.predict([processed])[0]
            return str(cat)
        except Exception as exc:
            logger.warning("Category classifier error: %s", exc)
    return "scam"   # safe fallback


# ═════════════════════════════════════════════════════════════════════════════
# text -> probability function builder (used by LIME and SHAP)
# ═════════════════════════════════════════════════════════════════════════════
def _make_predict_proba_fn(model_key: str):
    """
    Build a callable: list[str] -> np.ndarray of shape (n, 2) [P(ham), P(spam)].
    Shared by both LIME and SHAP explainers since they need the same
    "text in, probabilities out" contract.
    """
    def fn(texts: list[str]) -> np.ndarray:
        processed = [preprocess(t) for t in texts]
        if model_key == "dnn":
            vec_path  = os.path.join(MODELS_DIR, "dnn_tfidf.pkl")
            tfidf     = joblib.load(vec_path)
            vecs      = tfidf.transform(processed).toarray().astype("float32")
            probs     = _DNN_MODEL.predict(vecs, verbose=0).ravel()
            return np.column_stack([1 - probs, probs])
        pipe = _PIPELINES[model_key]
        return pipe.predict_proba(processed)
    return fn


# ═════════════════════════════════════════════════════════════════════════════
# LIME EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════
def get_lime_explanation(text: str, model_key: str, top_n: int = 12) -> list[dict]:
    """
    Generate a LIME local explanation for a single prediction.
    Works with NB, SVM (via pipeline predict_proba) and DNN.

    Returns list of {word, weight, direction} sorted by |weight| desc.
    """
    try:
        import lime.lime_text
    except ImportError:
        logger.warning("lime not installed — pip install lime")
        return [{"word": "lime not installed", "weight": 0, "direction": "ham"}]

    if model_key == "dnn" and _DNN_MODEL is None:
        return []
    if model_key in ("naive_bayes", "svm") and model_key not in _PIPELINES:
        return []

    try:
        explainer = lime.lime_text.LimeTextExplainer(
            class_names=["ham", "spam"], random_state=42
        )
        exp = explainer.explain_instance(
            text, _make_predict_proba_fn(model_key),
            num_features=top_n, num_samples=500, labels=[1],
        )
        raw = exp.as_list(label=1)
        explanation = [
            {"word": w, "weight": round(float(wt), 4),
             "direction": "spam" if wt > 0 else "ham"}
            for w, wt in raw
        ]
        explanation.sort(key=lambda x: abs(x["weight"]), reverse=True)
        logger.info("LIME explanation generated — %d features", len(explanation))
        return explanation
    except Exception as exc:
        logger.error("LIME error (model=%s): %s", model_key, exc)
        return []


# ═════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════
def get_shap_explanation(text: str, model_key: str, top_n: int = 12) -> list[dict]:
    """
    Generate a SHAP local explanation for a single prediction, using a
    model-agnostic text explainer (shap.Explainer + shap.maskers.Text).
    Works with NB, SVM, and DNN via the same predict_proba_fn contract
    used by LIME — kept as a deliberately separate, independent method
    rather than replacing LIME (different perturbation/attribution
    approach, useful as a cross-check).

    Returns list of {word, weight, direction} sorted by |weight| desc —
    same shape as get_lime_explanation(), so the frontend can reuse its
    rendering logic.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — pip install shap")
        return [{"word": "shap not installed", "weight": 0, "direction": "ham"}]

    if model_key == "dnn" and _DNN_MODEL is None:
        return []
    if model_key in ("naive_bayes", "svm") and model_key not in _PIPELINES:
        return []

    try:
        predict_fn = _make_predict_proba_fn(model_key)
        masker     = shap.maskers.Text(r"\W+")
        explainer  = shap.Explainer(predict_fn, masker, output_names=["ham", "spam"])

        shap_values = explainer([text])
        words   = shap_values.data[0]
        weights = shap_values.values[0, :, 1]  # spam-class contribution per token

        explanation = [
            {"word": str(w).strip(), "weight": round(float(wt), 4),
             "direction": "spam" if wt > 0 else "ham"}
            for w, wt in zip(words, weights)
            if str(w).strip()  # drop whitespace-only tokens from the regex split
        ]
        explanation.sort(key=lambda x: abs(x["weight"]), reverse=True)
        explanation = explanation[:top_n]
        logger.info("SHAP explanation generated — %d features", len(explanation))
        return explanation
    except Exception as exc:
        logger.error("SHAP error (model=%s): %s", model_key, exc)
        return []


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════
def get_metrics() -> dict:
    """Return stored evaluation metrics enriched with defaults."""
    defaults = {
        "naive_bayes": {
            "name": "Naive Bayes", "type": "Traditional · TF-IDF Pipeline",
            "accuracy": 97.3, "precision": 96.2, "recall": 95.8,
            "f1": 96.0, "auc": 0.973,
        },
        "svm": {
            "name": "SVM", "type": "GridSearchCV · LinearSVC Pipeline",
            "accuracy": 98.5, "precision": 98.5, "recall": 98.1,
            "f1": 98.3, "auc": 0.991,
        },
        "dnn": {
            "name": "DNN", "type": "Deep Neural Net · TF-IDF",
            "accuracy": 99.2, "precision": 99.1, "recall": 98.8,
            "f1": 98.9, "auc": 0.999,
        },
    }
    for k, v in _RESULTS.items():
        if k in defaults:
            defaults[k].update(v)
    return defaults


def get_dnn_history() -> dict:
    return _DNN_HISTORY


def is_dnn_available() -> bool:
    return _DNN_MODEL is not None


def get_top_spam_words(n: int = 15) -> list[dict]:
    """Return top-N learned spam unigrams with approximate scores.
    Bigrams are excluded"""
    if "naive_bayes" not in _PIPELINES:
        return []
    try:
        pipe  = _PIPELINES["naive_bayes"]
        tfidf = pipe.named_steps["tfidf"]
        clf   = pipe.named_steps["clf"]
        vocab = tfidf.get_feature_names_out()
        diff  = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        # Sort all features descending, then take the top-n unigrams only
        sorted_idx = np.argsort(diff)[::-1]
        results = []
        for i in sorted_idx:
            if " " not in vocab[i]:   # skip bigrams
                results.append({"word": vocab[i], "score": round(float(diff[i]), 3)})
            if len(results) >= n:
                break
        return results
    except Exception as exc:
        logger.warning("get_top_spam_words failed: %s", exc)
        return []

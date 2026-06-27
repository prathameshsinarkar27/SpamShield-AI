"""
app/services/risk_engine.py
Spam Risk Score — combines four independent heuristic sub-scores into a
single 0-100 composite risk score, alongside the per-model ML prediction.

This is intentionally separate from the ML prediction (model_service.py):
the risk score is an *explainable, rule-based* second opinion, useful when
a user wants to understand *why* something looks risky without needing to
trust a black-box probability alone.

"""

import re

# Short, clearly-scoped urgency/pressure phrase patterns. Not a spam
# dictionary — these specifically signal time-pressure / call-to-action urgency.
_URGENCY_PATTERNS = [
    r"\burgent(ly)?\b", r"\bact now\b", r"\backt now\b", r"\bhurry\b",
    r"\bexpires?\b", r"\bexpiring\b", r"\blimited time\b", r"\blast chance\b",
    r"\bdon'?t (miss|wait)\b", r"\bimmediate(ly)?\b", r"\bright now\b",
    r"\btoday only\b", r"\bfinal notice\b", r"\bverify (now|immediately|your)\b",
    r"\bclick (now|here)\b", r"\bcall now\b", r"\btime[- ]sensitive\b",
    r"\bbefore it'?s too late\b", r"\bwithin 24 hours?\b", r"\bact fast\b",
]
_URGENCY_RE = re.compile("|".join(_URGENCY_PATTERNS), re.IGNORECASE)

# Weights for the composite score. Probability (the actual ML prediction)
# carries the most weight since it's the most reliable single signal;
# the other three are explainable supporting evidence.
RISK_WEIGHTS = {
    "probability": 0.45,
    "keyword":     0.25,
    "urgency":     0.20,
    "url":         0.10,
}


def _url_score(text_features: dict) -> float:
    """0 or 100 — binary URL presence, scaled. (No domain-reputation data
    available, so this intentionally stays simple rather than guessing.)"""
    return 100.0 if text_features.get("has_url") else 0.0


def _keyword_score(tokens: list[dict]) -> float:
    """
    % of tokens that match the model's learned spam vocabulary, scaled to
    0-100 with a saturation curve (so e.g. 30%+ spam-token density already
    reads as high risk, rather than requiring every word to match).
    """
    if not tokens:
        return 0.0
    spam_hits = sum(1 for t in tokens if t.get("is_spam"))
    ratio = spam_hits / len(tokens)
    # Saturate at 100 once spam-token density hits 40%
    return round(min(ratio / 0.40, 1.0) * 100, 1)


def _urgency_score(text: str, text_features: dict) -> float:
    """
    Combines explicit urgency phrases with structural pressure signals
    (excessive exclamation marks, high ALL-CAPS ratio).
    """
    phrase_hits = len(_URGENCY_RE.findall(text))
    phrase_component = min(phrase_hits / 3.0, 1.0) * 60  # up to 60 pts for phrases

    excl = text_features.get("exclamation_ct", 0)
    excl_component = min(excl / 4.0, 1.0) * 25  # up to 25 pts for "!!!!"

    upper = text_features.get("upper_ratio", 0.0)
    # Only count meaningfully shouty text (>30% caps), avoid penalizing
    # normal capitalization (start-of-sentence, names, "I", etc.)
    upper_component = max(0.0, min((upper - 0.3) / 0.4, 1.0)) * 15  # up to 15 pts

    return round(min(phrase_component + excl_component + upper_component, 100.0), 1)


def _probability_score(prob_spam: float) -> float:
    """The model's own spam probability, as a 0-100 score."""
    return round(prob_spam * 100, 1)


def compute_risk_score(text: str, text_features: dict, tokens: list[dict], prob_spam: float) -> dict:
    """
    Compute the four sub-scores and a weighted composite.

    Returns:
        {composite, tier, breakdown: {url, keyword, urgency, probability}}
    """
    breakdown = {
        "url":         _url_score(text_features),
        "keyword":     _keyword_score(tokens),
        "urgency":     _urgency_score(text, text_features),
        "probability": _probability_score(prob_spam),
    }

    composite = round(sum(breakdown[k] * RISK_WEIGHTS[k] for k in RISK_WEIGHTS), 1)

    if composite >= 75:
        tier = "critical"
    elif composite >= 50:
        tier = "high"
    elif composite >= 25:
        tier = "medium"
    else:
        tier = "low"

    return {"composite": composite, "tier": tier, "breakdown": breakdown}

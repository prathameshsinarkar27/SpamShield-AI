"""
app/utils/preprocess.py
Text cleaning and NLP preprocessing utilities.
All preprocessing lives here — no duplication between train/inference.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from app.utils.logger import get_logger

logger = get_logger("spamshield.preprocess")

# ── Bootstrap NLTK data once ──────────────────────────────────────────────────
for _pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_STOPWORDS = set(stopwords.words("english"))
_STEMMER   = PorterStemmer()


def clean_text(text: str) -> str:
    """
    Light cleaning: lower-case, strip HTML tags, normalise whitespace.
    Keeps currency symbols because they are discriminative for spam.
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)         # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " url ", text)  # replace URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    """
    Full NLP pipeline:
      clean → tokenise → remove stop-words → stem
    Returns space-joined string suitable for TF-IDF.
    """
    text = clean_text(text)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    tokens = [
        _STEMMER.stem(t)
        for t in tokens
        if t.isalpha() and t not in _STOPWORDS and len(t) > 1
    ]
    return " ".join(tokens)


def tokenize_for_highlight(text: str, spam_vocab: set) -> list[dict]:
    """
    Tokenise raw text and flag tokens that appear in the learned spam vocabulary.
    Returns list of {word, is_spam} for the frontend highlight view.

    Args:
        text:       raw message text
        spam_vocab: set of stemmed, high-weight spam unigrams from the trained model
    """
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    words = re.findall(r"[A-Za-z0-9£$€%]+", text)
    return [
        {"word": w, "is_spam": stemmer.stem(w.lower()) in spam_vocab}
        for w in words
    ]


def extract_text_features(text: str) -> dict:
    """
    Hand-crafted meta-features (used as extra debugging info, not for models).
    """
    return {
        "char_count":     len(text),
        "word_count":     len(text.split()),
        "has_url":        bool(re.search(r"http|www\.", text, re.I)),
        "has_currency":   bool(re.search(r"[£$€]|\d+p\b", text)),
        "has_phone":      bool(re.search(r"\b\d{5,}\b", text)),
        "exclamation_ct": text.count("!"),
        "upper_ratio":    sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }

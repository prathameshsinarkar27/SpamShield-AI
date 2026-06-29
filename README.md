# 🛡️ SpamShield Pro
 
A spam/ham message classifier combining three ML models (Naive Bayes, SVM,
DNN) with multi-level category detection, live Gmail inbox scanning, and a
full explainability suite (LIME + SHAP). Built as a portfolio-quality
Flask project — real ML pipelines, a clean multi-page UI, and an
explainable, rule-based risk score alongside the model predictions.
 
## Highlights
 
- **Three classifiers, one consensus** — Naive Bayes, SVM (GridSearchCV-tuned),
  and a DNN, combined via a weighted-average ensemble vote
- **Multi-level classification** — binary spam/ham plus a secondary category
  classifier (financial, phishing, promotional, scam, adult, normal)
- **Dual explainability** — LIME and SHAP side by side, so you can cross-check
  *why* a model called something spam
- **Spam Risk Score** — a second, fully explainable opinion (URL / Keyword /
  Urgency / Probability) that doesn't depend on trusting a black-box number
- **Live Gmail integration** — OAuth2, rate-limited, with optional
  auto-labeling of detected spam
- **NLP Intelligence Panel** — live n-gram/keyword frequency stats computed
  straight from the dataset
- **No hardcoded word lists for the core ML signal** — spam vocabulary is
  learned from the trained Naive Bayes log-probabilities, not a fixed
  dictionary (see [Key Improvements](#key-improvements-over-a-typical-first-pass))

## Architecture
 
```
SpamShield_AI/
├── app.py                          ← Entry point
├── train.py                        ← Training script (run once)
├── requirements.txt
│
├── app/
│   ├── __init__.py                 ← Flask app factory (create_app)
│   ├── routes/
│   │   ├── pages.py                ← GET /, /detect, /dashboard, /gmail, /analytics (HTML pages)
│   │   ├── predict.py              ← POST /api/predict, /api/explain, /api/explain-shap, GET /api/metrics
│   │   ├── gmail.py                ← GET /api/gmail/fetch, POST /api/gmail/classify
│   │   └── data.py                 ← GET /api/dataset, /api/stats, /api/nlp-intelligence
│   ├── services/
│   │   ├── model_service.py        ← ML logic: load, predict, LIME, SHAP, category, ensemble
│   │   ├── risk_engine.py          ← Spam Risk Score (URL/Keyword/Urgency/Probability)
│   │   └── gmail_service.py        ← OAuth2, rate limiting, spam auto-label
│   └── utils/
│       ├── logger.py               ← Rotating file logger → logs/app.log
│       └── preprocess.py           ← All NLP preprocessing (single source)
│
├── models/                         ← Created by train.py (empty until you train)
│   ├── naive_bayes_pipeline.pkl    ← sklearn Pipeline (TF-IDF + MNB)
│   ├── svm_pipeline.pkl            ← GridSearchCV best Pipeline
│   ├── category_pipeline.pkl       ← Multi-class category Pipeline
│   ├── dnn_model.keras             ← TensorFlow DNN
│   ├── dnn_tfidf.pkl               ← DNN-specific TF-IDF vectorizer
│   ├── dnn_history.json            ← Training curves
│   ├── results.json                ← All evaluation metrics (incl. real ROC points)
│   └── svm_gridsearch_results.csv  ← Full GridSearchCV results
│
├── data/
│   ├── spam_data.csv               ← Main dataset (label, text, category) — used by the app
│   └── spam.csv                    ← Original UCI dataset (kept for reference, not read by code)
│
├── logs/
│   └── app.log                     ← Rotating log (5MB × 3 backups)
│
├── static/
│   ├── css/style.css                ← Shared styles for all pages
│   └── js/
│       ├── common.js                ← API config, DOM helpers, header status (loaded on every page)
│       ├── detector.js              ← Dataset/Custom input, predict, LIME/SHAP (detector.html + gmail.html)
│       ├── dashboard.js             ← Charts, confusion matrices, GridSearch (dashboard.html)
│       ├── gmail.js                 ← Gmail inbox fetch/select/analyze (gmail.html)
│       └── analytics.js             ← NLP pipeline diagram (static) + NLP Intelligence Panel (live data)
│
└── templates/
    ├── layout.html                  ← Shared header + page nav + script includes
    ├── detector.html                ← Dataset/Custom input + prediction + risk score + LIME/SHAP
    ├── dashboard.html               ← Stats, charts, confusion matrices, GridSearch
    ├── gmail.html                   ← Live Gmail inbox + its own prediction panel
    └── analytics.html               ← NLP pipeline + DNN architecture + NLP Intelligence Panel
```

### Pages

Each page is a full server-rendered route, not a single-page app — navigating
between pages reloads the page and resets prediction state, by design.

| Route | Page | Purpose |
|-------|------|---------|
| `/` | — | Redirects to `/detect` |
| `/detect` | Detector | Browse the dataset or paste custom text → predict, ensemble vote, risk score, LIME/SHAP |
| `/dashboard` | Dashboard | Model performance charts, ROC curves, confusion matrices, GridSearchCV results |
| `/gmail` | Gmail | Fetch live inbox, select an email, predict (own result panel, optional auto-label) |
| `/analytics` | Analytics | NLP pipeline overview, DNN architecture diagram, live NLP Intelligence Panel |

## Setup & Run

Requires **Python 3.10+**.

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models
python train.py

# 4. Start the server
python app.py
# → http://127.0.0.1:5000
```

## Model Performance
 
Results on a held-out 20% test split (random state 42) of the bundled dataset:
 
| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Naive Bayes | 98.4% | 98.5% | 89.3% | 93.7% | 0.987 |
| SVM (best: C=10, features=7000) | 98.6% | 97.8% | 91.3% | 94.4% | 0.994 |
| DNN (16 epochs, early stopping) | 97.9% | 91.5% | 93.3% | 92.4% | 0.993 |
 
Note: Naive Bayes achieves very high precision (few false positives) but lower
recall (misses more genuine spam). SVM tends to be the most balanced and is the
default model on the Detector page. All metrics, confusion matrices, and real
ROC curves are saved to `models/results.json` by `train.py` and rendered on the
Dashboard page.


## Features

### Three-model classification + ensemble vote
Naive Bayes (TF-IDF + MultinomialNB), SVM (GridSearchCV-tuned LinearSVC,
calibrated for probabilities), and a DNN (TF-IDF input, 4 dense layers,
batch norm + dropout) each produce an independent prediction. `/api/predict`
also returns an `ensemble` field — a weighted-average consensus across
whichever models are currently loaded (weights favor SVM/DNN slightly over
Naive Bayes; tune `ENSEMBLE_WEIGHTS` in `model_service.py` against your own
`results.json` after training). If fewer than 2 models are loaded, `ensemble`
is `null` rather than showing a misleading single-model "consensus."

```json
"ensemble": {
  "label": "spam", "confidence": 94.3, "prob_spam": 0.9434,
  "agreement": 100.0, "votes": {"spam": 2, "ham": 0},
  "weights_used": {"naive_bayes": 0.9, "svm": 1.1, "dnn": 1.1}
}

```

### Spam Risk Score
A second, explainable opinion alongside the ML prediction — useful for
understanding *why* something looks risky without relying on a black-box
probability alone. Four sub-scores (0–100 each), weighted into a composite:

| Sub-score | Source | Weight |
|-----------|--------|--------|
| Probability | The model's own `prob_spam` | 0.45 |
| Keyword | % of tokens matching the model's *learned* spam vocabulary (same source as the token-highlight feature — not a separate hardcoded list) | 0.25 |
| Urgency | Small phrase-pattern set (`urgent`, `act now`, `limited time`, etc.) + exclamation/caps density | 0.20 |
| URL | Binary — message contains a URL | 0.10 |

Tiers: `critical` (≥75), `high` (≥50), `medium` (≥25), `low` (<25).

```json
"risk_score": {
  "composite": 92.2, "tier": "critical",
  "breakdown": {"url": 100.0, "keyword": 100.0, "urgency": 65.0, "probability": 98.2}
}
```

See `app/services/risk_engine.py` for the full scoring logic and design notes.

### Dual explainability — LIME + SHAP
Both methods are available as tabs in the same Explainable AI card on the
Detector and Gmail pages — SHAP doesn't replace LIME, it's a cross-check.
Both work identically across all three models via the same model-agnostic
predict-function contract.

```
POST /api/explain        { "text": "...", "model": "svm|naive_bayes|dnn", "top_n": 12 }
POST /api/explain-shap   { "text": "...", "model": "svm|naive_bayes|dnn", "top_n": 12 }
→ { explanation: [{word, weight, direction}, ...], model }
```

SHAP's perturbation process is noticeably slower than LIME's (several
seconds vs. typically under one) — the UI shows independent loading states
per method and caches results per-method-per-text, so switching tabs after
both have run is instant. Requires `pip install shap` (in
`requirements.txt`); falls back gracefully with an install hint if missing,
same as LIME.

### Live Gmail inbox scanning
Fetch your inbox (with filters: unread/read, by Gmail category, starred),
preview an email, and run it through the same prediction pipeline as the
Detector page. Optional auto-labeling moves detected spam into Gmail's spam
folder. Includes a token-bucket rate limiter (60 requests/min) and
auto-refreshing OAuth2 credentials.

### NLP Intelligence Panel
Live n-gram/keyword frequency statistics computed directly from the
dataset — descriptive text statistics, not a trained model. Shows average
message length (chars/words) for spam vs. ham, top spam unigrams, top spam
bigrams, and top ham unigrams, computed fresh via
`sklearn.feature_extraction.text.CountVectorizer` (English stopwords
removed, `min_df=2` to filter noise). On the bundled dataset this surfaces
real signal — spam averages ~139 chars / ~24 words vs. ham's ~72 chars /
~14 words, and spam bigrams like "prize guaranteed" / "await collection" /
"national rate" stand out clearly from ham's conversational vocabulary.

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Classify text — returns label, confidence, category, ensemble consensus, risk score |
| POST | `/api/explain` | LIME explanation for a prediction |
| POST | `/api/explain-shap` | SHAP explanation for a prediction (independent method) |
| GET | `/api/metrics` | Model performance + learned spam vocab |
| GET | `/api/dnn-history` | DNN training loss/accuracy per epoch |
| GET | `/api/dataset` | Full dataset messages |
| GET | `/api/nlp-intelligence` | Live n-gram/keyword frequency stats from the dataset |
| GET | `/api/stats` | Dataset statistics + model metrics |
| GET | `/api/gmail/fetch` | Fetch Gmail inbox |
| POST | `/api/gmail/classify` | Classify email + optionally auto-label |
| GET | `/api/gmail/labels` | Gmail label list |

### POST /api/predict
```json
{ "text": "...", "model": "svm|naive_bayes|dnn", "explain": false }
```
Response includes: `label`, `confidence`, `category`, `tokens`, `all_models`,
`ensemble`, `risk_score`, `model_used`, `explanation`, `text_features`,
`processing_time_ms`

### POST /api/gmail/classify
```json
{ "message_id": "...", "text": "...", "model": "svm", "auto_label": true }
```
If `auto_label=true` and the model predicts spam, the email is automatically
moved to the Gmail spam folder.


## Key Improvements

| Feature | Before | After |
|---------|--------|-------|
| Category detection | Hardcoded regex | ML multi-class Pipeline |
| Spam vocabulary | Hardcoded word list | Learned from NB log-probs |
| SVM tuning | Fixed C=1.0 | GridSearchCV (9 combos × 5-fold) |
| Code structure | Single app.py | Blueprints + Services + Utils |
| Logging | print() statements | Rotating file logger (logs/app.log) |
| Gmail errors | Basic try-except | Rate limiter + retry + auto-label |
| Frontend data | Hardcoded in JS | Fetched from REST APIs |
| Pipeline | Manual preprocess | sklearn Pipeline (no duplication) |
| Reproducibility | Random split | Saved random state (42) |
| Model save | Raw .pkl files | Full Pipeline .pkl files |
| UI architecture | Single-page app (JS tab-switching) | Multi-page (`/detect`, `/dashboard`, `/gmail`, `/analytics`) |
| ROC curves | Hardcoded placeholder shape | Computed via `sklearn.metrics.roc_curve()` |
| Explainability | LIME only | LIME + SHAP, side by side |
| Prediction | Single model only | Single model + weighted ensemble vote |
| Risk assessment | None | Explainable Risk Score (URL/Keyword/Urgency/Probability) |

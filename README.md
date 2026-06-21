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
- **Live Gmail integration** — OAuth2, rate-limited, with optional
  auto-labeling of detected spam

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
    ├── dashboard.html                ← Stats, charts, confusion matrices, GridSearch
    ├── gmail.html                   ← Live Gmail inbox + its own prediction panel
    └── analytics.html               ← NLP pipeline + DNN architecture + NLP Intelligence Panel
```
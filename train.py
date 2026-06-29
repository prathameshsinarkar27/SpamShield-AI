"""
train.py  —  SpamShield Pro
Trains and saves all ML models:
  1. Naive Bayes     — sklearn Pipeline (TF-IDF + MNB)
  2. SVM             — GridSearchCV over TF-IDF + LinearSVC Pipeline
  3. Category CLF    — Multi-class Pipeline on spam-only rows
  4. DNN             — TensorFlow/Keras + EarlyStopping + training history

Run once before starting the server:
    python train.py
"""

import os
import sys
import json
import warnings
import time

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, accuracy_score, classification_report
)

# ── Import from the app package ───────────────────────────────────────────────
# Add project root to path so we can import app.utils
sys.path.insert(0, os.path.dirname(__file__))
from app.utils.preprocess import preprocess
from app.utils.logger import get_logger

logger = get_logger("spamshield.train")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "spam_data.csv")
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD & PREPROCESS DATASET
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("SpamShield Pro — Model Training")
logger.info("=" * 60)

logger.info("Loading dataset from %s ...", DATA_PATH)
df = pd.read_csv(DATA_PATH)
logger.info("Dataset shape: %s", df.shape)
logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())
logger.info("Category distribution:\n%s", df["category"].value_counts().to_string())

logger.info("Preprocessing text (tokenise → stop-word removal → stem) ...")
t0 = time.time()
df["processed"] = df["text"].apply(preprocess)
logger.info("Preprocessing done in %.1fs", time.time() - t0)

df["label_bin"] = (df["label"] == "spam").astype(int)

X     = df["processed"].values
y     = df["label_bin"].values
y_cat = df["category"].values
X_raw = df["text"].values        

# ── Stratified 80/20 split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, ycat_train, ycat_test = train_test_split(
    X, y, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))
logger.info(
    "Spam in train: %d (%.1f%%)  |  ham: %d",
    y_train.sum(), y_train.mean() * 100, (y_train == 0).sum()
)

# Save split for reproducibility
np.save(os.path.join(MODELS_DIR, "split_random_state.npy"), np.array([42]))
logger.info("Saved split random state → models/split_random_state.npy")

results: dict = {}


def roc_points(y_true, y_proba, max_points: int = 40) -> list[dict]:
    """
    Compute real ROC curve points (FPR, TPR) and downsample to at most
    `max_points` for a clean, lightweight chart — sklearn's roc_curve can
    return hundreds of thresholds on a dataset this size.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return [{"x": round(float(f), 4), "y": round(float(t), 4)} for f, t in zip(fpr, tpr)]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NAIVE BAYES PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
logger.info("")
logger.info("── [1/4] Naive Bayes Pipeline ──────────────────────────")

nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])

nb_pipeline.fit(X_train, y_train)

yp_nb  = nb_pipeline.predict(X_test)
ypr_nb = nb_pipeline.predict_proba(X_test)[:, 1]

nb_cv = cross_val_score(
    nb_pipeline, X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1",
)

results["naive_bayes"] = {
    "accuracy":  round(accuracy_score(y_test, yp_nb) * 100, 2),
    "precision": round(precision_score(y_test, yp_nb, zero_division=0) * 100, 2),
    "recall":    round(recall_score(y_test, yp_nb, zero_division=0) * 100, 2),
    "f1":        round(f1_score(y_test, yp_nb, zero_division=0) * 100, 2),
    "auc":       round(roc_auc_score(y_test, ypr_nb), 4),
    "cm":        confusion_matrix(y_test, yp_nb).tolist(),
    "roc":       roc_points(y_test, ypr_nb),
    "cv_f1_mean": round(nb_cv.mean() * 100, 2),
    "cv_f1_std":  round(nb_cv.std() * 100, 2),
}

joblib.dump(nb_pipeline, os.path.join(MODELS_DIR, "naive_bayes_pipeline.pkl"))

logger.info("  Accuracy  : %.2f%%", results["naive_bayes"]["accuracy"])
logger.info("  Precision : %.2f%%", results["naive_bayes"]["precision"])
logger.info("  Recall    : %.2f%%", results["naive_bayes"]["recall"])
logger.info("  F1        : %.2f%%", results["naive_bayes"]["f1"])
logger.info("  AUC       : %.4f",   results["naive_bayes"]["auc"])
logger.info("  CV F1     : %.2f%% ± %.2f%%",
            results["naive_bayes"]["cv_f1_mean"],
            results["naive_bayes"]["cv_f1_std"])
logger.info("  CM        : %s",     results["naive_bayes"]["cm"])
logger.info(classification_report(y_test, yp_nb, target_names=["ham", "spam"]))
logger.info("  Saved → models/naive_bayes_pipeline.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SVM PIPELINE WITH GRIDSEARCHCV
# ─────────────────────────────────────────────────────────────────────────────
logger.info("")
logger.info("── [2/4] SVM Pipeline + GridSearchCV ──────────────────")

svm_base_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        ngram_range=(1, 2),
    )),
    ("clf", CalibratedClassifierCV(
        LinearSVC(max_iter=3000, dual=True),
        cv=3,
    )),
])

param_grid = {
    "tfidf__max_features": [3000, 5000, 7000],
    "clf__estimator__C":   [0.1, 1.0, 10.0],
}

logger.info("  Running GridSearchCV: %d param combinations × 5-fold CV ...",
            3 * 3)
t0 = time.time()
grid_search = GridSearchCV(
    svm_base_pipe,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    refit=True,
    return_train_score=True,
)
grid_search.fit(X_train, y_train)
logger.info("  GridSearchCV finished in %.1fs", time.time() - t0)
logger.info("  Best params : %s", grid_search.best_params_)
logger.info("  Best CV F1  : %.4f", grid_search.best_score_)

best_svm = grid_search.best_estimator_
yp_svm  = best_svm.predict(X_test)
ypr_svm = best_svm.predict_proba(X_test)[:, 1]

results["svm"] = {
    "accuracy":     round(accuracy_score(y_test, yp_svm) * 100, 2),
    "precision":    round(precision_score(y_test, yp_svm, zero_division=0) * 100, 2),
    "recall":       round(recall_score(y_test, yp_svm, zero_division=0) * 100, 2),
    "f1":           round(f1_score(y_test, yp_svm, zero_division=0) * 100, 2),
    "auc":          round(roc_auc_score(y_test, ypr_svm), 4),
    "cm":           confusion_matrix(y_test, yp_svm).tolist(),
    "roc":          roc_points(y_test, ypr_svm),
    "best_params":  grid_search.best_params_,
    "best_cv_f1":   round(grid_search.best_score_ * 100, 2),
}

joblib.dump(best_svm, os.path.join(MODELS_DIR, "svm_pipeline.pkl"))

# Also save full grid results for analysis
cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_df.to_csv(os.path.join(MODELS_DIR, "svm_gridsearch_results.csv"), index=False)

logger.info("  Accuracy  : %.2f%%", results["svm"]["accuracy"])
logger.info("  Precision : %.2f%%", results["svm"]["precision"])
logger.info("  Recall    : %.2f%%", results["svm"]["recall"])
logger.info("  F1        : %.2f%%", results["svm"]["f1"])
logger.info("  AUC       : %.4f",   results["svm"]["auc"])
logger.info("  CM        : %s",     results["svm"]["cm"])
logger.info(classification_report(y_test, yp_svm, target_names=["ham", "spam"]))
logger.info("  Saved → models/svm_pipeline.pkl")
logger.info("  Grid results → models/svm_gridsearch_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MULTI-CLASS CATEGORY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
logger.info("")
logger.info("── [3/4] Category Classifier (multi-class) ────────────")

# Train only on spam rows (ham rows are labelled 'normal' — trivially correct)
spam_mask_train = y_train == 1
spam_mask_test  = y_test  == 1

X_cat_train = X_train[spam_mask_train]
y_cat_train = ycat_train[spam_mask_train]
X_cat_test  = X_test[spam_mask_test]
y_cat_test  = ycat_test[spam_mask_test]

logger.info("  Spam train samples: %d  |  categories: %s",
            len(X_cat_train),
            dict(pd.Series(y_cat_train).value_counts()))

cat_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )),
    ("clf", CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000),
        cv=3,
    )),
])
cat_pipeline.fit(X_cat_train, y_cat_train)

yp_cat  = cat_pipeline.predict(X_cat_test)
cat_acc = accuracy_score(y_cat_test, yp_cat)

results["category"] = {
    "accuracy": round(cat_acc * 100, 2),
    "f1": round(
        f1_score(
            y_cat_test,
            yp_cat,
            average="weighted",
            zero_division=0
        ) * 100,
        2
    )
}

joblib.dump(cat_pipeline, os.path.join(MODELS_DIR, "category_pipeline.pkl"))

logger.info("  Category accuracy: %.2f%%", cat_acc * 100)
logger.info(classification_report(y_cat_test, yp_cat))
logger.info("  Saved → models/category_pipeline.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DEEP NEURAL NETWORK
# ─────────────────────────────────────────────────────────────────────────────
logger.info("")
logger.info("── [4/4] Deep Neural Network (TensorFlow/Keras) ────────")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Input, Activation
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2

    logger.info("  TensorFlow version: %s", tf.__version__)
    tf.random.set_seed(42)

    # Dedicated TF-IDF for DNN (dense float32 arrays)
    dnn_tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_tr_d = dnn_tfidf.fit_transform(X_train).toarray().astype("float32")
    X_te_d = dnn_tfidf.transform(X_test).toarray().astype("float32")
    joblib.dump(dnn_tfidf, os.path.join(MODELS_DIR, "dnn_tfidf.pkl"))

    # ── Architecture ──────────────────────────────────────────────────────
    input_dim = X_tr_d.shape[1]
    dnn = Sequential([
        Input(shape=(input_dim,)),

        Dense(512, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.40),

        Dense(256, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.35),

        Dense(128, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.30),

        Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.20),

        Dense(1, activation="sigmoid"),
    ], name="SpamShield_DNN")

    dnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    dnn.summary()

    # Class weights to handle imbalance
    neg, pos = np.bincount(y_train)
    class_weight = {0: 1.0, 1: round(neg / pos, 4)}
    logger.info("  Class weights: ham=1.0, spam=%.2f", class_weight[1])

    history = dnn.fit(
        X_tr_d, y_train,
        validation_split=0.15,
        epochs=40,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ],
        verbose=1,
    )

    ypr_dnn = dnn.predict(X_te_d, verbose=0).ravel()
    yp_dnn  = (ypr_dnn >= 0.5).astype(int)

    results["dnn"] = {
        "accuracy":       round(accuracy_score(y_test, yp_dnn) * 100, 2),
        "precision":      round(precision_score(y_test, yp_dnn, zero_division=0) * 100, 2),
        "recall":         round(recall_score(y_test, yp_dnn, zero_division=0) * 100, 2),
        "f1":             round(f1_score(y_test, yp_dnn, zero_division=0) * 100, 2),
        "auc":            round(roc_auc_score(y_test, ypr_dnn), 4),
        "cm":             confusion_matrix(y_test, yp_dnn).tolist(),
        "roc":            roc_points(y_test, ypr_dnn),
        "epochs_trained": len(history.history["loss"]),
        "architecture":   (
            "Dense(512)+BN+ReLU+Drop(0.40) → "
            "Dense(256)+BN+ReLU+Drop(0.35) → "
            "Dense(128)+BN+ReLU+Drop(0.30) → "
            "Dense(64)+L2+Drop(0.20) → Dense(1,sigmoid)"
        ),
    }

    # Save model + history
    dnn.save(os.path.join(MODELS_DIR, "dnn_model.keras"))
    hist_out = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODELS_DIR, "dnn_history.json"), "w") as fh:
        json.dump(hist_out, fh, indent=2)

    logger.info("  Accuracy  : %.2f%%", results["dnn"]["accuracy"])
    logger.info("  Precision : %.2f%%", results["dnn"]["precision"])
    logger.info("  Recall    : %.2f%%", results["dnn"]["recall"])
    logger.info("  F1        : %.2f%%", results["dnn"]["f1"])
    logger.info("  AUC       : %.4f",   results["dnn"]["auc"])
    logger.info("  Epochs    : %d",     results["dnn"]["epochs_trained"])
    logger.info("  CM        : %s",     results["dnn"]["cm"])
    logger.info(classification_report(y_test, yp_dnn, target_names=["ham", "spam"]))
    logger.info("  Saved → models/dnn_model.keras + models/dnn_history.json")

except ImportError:
    logger.warning("TensorFlow not installed — skipping DNN training")
    logger.warning("Install with: pip install tensorflow")
except Exception as exc:
    logger.error("DNN training failed: %s", exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE COMBINED RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "results.json"), "w") as fh:
    json.dump(results, fh, indent=2)

logger.info("")
logger.info("=" * 60)
logger.info("Training complete — summary")
logger.info("=" * 60)
for name, r in results.items():
    if "accuracy" in r:
        logger.info(
            "  %-14s  Acc=%5.1f%%  F1=%5.1f%%  AUC=%s",
            name, r["accuracy"], r["f1"],
            r.get("auc", "N/A")
        )
logger.info("")
logger.info("All artifacts saved to models/")
logger.info("Next step: python app.py")

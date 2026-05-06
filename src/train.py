"""
src/train.py
------------
End-to-end training script:
  1. Load / generate data
  2. Drop irrelevant raw columns
  3. Feature engineering  (FamilySize, IsAlone, Title)
  4. Train/test split
  5. sklearn Pipeline:  ColumnTransformer  →  RandomForestClassifier
  6. RandomizedSearchCV  (hyperparameter tuning)
  7. Threshold optimisation  (maximise F1 on validation set  →  fixes low Recall)
  8. Evaluate on test set
  9. Persist pipeline + metadata
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features    import TitanicFeatureEngineer
from src.preprocessor import build_preprocessor, RAW_INPUT_FEATURES
from data.dataset    import get_data

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "titanic_pipeline.pkl")
META_PATH    = os.path.join(MODEL_DIR, "metadata.json")

# Columns to drop before any modelling
DROP_COLS    = ["PassengerId", "Ticket", "Cabin"]
TARGET       = "Survived"
RANDOM_STATE = 42


# ── Data ───────────────────────────────────────────────────────────────────────
def load_dataset():
    df = get_data()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Keep only columns the pipeline uses + target
    keep = RAW_INPUT_FEATURES + [TARGET]
    keep = [c for c in keep if c in df.columns]
    df   = df[keep]

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


# ── Pipeline ───────────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Three-step sklearn Pipeline:
      step 1 — TitanicFeatureEngineer  (adds FamilySize, IsAlone, Title)
      step 2 — ColumnTransformer       (impute, scale, encode)
      step 3 — RandomForestClassifier  (tuned by RandomizedSearchCV)
    """
    return Pipeline([
        ("engineer",     TitanicFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("classifier",   RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def _param_grid() -> dict:
    return {
        "classifier__n_estimators":      [200, 300, 400, 500],
        "classifier__max_depth":         [None, 6, 8, 10, 12, 15],
        "classifier__min_samples_split": [2, 4, 6, 10],
        "classifier__min_samples_leaf":  [1, 2, 3, 4],
        "classifier__max_features":      ["sqrt", "log2", 0.4],
        "classifier__class_weight":      [None, "balanced", {0:1, 1:1.5}],
    }


# ── Threshold optimisation ─────────────────────────────────────────────────────
def _best_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Scan thresholds [0.30 … 0.70] and pick the one that maximises F1.
    Default threshold of 0.50 often under-predicts the minority class (survivors).
    """
    proba = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.50, 0.0
    for t in np.arange(0.30, 0.71, 0.01):
        preds = (proba >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2)


# ── Evaluation ────────────────────────────────────────────────────────────────
def _evaluate(model, X_test, y_test, threshold: float) -> tuple[dict, str]:
    proba  = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),         4),
        "precision": round(precision_score(y_test, y_pred),        4),
        "recall":    round(recall_score(y_test, y_pred),           4),
        "f1_score":  round(f1_score(y_test, y_pred),               4),
        "roc_auc":   round(roc_auc_score(y_test, proba),           4),
    }
    return metrics, classification_report(y_test, y_pred)


# ── Feature importance ────────────────────────────────────────────────────────
def _feature_importance(model) -> dict:
    try:
        clf        = model.named_steps["classifier"]
        pre        = model.named_steps["preprocessor"]
        feat_names = pre.get_feature_names_out()
        importances = clf.feature_importances_
        ranked = sorted(zip(feat_names, importances),
                        key=lambda x: x[1], reverse=True)
        return {k: round(float(v), 4) for k, v in ranked[:10]}
    except Exception:
        return {}


# ── Main ───────────────────────────────────────────────────────────────────────
def train(test_size: float = 0.2,
          val_size:  float = 0.1,
          n_iter:    int   = 40,
          cv:        int   = 5) -> dict:

    print("=" * 60)
    print("  TITANIC ML v2 — MODEL TRAINING")
    print("=" * 60)

    # 1. Data
    print("\n[1/6] Loading dataset …")
    X, y = load_dataset()
    print(f"      Samples        : {len(X)}")
    print(f"      Survival rate  : {y.mean():.2%}")
    print(f"      Missing Age    : {X['Age'].isna().sum()}")
    print(f"      Features (raw) : {list(X.columns)}")

    # 2. Split  →  train + val + test
    print("\n[2/6] Splitting train / val / test …")
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=RANDOM_STATE, stratify=y_tv)
    print(f"      Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

    # 3. Pipeline
    print("\n[3/6] Building pipeline …")
    pipeline = build_pipeline()

    # 4. Hyperparameter search
    print(f"\n[4/6] RandomizedSearchCV  (n_iter={n_iter}, cv={cv}) …")
    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    search   = RandomizedSearchCV(
        estimator           = pipeline,
        param_distributions = _param_grid(),
        n_iter              = n_iter,
        cv                  = cv_strat,
        scoring             = "roc_auc",
        n_jobs              = -1,
        random_state        = RANDOM_STATE,
        verbose             = 1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"\n      Best CV ROC-AUC : {search.best_score_:.4f}")
    print(f"      Best params     : {search.best_params_}")

    # 5. Threshold optimisation
    print("\n[5/6] Optimising decision threshold on validation set …")
    threshold = _best_threshold(best, X_val, y_val)
    print(f"      Optimal threshold : {threshold}  (default was 0.50)")

    # 6. Evaluate
    print("\n[6/6] Evaluating on held-out test set …")
    metrics, report = _evaluate(best, X_test, y_test, threshold)
    importance      = _feature_importance(best)

    print("\n  ┌──────────────────────────────┐")
    print(f"  │  Accuracy  : {metrics['accuracy']:.4f}            │")
    print(f"  │  Precision : {metrics['precision']:.4f}            │")
    print(f"  │  Recall    : {metrics['recall']:.4f}            │")
    print(f"  │  F1-Score  : {metrics['f1_score']:.4f}            │")
    print(f"  │  ROC-AUC   : {metrics['roc_auc']:.4f}            │")
    print(f"  │  Threshold : {threshold}              │")
    print("  └──────────────────────────────┘")
    print("\nClassification Report:\n", report)

    if importance:
        print("Top-10 feature importances:")
        for feat, imp in importance.items():
            bar = "█" * int(imp * 40)
            print(f"  {feat:<30} {imp:.4f}  {bar}")

    # Persist
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print(f"\n✅  Model saved  → {MODEL_PATH}")

    metadata = {
        **metrics,
        "threshold":   threshold,
        "cv_roc_auc":  round(search.best_score_, 4),
        "best_params": search.best_params_,
        "top_features": importance,
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅  Metadata saved → {META_PATH}")

    return metadata


if __name__ == "__main__":
    train()

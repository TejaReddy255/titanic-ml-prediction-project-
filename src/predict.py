"""
src/predict.py
--------------
Prediction module — accepts raw dict input, validates, and returns prediction.

The optimal decision threshold (found during training) is read from metadata.json.
This ensures train/inference consistency — the same threshold used at eval time
is applied at runtime.
"""

import os
import sys
import json
import joblib
import pandas as pd
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessor import RAW_INPUT_FEATURES

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_pipeline.pkl")
META_PATH  = os.path.join(MODEL_DIR, "metadata.json")

# ── Schema ─────────────────────────────────────────────────────────────────────
FIELD_TYPES: dict[str, type] = {
    "Pclass":   int,
    "Sex":      str,
    "Age":      float,
    "Fare":     float,
    "Embarked": str,
    "SibSp":    int,
    "Parch":    int,
    "Name":     str,
}

VALID_VALUES: dict[str, list] = {
    "Pclass":   [1, 2, 3],
    "Sex":      ["male", "female"],
    "Embarked": ["C", "Q", "S"],
}

# Fields that are optional (will be imputed / synthesised if absent)
OPTIONAL_FIELDS = {"Age", "Embarked", "Name", "SibSp", "Parch"}

# Minimum public-facing API fields (users don't have to supply Name / SibSp / Parch)
PUBLIC_REQUIRED = {"Pclass", "Sex", "Fare"}


# ── Singleton loaders ──────────────────────────────────────────────────────────
_model_cache: Any   = None
_meta_cache:  dict  = {}


def load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No model at {MODEL_PATH}. Run `python src/train.py` first.")
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def load_metadata() -> dict:
    global _meta_cache
    if not _meta_cache and os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _meta_cache = json.load(f)
    return _meta_cache


def get_threshold() -> float:
    return load_metadata().get("threshold", 0.50)


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_input(data: dict) -> dict:
    """
    Coerce and validate raw API input.
    Returns cleaned dict with all fields the pipeline expects.
    Raises ValueError with a descriptive message on bad input.
    """
    cleaned: dict = {}

    for field in FIELD_TYPES:
        value = data.get(field)

        # Missing field handling
        if value is None or value == "":
            if field in OPTIONAL_FIELDS:
                cleaned[field] = None
                continue
            if field in PUBLIC_REQUIRED:
                raise ValueError(f"Missing required field: '{field}'")
            cleaned[field] = None   # non-required internal field
            continue

        # Type coercion
        try:
            coerced = FIELD_TYPES[field](value)
        except (ValueError, TypeError):
            raise ValueError(
                f"'{field}' must be {FIELD_TYPES[field].__name__}, "
                f"got {type(value).__name__} ({value!r})")

        # Categorical validation
        if field in VALID_VALUES and coerced not in VALID_VALUES[field]:
            raise ValueError(
                f"'{field}' must be one of {VALID_VALUES[field]}, got {coerced!r}")

        # Range checks
        if field == "Age"      and coerced is not None and not (0 <= coerced <= 120):
            raise ValueError(f"'Age' must be 0–120, got {coerced}")
        if field == "Fare"     and coerced is not None and coerced < 0:
            raise ValueError(f"'Fare' must be ≥ 0, got {coerced}")
        if field == "SibSp"    and coerced is not None and not (0 <= coerced <= 10):
            raise ValueError(f"'SibSp' must be 0–10, got {coerced}")
        if field == "Parch"    and coerced is not None and not (0 <= coerced <= 10):
            raise ValueError(f"'Parch' must be 0–10, got {coerced}")

        cleaned[field] = coerced

    # Provide sensible defaults for optional internal fields
    if cleaned.get("SibSp") is None:
        cleaned["SibSp"] = 0
    if cleaned.get("Parch") is None:
        cleaned["Parch"] = 0
    if cleaned.get("Name") is None:
        # Synthesise a placeholder name — Title becomes "Mr" or "Mrs" from Sex
        title = "Mrs" if cleaned.get("Sex") == "female" else "Mr"
        cleaned["Name"] = f"Unknown, {title}. Unknown"

    return cleaned


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(data: dict) -> dict:
    """
    Parameters
    ----------
    data : dict  — raw API input (validated internally)

    Returns
    -------
    dict:
        prediction  : int   (0 or 1)
        probability : float (survival probability)
        label       : str   ("Survived" / "Did Not Survive")
        threshold   : float (decision threshold used)
    """
    cleaned = validate_input(data)

    # Build DataFrame with columns the pipeline expects
    input_cols = [c for c in RAW_INPUT_FEATURES if c in FIELD_TYPES]
    row        = {col: cleaned.get(col) for col in input_cols}
    df         = pd.DataFrame([row], columns=input_cols)

    model     = load_model()
    threshold = get_threshold()

    proba      = float(model.predict_proba(df)[0][1])
    prediction = int(proba >= threshold)

    return {
        "prediction":  prediction,
        "probability": round(proba, 4),
        "label":       "Survived" if prediction == 1 else "Did Not Survive",
        "threshold":   threshold,
    }


# ── CLI demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        {"Pclass": 1, "Sex": "female", "Age": 29,   "Fare": 211.34, "Embarked": "S", "SibSp": 0, "Parch": 0},
        {"Pclass": 3, "Sex": "male",   "Age": 22,   "Fare": 7.25,   "Embarked": "S", "SibSp": 1, "Parch": 0},
        {"Pclass": 2, "Sex": "female", "Age": 14,   "Fare": 30.07,  "Embarked": "C", "SibSp": 1, "Parch": 1},
        {"Pclass": 3, "Sex": "male",   "Age": None,  "Fare": 8.05},   # minimal input
    ]
    print("\n── Prediction demo ──────────────────────────────────")
    for s in samples:
        r   = predict(s)
        bar = "█" * int(r["probability"] * 20)
        print(f"\n  Input : {s}")
        print(f"  → {r['label']}  prob={r['probability']:.2%}  {bar}")

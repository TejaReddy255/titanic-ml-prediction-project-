"""
src/preprocessor.py
-------------------
ColumnTransformer that handles all preprocessing after feature engineering.

Features used (post-engineering):
  Numeric     : Age, Fare, FamilySize
  Binary      : IsAlone                   (pass-through — already 0/1)
  Categorical : Pclass, Sex, Embarked, Title

Columns dropped (with justification):
  PassengerId : arbitrary row ID — zero signal
  Ticket      : unstructured alphanumeric — no consistent pattern to learn
  Cabin       : 77%+ missing; deck letter could help but adds fragility
  Name        : consumed by TitanicFeatureEngineer → Title
  SibSp/Parch : consumed by TitanicFeatureEngineer → FamilySize, IsAlone
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ── Feature lists (post-engineering column names) ──────────────────────────────
NUMERIC_FEATURES     = ["Age", "Fare", "FamilySize"]
BINARY_FEATURES      = ["IsAlone"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title"]

# Columns the model pipeline receives (raw API input + engineered)
RAW_INPUT_FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked", "SibSp", "Parch", "Name"]

# All post-engineering features (subset that enters the ColumnTransformer)
ALL_MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def _numeric_pipeline() -> Pipeline:
    """Median imputation (robust to outliers) → z-score scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def _categorical_pipeline() -> Pipeline:
    """
    Most-frequent imputation → OneHotEncoding.
    handle_unknown='ignore' ensures safety on unseen categories at inference.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor() -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer.
    Binary feature (IsAlone) passes through unchanged — it's already 0/1.
    """
    return ColumnTransformer(
        transformers=[
            ("num",     _numeric_pipeline(),    NUMERIC_FEATURES),
            ("bin",     "passthrough",           BINARY_FEATURES),
            ("cat",     _categorical_pipeline(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

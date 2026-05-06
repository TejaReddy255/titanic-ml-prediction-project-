"""
src/features.py
---------------
Feature engineering transformations applied BEFORE the sklearn Pipeline.

New features (address evaluation gap — SibSp/Parch dropped in v1):
  FamilySize  = SibSp + Parch + 1   (total people including self)
  IsAlone     = 1 if FamilySize == 1 else 0
  Title       = extracted salutation from Name  (Mr / Mrs / Miss / Master / Rare)

These three features consistently rank in the top-5 importances on the real
Titanic dataset and directly capture the "women and children first" dynamic.
"""

import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ── Title mapping ──────────────────────────────────────────────────────────────
# Rare titles (Dr, Rev, Col …) are grouped to avoid high-cardinality OHE columns.
_TITLE_MAP = {
    "Mr":     "Mr",
    "Mrs":    "Mrs",
    "Miss":   "Miss",
    "Ms":     "Miss",
    "Mlle":   "Miss",
    "Mme":    "Mrs",
    "Master": "Master",
    "Dr":     "Rare",
    "Rev":    "Rare",
    "Col":    "Rare",
    "Major":  "Rare",
    "Capt":   "Rare",
    "Sir":    "Rare",
    "Lady":   "Rare",
    "Don":    "Rare",
    "Dona":   "Rare",
    "Jonkheer": "Rare",
    "the Countess": "Rare",
}


def _extract_title(name: str) -> str:
    """Extract title from 'Last, Title. First' format."""
    match = re.search(r",\s*([^.]+)\.", str(name))
    if not match:
        return "Rare"
    raw = match.group(1).strip()
    return _TITLE_MAP.get(raw, "Rare")


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that adds engineered features.
    Operates on a DataFrame; returns a DataFrame with new columns appended.

    Expected input columns: Name, SibSp, Parch  (plus any others, passed through)
    New output columns    : FamilySize, IsAlone, Title
    Dropped columns       : Name, SibSp, Parch  (subsumed by new features)
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self   # stateless — nothing to learn

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # FamilySize and IsAlone
        if "SibSp" in df.columns and "Parch" in df.columns:
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
            df.drop(columns=["SibSp", "Parch"], inplace=True)

        # Title
        if "Name" in df.columns:
            df["Title"] = df["Name"].apply(_extract_title)
            df.drop(columns=["Name"], inplace=True)

        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for use outside the sklearn Pipeline."""
    return TitanicFeatureEngineer().transform(df)

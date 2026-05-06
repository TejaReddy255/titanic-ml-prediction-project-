"""
data/dataset.py
---------------
Titanic dataset loader.
Loads data/titanic.csv — the real Kaggle dataset.
"""

import os
import pandas as pd

DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DATA_DIR, "titanic.csv")


def get_data() -> pd.DataFrame:
    """
    Load the Titanic dataset from data/titanic.csv.
    Raises FileNotFoundError if the file is missing.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Please place the Kaggle titanic.csv file in the data/ folder."
        )

    print(f"[data] Loaded dataset from {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    df = get_data()
    print(df.head(3).to_string())
    print(f"\nShape       : {df.shape}")
    print(f"Missing Age : {df['Age'].isna().sum()}")
    print(f"Survival    : {df['Survived'].mean():.2%}")

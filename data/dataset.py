"""
data/dataset.py
---------------
Titanic dataset loader.

REAL DATA (recommended):
  Drop the official Kaggle CSV at data/titanic.csv  →  it will be used automatically.
  Download from: https://www.kaggle.com/c/titanic/data  (train.csv → rename to titanic.csv)

SYNTHETIC FALLBACK:
  If data/titanic.csv is absent, a 891-row synthetic dataset is generated that
  faithfully reproduces the real dataset's class/sex survival statistics.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(DATA_DIR, "titanic.csv")

# ── Real Titanic survival rates (source: Kaggle ground truth) ─────────────────
# Used to calibrate the synthetic generator so survival distributions match.
_SURVIVAL_RATES = {
    (1, "female"): 0.968, (1, "male"): 0.369,
    (2, "female"): 0.921, (2, "male"): 0.157,
    (3, "female"): 0.500, (3, "male"): 0.135,
}

_TITLES = ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Col", "Major", "Sir"]
_TITLE_WEIGHTS = [0.582, 0.196, 0.131, 0.046, 0.016, 0.012, 0.008, 0.006, 0.003]


def _generate(n: int = 891, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Titanic dataset calibrated to real survival rates.
    Column schema matches the Kaggle train.csv exactly.
    """
    rng = np.random.RandomState(seed)

    pclass = rng.choice([1, 2, 3], n, p=[0.242, 0.206, 0.552])
    sex    = rng.choice(["male", "female"], n, p=[0.647, 0.353])

    # Age: bimodal — children cluster around 4, adults around 30
    child_mask = rng.random(n) < 0.08
    age_raw    = np.where(child_mask,
                          np.clip(rng.normal(4, 2, n), 0.17, 14),
                          np.clip(rng.normal(30, 13, n), 15, 80))
    age = np.where(rng.random(n) < 0.198, np.nan, np.round(age_raw, 1))

    sibsp = rng.choice([0,1,2,3,4,5,8], n, p=[0.682,0.234,0.031,0.018,0.018,0.011,0.006])
    parch = rng.choice([0,1,2,3,4,5,6], n, p=[0.760,0.132,0.089,0.009,0.005,0.002,0.003])

    fare_base = np.where(pclass==1, rng.exponential(60,n) + 25,
                np.where(pclass==2, rng.exponential(14,n) + 10,
                                    rng.exponential(8, n) +  5))
    fare = np.clip(np.round(fare_base, 4), 5, 512)

    emb_raw = rng.choice(["S","C","Q"], n, p=[0.725,0.188,0.087]).astype(object)
    emb_raw[rng.random(n) < 0.002] = None
    embarked = emb_raw

    cabin_raw = [f"{rng.choice(list('ABCDEF'))}{rng.randint(10,150)}"
                 for _ in range(n)]
    cabin = np.where(rng.random(n) < 0.771, None, cabin_raw)

    titles = rng.choice(_TITLES, n, p=_TITLE_WEIGHTS)
    last   = [f"Passenger_{i}" for i in range(n)]
    name   = [f"{last[i]}, {titles[i]}. Synthetic_{i}" for i in range(n)]

    # Survival — calibrated per (Pclass, Sex) cell
    survived = np.zeros(n, dtype=int)
    for (pc, sx), rate in _SURVIVAL_RATES.items():
        mask = (pclass == pc) & (sex == sx)
        n_cell = mask.sum()
        if n_cell:
            noise = rng.normal(0, 0.04, n_cell)
            p = np.clip(rate + noise, 0.05, 0.98)
            survived[mask] = (rng.random(n_cell) < p).astype(int)

    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived":    survived,
        "Pclass":      pclass,
        "Name":        name,
        "Sex":         sex,
        "Age":         age,
        "SibSp":       sibsp,
        "Parch":       parch,
        "Ticket":      [f"T{rng.randint(1000,9999)}" for _ in range(n)],
        "Fare":        fare,
        "Cabin":       cabin,
        "Embarked":    embarked,
    })


def get_data() -> pd.DataFrame:
    """
    Load the Titanic dataset.
    Uses data/titanic.csv if present (real Kaggle data), otherwise generates
    a calibrated synthetic dataset and saves it for subsequent calls.
    """
    if os.path.exists(DATA_PATH):
        print(f"[data] Loaded real dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("[data] titanic.csv not found — generating calibrated synthetic dataset")
    print("[data] TIP: replace data/titanic.csv with the real Kaggle train.csv for best results")
    df = _generate()
    df.to_csv(DATA_PATH, index=False)
    rate = df["Survived"].mean()
    print(f"[data] Generated {len(df)} rows  |  survival rate: {rate:.2%}")
    return df


if __name__ == "__main__":
    df = get_data()
    print(df.head(3).to_string())
    print(f"\nShape       : {df.shape}")
    print(f"Missing Age : {df['Age'].isna().sum()}")
    print(f"Survival    : {df['Survived'].mean():.2%}")
